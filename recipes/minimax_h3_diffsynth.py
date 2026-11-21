#!/usr/bin/env python3
"""Colab Ultra — Reusable High-Throughput MiniMax-H3-Image Batch Runner (L4 / G4 / A100).

Drop a `prompts.json` into your project's `code/` directory and run directly without writing new code:
    colab-ultra exec -s l4 -p <project> -f ~/.gemini/config/skills/colab-ultra/recipes/minimax_h3_diffsynth.py

Minimal `prompts.json` schema:
{
  "run_name": "batch01",
  "width": 2048,
  "height": 2048,
  "seed_base": 100,
  "dit_variant": "dit_4p5bit_int8",
  "compile_dit": true,
  "vae_batch_size": 2,
  "prompts": [
    {"slug": "01_scene", "prompt": "Your prompt here..."}
  ]
}

Image-conditioned edit ("edit"/restyle) is supported per item: add `"edit_image": "<path or [paths]>"` (relative to
the project root / code / data / runs, or absolute) and optionally per-item `width`/`height`/`seed`/`output_format`
(`jpg` default, `png`, or lossless `webp`; set top-level `"output_format"` to change the default). Edit items run the
full DiffSynth unit chain (vision tower + reference latents); plain items keep the fast batched text-encoder path.
  {"slug": "tile_r1_c2", "edit_image": "runs/v5_16k_base/tile_r1_c2.webp", "width": 4096, "height": 4096,
   "output_format": "webp", "prompt": "Restyle <Picture 1> ..."}

Verified L4 baseline (10x 2048x2048, dit_4p5bit_int8 + compile + batched TE/VAE):
  - Steady-state inference: ~24.5s/img (~298s total for 10 images)
  - Total end-to-end (cold up -> 10 images -> down): ~10.5 min, ~0.28 CU (~0.028 CU/img)
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
try:
    from cu_remote import recipe, stage, vram_policy
except ModuleNotFoundError as exc:
    if exc.name != "cu_remote":
        raise
    # Source checkout for offline dry runs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from colab_ultra.remote import recipe, stage, vram_policy

PROCESSOR_FILES = (
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def run_cmd(cmd: list[str], check: bool = True) -> int:
    print(f"[Cmd] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check).returncode


def move_tensors(obj, device: str):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_tensors(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_tensors(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_tensors(v, device) for k, v in obj.items()}
    return obj


def main() -> None:
    cfg, cwd, project_root = recipe.load_project()
    run_name = cfg.get("run_name", time.strftime("h3_%Y%m%d_%H%M%S"))
    width = int(cfg.get("width", 2048))
    height = int(cfg.get("height", 2048))
    seed_base = int(cfg.get("seed_base", 100))
    dit_variant = str(cfg.get("dit_variant", "dit_4p5bit_int8"))
    compile_dit = bool(cfg.get("compile_dit", True))
    vae_batch_size = max(1, int(cfg.get("vae_batch_size", 2)))
    use_sage = bool(cfg.get("use_sage_attention", False))
    items = cfg["prompts"]
    default_fmt = str(cfg.get("output_format", "jpg")).lower().lstrip(".")
    out_formats = {
        it.get("slug", f"img_{i + 1:02d}"): str(it.get("output_format", default_fmt)).lower().lstrip(".")
        for i, it in enumerate(items)
    }

    out_dir = project_root / "runs" / run_name

    vae_fp32 = bool(cfg.get("vae_fp32", False))  # full-precision original VAE (reduces 16px lattice artifacts)
    use_vae_int8 = not vae_fp32
    dit_nf4 = bool(cfg.get("dit_nf4", False))  # bf16 DiT + bitsandbytes NF4 (4-bit weight-only, bf16 activations); no EntroPack for the DiT
    diffsynth_repo = str(cfg.get("diffsynth_repo", "https://github.com/lgyStoic/DiffSynth-Studio.git"))
    diffsynth_branch = str(cfg.get("diffsynth_branch", "t4-sm75-int8-support"))

    # Pick largest ephemeral local disk (/content on Colab, /tmp on Kaggle where /kaggle/working is ~19 GB)
    content_free_gb = shutil.disk_usage("/content").free / (1024 ** 3) if os.path.exists("/content") else 0.0
    tmp_free_gb = shutil.disk_usage("/tmp").free / (1024 ** 3) if os.path.exists("/tmp") else 0.0
    nvme_root = Path("/content/models") if content_free_gb >= 35.0 or content_free_gb >= tmp_free_gb else Path("/tmp/models")
    print(f"[Disk] staging root={nvme_root} (/content free={content_free_gb:.1f} GiB, /tmp free={tmp_free_gb:.1f} GiB)", flush=True)

    staging_err: list[Exception] = []
    stage_elapsed = [0.0]

    vae_hf_entry = (
        (
            "DiffSynth-Studio/MiniMax-H3-Image-EntroPack/models",
            "vae_int8.safetensors",
            "https://huggingface.co/DiffSynth-Studio/MiniMax-H3-Image-EntroPack/resolve/main/models/vae_int8.safetensors",
        )
        if use_vae_int8
        else (
            "MiniMax/MiniMax-H3-Image/vae",
            "diffusion_pytorch_model.safetensors",
            "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/vae/diffusion_pytorch_model.safetensors",
        )
    )

    # Prefer HF GCP CDN for all public weights + processor (270-450 MB/s, zero project coupling)
    dit_file = "bf16_table_pruned" if dit_nf4 else dit_variant
    hf_urls = [
        (
            "DiffSynth-Studio/MiniMax-H3-Image-EntroPack/models",
            f"{dit_file}.safetensors",
            f"https://huggingface.co/DiffSynth-Studio/MiniMax-H3-Image-EntroPack/resolve/main/models/{dit_file}.safetensors",
        ),
        (
            "DiffSynth-Studio/MiniMax-H3-Image-EntroPack/models",
            f"{dit_file}.json",
            f"https://huggingface.co/DiffSynth-Studio/MiniMax-H3-Image-EntroPack/resolve/main/models/{dit_file}.json",
        ),
        vae_hf_entry,
        (
            "MiniMax/MiniMax-H3-Image",
            "pdd_fused_heads.safetensors",
            "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/pdd_fused_heads.safetensors",
        ),
        (
            "DiffSynth-Studio/EntroPackPreQuants/models/MiniMax-H3",
            "text_encoder_4bpp.safetensors",
            "https://huggingface.co/DiffSynth-Studio/EntroPackPreQuants/resolve/main/models/MiniMax-H3/text_encoder_4bpp.safetensors",
        ),
        (
            "DiffSynth-Studio/EntroPackPreQuants/models/MiniMax-H3",
            "text_encoder_4bpp.json",
            "https://huggingface.co/DiffSynth-Studio/EntroPackPreQuants/resolve/main/models/MiniMax-H3/text_encoder_4bpp.json",
        ),
    ] + [
        (
            "MiniMax/MiniMax-H3-Image/processor",
            fname,
            f"https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/processor/{fname}",
        )
        for fname in PROCESSOR_FILES
    ]

    if dit_nf4:
        hf_urls = [u for u in hf_urls if u[1] != f"{dit_file}.json"]  # bf16 DiT has no EntroPack sidecar

    plans = [stage.plan(url, nvme_root / folder / name,
                        min_bytes=stage.MIN_BYTES if name.endswith(".safetensors") else 0)
             for folder, name, url in hf_urls]
    if recipe.dry_run():
        recipe.describe(cfg, run_name)
        print("git clone -b", diffsynth_branch, diffsynth_repo)
        for item in plans:
            print(item.backend, item.source, "->", item.dest)
        print(f"MiniMaxH3ImagePipeline: vae_fp32={vae_fp32} dit_nf4={dit_nf4} -> {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import torch

    gpu_major, gpu_minor = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    is_pre_ampere = torch.cuda.is_available() and gpu_major < 8
    run_dtype = torch.bfloat16
    if is_pre_ampere:
        # lgyStoic/DiffSynth-Studio@t4-sm75-int8-support natively enables cuBLAS INT8 Tensor Core
        # & MemEfficient FP16 SDPA on Turing (sm_75 / Tesla T4) while keeping bfloat16 outer activations.
        compile_dit = False

    print(
        f"[Start] {time.strftime('%Y-%m-%d %H:%M:%S')} | project={project_root.name} | run={run_name} | "
        f"gpu={gpu_name} (sm_{gpu_major}{gpu_minor}, dtype={run_dtype}) | count={len(items)} | "
        f"res={width}x{height} | dit={dit_variant} | vae_int8={use_vae_int8} | compile={compile_dit} | vae_bs={vae_batch_size}",
        flush=True,
    )

    def _bg_stage() -> None:
        try:
            t_s = time.time()
            stage.run_many(plans)
            stage_elapsed[0] = time.time() - t_s
        except Exception as exc:
            staging_err.append(exc)

    t_stage = threading.Thread(target=_bg_stage, daemon=True)
    t_stage.start()

    # Main thread: clone DiffSynth-Studio + install deps via uv (overlapped with NVMe download)
    t_setup_start = time.time()
    repo_dir = Path("/content/DiffSynth-Studio")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run_cmd(["git", "clone", "--depth", "1", "-b", diffsynth_branch, diffsynth_repo, str(repo_dir)])

    pip_pkgs = [
        "transformers>=4.57.0",
        "entropack>=0.1.7",
        *(['bitsandbytes'] if dit_nf4 else []),
        "modelscope>=1.18.0",
        "huggingface_hub>=0.25.0",
        "ftfy",
        "safetensors",
        "einops",
        "accelerate",
        "peft",
        "sentencepiece",
        "protobuf",
        "imageio",
        "pillow",
    ]
    uv_bin = shutil.which("uv")
    if uv_bin:
        run_cmd([uv_bin, "pip", "install", "--system", "-q", *pip_pkgs])
        run_cmd([uv_bin, "pip", "install", "--system", "-q", "--no-deps", "-e", str(repo_dir)])
        if use_sage:
            run_cmd([uv_bin, "pip", "install", "--system", "-q", "sageattention==1.0.6"], check=False)
    else:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", *pip_pkgs])
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo_dir)])
        if use_sage:
            run_cmd([sys.executable, "-m", "pip", "install", "-q", "sageattention==1.0.6"], check=False)

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    importlib.invalidate_caches()
    t_setup = time.time() - t_setup_start
    print(f"[Timer] setup_env (overlapped): {t_setup:.2f}s", flush=True)

    active_attn = "torch"
    if use_sage:
        try:
            import sageattention  # noqa: F401
            os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "sage_attention"
            active_attn = "sage_attention"
        except Exception as exc:
            print(f"[Warn] sageattention unavailable ({exc}), falling back to torch SDPA", flush=True)
    print(f"[Attn] implementation={active_attn}", flush=True)

    t_stage.join()
    if staging_err:
        raise staging_err[0]
    t_prep_wall = time.time() - t0
    print(f"[Timer] nvme_stage (background): {stage_elapsed[0]:.2f}s | wall_prep_total: {t_prep_wall:.2f}s", flush=True)

    import torch
    from diffsynth.configs import MODEL_CONFIGS
    from diffsynth.pipelines.minimax_h3_image import (
        MINIMAX_H3_IMAGE_PIXEL_MEAN,
        MINIMAX_H3_IMAGE_PIXEL_STD,
        MINIMAX_H3_IMAGE_TEXT_TAG,
        MiniMaxH3ImagePipeline,
        MiniMaxH3ImageUnit_PromptEmbedder,
        ModelConfig,
    )
    from PIL import Image
    from tqdm import tqdm
    dit_path = nvme_root / f"DiffSynth-Studio/MiniMax-H3-Image-EntroPack/models/{dit_file}.safetensors"
    te_path = nvme_root / "DiffSynth-Studio/EntroPackPreQuants/models/MiniMax-H3/text_encoder_4bpp.safetensors"
    pdd_path = nvme_root / "MiniMax/MiniMax-H3-Image/pdd_fused_heads.safetensors"
    vae_path = (
        nvme_root / "DiffSynth-Studio/MiniMax-H3-Image-EntroPack/models/vae_int8.safetensors"
        if use_vae_int8
        else nvme_root / "MiniMax/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors"
    )
    proc_dir = nvme_root / "MiniMax/MiniMax-H3-Image/processor"

    comp_sizes_gb = {
        "dit": (11.5 if dit_nf4 else dit_path.stat().st_size / (1024 ** 3)),  # NF4 resident size ~11.5 GiB
        "text_encoder": te_path.stat().st_size / (1024 ** 3),
        "pdd_heads": pdd_path.stat().st_size / (1024 ** 3),
        "vae": vae_path.stat().st_size / (1024 ** 3),
    }
    plan = vram_policy.build_vram_plan(comp_sizes_gb, resolution=(width, height), dtype=run_dtype)
    print(vram_policy.format_vram_plan(plan), flush=True)
    vae_tiling = bool(cfg.get("vae_tiling", plan["vae_tiling"]))

    def register_local_prequantized(safetensors_path: Path, vram_cfg: dict) -> ModelConfig:
        sidecar_path = safetensors_path.with_suffix(".json")
        with open(sidecar_path, encoding="utf-8") as handle:
            entry = json.load(handle)
        if not any(config.get("model_hash") == entry["model_hash"] for config in MODEL_CONFIGS):
            MODEL_CONFIGS.append(entry)
        return ModelConfig(path=str(safetensors_path), **vram_cfg)

    t_load_start = time.time()
    pipe = MiniMaxH3ImagePipeline.from_pretrained(
        torch_dtype=run_dtype,
        device="cuda",
        model_configs=[
            (ModelConfig(path=str(dit_path), **plan["configs"]["dit"], quantize=__import__("diffsynth.core.quant", fromlist=["QuantizeConfig"]).QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=["video_patch_proj", "condition_proj", "token_refiner.blocks.0.attn.qkv_proj", "token_refiner.blocks.0.attn.out_proj", "token_refiner.blocks.0.mlp.fc1", "token_refiner.blocks.0.mlp.fc2", "token_refiner.blocks.1.attn.qkv_proj", "token_refiner.blocks.1.attn.out_proj", "token_refiner.blocks.1.mlp.fc1", "token_refiner.blocks.1.mlp.fc2"])) if dit_nf4 else register_local_prequantized(dit_path, plan["configs"]["dit"])),
            register_local_prequantized(te_path, plan["configs"]["text_encoder"]),
            ModelConfig(path=str(pdd_path), **plan["configs"]["pdd_heads"]),
            ModelConfig(path=str(vae_path), **({**plan["configs"]["vae"], **{k: torch.float32 for k in ("offload_dtype", "onload_dtype", "preparing_dtype", "computation_dtype")}} if vae_fp32 else plan["configs"]["vae"])),
        ],
        processor_config=ModelConfig(path=str(proc_dir)),
        vram_limit=plan["vram_limit"],
    )
    t_load = time.time() - t_load_start
    if vae_fp32:
        print("[VAE] fp32 mode: casting VAE inputs to float32", flush=True)
        _enc, _dec = pipe.vae.encode, pipe.vae.decode
        pipe.vae.encode = lambda x, *a, **k: _enc(x.float(), *a, **k)
        pipe.vae.decode = lambda z, *a, **k: _dec(z.float(), *a, **k)
    print(f"[Timer] load_pipeline (NVMe): {t_load:.2f}s", flush=True)

    # Stage 1: Text encoding. Pure text-to-image items share ONE batched forward pass over the TextEncoder.
    # Items with `edit_image` (image-conditioned restyle / "edit") need the vision tower + reference latents, so they
    # run the full pipeline unit chain one by one (EditImageResize -> PromptEmbedder(vision) -> ReferenceEncoder ...).
    t_stage1_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    prepared = []

    def _load_edit_images(spec):
        specs = spec if isinstance(spec, (list, tuple)) else [spec]
        return [Image.open(recipe.resolve_image(x, cwd, project_root)).convert("RGB") for x in specs]

    with torch.no_grad():
        pipe.load_models_to_device(["text_encoder"])
        text_idx = [i for i, it in enumerate(items) if not it.get("edit_image")]
        hidden = None
        token_lists = {}
        if text_idx:
            raw_texts = {i: (items[i]["prompt"] if items[i]["prompt"] else " ") for i in text_idx}
            token_lists = {
                i: pipe.tokenizer(raw_texts[i], add_special_tokens=False)["input_ids"][:4096] for i in text_idx
            }
            max_len = max(len(ids) for ids in token_lists.values())
            pad_id = pipe.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = pipe.tokenizer.eos_token_id or 0
            input_ids = torch.full((len(text_idx), max_len), pad_id, dtype=torch.long, device=pipe.device)
            attention_mask = torch.zeros((len(text_idx), max_len), dtype=torch.long, device=pipe.device)
            for row, i in enumerate(text_idx):
                ids = token_lists[i]
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=pipe.device)
                attention_mask[row, : len(ids)] = 1
            hidden = pipe.text_encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            row_of = {i: row for row, i in enumerate(text_idx)}
            del input_ids, attention_mask
        else:
            row_of = {}

        non_te_units = [u for u in pipe.units if not isinstance(u, MiniMaxH3ImageUnit_PromptEmbedder)]
        pipe.scheduler.set_timesteps(6, distilled=True)
        for i, item in enumerate(items):
            idx = i + 1
            slug = item.get("slug", f"img_{idx:02d}")
            i_w = int(item.get("width", width))
            i_h = int(item.get("height", height))
            inputs_shared = {
                "cfg_scale": 1.0,
                "edit_image": None,
                "height": i_h,
                "width": i_w,
                "min_generation_area": 2048 * 2048,
                "max_sequence_length": 4096,
                "seed": int(item.get("seed", seed_base + idx)),
                "rand_device": "cpu",
                "latents": None,
                "reference_noise": None,
                "tiling": bool(item.get("vae_tiling", vae_tiling or max(i_w, i_h) > 2048)),
            }
            if item.get("edit_image"):
                inputs_shared["edit_image"] = _load_edit_images(item["edit_image"])
                # reference-latent encoding uses the VAE, and the prompt unit needs the text encoder/vision tower
                pipe.load_models_to_device(["text_encoder", "vae"])
                inputs_posi = {"prompt": item["prompt"]}
                inputs_nega = {"negative_prompt": ""}
                for unit in pipe.units:  # full chain incl. PromptEmbedder with vision tokens
                    inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                        unit, pipe, inputs_shared, inputs_posi, inputs_nega
                    )
                print(f"[Stage1 Edit {idx:02d}/{len(items)}] {slug} | refs={len(inputs_shared['edit_image'])} | out={i_w}x{i_h}", flush=True)
            else:
                ids = token_lists[i]
                prompt_embeds = hidden[row_of[i], : len(ids)].to(dtype=pipe.torch_dtype)
                text_token_tags = torch.full((len(ids),), MINIMAX_H3_IMAGE_TEXT_TAG, dtype=torch.long)
                inputs_posi = {
                    "prompt": item["prompt"],
                    "prompt_embeds": prompt_embeds,
                    "text_token_tags": text_token_tags,
                }
                inputs_nega = {"negative_prompt": ""}
                for unit in non_te_units:
                    inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                        unit, pipe, inputs_shared, inputs_posi, inputs_nega
                    )
            inputs_shared.pop("edit_image", None)  # PIL refs are no longer needed after reference latents exist
            prepared.append((
                slug,
                move_tensors(inputs_shared, "cpu"),
                move_tensors(inputs_posi, "cpu"),
                move_tensors(inputs_nega, "cpu"),
            ))
        del hidden
        pipe.load_models_to_device([])
        torch.cuda.empty_cache()

    t_stage1 = time.time() - t_stage1_start
    peak_s1 = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[Timer] stage1_encode_batch({len(items)}): {t_stage1:.2f}s ({t_stage1/len(items):.2f}s/img) | peak_vram={peak_s1:.2f} GiB", flush=True)

    # Stage 2: GPU-Resident DiT + PDD (with optional torch.compile)
    t_stage2_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    latents_out = []
    denoise_secs = []
    compile_active = False
    with torch.no_grad():
        pipe.load_models_to_device(pipe.in_iteration_models)
        if compile_dit:
            try:
                pipe.compile(mode="default", dynamic=True, fullgraph=False)
                compile_active = True
                print("[Compile] torch.compile(pipe.dit, mode='default', dynamic=True) enabled", flush=True)
            except Exception as exc:
                print(f"[Warn] pipe.compile failed ({exc}), running eager", flush=True)

        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for idx, (slug, s_cpu, p_cpu, n_cpu) in enumerate(prepared, start=1):
            t_d = time.time()
            pipe.scheduler.set_timesteps(6, distilled=True)
            inputs_shared = move_tensors(s_cpu, pipe.device)
            inputs_posi = move_tensors(p_cpu, pipe.device)
            inputs_nega = move_tensors(n_cpu, pipe.device)
            for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc=f"DiT {idx:02d}/{len(items)}")):
                timestep = timestep.to(dtype=torch.float32)
                noise_pred = pipe.cfg_guided_model_fn(
                    pipe.model_fn,
                    1.0,
                    inputs_shared,
                    inputs_posi,
                    inputs_nega,
                    **models,
                    timestep=timestep,
                )
                inputs_shared["latents"] = pipe.step(
                    pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared
                )
            dt = time.time() - t_d
            denoise_secs.append(dt)
            latents_out.append((
                slug,
                inputs_shared["latents"].to("cpu"),
                inputs_shared["output_width"],
                inputs_shared["output_height"],
            ))
            print(f"[Stage2 Denoise {idx:02d}/{len(items)}] {slug} | {dt:.2f}s", flush=True)

        pipe.load_models_to_device([])
        torch.cuda.empty_cache()
    t_stage2 = time.time() - t_stage2_start
    peak_s2 = torch.cuda.max_memory_allocated() / (1024 ** 3)
    steady_s2 = sum(denoise_secs[1:]) / max(1, len(denoise_secs) - 1) if len(denoise_secs) > 1 else denoise_secs[0]
    print(
        f"[Timer] stage2_denoise_all: {t_stage2:.2f}s (avg={t_stage2/len(items):.2f}s/img, steady={steady_s2:.2f}s/img) | "
        f"peak_vram={peak_s2:.2f} GiB | compiled={compile_active}",
        flush=True,
    )

    # Stage 3: Batched Full-Frame VAE Decode (with automatic OOM fallback)
    t_stage3_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    results = []
    with torch.no_grad():
        pipe.load_models_to_device(["vae"])
        latents_mean = pipe.vae.latents_mean.to(device=pipe.device, dtype=torch.float32).view(1, -1, 1, 1)
        latents_std = pipe.vae.latents_std.to(device=pipe.device, dtype=torch.float32).view(1, -1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_IMAGE_PIXEL_MEAN, device=pipe.device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_IMAGE_PIXEL_STD, device=pipe.device).view(1, -1, 1, 1)

        def _decode_and_save_batch(batch_items, start_idx: int, use_tiling: bool) -> None:
            t_b = time.time()
            z = torch.cat([x[1][:, :, 0] for x in batch_items], dim=0).to(device=pipe.device, dtype=torch.float32)
            z = z * latents_std + latents_mean
            imgs = pipe.vae.decode(z, tiling=use_tiling)
            imgs = (imgs.float() * pixel_std + pixel_mean).clamp(0, 1)
            dt_b = (time.time() - t_b) / len(batch_items)
            for j, (slug, _, out_w, out_h) in enumerate(batch_items):
                pil_img = pipe.vae_output_to_image(imgs[j : j + 1, :, None], pattern="B C T H W", min_value=0, max_value=1)
                if pil_img.size != (out_w, out_h):
                    pil_img = pil_img.resize((out_w, out_h), Image.LANCZOS)
                fmt = out_formats.get(slug, "jpg")
                out_path = out_dir / f"{slug}.{fmt}"
                if fmt == "webp":
                    pil_img.save(str(out_path), lossless=True, quality=100, method=4)
                elif fmt == "png":
                    pil_img.save(str(out_path), compress_level=3)
                else:
                    pil_img.save(str(out_path), quality=95)
                sz_kb = out_path.stat().st_size / 1024.0
                idx_global = start_idx + j
                results.append({
                    "slug": slug,
                    "denoise_sec": round(denoise_secs[idx_global], 2),
                    "decode_sec": round(dt_b, 2),
                    "size_kb": round(sz_kb, 1),
                })
                print(
                    f"[Stage3 Decode {idx_global + 1:02d}/{len(items)}] {slug}.jpg | {dt_b:.2f}s/img "
                    f"(bs={len(batch_items)}, tiling={use_tiling}) | {sz_kb:.1f} KB",
                    flush=True,
                )

        i = 0
        eff_bs = vae_batch_size
        eff_tiling = vae_tiling
        while i < len(latents_out):
            chunk = latents_out[i : i + eff_bs]
            try:
                _decode_and_save_batch(chunk, i, eff_tiling)
                i += len(chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if eff_bs > 1:
                    print(f"[Warn] VAE OOM at bs={eff_bs}, falling back to bs=1", flush=True)
                    eff_bs = 1
                elif not eff_tiling:
                    print("[Warn] VAE OOM at tiling=False, falling back to tiling=True", flush=True)
                    eff_tiling = True
                else:
                    raise

        pipe.load_models_to_device([])
        torch.cuda.empty_cache()
    t_stage3 = time.time() - t_stage3_start
    peak_s3 = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[Timer] stage3_decode_all: {t_stage3:.2f}s ({t_stage3/len(items):.2f}s/img) | peak_vram={peak_s3:.2f} GiB", flush=True)

    os.sync()
    t_infer_total = t_stage1 + t_stage2 + t_stage3
    t_total = time.time() - t0
    summary = {
        "project": project_root.name,
        "run_name": run_name,
        "dit_variant": dit_variant,
        "compiled_dit": compile_active,
        "vae_tiling": eff_tiling,
        "vae_batch_size": eff_bs,
        "attention_backend": active_attn,
        "vram_policy": plan["mode"],
        "setup_env_sec": round(t_setup, 2),
        "nvme_stage_sec": round(stage_elapsed[0], 2),
        "wall_prep_sec": round(t_prep_wall, 2),
        "load_pipeline_sec": round(t_load, 2),
        "stage1_encode_all_sec": round(t_stage1, 2),
        "stage2_denoise_all_sec": round(t_stage2, 2),
        "stage2_steady_sec": round(steady_s2, 2),
        "stage3_decode_all_sec": round(t_stage3, 2),
        "batch_imgs_sec": round(t_infer_total, 2),
        "avg_per_img_sec": round(t_infer_total / len(items), 2),
        "peak_vram_gib": {"stage1": round(peak_s1, 2), "stage2": round(peak_s2, 2), "stage3": round(peak_s3, 2)},
        "total_script_sec": round(t_total, 2),
        "images": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.sync()
    print(
        f"[Summary] dit={dit_variant} compiled={compile_active} | prep={t_prep_wall:.2f}s | load={t_load:.2f}s | "
        f"infer_{len(items)}imgs={t_infer_total:.2f}s (S1={t_stage1:.1f}s + S2={t_stage2:.1f}s[steady={steady_s2:.1f}s] + S3={t_stage3:.1f}s, "
        f"avg={t_infer_total/len(items):.2f}s/img) | total_script={t_total:.2f}s",
        flush=True,
    )
    print("[Done] minimax_h3_batch_finished_ok", flush=True)


if __name__ == "__main__":
    main()
