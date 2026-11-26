#!/usr/bin/env python3
"""Generate or edit MiniMax-H3 images with the standalone inference library.

Run `colab-ultra sync-remote`, then put `prompts.json` in the project's `code/`
directory and run:
    colab-ultra exec -s l4 -p <project> -f recipes/minimax_h3_infer.py

Example prompts.json:
{
  "run_name": "batch01", "width": 2048, "height": 2048, "skip_existing": true,
  "prompts": [
    {"slug": "boat", "prompt": "A fishing boat in morning fog"},
    {"slug": "restyle", "prompt": "Restyle <Picture 1> as a watercolor",
     "edit_image": "boat.png", "output_format": "webp"}
  ]
}

Items can override width, height, seed, and output_format (PNG by default).
edit_image accepts a path or an ordered list of paths. Use prompt_embeds to
specify a .pt file, or let the recipe cache embeddings in data/h3_embeds_cache/.
The default attention_backend is "auto", which prefers SageAttention.
Set H3_PROMPTS_JSON for local `--dry-run`, or use CU_DRY_RUN=1.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
try:
    from cu_remote import models, recipe
except ModuleNotFoundError as exc:
    if exc.name != "cu_remote":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from colab_ultra.remote import models, recipe

PROCESSOR_FILES = (
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
IMAGE_GROUP = ("minimax-h3-image", "bf16")
TE_GROUP = ("minimax-h3-qwen3vl-32b", "entropack-4bpp")
BF16_FILES = {
    name: f"hf://MiniMaxAI/MiniMax-H3-Image/{name}"
    for name in (
        "pdd_config.json",
        "pdd_fused_heads.safetensors",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/model.safetensors.index.json",
        *(f"transformer/model-{i:05d}-of-00012.safetensors" for i in range(1, 13)),
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        *(f"processor/{name}" for name in PROCESSOR_FILES),
    )
}
TE_FILES = {
    name: f"hf://DiffSynth-Studio/EntroPackPreQuants/models/MiniMax-H3/{name}"
    for name in ("text_encoder_4bpp.safetensors", "text_encoder_4bpp.json")
}
DIFFSYNTH_DIR = Path("/content/DiffSynth-Studio")
SAVE_OPTIONS = {
    "png": {"compress_level": 3},
    "jpg": {"quality": 95},
    "jpeg": {"quality": 95},
    "webp": {"lossless": True, "quality": 100, "method": 4},
}

# Run in a fresh interpreter so the text and vision models exit before loading DiT.
ENCODE_TASKS = """
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import torch
from PIL import Image
from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.minimax_h3_image import (
    MiniMaxH3ImagePipeline,
    MiniMaxH3ImageUnit_EditImageResize,
    MiniMaxH3ImageUnit_PromptEmbedder,
    ModelConfig,
)

tasks = json.load(sys.stdin)
te_path = Path(sys.argv[2])
entry = json.loads(te_path.with_suffix(".json").read_text(encoding="utf-8"))
if not any(config.get("model_hash") == entry["model_hash"] for config in MODEL_CONFIGS):
    MODEL_CONFIGS.append(entry)

pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(
        path=str(te_path),
        offload_dtype=torch.bfloat16, offload_device="cpu",
        onload_dtype=torch.bfloat16, onload_device="cuda",
        preparing_dtype=torch.bfloat16, preparing_device="cuda",
        computation_dtype=torch.bfloat16, computation_device="cuda",
    )],
    processor_config=ModelConfig(path=sys.argv[3]),
    vram_limit=20.0,
)


def save_embedding(task, embeds, tags):
    dest = Path(task["cache_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir="/tmp") as local:
        torch.save({"prompt": task["prompt"], "prompt_embeds": embeds.to(torch.bfloat16).cpu(),
                    "text_token_tags": tags.cpu()}, local.name)
        with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
            staged = Path(staging) / dest.name
            shutil.copy2(local.name, staged)
            staged.replace(dest)
    print(f"[TE] saved {dest} shape={tuple(embeds.shape)}", flush=True)


with torch.no_grad():
    pipe.load_models_to_device(["text_encoder"])
    text_tasks = [task for task in tasks if not task["edit_images"]]
    if text_tasks:
        token_lists = pipe.tokenizer([task["prompt"] or " " for task in text_tasks],
                                     add_special_tokens=False)["input_ids"]
        token_lists = [ids[:4096] for ids in token_lists]
        pad_id = pipe.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = pipe.tokenizer.eos_token_id or 0
        shape = (len(text_tasks), max(map(len, token_lists)))
        input_ids = torch.full(shape, pad_id, dtype=torch.long, device=pipe.device)
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(token_lists):
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=pipe.device)
            attention_mask[row, :len(ids)] = 1
        hidden = pipe.text_encoder.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False,
            use_cache=False, return_dict=True,
        ).last_hidden_state
        for task, ids, embeds in zip(text_tasks, token_lists, hidden):
            save_embedding(task, embeds[:len(ids)], torch.ones(len(ids), dtype=torch.long))
        del input_ids, attention_mask, hidden, embeds

    resize = MiniMaxH3ImageUnit_EditImageResize()
    embed = MiniMaxH3ImageUnit_PromptEmbedder()
    for task in tasks:
        if not task["edit_images"]:
            continue
        images = []
        for path in task["edit_images"]:
            with Image.open(path) as image:
                images.append(image.copy())
        vision_images = resize.process(pipe, images)["vision_images"]
        encoded = embed.process(pipe, task["prompt"], vision_images, 4096)
        # DiffSynth tags vision tokens 0 and text tokens 1, as h3_infer expects.
        save_embedding(task, encoded["prompt_embeds"], encoded["text_token_tags"])
"""


def embedding_path(item: dict, edit_paths: list[Path], cwd: Path, project_root: Path) -> Path:
    if "prompt_embeds" in item:
        path = Path(item["prompt_embeds"]).expanduser()
        candidates = (cwd / path, project_root / path, project_root / "data" / path)
        return next((p for p in candidates if p.is_file()), candidates[0]).resolve()

    edits = []
    for path in edit_paths:
        stat = path.stat()
        edits.append((str(path), stat.st_size, stat.st_mtime_ns))
    key = json.dumps({"prompt": item["prompt"], "edits": edits}, sort_keys=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return project_root / "data" / "h3_embeds_cache" / f"{digest[:16]}.pt"


def setup_encoder(repo: str, branch: str) -> None:
    if not DIFFSYNTH_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1", "-b", branch, repo, str(DIFFSYNTH_DIR)], check=True)
    uv_bin = shutil.which("uv")
    install = [uv_bin, "pip", "install", "--system"] if uv_bin else [sys.executable, "-m", "pip", "install"]
    subprocess.run([*install, "-q", "transformers>=4.57.0", "entropack>=0.1.7", "modelscope>=1.18.0",
                    "huggingface_hub>=0.25.0", "ftfy", "einops", "accelerate", "peft", "sentencepiece",
                    "protobuf", "pillow"], check=True)


def ensure_sageattention(attention_backend: str) -> None:
    if attention_backend not in ("auto", "sageattention") or importlib.util.find_spec("sageattention") is not None:
        return

    uv_bin = shutil.which("uv")
    install = [uv_bin, "pip", "install", "--system"] if uv_bin else [sys.executable, "-m", "pip", "install"]
    try:
        result = subprocess.run([*install, "-q", "sageattention==1.0.6"], check=False)
        if result.returncode:
            print(f"[Warn] sageattention installation failed (exit {result.returncode})", flush=True)
    except OSError as exc:
        print(f"[Warn] sageattention installation failed ({exc})", flush=True)
    finally:
        importlib.invalidate_caches()


def has_packed_attention(backend: str, kernel_name: str) -> bool:
    if importlib.util.find_spec(backend) is None:
        return False
    try:
        return callable(getattr(importlib.import_module(backend), kernel_name, None))
    except (ImportError, OSError, RuntimeError):
        return False


def encode_tasks(tasks: list[dict], bf16_dir: Path, te_dir: Path) -> None:
    cmd = [sys.executable, "-u", "-c", ENCODE_TASKS, str(DIFFSYNTH_DIR),
           str(te_dir / "text_encoder_4bpp.safetensors"), str(bf16_dir / "processor")]
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        manifest = Path(tmp) / "h3_te_tasks.json"
        manifest.write_text(json.dumps(tasks), encoding="utf-8")
        with manifest.open(encoding="utf-8") as source:
            # Each subprocess keeps its own manifest open even if another run replaces the path.
            manifest.replace("/tmp/h3_te_tasks.json")
            with subprocess.Popen(cmd, stdin=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                if proc.wait():
                    raise subprocess.CalledProcessError(proc.returncode, cmd[:2], output="See Stage 1 logs")


def main() -> None:
    cfg, cwd, project_root = recipe.load_project()
    run_name = cfg.get("run_name", time.strftime("h3_infer_%Y%m%d_%H%M%S"))
    width = int(cfg.get("width", 2048))
    height = int(cfg.get("height", 2048))
    seed_base = int(cfg.get("seed_base", 100))
    default_fmt = str(cfg.get("output_format", "png")).lower().lstrip(".")
    skip_existing = bool(cfg.get("skip_existing", True))
    compile_mode = str(cfg.get("compile_mode", "regional" if cfg.get("compile_dit", True) else "none"))
    if "attention_backend" in cfg:
        attention_backend = str(cfg["attention_backend"])
    elif cfg.get("use_sage_attention") is False or cfg.get("sage_attn") is False:
        attention_backend = "sdpa"
    else:
        attention_backend = "auto"
    resident_blocks = cfg.get("resident_blocks", "auto")
    staged_vram = bool(cfg.get("staged_vram", True))
    prune_adaln = cfg.get("prune_adaln", "auto")
    tiling = cfg.get("tiling", None)
    diffsynth_repo = str(cfg.get("diffsynth_repo", "https://github.com/lgyStoic/DiffSynth-Studio.git"))
    diffsynth_branch = str(cfg.get("diffsynth_branch", "t4-sm75-int8-support"))
    out_dir = project_root / "runs" / run_name

    if recipe.dry_run():
        recipe.describe(cfg, run_name)
        models.describe(*IMAGE_GROUP, BF16_FILES)
        models.describe(*TE_GROUP, TE_FILES)
        print(f"MiniMaxH3ImagePipeline (h3_infer): compile_mode={compile_mode} "
              f"attention_backend={attention_backend} skip_existing={skip_existing} -> {out_dir}")
        return

    outputs = []
    batch_items = []
    uncached = {}
    for idx, item in enumerate(cfg["prompts"], 1):
        slug = item.get("slug", f"img_{idx:02d}")
        fmt = str(item.get("output_format", default_fmt)).lower().lstrip(".")
        out_path = out_dir / f"{slug}.{fmt}"
        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            print(f"[Skip] {out_path}", flush=True)
            continue

        specs = item.get("edit_image")
        if isinstance(specs, str):
            specs = [specs] if specs else []
        edit_paths = [recipe.resolve_image(spec, cwd, project_root).resolve() for spec in (specs or [])]
        cache_path = embedding_path(item, edit_paths, cwd, project_root)
        edit_images = [str(p) for p in edit_paths]
        if not cache_path.is_file():
            uncached.setdefault(cache_path, {"prompt": item["prompt"], "edit_images": edit_images,
                                             "cache_path": str(cache_path)})
        batch_item = {"prompt_embeds": cache_path, "seed": int(item.get("seed", seed_base + idx)),
                      "height": int(item.get("height", height)), "width": int(item.get("width", width))}
        if edit_images:
            batch_item["edit_images"] = edit_images
        batch_items.append(batch_item)
        outputs.append((slug, fmt, out_path))

    if not batch_items:
        return

    content_free_gb = shutil.disk_usage("/content").free / (1024 ** 3) if Path("/content").exists() else 0.0
    tmp_free_gb = shutil.disk_usage("/tmp").free / (1024 ** 3) if Path("/tmp").exists() else 0.0
    nvme_root = Path("/content/models") if content_free_gb >= 35.0 or content_free_gb >= tmp_free_gb else Path("/tmp/models")
    print(f"[Disk] staging root={nvme_root} (/content free={content_free_gb:.1f} GiB, /tmp free={tmp_free_gb:.1f} GiB)",
          flush=True)
    bf16_dir = nvme_root.joinpath(*IMAGE_GROUP)
    te_dir = nvme_root.joinpath(*TE_GROUP)

    def stage_models() -> None:
        files = {name: url for name, url in BF16_FILES.items() if uncached or not name.startswith("processor/")}
        models.ensure(*IMAGE_GROUP, files, to=bf16_dir)
        if uncached:
            models.ensure(*TE_GROUP, TE_FILES, to=te_dir)

    with ThreadPoolExecutor(max_workers=1) as pool:
        staged = pool.submit(stage_models)
        if uncached:
            setup_encoder(diffsynth_repo, diffsynth_branch)
        import torch

        major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
        if major >= 8:
            ensure_sageattention(attention_backend)
        staged.result()
    if uncached:
        encode_tasks(list(uncached.values()), bf16_dir, te_dir)

    for var, name in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
        cache_dir = Path("/content/.cache/h3_infer") / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(cache_dir)

    try:
        from h3_infer import MiniMaxH3ImagePipeline
    except ModuleNotFoundError as exc:
        if exc.name != "h3_infer":
            raise
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines"))
        from h3_infer import MiniMaxH3ImagePipeline

    from h3_infer.ops.attention import recommended_attention_backend

    active_attn = "sdpa"
    if attention_backend == "auto":
        active_attn = recommended_attention_backend(major)
    elif attention_backend == "sageattention" and major >= 8 and (
        has_packed_attention("sageattention", "sageattn_varlen")
        or has_packed_attention("sageattention", "sageattn")
    ):
        active_attn = "sageattention"
    elif (attention_backend == "flash_attn" and major in (8, 9)
          and has_packed_attention("flash_attn", "flash_attn_varlen_func")):
        active_attn = "flash_attn"
    print(f"[Attn] requested={attention_backend} active={active_attn}", flush=True)

    pipe = MiniMaxH3ImagePipeline.from_pretrained(
        bf16_dir, device="cuda", attention_backend=active_attn, offload_dit=True,
        prune_adaln=prune_adaln, staged_vram=staged_vram, resident_blocks=resident_blocks, compile_mode=compile_mode,
    )

    def report_step(item_idx, step, _latents, _velocity):
        slug = outputs[item_idx][0]
        print(f"[H3Infer {time.strftime('%H:%M:%S')}] [{item_idx + 1}/{len(outputs)}] {slug} step {step + 1}/6",
              flush=True)

    results = pipe.generate_batch(batch_items, tiling=tiling, output_type="pil", return_stats=True,
                                  step_callback=report_step)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        for (slug, fmt, out_path), (image, stats) in zip(outputs, results, strict=True):
            staged_path = Path(tmp) / out_path.name
            image.save(staged_path, **SAVE_OPTIONS[fmt])
            staged_path.replace(out_path)
            images.append({"slug": slug, "file": out_path.name, **asdict(stats)})
        summary_path = out_dir / "summary.json"
        staged_summary = Path(tmp) / summary_path.name
        summary = {"run_name": run_name, "compile_mode": compile_mode, "attention_backend": attention_backend,
                   "active_attention_backend": active_attn,
                   "skipped": len(cfg["prompts"]) - len(outputs), "images": images}
        staged_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged_summary.replace(summary_path)
    os.sync()
    print(f"[Summary] {len(images)} generated, {summary['skipped']} skipped -> {summary_path}", flush=True)
    print("[Done] minimax_h3_infer_finished_ok", flush=True)


if __name__ == "__main__":
    main()
