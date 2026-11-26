#!/usr/bin/env python3
"""Measure BF16 MiniMax-H3-Image inference against saved DiffSynth tensors.

Run on an A100, L4, or H100 with the BF16 release staged on local disk.
Save reference tensors as a weights-only .pt dict with step_1_latents
and final_latents in [1,C,1,H,W].
An optional step_1_velocity field uses DiffSynth's noise-ward sign.
Supply initial_latents in that dict or via --initial-latents to share noise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

engines_root = str(Path(__file__).resolve().parents[1] / "engines")
if engines_root not in sys.path:
    sys.path.insert(0, engines_root)

import torch
from torch.nn import functional as F

from h3_infer.ops.attention import BACKENDS
from h3_infer.pipeline import MIN_GENERATION_AREA, MiniMaxH3ImagePipeline
from h3_infer.runtime.colab import detect_gpu, staging_paths


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _resident_blocks(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected 'auto' or an integer") from error


def _metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f"Reference shape {tuple(reference.shape)} differs from output {tuple(actual.shape)}")
    actual, reference = actual.float().flatten(), reference.float().flatten()
    return {
        "cosine_similarity": F.cosine_similarity(actual, reference, dim=0).item(),
        "max_abs_diff": (actual - reference).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Local BF16 release directory")
    parser.add_argument("--prompt-embeds", type=Path, required=True, help="Saved text_embeds.pt")
    parser.add_argument("--reference", type=Path, help="DiffSynth .pt dict (see module docstring)")
    parser.add_argument("--initial-latents", type=Path, help="Optional .pt tensor for shared initial noise")
    parser.add_argument("--ref-latents", type=Path, help="Optional .pt tensor/list of edit conditioning latents")
    parser.add_argument("--pdd-heads", type=Path, help="Override the PDD head checkpoint path")
    parser.add_argument("--output-image", type=Path, help="Save the resized VAE output")
    parser.add_argument("--save-tensors", type=Path, help="Save first-step and final latents plus first velocity")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--min-generation-area", type=int, default=MIN_GENERATION_AREA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=BACKENDS, default="auto")
    parser.add_argument("--offload-dit", action=argparse.BooleanOptionalAction, default=None,
                        help="Stream CPU-backed DiT weights (default: select by free VRAM)")
    parser.add_argument("--prune-adaln", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--staged-vram", action=argparse.BooleanOptionalAction, default=None,
                        help="Stage DiT and VAE separately (default: enabled on CUDA)")
    parser.add_argument("--resident-blocks", type=_resident_blocks, default="auto",
                        help="CUDA resident prefix for offloaded DiT blocks")
    parser.add_argument("--compile", choices=("none", "regional", "full"), default="none", dest="compile_mode")
    args = parser.parse_args()

    device = torch.device(args.device)
    gpu = detect_gpu(device)
    if device.type == "cuda" and not gpu.supports_bf16:
        parser.error("BF16 parity requires an sm_80 or newer CUDA GPU")
    for name, path in staging_paths().cache_environment().items():
        os.environ.setdefault(name, path)
    reference = _load(args.reference) if args.reference else {}
    if not isinstance(reference, dict):
        parser.error("--reference must contain a tensor dictionary")
    keys = ("step_1_latents", "final_latents", "step_1_velocity")
    if args.reference and not any(key in reference for key in keys):
        parser.error("--reference must provide step_1_latents, final_latents, or step_1_velocity")
    latents = _load(args.initial_latents) if args.initial_latents else reference.get("initial_latents")

    pipe = MiniMaxH3ImagePipeline.from_pretrained(
        args.model_dir, device=device, attention_backend=args.attention,
        offload_dit=args.offload_dit, pdd_heads_path=args.pdd_heads,
        prune_adaln={"auto": "auto", "always": True, "never": False}[args.prune_adaln],
        staged_vram=args.staged_vram, resident_blocks=args.resident_blocks, compile_mode=args.compile_mode,
    )
    observed = {}

    def capture(step, value, velocity):
        if step == 0:
            observed["step_1_latents"] = value.cpu()
            observed["step_1_velocity"] = -velocity.cpu()
        if step + 1 == pipe.scheduler.num_inference_steps:
            observed["final_latents"] = value.cpu()

    result, stats = pipe(
        args.prompt_embeds, latents=latents,
        ref_latents=_load(args.ref_latents) if args.ref_latents else None,
        height=args.height, width=args.width, min_generation_area=args.min_generation_area,
        seed=args.seed, return_stats=True, output_type="pil" if args.output_image else "latent",
        step_callback=capture if reference or args.save_tensors else None,
    )
    if args.output_image:
        result.save(args.output_image)
    if args.save_tensors:
        if latents is not None:
            observed["initial_latents"] = latents.cpu()
        torch.save(observed, args.save_tensors)
    report = {
        "gpu": gpu.device_name,
        "compute_capability": gpu.compute_capability,
        "attention": args.attention,
        "recommended_attention": gpu.attention_backend,
        "offload_dit": pipe.dit.offload,
        "pruned_adaln": pipe.dit.pruned_adaln,
        "staged_vram": pipe.staged_vram,
        "compile_mode": args.compile_mode,
        "single_step": {"latency_ms": stats.step_ms[0]},
        "six_step": {"latency_ms": sum(stats.step_ms)},
        "per_step_latency_ms": stats.step_ms,
        "peak_vram_bytes": stats.peak_vram_bytes,
    }
    if "step_1_latents" in reference:
        report["single_step"].update(_metrics(observed["step_1_latents"], reference["step_1_latents"]))
    if "final_latents" in reference:
        report["six_step"].update(_metrics(observed["final_latents"], reference["final_latents"]))
    if "step_1_velocity" in reference:
        report["single_step_velocity"] = _metrics(observed["step_1_velocity"], reference["step_1_velocity"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
