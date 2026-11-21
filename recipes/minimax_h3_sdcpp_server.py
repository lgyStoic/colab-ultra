#!/usr/bin/env python3
"""MiniMax-H3-Image via stable-diffusion.cpp sd-server + unsloth GGUF (UD-Q4_K_XL, release VAE, Q4_K_M LLM).

Features:
  - Compiles & caches sd-server in Drive; copies to /tmp with +x on reuse.
  - Keeps models loaded once in sd-server (eliminates ~75s/tile disk reload).
  - Enables --sage-attn + --max-vram (configurable, default 19 GB).
  - Encodes lossless WebP (-compression_level 1) in background threads overlapped with GPU inference.
  - Skips any tile whose output file already exists (> 1 MB).
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
try:
    from cu_remote import bg, binaries, recipe, stage
except ModuleNotFoundError as exc:
    if exc.name != "cu_remote":
        raise
    # Source checkout for offline dry runs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from colab_ultra.remote import bg, binaries, recipe, stage

DRIVE = Path("/content/drive/MyDrive/ColabUltra")
BIN_DIR = DRIVE / "bin" / "sdcpp"
MODELS = Path("/content/models/h3gguf")

DIT_NAME = os.environ.get("H3_DIT_GGUF", "minimax_h3_image_pruned-UD-Q4_K_XL.gguf")
FILES = {
    "dit": (MODELS / DIT_NAME, f"https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF/resolve/main/{DIT_NAME}"),
    "vae": (MODELS / "diffusion_pytorch_model.safetensors",
            "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/vae/diffusion_pytorch_model.safetensors"),
    "llm": (MODELS / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            "https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/main/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"),
}


def build_binaries():
    """Build both CUDA executables for the current VM architecture."""
    src = Path("/content/stable-diffusion.cpp")
    if not (src / "CMakeLists.txt").is_file():
        subprocess.run(["git", "clone", "--depth", "1", "--recursive",
                        "https://github.com/leejet/stable-diffusion.cpp.git", str(src)], check=True)
    cap = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  text=True).splitlines()[0].strip().replace(".", "")
    subprocess.run(["cmake", "-B", "build", "-DSD_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_CUDA_ARCHITECTURES={cap}"], cwd=src, check=True)
    subprocess.run(["cmake", "--build", "build", "--config", "Release", "--target", "sd-cli", "sd-server",
                    "-j", str(os.cpu_count() or 1)], cwd=src, check=True)
    return {name: next((src / "build").rglob(name)) for name in ("sd-cli", "sd-server")}


def wait_server(port: int, proc: subprocess.Popen, timeout_s: int = 300) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"sd-server exited prematurely with rc={proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"sd-server not ready after {timeout_s}s")


def main() -> None:
    """Keep one sd-server resident for all unfinished tiles in a project."""
    cfg, cwd, project_root = recipe.load_project(("webp", "png"))
    run_name = cfg.get("run_name", time.strftime("sdcpp_%Y%m%d_%H%M%S"))
    out_dir = project_root / "runs" / run_name

    todo = []
    for i, it in enumerate(cfg["prompts"], 1):
        slug = it.get("slug", f"img_{i:02d}")
        fmt = str(it.get("output_format", cfg.get("output_format", "webp"))).lower().lstrip(".")
        dst = out_dir / f"{slug}.{fmt}"
        if dst.is_file() and dst.stat().st_size > 1_000_000:
            print(f"[Skip] {slug} already exists ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
            continue
        todo.append((i, it, slug, fmt, dst))

    print(f"[Start] {time.strftime('%F %T')} | run={run_name} | total={len(cfg['prompts'])} | todo={len(todo)}", flush=True)
    port = int(cfg.get("server_port", 1234))
    max_vram = str(cfg.get("max_vram", "19"))
    if not 0 < port < 65536 or float(max_vram) <= 0:
        raise ValueError("Expected a valid server_port and positive max_vram in GB")
    use_sage = bool(cfg.get("sage_attn", True))
    srv_cmd = [
        "/tmp/sd-server", "--listen-port", str(port),
        "--diffusion-model", str(FILES["dit"][0]), "--vae", str(FILES["vae"][0]), "--llm", str(FILES["llm"][0]),
        "--cfg-scale", "1.0", "--steps", "6", "--sampling-method", "euler", "--sigmas", "minimax-h3-image",
        "--rng", "cpu", "--max-vram", max_vram, "-v",
    ]
    if use_sage:
        srv_cmd.append("--sage-attn")

    plans = [stage.plan(url, path) for path, url in FILES.values()]
    if recipe.dry_run():
        recipe.describe(cfg, run_name)
        for item in plans:
            print(item.backend, item.source, "->", item.dest)
        print(" ".join(srv_cmd))
        for _, _, slug, fmt, dst in todo:
            print(f"POST /v1/images/edits {slug} -> {dst} ({fmt})")
        return
    if not todo:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        staged = pool.submit(stage.run_many, plans)
        srv_cmd[0] = str(binaries.ensure_binary("sd-server", BIN_DIR, build_binaries))
        staged.result()

    print(f"[Server] starting: {' '.join(srv_cmd)}", flush=True)
    srv_log = open("/content/sd_server.log", "w")
    proc = subprocess.Popen(srv_cmd, stdout=srv_log, stderr=subprocess.STDOUT)
    try:
        wait_server(port, proc, timeout_s=360)
        print(f"[Server] ready in {time.time() - t0:.1f}s", flush=True)

        with bg.encoders() as enqueue:
            for idx, (orig_i, it, slug, fmt, dst) in enumerate(todo, 1):
                w, h = int(it.get("width", cfg.get("width", 4096))), int(it.get("height", cfg.get("height", 4096)))
                seed = int(it.get("seed", cfg.get("seed_base", 100) + orig_i))
                extra = json.dumps({"seed": seed})
                prompt_payload = f"{it['prompt']} <sd_cpp_extra_args>{extra}</sd_cpp_extra_args>"
                resp_json = Path(f"/content/{slug}_resp.json")
                t1 = time.time()
                curl_cmd = [
                    "curl", "-sS", "-f", "--max-time", "1800",
                    "-X", "POST", f"http://127.0.0.1:{port}/v1/images/edits",
                    "-F", f"prompt={prompt_payload}",
                    "-F", f"size={w}x{h}",
                    "-F", "output_format=png",
                ]
                if it.get("edit_image"):
                    curl_cmd += ["-F", f"image[]=@{recipe.resolve_image(it['edit_image'], cwd, project_root)}"]
                curl_cmd += ["-o", str(resp_json)]
                rc = subprocess.run(curl_cmd, check=False).returncode
                if rc != 0 or not resp_json.is_file():
                    raise RuntimeError(f"{slug}: image request failed (curl rc={rc})")
                data = json.loads(resp_json.read_text(encoding="utf-8"))
                resp_json.unlink(missing_ok=True)
                b64 = data["data"][0]["b64_json"]
                png_path = Path(f"/content/{slug}.png")
                png_path.write_bytes(base64.b64decode(b64))
                dt = time.time() - t1
                enqueue(png_path, dst)
                print(f"[Done {idx:02d}/{len(todo)}] {slug} | infer={dt:.1f}s (webp encoding in bg)", flush=True)
        print(f"[Done] sdcpp_gguf_server_finished | total={time.time() - t0:.1f}s", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        srv_log.close()


if __name__ == "__main__":
    main()
