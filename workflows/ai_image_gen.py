#!/usr/bin/env python3
"""
Scenario 2: 免 ComfyUI 极速 AI 文生图引擎 (20TB 云盘永久模型缓存 + 一键出图)
No need to learn complex ComfyUI node wiring!
Directly generates high-resolution images on A100/H100/L4 using HuggingFace Diffusers,
automatically caching model weights inside /content/drive/MyDrive/ColabUltra/hf_cache
and saving outputs to /content/drive/MyDrive/ColabUltra/outputs/images/.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

DRIVE_HUB = "/content/drive/MyDrive/ColabUltra"
HEARTBEAT_FILE = "/content/.task_heartbeat.json"


def update_heartbeat(status: str, active: bool = True):
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            json.dump({"task": "AI-Image-Gen", "status": status, "active": active, "timestamp": time.time()}, f)
    except Exception:
        pass


def ensure_diffusers():
    try:
        import diffusers  # noqa
        import accelerate  # noqa
    except ImportError:
        print("📦 正在通过 uv 极速安装 diffusers 与 accelerate...")
        subprocess.check_call(["uv", "pip", "install", "--system", "-q", "diffusers", "accelerate", "transformers", "safetensors"])


def generate_image(
    prompt: str = "A futuristic glass laboratory floating above clouds at sunset, cinematic lighting, 8k masterpiece, photorealistic",
    model_id: str = "stabilityai/sdxl-turbo",
    steps: int = 4,
    width: int = 1024,
    height: int = 1024,
):
    if os.path.isdir(DRIVE_HUB):
        os.environ["HF_HOME"] = os.path.join(DRIVE_HUB, "hf_cache")
        out_dir = os.path.join(DRIVE_HUB, "outputs", "images")
    else:
        out_dir = "/content/outputs/images"
    os.makedirs(out_dir, exist_ok=True)

    update_heartbeat("正在检查环境并加载生图模型...", active=True)
    ensure_diffusers()

    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 60)
    print(f"🎨 AI 极速生图引擎 | 模型: {model_id}")
    print(f"📝 提示词: {prompt}")
    print(f"📂 模型缓存目录: {os.environ.get('HF_HOME', '默认缓存')}")
    print("=" * 60)

    t0 = time.perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype, variant="fp16" if "turbo" in model_id else None)
    pipe.to(device)
    load_sec = time.perf_counter() - t0
    print(f"⚡ 模型加载耗时: {load_sec:.2f} 秒")

    update_heartbeat(f"正在生成图像 ({width}x{height})...", active=True)
    t1 = time.perf_counter()
    guidance = 0.0 if "turbo" in model_id.lower() else 7.0
    image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, width=width, height=height).images[0]
    gen_sec = time.perf_counter() - t1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"gen_{ts}.png")
    latest_path = "/content/latest_generated.png"
    image.save(out_path)
    image.save(latest_path)

    update_heartbeat(f"生图完成 ({gen_sec:.2f}s) -> {out_path}", active=False)
    print(f"✅ 图像生成完成！纯渲染耗时: {gen_sec:.2f} 秒")
    print(f"💾 已永久归档至路径: {out_path}")
    print(f"🖼️ 预览副本路径: {latest_path}")
    print("=" * 60)


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "A cyberpunk AI server room glowing with neon blue circuit breakers, ultra detailed, 8k resolution, cinematic atmosphere"
    )
    generate_image(prompt=user_prompt)
