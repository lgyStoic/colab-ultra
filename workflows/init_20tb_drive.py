#!/usr/bin/env python3
"""
Scenario 3: 20 TB Google Drive AI Hub Initializer (云端永久母舰目录与缓存配置)
Run inside Colab VM (after drive is mounted at /content/drive) to:
1. Verify 20 TB Google Drive capacity & speed.
2. Create standardized permanent directories under /content/drive/MyDrive/ColabUltra/.
3. Redirect HuggingFace (HF_HOME) and PyTorch (TORCH_HOME) caches to the 20 TB Drive
   so any model downloaded once is permanently preserved across all future VMs.
"""
import json
import os
import time

DRIVE_MOUNT = "/content/drive"
MY_DRIVE = "/content/drive/MyDrive"
HUB_ROOT = "/content/drive/MyDrive/ColabUltra"

DIRS = [
    "models/llm",          # 大语言模型权重 (Qwen, Llama, DeepSeek 等)
    "models/checkpoints",  # 生图/生视频基础大模型 (FLUX, SD3.5, Wan2.1 等)
    "models/loras",        # LoRA 微调权重
    "models/vae_clip",     # VAE / Text Encoder 组件
    "datasets",            # 训练与微调数据集
    "outputs/images",      # AI 生图输出永久归档
    "outputs/videos",      # AI 生视频输出永久归档
    "outputs/training",    # 模型训练 Checkpoints 与日志
    "hf_cache",            # HuggingFace 全局缓存 (HF_HOME)
    "torch_cache",         # PyTorch Hub 全局缓存 (TORCH_HOME)
    "workspace",           # 永久工作目录 (软链接至 /content/workspace，断电重启零丢失)
    "emergency_backups",   # 智能电闸拉闸前自动抢救备份区
    "emergency_backups/latest_root_files",  # 根目录散落脚本/笔记本自动快照与恢复区
]


def setup_drive_hub():
    print("=" * 60)
    print("🚀 正在初始化 Google AI Ultra 20TB 云端永久母舰...")
    print("=" * 60)

    if not os.path.isdir(MY_DRIVE):
        print("⚠️ 检测到 /content/drive/MyDrive 尚未挂载，尝试挂载中...")
        try:
            from google.colab import drive
            drive.mount(DRIVE_MOUNT)
        except Exception as e:
            print(f"❌ 自动挂载需要授权，请先在本地运行: colab drivemount -s <会话名> ({e})")
            return False

    st = os.statvfs(DRIVE_MOUNT)
    total_tb = (st.f_blocks * st.f_frsize) / (1024 ** 4)
    free_tb = (st.f_bavail * st.f_frsize) / (1024 ** 4)
    used_tb = total_tb - free_tb

    print(f"✅ Google Drive 挂载就绪！")
    print(f"📦 云盘总容量: {total_tb:.2f} TB | 已用: {used_tb:.2f} TB | 剩余可用: {free_tb:.2f} TB")
    print("-" * 60)

    for sub in DIRS:
        full_path = os.path.join(HUB_ROOT, sub)
        os.makedirs(full_path, exist_ok=True)
        print(f"📁 已就绪目录: {full_path}")

    # Symlink /content/workspace and /content/outputs directly to 20TB Drive so nothing is ever lost on shutdown
    for link_name, target_sub in [("workspace", "workspace"), ("outputs", "outputs")]:
        link_path = os.path.join("/content", link_name)
        target_path = os.path.join(HUB_ROOT, target_sub)
        if not os.path.exists(link_path) and not os.path.islink(link_path):
            os.symlink(target_path, link_path)
            print(f"🔗 已绑定永久软链接: {link_path} -> {target_path}")

    # Restore any loose root files (/content/*.py, *.ipynb, *.json, *.log) rescued from previous session
    latest_rescue_dir = os.path.join(HUB_ROOT, "emergency_backups", "latest_root_files")
    if os.path.isdir(latest_rescue_dir):
        import shutil
        restored = []
        for fname in os.listdir(latest_rescue_dir):
            src = os.path.join(latest_rescue_dir, fname)
            dst = os.path.join("/content", fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                restored.append(fname)
        if restored:
            print(f"♻️ 已自动恢复上次断电前的 {len(restored)} 个根目录文件: {', '.join(restored)}")

    # Write environment profile so all future Python / shell invocations use 20TB Drive cache
    env_vars = {
        "COLAB_ULTRA_ROOT": HUB_ROOT,
        "HF_HOME": os.path.join(HUB_ROOT, "hf_cache"),
        "TORCH_HOME": os.path.join(HUB_ROOT, "torch_cache"),
    }
    for k, v in env_vars.items():
        os.environ[k] = v

    profile_snippet = "\n".join([f'export {k}="{v}"' for k, v in env_vars.items()]) + "\n"
    with open("/content/.ultra_env.sh", "w") as f:
        f.write(profile_snippet)

    # Quick internal network write/read speed test to Google Drive
    test_file = os.path.join(HUB_ROOT, ".speedtest.tmp")
    payload = b"0" * (32 * 1024 * 1024)  # 32 MB
    t0 = time.perf_counter()
    with open(test_file, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    write_mb_s = 32.0 / max(0.001, time.perf_counter() - t0)
    os.remove(test_file)

    print("-" * 60)
    print(f"⚡ 云盘内网写入实测速率: {write_mb_s:.1f} MB/s")
    print(f"🔗 HF_HOME 已重定向至: {env_vars['HF_HOME']}")
    print("🎉 20TB 云端母舰初始化完毕！所有下载的模型将永久保留在云盘中。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    setup_drive_hub()
