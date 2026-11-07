import sys
import torch

print("=" * 50)
print("🚀 Colab 远程运行环境检测成功！")
print(f"Python 版本: {sys.version.split()[0]}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"显卡型号: {device_name}")
    print(f"可用显存 (VRAM): {vram:.2f} GB")
    print("TF32 加速支持:", torch.cuda.is_bf16_supported())
print("=" * 50)
