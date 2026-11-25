# 硬件规格与量化精度

## Dense Tensor Core 峰值

除 INT8 使用 TOPS 外，各精度的峰值均以 TFLOPS 计。VRAM 和 RAM 以 GB 计，显存带宽以 GB/s 计。

| 硬件 | SM | VRAM | 显存带宽 | FP16 | BF16 | INT8 (TOPS) | FP8 | NVFP4 | TDP (W) | 空载阈值 (W) | CU/h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T4X2（Kaggle） | sm_75 | 32 | 640 | 130 | — | 260 | — | — | 140 | 28 | 0 |
| T4 | sm_75 | 16 | 320 | 65 | — | 130 | — | — | 70 | 28 | 1.19 |
| L4 | sm_89 | 24 | 300 | 121 | 121 | 242.5 | 242.5 | — | 72 | 28 | 1.71 |
| A100 80 GB | sm_80 | 80 | 2039 | 312 | 312 | 624 | — | — | 400 | 75 | 8.71 |
| G4（RTX PRO 6000 Blackwell Server Edition） | sm_120 | 96 | 1597 | 504 | 504 | 1008 | 1008 | 2015 | 600 | 75 | 10.05 |
| H100 80 GB | sm_90 | 80 | 3350 | 989 | 989 | 1979 | 1979 | — | 700 | 75 | 18.05 |
| CPU-STD | — | 0 | 0 | — | — | — | — | — | — | — | 0.08 |
| CPU-HIGH | — | 0 | 0 | — | — | — | — | — | — | — | 0.26 |
| TPU-V5E1 | — | 0 | 0 | — | — | — | — | — | — | — | 3.09 |
| TPU-V6E1 | — | 0 | 0 | — | — | — | — | — | — | — | 5.09 |

T4X2 的显存、带宽、算力和 TDP 为双卡合计。单卡有 16 GB 显存、320 GB/s 带宽和 70 W TDP，空载阈值按单卡计。A100 和 H100 的表值对应 SXM 版，PCIe 版的显存带宽分别为 1935 GB/s 和 2000 GB/s。

Dense 峰值按稠密矩阵计算。Ampere 及更新架构对支持的精度和内核可利用 2:4 结构化稀疏（每组四个权重有两个为零），理论峰值可达同精度 Dense 的 2 倍。实际吞吐还取决于内核和显存带宽。CPU 和 TPU 行的 VRAM 与显存带宽记为 0，峰值栏留空。

## 量化精度速查

| 硬件 | 优先考虑 |
| --- | --- |
| T4 / T4X2 | INT8 / FP16 |
| L4 | FP8 / INT8 |
| A100 | BF16 / INT8 |
| G4 | NVFP4 / FP8 / INT8 |
| H100 | FP8 / INT8 / BF16 |

## 命令行与运行时查询

```sh
colab-ultra specs
colab-ultra specs --precision int8
colab-ultra specs --min-vram 80 --kind gpu --json
```

指定 `--precision` 会筛出原生支持该精度的条目，并显示峰值算力除以 CU/h 的结果。免费 T4X2 显示 `FREE`，其 JSON `efficiency_per_cu` 为 `null`。

VM 上可通过已同步的运行库识别当前 GPU。查询优先使用 PyTorch CUDA，其次使用 `nvidia-smi`，没有受支持的 GPU 时返回 `None`。

```python
from cu_remote.hardware import current

spec = current()
if spec is not None and spec.supports_fp8:
    print(spec.key, spec.metric_for("fp8"))
```
