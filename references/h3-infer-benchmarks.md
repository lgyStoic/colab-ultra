# MiniMax-H3-Image 稳态性能基准表

统一口径：仅保留去除首次 compile 噪声后的**稳态单步耗时（`s/step`）**与**无重编译 6 步理论/实测 DiT 耗时（`6 × steady`）**，按 **机型 × 测试 × 方法** 归类（2048×2048 PNG，6 步 PDD，`sageattention`）。

---

## 一、端到端实测矩阵（机型 × 测试 × 方法）

| 机型 | 测试任务 | 方法 | 权重体积 | 常驻 / 显存 | **稳态单步** | **无编译 6 步估算** | 实测最好整图 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Colab L4 (22.5GB)** | **T2I 纯文生图** | `bf16` | ~66 GB | 17.88 GB | **4.06 s/步** | **24.4s** | **26.90s** (`hetero_demo_01`) |
| **Colab L4 (22.5GB)** | **T2I 纯文生图** | **`int8_convrot` (cuBLASLt `_int_mm`+Triton反量化)** | **20.98 GB** | **41/50 块 · 18.31 GB** | **3.02–3.12 s/步** | **18.1–18.7s** | **19.33s**（`hetero_int8_mix10` 混排零重编） |
| **Colab L4 (22.5GB)** | **Edit 2K 改图** | `bf16` | ~66 GB | 18.18 GB | **9.26–9.59 s/步** | **55.6–57.5s** | **60.27s** (`hetero_edit_dual_01`) |
| **Colab L4 (22.5GB)** | **Edit 2K 改图** | `int8_convrot` (未融合) | 20.98 GB | 41/50 块 · 19.31 GB | **9.38–9.64 s/步** | **56.3–57.8s** | 96.32s（含二次重编） |
| **Colab L4 (22.5GB)** | **Edit 2K 改图** | `int8_convrot` (Triton融合+动态图) | 20.98 GB | 41/50 块 · 18.59 GB | 9.01–9.61 s/步 | 54.1–57.7s | 57.42s（次图零重编） |
| **Colab L4 (22.5GB)** | **Edit 2K 改图** | **`int8_convrot` (cuBLASLt `_int_mm`+Triton反量化)** | **20.98 GB** | **41/50 块 · 19.08 GB** | **7.40–7.57 s/步** | **44.4–45.4s** | **45.14s**（次图零重编） |

### 配套异构阶段耗时（`embed` & `vae`，2048×2048）

| 阶段 | 机型与配置 | 实测稳态耗时 | 成本 |
| :--- | :--- | :---: | :---: |
| **`embed` (Qwen3-VL-32B 4bpp, DiffSynth)** | Kaggle 免费 T4 GPU0 (CPU offload) | 68.4–76.9 s/条 | 0 CCU/h |
| **`embed` (Comfy `INT8-ConvRot` 50层, 零DiffSynth)** | **Kaggle 免费双 T4 (22/28 层常驻双卡)** | **T2I `2.68–3.06 s/条` · Edit `6.82–6.98 s/条`** | **0 CCU/h** |
| **`vae` 解码** | Colab CPU (`8核50GB`, `float32`, `tile_batch=4`) | 118.2–153.3 s/图 | 0.26 CCU/h |
| **`vae` 解码** | **Kaggle 免费双 T4 (`vae-0`/`vae-1` 并行, `float32`, `tile_batch=16`)** | **单卡 `16.8–23.7 s/图`（双卡并发均摊 ~10–11s/图）** | **0 CCU/h** |

---

## 二、Colab L4 单 Block 四层投影算子实测对比（`torchao` W8A8 vs Triton INT8 vs BF16）

测试环境：NVIDIA L4 (`sm_89`, 72W 功耗墙), PyTorch `2.14.1+cu130`, `torchao 0.10.0`, `triton 3.8.0`；数据源：[torchao_w8a8_bench.json](file:///Users/garry/ColabUltra/h3_pipeline_demo/runs/torchao_w8a8_bench.json)。
每 Block 统计 `qkv_proj (5376→21504)` + `out_proj (7168→5376)` + `mlp_fc1 (5376→28672)` + `mlp_fc2 (14336→5376)` 四层纯线性层合计（50 块 × 1-Block 耗时 ≈ 单步纯矩阵乘理论下限）：

| 机型 | 测试规模 | 方法 | 单 Block 耗时 | 50 块线性合计 / 步 | 实测 L4 功耗 | 说明与权重需求 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Colab L4** | **Edit (`M=8500`)** | `bf16` (cuBLAS) | 104.02 ms | **5.20 s/步** | 28–70 W | 需 66 GB（或 40 GB pruned BF16） |
| **Colab L4** | **Edit (`M=8500`)** | `triton_int8_convrot` (`tl.dot` 融合) | 102.72 ms | **5.14 s/步** | 54–73 W | 直接兼容 20.98 GB Comfy INT8-ConvRot（旧实现） |
| **Colab L4** | **Edit (`M=8500`)** | **`cublaslt_int8_convrot` (`_int_mm`+Triton反量化)** | **77.43 ms** | **3.87 s/步** | — | **当前默认实现**：较 BF16 快 1.34×，兼容 20.98 GB Comfy INT8-ConvRot，无新增依赖 |
| **Colab L4** | **Edit (`M=8500`)** | `triton_int8_no_rot` (无旋转对比) | 96.88 ms | 4.84 s/步 | — | 说明在线 ConvRot 旋转本身仅占 ~0.30 s/步 |
| **Colab L4** | **Edit (`M=8500`)** | **`torchao` W8A8 INT8 (`torch.compile`)** | **66.53 ms** | **3.33 s/步** | 39–52 W | **比 BF16/Triton 快 ~1.55×**（内部走 `torch._int_mm` cuBLASLt + Inductor）；eager 慢至 279.1 ms |
| **Colab L4** | **Edit (`M=8500`)** | **`torchao` W8A8 FP8 (`PerTensor`+`compile`)** | **64.83 ms** | **3.24 s/步** | 42–67 W | `PerRow` 在 `sm_89` 被 `torchao` 拦截，只能 `PerTensor`；速度与 `torchao` INT8 几乎一致 |
| **Colab L4** | **T2I (`M=4100`)** | `bf16` (cuBLAS) | 54.96 ms | **2.75 s/步** | 61–74 W | — |
| **Colab L4** | **T2I (`M=4100`)** | `triton_int8_convrot` (`tl.dot`) | 51.11 ms | **2.56 s/步** | 68–72 W | 旧实现 |
| **Colab L4** | **T2I (`M=4100`)** | **`cublaslt_int8_convrot` (`_int_mm`+Triton反量化)** | **36.80 ms** | **1.84 s/步** | — | **较 BF16 快 1.49×（超越 `torchao` FP8）** |
| **Colab L4** | **T2I (`M=4100`)** | **`torchao` W8A8 FP8 (`PerTensor`+`compile`)** | **38.73 ms** | **1.94 s/步** | 71–72 W | 较 BF16 快 ~1.42×；注：`torchao_int8` 本轮因命中 `recompile_limit=8` 退回 eager（151.3 ms） |

### 关键结论
1. **L4 真正拉开速度差距的不是 FP8，而是 `torch._int_mm`（cuBLASLt INT8 GEMM）+ Inductor 融合量化/反量化 epilogue**：`M=8500` 下 `torchao` INT8 (`3.33s`) 与 `torchao` FP8 (`3.24s`) 几乎打平，均比我们的纯 Triton `tl.dot` GEMM (`5.14s`) 快约 **1.8 s/步（整图 6 步省 ~11 秒）**；
2. **`torchao` 在 L4 上限制了 FP8 粒度**：`torchao 0.10.0` 的 `PerRow` 动态 FP8 硬要求 `sm_90+`（H100），L4 只能用 `PerTensor`，抗离群值精度不如 `INT8-ConvRot`（`int8 [N, K]` + 逐通道 `scale [N]`）；
3. **`torchao` 所需权重**：它直接接收普通 `BF16` 权重在线 `quantize_()`（需要 40–66 GB BF16），或者存成它的 `LinearActivationQuantizedTensor` / `AffineQuantizedTensor`（内部存 `int8 [N, K]` + `bf16 scale [N]`），无法零转换直接吃 Comfy-Org 20.98 GB 的 `int8_convrot` 单文件。相反，如果我们把 **`h3_infer` 里现有的 Triton `tl.dot` GEMM 换回 `torch._int_mm` cuBLASLt + 轻量化 Triton/Inductor 反量化融合**，就能**直接复用已下载的 20.98 GB Comfy INT8-ConvRot 权重（不丢画质、不引 `torchao` 依赖），同时拿到接近 `3.3–3.6 s/步` 的 cuBLASLt 速度**！
