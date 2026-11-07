# ⚡ Colab Ultra (`colab-ultra`)

**Agent-First Google Colab GPU 算力中枢 × 显卡物理瓦数智能电闸 × 20TB 云盘永久工作区**

![Colab Ultra 功能全景图](assets/colab-ultra-overview.jpg)

`colab-ultra` 是建立在 Google 官方 [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) 之上的**下一代 Agent 优先（Agent-Native）算力调度与防爆仓护栏系统**。它既可供人类开发者在终端一键开卡，也可作为 [`SKILL.md`](SKILL.md) 直接挂载给 Claude Code、Codex、Antigravity、Cursor 等 AI Agent 自主调用 A100 / H100 算力，同时彻底杜绝「Agent 任务跑完或死锁忘关机，一觉醒来算力点（CCU）烧光」的痛点。

---

## ✨ 相比官方 `google-colab-cli` 的 5 大核心升级

| 维度 | 官方 `google-colab-cli` (v0.7.5) | ⚡ `colab-ultra` (本项目) |
| :--- | :--- | :--- |
| **1. 空载/死锁检测指标** | ❌ 无任何硬件监控（仅能看账号余额） | ✅ **显卡物理功耗（瓦数 W）真值判定**：识破 NCCL 死锁 `100%` 假利用率（A100/H100 `< 75W`、T4/L4 `< 28W` 判为空载） |
| **2. 自动断电拉闸保护** | ❌ 无自动关机，合盖或断开即持续扣费 | ✅ **5 分钟预警 + 10 分钟自动抢救文件并强制拔电源**（支持本机守护 + **VPS 24h 离线哨兵**） |
| **3. 启动速度与云盘挂载** | 🐛 `--auth=adc` 每次卡 13.2 秒（见上游 [PR #218](https://github.com/googlecolab/google-colab-cli/pull/218)），`drivemount` 报 HTTP 400 | ✅ `colab-ultra init` **自动修补官方 CLI 挂载 Bug + 260 倍凭证读取提速（13.2s → 0.05s）**，开机默认挂载云盘并软链 `/content/workspace` 与 `/content/outputs` |
| **4. 断电前急救箱** | ❌ 实例销毁即丢失 `/content` 全部成果 | ✅ 拉闸前 2 秒自动将 `/content/outputs` 及根目录 `.py/.ipynb/.safetensors/.pt` 抢救至云盘，下次开机自动还原 |
| **5. AI Agent 原生支持** | ❌ 仅面向人类交互，缺少结构化硬件诊断 | ✅ 全命令支持 `--json`（一次调用同时返回执行结果 + 实时瓦数/显存/云盘/CCU 余额），内置 [`SKILL.md`](SKILL.md) |

---

## 📂 仓库目录结构

```text
colab-ultra/
├── README.md                     # 项目文档与快速上手指南
├── SKILL.md                      # AI Agent (Claude Code / Codex / Antigravity) 原生技能定义
├── config.example.json           # 零隐私配置模板（实际配置存放在仓库外 ~/.config/colab-ultra/config.json）
├── assets/
│   └── colab-ultra-overview.jpg  # 功能架构全景图
├── colab_ultra/
│   ├── breaker.py                # 主 CLI 入口 (up / status / exec / watch / down / init)
│   ├── telemetry_probe.py        # 远端 Colab 实例每 3 秒静默采样 GPU 瓦数/显存/云盘的探针
│   ├── patch_colab_cli.py        # 官方 google-colab-cli 0.05s 极速 ADC 启动与云盘挂载补丁
│   └── vps_sentinel.py           # 24 小时 VPS 离线巡检与自动拉闸哨兵脚本
└── workflows/
    ├── init_20tb_drive.py        # 开机自动挂载 Google Drive、软链工作区并恢复断电抢救文件
    ├── check_env.py              # 远端 PyTorch / CUDA / 显存与 BF16 环境快速自检
    ├── a100_benchmark.py         # A100/H100 FP32/TF32/BF16 GEMM 算力与 FlashAttention-2 基准测试
    ├── cpu_data_pipeline.py      # 多核高内存纯 CPU 数据清洗/分片并行处理基准工作流
    ├── ai_image_gen.py           # 开源生图工作流示例（自动持久化至云盘）
    └── llm_runner.py             # 开源大模型推理工作流示例（共享云盘 HuggingFace 缓存）
```

---

## 📊 Google One AI Ultra / Colab Pro+ 纯 CPU 与大数据处理实测规格表

做数据清洗、Parquet/Arrow 转换、去重分片或爬虫预处理时，完全不需要烧昂贵的 A100（`5.39 CU/h`）。`colab-ultra` 内置了针对**纯 CPU 与超大内存数据处理**的分级调度与 CPU 负载保护（不会因为没有 GPU 瓦数而误杀正在跑数据的 CPU 节点，跑完空闲 10 分钟同样自动备份并关机）：

| 模式 | 启动命令 | CPU 规格 | 内存 (RAM) | 本地磁盘 | 扣费速率 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **标准 CPU** | `colab-ultra up --cpu --std-mem` | **2 vCPU** (Intel Xeon) | **12.7 GB** | 225 GB | **`0.08 CU/h`** | 极低成本长时下载、解压、API 批处理（1 CU 可跑 12.5 小时） |
| **高内存 CPU (默认推荐)** | `colab-ultra up --cpu` | **8 vCPU** (AMD EPYC 7B12) | **51.0 GB** | 225 GB | **`0.26 CU/h`** | 日常多进程数据预处理、Polars/Pandas 大表清洗（比 A100 便宜 20 倍） |
| **5 节点 CPU 并发集群** | `colab-ultra up --cpu --workers 5 -s data` | **40 vCPU** (5 × 8 核) | **255 GB** (5 × 51G) | 1.1 TB + 共享云盘 | **`1.30 CU/h`** | 大规模数据集分片并行处理（5 台机器同时读写同一个 Google Drive） |
| **44 核超大内存单机** | `colab-ultra up --tpu v6e1 -s bigmem` | **44 vCPU** (AMD EPYC 9B14 Zen4) | **172.0 GB** | 225 GB | **`5.09 CU/h`** | 单机需 >100GB 内存的巨型内存数据库（DuckDB / 图计算 / 全量去重） |

---

## 🚀 3 分钟快速上手

### 1. 安装基础依赖与 `colab-ultra`
```bash
# 1. 安装官方底层 google-colab-cli
uv tool install google-colab-cli
uv tool run --from google-colab-cli pip install rich websocket-client requests

# 2. 克隆本仓库并创建全局软链
git clone https://github.com/lgyStoic/colab-ultra.git
cd colab-ultra
chmod +x colab_ultra/breaker.py
ln -sf "$(pwd)/colab_ultra/breaker.py" ~/.local/bin/colab-ultra
```

### 2. 一键初始化（自动修补官方 CLI + 配置凭证 + 设置电闸）
```bash
# 纯本地模式（默认 5 分钟预警、10 分钟自动备份并关机）：
colab-ultra init

# 如果你有自己的 Linux VPS，一键开启 24h 离线哨兵同步：
colab-ultra init --vps root@your-vps-ip --alert-minutes 5 --stop-minutes 10
```
> 🔒 **隐私安全设计**：你的所有个人配置（如 `vps_host`、SSH 密钥路径、Webhook URL）均自动保存在仓库外部的 `~/.config/colab-ultra/config.json`（权限 `600`），代码库内永远保持零隐私状态。

---

## 💻 日常使用指南（人类 & AI Agent 通用）

### 1. 开机并自动挂载云盘永久工作区
```bash
# 默认启动省钱的 Tesla T4 并自动挂载 Google Drive：
colab-ultra up -s work

# 启动 8 核 51GB 高内存纯 CPU 节点做数据清洗（仅 0.26 CU/h）：
colab-ultra up --cpu -s data

# 一次性并发拉起 5 台 8 核 51GB 纯 CPU 集群（合计 40 核 + 255GB 内存，共享云盘）：
colab-ultra up --cpu --workers 5 -s data

# 启动 44 核 AMD EPYC Zen4 + 172GB 超大内存单机（借道 TPU v6e1 宿主机）：
colab-ultra up --tpu v6e1 -s bigmem

# 启动 A100 / H100 跑大模型或 DiT LoRA 微调：
colab-ultra up -s ultra --gpu A100
```

### 2. 查看实时瓦数、温度、显存与云盘状态
```bash
# 终端彩色仪表盘：
colab-ultra status

# AI Agent 专用结构化 JSON 输出：
colab-ultra status --json
```

### 3. 远程执行代码/脚本并返回实时硬件遥测（Agent-First）
```bash
colab-ultra exec -s ultra "import torch; print(torch.cuda.get_device_name(0))" --json
colab-ultra exec -s data -f workflows/cpu_data_pipeline.py --json
colab-ultra exec -s ultra -f workflows/check_env.py --json
colab-ultra exec -s ultra -f workflows/a100_benchmark.py --json
colab-ultra exec -s ultra -f workflows/ai_image_gen.py --json
```

### 4. 开启智能瓦数电闸守护 / 手动安全拉闸
```bash
# 在本地启动守护（满 5 分钟低瓦数语音+弹窗报警，满 10 分钟自动备份云盘并断电）：
colab-ultra watch

# 任务结束，立即安全备份并关闭所有机器（0 扣费）：
colab-ultra down --all
```
