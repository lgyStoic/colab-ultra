# Colab Ultra

在本机用一套命令统一调度 **Google Colab（CPU / GPU / TPU）** 与 **Kaggle 免费双卡 T4×2**：把代码送到云端前台执行、实时落盘日志、自动回传产物，并用硬件功耗、磁盘 I/O 与任务心跳守住算力防空烧。

![Colab Ultra 工作区与算力监控](assets/colab-ultra-overview.jpg)

---

## 我们解决了哪些问题

| 真实痛点 | Colab Ultra 怎么解决的 |
| --- | --- |
| **1. 付费 GPU 空烧扣费、假关机（孤儿实例）** | `watch` 综合 GPU 功耗、CPU、磁盘读写和任务心跳判定空载（默认 5 分钟告警、10 分钟自动拉闸），配合独立 VPS 24 小时哨兵防本机休眠失联；`down` 内置 120s 超时保护并向云端核验真实释放（含网页端孤儿实例），紧急关机可用 `--no-backup` 跳过产物回传直接拉闸。 |
| **2. 有免费 Kaggle 双卡，却误花钱开 Colab T4** | 请求单台普通 `T4` 时默认优先申请每周 30h 免费的 Kaggle 双卡 T4×2（30 GB 显存，0 CU/h），失败自动回退 Colab T4；额度接口抖动时读取本地缓存 `kaggle_quota_cache.json` 兜底；显式加 `--colab` 且 Kaggle 仍有额度时，CLI 会直接黄字提醒费率（约 1.19 CU/h）。 |
| **3. 长任务断连丢日志、SSL 抖动、30 分钟空闲回收** | 所有脚本通过进程内 Jupyter WebSocket 前台运行，保持内核持续 `BUSY` 不触发平台空闲回收；每次 `exec` 用独立新 inode 将 `stdout`/`stderr` 实时写入 `/content/.cu_last_exec.log`，断连或内核忙时可用 `colab-ultra logs` 随时回看；轮询与传输层自动重试瞬态 `SSLEOFError` / WebSocket 断连，Drive 初始化也改为直连 WebSocket + 目录列表核验以绕开上游关闭假超时。 |
| **4. 权重直写 Drive 极慢、多项目重复囤模型** | 建立全局共享模型库 `ColabUltra/models/<模型>/<量化版本>/` 与 `manifest.json`（配套 `colab-ultra models ls/verify/rm` 零读正文核对、本地迁移去重工具）；Hugging Face / HTTP 默认用 64 并发 `hf_transfer`（实测 34.16 GB 耗时 99.5s / 343.5 MB/s，比 16 并发 `aria2` 的 286.0 MB/s 快 20.1%），先落 VM 本地 NVMe 再顺序写 Drive（440–495 MB/s，对比直写 Drive FUSE 仅 32.6 MB/s）。 |
| **5. Kaggle 无 Drive 挂载，权重与产物两张皮** | Kaggle 容器内模拟与 Colab 一致的 `/content/drive/MyDrive/ColabUltra/<项目>/{code,data,runs}`；`exec` 注入短时 Drive OAuth Token（结束后自动清理），`cu_remote.models.ensure()` 优先走 Google 内网 Drive API 直拉共享模型（缺失文件逐个回退 HF/HTTP），运行中与关机前自动把 `runs/` 产物同步回 Drive 与本地并防护过期会话遮蔽。 |
| **6. AI Agent 协作爱设定时器空转、状态不透明** | 完整适配 Agent 工作流：`exec` 前台长连接结束或出错即自动唤醒 Agent，禁止定时/循环轮询；`status --json` / `logs --json` 返回结构化瓦数、心跳、余额与尾部日志；所有推理脚本支持 `--dry-run` 在本机离线校验配置与下载计划后再开机。 |

---

## 3 分钟快速上手

在 Mac 上准备好 Python、`uv` 与 Google Cloud CLI（`gcloud`）。需要本地与云盘双向同步时，用 Google Drive for Desktop 登录同一 Google 账户；使用免费双卡时将 Kaggle API Token 放入 `~/.kaggle/access_token`。

在仓库根目录执行安装（已有 `~/.local/bin/colab-ultra` 软链可跳过 `ln -s`）：

```bash
uv tool install --with requests --with rich --with websocket-client google-colab-cli==0.7.5
mkdir -p ~/.local/bin
ln -s "$PWD/colab_ultra/breaker.py" ~/.local/bin/colab-ultra
export PATH="$HOME/.local/bin:$PATH"
colab-ultra init --with-drive-scope
```

`init` 会校验客户端兼容接口、配置 ADC 与告警、同步 `cu_remote` 运行库到本地 Drive，并授权 Kaggle 走 Drive API 拉取共享模型。随后逐条运行下面四条命令跑通全流程：

```bash
colab-ultra up -s demo
colab-ultra exec -s demo -f workflows/check_env.py
colab-ultra logs -s demo -n 20
colab-ultra down -s demo
```

长任务可在第二个终端用 `colab-ultra logs -s demo -n 100` 随时查看输出。每次 `exec` 会覆盖上一次日志；需要长期保留的结果请写入项目的 `runs/` 目录，确认回传后再关机。

---

## 常用命令速查

| 命令 | 用途 |
| --- | --- |
| `colab-ultra status` | 查看活跃实例、GPU/CPU 负载、心跳与 CU/Kaggle 余额，加 `--json` 输出结构化数据。 |
| `colab-ultra specs` | 查询硬件规格、精度支持和单位 CU 的峰值算力，可按显存和硬件类型筛选。 |
| `colab-ultra up -s demo` | 启动会话。默认 T4 优先 Kaggle 双卡；可用 `--gpu L4` / `--cpu` / `--colab` / `--kaggle` 指定硬件或渠道。 |
| `colab-ultra exec -s demo -p tiles -f job.py` | 把脚本送到云端并在项目 `tiles/code/` 下前台执行、实时回传输出（加 `--timeout` 调长限时）。 |
| `colab-ultra logs -s demo -n 100` | 读取最近一次 `exec` 日志末尾 100 行与当前心跳任务（`-n 0` 读全文，`--json` 结构化输出，内核 `BUSY` 也可用）。 |
| `colab-ultra pull org/repo --dest models/my-model/bf16` | 用临时 0-GPU CPU 节点按 Google 内网速度下载仓库或文件到共享模型库，下完立即自动关机。 |
| `colab-ultra models ls` | 本机扫描 `~/ColabUltra/models/`，按元数据列出各模型版本的文件数、GB 体积与清单状态（加 `--json` 输出结构化结果）。 |
| `colab-ultra models verify [MODEL/VARIANT]` | 核对磁盘文件存在性与字节数是否符合 `manifest.json`，异常返回退出码 `1`。 |
| `colab-ultra models rm MODEL/VARIANT` | 预览待删除的模型版本及体积，加 `--yes` 才真正删除并在父目录为空时清理。 |
| `colab-ultra down -s demo` | 回传输出（加 `--no-backup` 可跳过直接强关）、释放实例并核验云端算力已停表。 |
| `colab-ultra watch` | 持续巡检功耗、CPU、I/O 与心跳并自动告警/拉闸。 |
| `colab-ultra sync-remote` | 把当前仓库的 `colab_ultra/remote/` 同步到同步盘 `scripts/cu_remote/`。 |

---

## 共享模型库与项目目录

本机 `~/ColabUltra/` 映射 Google Drive 的 `MyDrive/ColabUltra/`；Colab 挂载在 `/content/drive/MyDrive/ColabUltra/`，Kaggle 在本地磁盘模拟同名路径并自动同步项目文件与模型：

```text
~/ColabUltra/
├── models/                    所有项目共用的全局模型库
│   └── <模型>/<量化版本>/
│       ├── manifest.json      记录来源 URL、字节数与 HF LFS sha256
│       └── ...                权重与配套配置/分词器文件
├── scripts/cu_remote/         VM 上直接 import 的共享运行库
└── <项目>/
    ├── code/                  推理脚本、prompts.json
    ├── data/                  输入数据（>80 MB 建议走 Drive API 或 HF/URL 直拉）
    └── runs/                  输出产物（自动保留/回传至本机）
```

---

## 接入自定义推理脚本

从 [recipes/_template.py](recipes/_template.py) 复制一份新脚本（如 `recipes/my_infer.py`），在 `MODELS` 中声明需要的多版本模型，并在 `infer(item, out_path)` 里写自己的模型加载与推理逻辑：

```python
MODELS = [
    ("minimax-h3-image", "bf16", {
        "vae/diffusion_pytorch_model.safetensors":
            "hf://MiniMaxAI/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors",
    }),
]
```

模板内部调用 `cu_remote.models.ensure(model, variant, files, to=MODEL_DIR / model / variant)`：返回按相对路径索引的本地文件路径字典（Colab 自动从共享库先校验再拷入 NVMe，Kaggle 自动优先走 Drive API 再按需回退 HF/HTTP）。

将提示词写入 `~/ColabUltra/tiles/code/prompts.json` 后，先在本机做零算力预检，再推到云端执行：

```bash
H3_PROMPTS_JSON=~/ColabUltra/tiles/code/prompts.json python3 recipes/my_infer.py --dry-run
colab-ultra exec -s demo -p tiles -f recipes/my_infer.py --timeout 7200
```

---

## 仓库结构与进阶文档

| 路径 / 文档 | 说明 |
| --- | --- |
| `colab_ultra/` | 本机 CLI（`cli.py`、`breaker.py`、`models_cli.py`）、传输与生命周期（`session.py`、`backends/`）、配置与挂载（`config.py`、`drive.py`、`staging.py`、`telemetry.py`）与 VM 共享库源码（`remote/`） |
| `recipes/` | 开箱即用的 MiniMax-H3（DiffSynth）配方与自定义模板，详见 [recipes 使用说明](recipes/README.md) |
| `tools/migrate_models.py` | 将旧版分散在项目或分类目录下的模型迁移、去重至全局共享模型库 |
| `workflows/` | 环境自检、20TB Drive 初始化等辅助脚本 |
| [SKILL.md](SKILL.md) | 完整命令约定、T4 默认路由规则、Agent 零轮询协作规范与实机踩坑记录 |
| [references/model-library.md](references/model-library.md) | 共享模型库命名规范、`manifest.json` 结构、`models.ensure()` 用法与迁移指南 |
| [references/fast-nvme-staging.md](references/fast-nvme-staging.md) | `hf_transfer`（64 并发）与 `aria2`（16 并发）实机基准数据与 NVMe 暂存说明 |
| [references/hardware-specs.md](references/hardware-specs.md) | Colab 与 Kaggle 硬件规格、量化精度选择和运行时 GPU 查询 |
