# Colab Ultra

在本机用一套命令统一调度 **Google Colab（CPU / GPU / TPU）** 与 **Kaggle 免费双卡 T4×2**：把代码送到云端前台执行、实时落盘日志、自动回传产物，并用硬件功耗、磁盘 I/O 与任务心跳守住算力防空烧。

![Colab Ultra 工作区与算力监控](assets/colab-ultra-overview.jpg)

---

## 核心能力

- **算力防空烧与真关机**：持续监控 GPU 功耗、CPU、磁盘 I/O 与任务心跳，低负载自动告警并拉闸，支持独立 VPS 哨兵防本机休眠失联；关机时向云端核验真实释放（含孤儿实例），避免假关机持续扣费。
- **免费双卡优先与智能路由**：请求普通 T4 时默认优先路由免费 Kaggle 双卡 T4×2，失败自动回退 Colab，并在有免费额度仍手动选付费渠道时给出费率提示。
- **前台长连接与断连可恢复**：通过 Jupyter WebSocket 前台执行脚本，保持内核持续活跃以规避平台空闲回收；支持网络抖动自动重试，以及断连或内核繁忙时随时回看实时日志。
- **全局共享模型库与高速暂存**：多项目共用 `ColabUltra/models/` 与清单校验，避免重复下载；大权重优先走高速并发下载落 VM 本地 NVMe 再顺序写盘，Kaggle 容器通过 Drive API 共享同一模型库并自动同步产物。
- **原生适配 AI Agent 协作**：长任务结束或出错即自动唤醒调用方（无需轮询），提供结构化 JSON 状态与日志输出，并支持本地零算力 `--dry-run` 预检。

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

随后按最小闭环跑通全流程：

```bash
colab-ultra up -s demo
colab-ultra exec -s demo -f workflows/check_env.py
colab-ultra logs -s demo -n 20
colab-ultra down -s demo
```

> 完整子命令与参数说明以 CLI 实时帮助为准，运行 `colab-ultra --help` 或 `colab-ultra <子命令> --help` 查看；完整命令约定与 Agent 协作规范见 [SKILL.md](SKILL.md)。

---

## 工作区目录约定

本机 `~/ColabUltra/` 映射 Google Drive 的 `MyDrive/ColabUltra/`；Colab 挂载在 `/content/drive/MyDrive/ColabUltra/`，Kaggle 在本地磁盘模拟同名路径并自动同步：

```text
~/ColabUltra/
├── models/                    所有项目共用的全局模型库（按 <模型>/<量化版本>/ 与 manifest.json 组织）
├── scripts/cu_remote/         VM 上直接 import 的共享运行库
└── <项目>/
    ├── code/                  任务脚本与配置
    ├── data/                  输入数据
    └── runs/                  输出产物（自动保留/回传至本机）
```

---

## 仓库导航与参考文档

| 路径 / 文档 | 说明 |
| --- | --- |
| `colab_ultra/` | 本机 CLI、调度与生命周期管理，以及 VM 共享运行库源码（`remote/`） |
| `recipes/` | 开箱即用的推理/任务配方与自定义模板，接入方式见 [recipes/README.md](recipes/README.md) |
| `workflows/` & `tools/` | 环境自检、初始化与模型迁移等辅助脚本 |
| [SKILL.md](SKILL.md) | 完整命令约定、硬件路由规则、Agent 零轮询协作规范与实机踩坑记录 |
| [references/](references/) | 专题参考文档目录（硬件规格与算力选型、共享模型库规范、高速下载与 NVMe 暂存基准等） |
