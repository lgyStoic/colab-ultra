# Colab Ultra

Colab Ultra 在本机管理 Google Colab 和 Kaggle 会话。它把项目脚本送到云端执行，实时显示日志，用硬件负载和任务心跳判断空闲，并在关机后核实实例是否已释放。

![Colab Ultra 工作区与算力监控](assets/colab-ultra-overview.jpg)

## 安装

在本仓库根目录运行以下命令。当前版本使用 `google-colab-cli` 0.7.5 的客户端接口，通过 ADC（Google 应用默认凭据）认证。Google Drive for Desktop 将项目同步到本机，Kaggle 凭据和远端 VPS 哨兵按需配置。

```bash
uv tool install --with requests --with rich --with websocket-client google-colab-cli==0.7.5
mkdir -p ~/.local/bin
ln -s "$PWD/colab_ultra/breaker.py" ~/.local/bin/colab-ultra
colab-ultra init
```

已有的 `~/.local/bin/colab-ultra` 链接可以继续使用。`init` 检查客户端接口、配置凭据和告警，并在检测到本地 Drive 时部署运行库。仅更新运行库时执行 `colab-ultra sync-remote`。配置文件默认为 `~/.config/colab-ultra/config.json`，可用 `COLAB_ULTRA_CONFIG` 更改路径。

## 运行项目

把配置放在 `~/ColabUltra/<project>/code/prompts.json`。在仓库根目录可先离线查看计划。

```bash
H3_PROMPTS_JSON=~/ColabUltra/tiles/code/prompts.json python3 recipes/minimax_h3_sdcpp_server.py --dry-run
```

以下命令分别用于查看算力、启动、执行和释放，逐条运行以保留实时日志。

```bash
colab-ultra status --json
colab-ultra up -s l4 --gpu L4
colab-ultra exec -s l4 -p tiles -f recipes/minimax_h3_sdcpp_server.py --timeout 7200
colab-ultra down -s l4
```

`watch` 根据功耗、CPU、I/O 和心跳发出告警或释放空闲机器。`pull` 使用临时 CPU 节点下载文件到 Drive。共享模型用 `--dest models/<model>/<variant>`，项目专用文件用 `-p <project>`。`colab-ultra --help` 和各子命令的 `--help` 列出完整选项。用法、计费和下载注意事项见 [SKILL.md](SKILL.md)。

## 目录

| 路径 | 用途 |
| --- | --- |
| `colab_ultra/breaker.py`、`cli.py` | 本机命令入口、监控循环、负载判定和参数解析 |
| `colab_ultra/backends/`、`session.py` | Colab、Kaggle、Jupyter 传输以及会话生命周期 |
| `config.py`、`drive.py`、`staging.py`（位于 `colab_ultra/`） | 配置、持久 Drive 挂载、运行库同步和下载任务 |
| `colab_ultra/telemetry.py`、`alerts.py` | 探针数据、仪表盘、通知和 webhook |
| `colab_ultra/compat/` | 已安装客户端的版本和接口检查 |
| `colab_ultra/remote/` | 部署到 Drive 后可在 VM 上导入的 `cu_remote` 包 |
| `recipes/` | DiffSynth 与常驻 sd-server 示例，以及新任务模板 |
| `workflows/` | 环境检查、Drive 初始化、图像生成和 CPU 数据处理脚本 |
| `tests/` | 离线测试，使用模拟的网络、进程和会话 |

共享权重放在 `~/ColabUltra/models/<model>/<variant>/`，每个版本用 `manifest.json` 记录文件来源和大小。项目代码、输入和输出分别放在项目的 `code/`、`data/` 和 `runs/` 下。旧目录的迁移方法见[共享模型库](references/model-library.md)。详见 [recipe 使用说明](recipes/README.md) 和 [VM 暂存与测速示例](references/fast-nvme-staging.md)。
