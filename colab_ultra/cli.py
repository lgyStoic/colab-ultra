import argparse
import json
import os
import subprocess

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from colab_ultra.alerts import send_alert
from colab_ultra.backends.kaggle import get_kaggle_quota
from colab_ultra.breaker import watch_loop
from colab_ultra.compat.colab_cli_patch import check_compatibility
from colab_ultra.config import (
    BASE_DIR,
    CONFIG_PATH,
    DEFAULT_GCLOUD_ADC,
    PROBE_LOCAL_PATH,
    VPS_SENTINEL_LOCAL_PATH,
    load_config,
    save_config,
)
from colab_ultra.dashboard import add_dashboard_parser, cmd_dashboard
from colab_ultra.drive import detect_local_drive_hub
from colab_ultra.models_cli import add_models_parser, cmd_models
from colab_ultra.queue_cli import add_queue_parser, cmd_queue
from colab_ultra.remote import hardware
from colab_ultra.ssh_cli import add_ssh_parser, cmd_ssh
from colab_ultra.session import (
    cmd_down,
    cmd_exec,
    cmd_logs,
    cmd_pull,
    cmd_up,
)
from colab_ultra.staging import sync_remote
from colab_ultra.telemetry import (
    build_agent_status_payload,
    inspect_all_vms,
    render_dashboard,
    sync_dashboard_json,
)

console = Console()


def cmd_init(args):
    """Configure credentials, Drive runtime, alerts and the optional VPS sentinel."""
    cfg = load_config()
    if args.alert_minutes is not None:
        cfg["alert_after_minutes"] = args.alert_minutes
    if args.stop_minutes is not None:
        cfg["stop_after_minutes"] = args.stop_minutes
    if args.vps is not None:
        cfg["vps_host"] = args.vps
    if args.webhook is not None:
        cfg["webhook_url"] = args.webhook

    console.print(
        Panel(
            "🚀 [bold cyan]Colab Ultra 一键初始化向导 (colab-ultra init)[/bold cyan]\n"
            "自动检查：0) 官方 CLI 兼容性  1) Google Colab + Kaggle 免费双卡凭证  2) 显卡瓦数断路器配置  3) 远端 VPS 24h 哨兵同步",
            border_style="cyan",
        )
    )

    check_compatibility()

    # Ensure Agent SKILL.md symlinks are installed for Codex / Antigravity / Agents
    for skill_dir in ("~/.codex/skills", "~/.gemini/config/skills", "~/.agents/skills"):
        parent = os.path.expanduser(skill_dir)
        if os.path.isdir(parent):
            link_path = os.path.join(parent, "colab-ultra")
            if not os.path.exists(link_path):
                try:
                    os.symlink(BASE_DIR, link_path)
                except Exception:
                    pass

    adc_path = DEFAULT_GCLOUD_ADC
    scopes_needed = (
        "openid,"
        "https://www.googleapis.com/auth/cloud-platform,"
        "https://www.googleapis.com/auth/userinfo.email,"
        "https://www.googleapis.com/auth/colaboratory"
    )
    if getattr(args, "with_drive_scope", False):
        scopes_needed += ",https://www.googleapis.com/auth/drive"

    if os.path.exists(adc_path) and not args.auth and not getattr(args, "with_drive_scope", False):
        console.print("✅ [green]步骤 1/3: 已检测到本地 Google ADC 凭证[/green]")
    else:
        console.print("[cyan]🔑 正在拉起 Google 浏览器授权...[/cyan]")
        subprocess.call(["gcloud", "auth", "application-default", "login", f"--scopes={scopes_needed}"])

    kq = get_kaggle_quota()
    if kq.get("configured") and "gpu_remaining_hours" in kq:
        console.print(
            f"✅ [green]Kaggle 免费双卡 T4x2 (30GB) 已就绪: 本周剩余 {kq['gpu_remaining_hours']:.1f}h / {kq['gpu_total_hours']:.0f}h (0 CU/h)[/green]"
        )
    else:
        console.print("[yellow]ℹ️ 未检测到 ~/.kaggle/access_token，仅使用 Google Colab 算力池。[/yellow]")

    local_hub = detect_local_drive_hub()
    if local_hub:
        sync_remote(local_hub)
        console.print(f"✅ [green]本地云盘挂载已就绪: ~/ColabUltra -> {local_hub}[/green]")
    else:
        console.print(
            "[yellow]ℹ️ 本地尚未检测到 Google Drive for Desktop 挂载 (~/Library/CloudStorage/GoogleDrive-*)。"
            "建议执行 `brew install --cask google-drive` 并登录同一账号以启用本地 ~/ColabUltra 零算力直读直写。[/yellow]"
        )

    save_config(cfg)
    console.print(
        f"✅ [green]步骤 2/3: 智能电闸规则已保存至 {CONFIG_PATH} (仓库外隔离存储)[/green]\n"
        f"   • 物理功耗判定: A100/H100 < 75W 视为空载 | L4/T4 < 28W (双卡单卡 < 30W) 视为空载\n"
        f"   • 预警时间: {cfg['alert_after_minutes']} 分钟 | 自动备份并拉闸时间: {cfg['stop_after_minutes']} 分钟"
    )

    vps_host = (cfg.get("vps_host", "") or "").strip()
    ssh_key = os.path.expanduser(cfg.get("vps_ssh_key", "~/.ssh/id_ed25519") or "~/.ssh/id_ed25519")
    vps_poller = VPS_SENTINEL_LOCAL_PATH
    if vps_host and os.path.exists(ssh_key) and os.path.exists(vps_poller) and os.path.exists(adc_path):
        console.print(f"🔄 [cyan]步骤 3/3: 正在将最新凭证与 24h 智能电闸哨兵同步至 VPS ({vps_host})...[/cyan]")
        try:
            subprocess.check_call([
                "scp", "-q", "-i", ssh_key, "-o", "StrictHostKeyChecking=no",
                adc_path, PROBE_LOCAL_PATH, vps_poller, f"{vps_host}:/tmp/"
            ])
            subprocess.check_call([
                "ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no", vps_host,
                "sudo mv /tmp/application_default_credentials.json /etc/colab-adc.json && "
                "sudo chmod 600 /etc/colab-adc.json && "
                "sudo mv /tmp/telemetry_probe.py /usr/local/share/colab_probe.py && "
                "sudo mv /tmp/vps_sentinel.py /usr/local/bin/update-colab-stats.py && "
                "sudo chmod +x /usr/local/bin/update-colab-stats.py && "
                "sudo /usr/local/bin/update-colab-stats.py"
            ])
            console.print(f"✅ [green]步骤 3/3: VPS ({vps_host}) 24h 独立值班哨兵已同步就绪！[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ VPS 同步失败: {e}[/yellow]")
    else:
        console.print("ℹ️ [dim]步骤 3/3: 未配置远端 VPS 或跳过同步，使用本地守护模式。[/dim]")

    ccu, vms = inspect_all_vms()
    render_dashboard(ccu, vms)


def cmd_specs(args):
    """Show the hardware catalog without contacting a backend."""
    specs = hardware.list_specs(
        precision=args.precision, min_vram_gb=args.min_vram, kind=args.kind,
    )
    if args.json:
        rows = [hardware.as_dict(spec) for spec in specs]
        if args.precision is not None:
            for row, spec in zip(rows, specs):
                row["selected_precision"] = args.precision
                row["efficiency_per_cu"] = spec.tflops_per_cu(args.precision)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    table = Table(
        box=box.SIMPLE,
        padding=0,
        caption="Dense Tensor Core peak. Sparse 2:4 can double throughput on Ampere and newer.",
    )
    table.add_column("Key", no_wrap=True)
    table.add_column("Arch / SM", overflow="fold")
    table.add_column("VRAM / RAM", justify="right", overflow="fold")
    for title in ("BW (GB/s)", "FP16", "BF16", "INT8 (TOPS)", "FP8", "NVFP4", "CU/h"):
        table.add_column(title, justify="right", overflow="fold")
    if args.precision is not None:
        table.add_column(f"{args.precision.upper()}/CU", justify="right", overflow="fold")

    for spec in specs:
        metrics = (
            spec.fp16_tflops, spec.bf16_tflops, spec.int8_tops, spec.fp8_tflops, spec.nvfp4_tflops,
        )
        row = [
            spec.key,
            f"{spec.arch} / {spec.sm}" if spec.sm else spec.arch,
            f"{spec.vram_gb:g} / {spec.ram_gb:g}",
            f"{spec.bandwidth_gb_s:g}",
            *(f"{value:g}" if value is not None else "-" for value in metrics),
            f"{spec.cu_per_hour:g}",
        ]
        if args.precision is not None:
            efficiency = spec.tflops_per_cu(args.precision)
            if efficiency is not None:
                row.append(f"{efficiency:.2f}")
            elif spec.cu_per_hour == 0 and spec.metric_for(args.precision) is not None:
                row.append("FREE")
            else:
                row.append("-")
        table.add_row(*row)
    console.print(table)


def build_parser():
    """Build the public CLI without contacting a backend."""
    parser = argparse.ArgumentParser(description="Colab Ultra: Agent-First Smart Breaker & 20TB Drive Hub (Colab + Free Kaggle T4x2)")
    sub = parser.add_subparsers(dest="cmd")

    # init
    p_init = sub.add_parser("init", help="一键初始化向导（检查官方 CLI 兼容性、配置凭证、瓦数断路器阈值、同步 VPS 哨兵）")
    p_init.add_argument("--auth", action="store_true", help="强制重新运行浏览器 OAuth 授权")
    p_init.add_argument("--with-drive-scope", action="store_true", help="同时授权 Google Drive API 权限以开启 Kaggle 云对云超大模型直传")
    p_init.add_argument("--alert-minutes", type=float, default=None, help="设置低功耗预警分钟数 (默认 5 分钟)")
    p_init.add_argument("--stop-minutes", type=float, default=None, help="设置低功耗自动拉闸分钟数 (默认 10 分钟)")
    p_init.add_argument("--vps", type=str, default=None, help="绑定远程 VPS 地址以实现 24h 离线断电保护")
    p_init.add_argument("--webhook", type=str, default=None, help="设置 Bark/飞书/Telegram 告警 Webhook URL")

    # status / top
    p_status = sub.add_parser("status", help="查看实时算力账单与所有 Colab / Kaggle 机器的 GPU 瓦数/CPU/云盘负载")
    p_status.add_argument("--json", action="store_true", help="输出 Agent 友好的结构化 JSON 诊断数据")

    # watch
    p_watch = sub.add_parser("watch", help="启动智能电闸后台轮询守护（低负载告警 + 超时自动拉闸）")
    p_watch.add_argument("--interval", type=int, default=None, help="轮询间隔秒数 (默认 15s)")
    p_watch.add_argument("--gpu-threshold", type=float, default=None, help="最低 GPU 利用率百分比阈值 (默认 5%%)")
    p_watch.add_argument("--alert-minutes", type=float, default=None, help="低负载多少分钟触发报警 (默认 5 分钟)")
    p_watch.add_argument("--stop-minutes", type=float, default=None, help="低负载多少分钟自动拉闸关机 (默认 10 分钟)")
    p_watch.add_argument("--once", action="store_true", help="仅执行一次巡检与阈值判定后退出")

    # up
    p_up = sub.add_parser("up", help="一键合闸开机（支持免费 Kaggle T4x2 / Colab GPU / 8核CPU / 44核TPU）、统一工作区与探针")
    p_up.add_argument("-s", "--session", default="ultra", help="会话名称 (默认 ultra；多节点时自动命名为 <session>-1..N)")
    p_up.add_argument("--gpu", default="T4", help="硬件型号: T4 (默认优先用免费 Kaggle T4x2), T4x2, L4, G4, A100, H100, CPU")
    p_up.add_argument("--kaggle", action="store_true", default=False, help="强制使用免费 Kaggle 双卡 T4x2 (30GB 显存, 0 CU/h)")
    p_up.add_argument("--colab", action="store_true", default=False, help="强制使用 Google Colab 节点 (跳过免费 Kaggle 优先路由)")
    p_up.add_argument("--cpu", action="store_true", default=False, help="启动纯 CPU 数据处理节点 (默认自动开启 8核50GB 高内存模式，仅 0.26 CU/h)")
    p_up.add_argument("--tpu", default=None, help="启动超高核 TPU 宿主机做大数据处理: v5e1 (24核47GB, 3.09 CU/h) 或 v6e1 (44核172GB, 5.09 CU/h)")
    p_up.add_argument("--high-mem", action="store_true", default=False, help="开启高内存模式 (--cpu 默认自动开启)")
    p_up.add_argument("--std-mem", action="store_true", default=False, help="在 --cpu 模式下强制使用标准内存 (2核12GB, 0.08 CU/h)")
    p_up.add_argument("--workers", type=int, default=1, help="并发拉起多台节点数量 (例如 --cpu --workers 5 可并发拉起 40核250GB 集群)")
    p_up.add_argument("--no-drive", action="store_true", default=False, help="跳过自动挂载 20TB Google Drive")
    p_up.add_argument("--json", action="store_true", help="以结构化 JSON 返回实例就绪与硬件状态 (Agent 专用)")

    # exec (Agent-first remote execution + telemetry return)
    p_exec = sub.add_parser("exec", help="在 Colab 实例中执行代码或脚本，并自动返回最新 GPU 瓦数/CPU核数/显存/云盘状态")
    p_exec.add_argument("-s", "--session", default="ultra", help="会话名称 (默认 ultra)")
    p_exec.add_argument("-p", "--project", default=None, help="项目名称（直接切换到 /content/drive/MyDrive/ColabUltra/<project>/code 下执行）")
    p_exec.add_argument("-f", "--file", default=None, help="要执行的脚本路径")
    p_exec.add_argument("--timeout", type=float, default=3600.0, help="远程执行超时秒数 (默认 3600s)")
    p_exec.add_argument("--restart-kernel", action="store_true", help="执行前自动重启 Jupyter 内核以释放残留显存")
    p_exec.add_argument("code", nargs="?", default=None, help="要直接执行的 Python 代码片段")
    p_exec.add_argument("--json", action="store_true", help="输出包含 stdout/stderr 与实时硬件瓦数的 JSON (Agent 专用)")

    # pull (Zero-burn cloud download of models/data directly to Drive)
    p_pull = sub.add_parser("pull", help="在 0-CU 云端 CPU 实例上极速下载大文件/模型权重至 Google Drive，下完自动拉闸关机")
    p_pull.add_argument("source", help="下载地址 (HTTP/HTTPS) 或 Hugging Face 仓库/文件名")
    p_pull.add_argument("-p", "--project", default=None, help="写入指定项目；省略时写入 ColabUltra 根目录下")
    p_pull.add_argument("--dest", default="models", help="目标子目录。共享模型用 models/<model>/<variant>；指定项目时默认 models")
    p_pull.add_argument("--filename", default=None, help="指定文件名 (可选)")
    p_pull.add_argument("-s", "--session", default="pull-worker", help="临时 CPU 会话名称 (默认 pull-worker)")
    p_pull.add_argument("--json", action="store_true", help="输出结构化 JSON (Agent 专用)")

    # down
    p_down = sub.add_parser("down", help="安全备份输出至云盘并拉闸关机（含网页端孤儿实例），停止扣费")
    p_down.add_argument("-s", "--session", default=None, help="指定会话名或节点前缀")
    p_down.add_argument("--all", action="store_true", help="关闭账号下所有运行中的 Colab 机器")
    p_down.add_argument("--no-backup", action="store_true", help="跳过产物回传并直接关机")
    p_down.add_argument("--json", action="store_true", help="以结构化 JSON 返回关机结果与剩余算力 (Agent 专用)")

    # test-alert
    sub.add_parser("test-alert", help="测试低负载多通道告警（Mac 系统通知 + 语音播报 + Webhook）")

    sub.add_parser("sync-remote", help="同步 cu_remote 运行库到本地 Google Drive")
    add_models_parser(sub)
    add_queue_parser(sub)
    add_dashboard_parser(sub)
    logs = sub.add_parser("logs", help="读取会话最近一次 exec 的远端日志")
    logs.add_argument("-s", "--session", default="ultra", help="会话名称 (默认 ultra)")
    logs.add_argument("-n", "--lines", type=int, default=100, help="显示最后 N 行，0 表示全部")
    logs.add_argument("--json", action="store_true", help="输出 JSON")

    specs = sub.add_parser("specs", help="查看各硬件规格、BF16/INT8/FP8/NVFP4 算力与性价比")
    specs.add_argument("--precision", choices=["fp16", "bf16", "int8", "fp8", "nvfp4"], default=None)
    specs.add_argument("--min-vram", type=float, default=0.0, help="最低显存容量 (GB)")
    specs.add_argument("--kind", choices=["gpu", "cpu", "tpu"], default=None)
    specs.add_argument("--json", action="store_true", help="输出 JSON")
    add_ssh_parser(sub)
    return parser


def main(argv=None):
    """Dispatch a workstation command."""
    args = build_parser().parse_args(argv)
    if args.cmd == "sync-remote":
        print(sync_remote())
        return
    if args.cmd in (None, "status"):
        ccu, vms = inspect_all_vms()
        if getattr(args, "json", False):
            sync_dashboard_json(ccu, vms)
            print(json.dumps(build_agent_status_payload(ccu, vms), indent=2, ensure_ascii=False))
        else:
            render_dashboard(ccu, vms)
    elif args.cmd == "init":
        cmd_init(args)
    elif args.cmd == "watch":
        watch_loop(args)
    elif args.cmd == "up":
        cmd_up(args)
    elif args.cmd == "exec":
        cmd_exec(args)
    elif args.cmd == "logs":
        cmd_logs(args)
    elif args.cmd == "models":
        cmd_models(args)
    elif args.cmd == "queue":
        cmd_queue(args)
    elif args.cmd == "dashboard":
        cmd_dashboard(args)
    elif args.cmd == "specs":
        cmd_specs(args)
    elif args.cmd == "ssh":
        cmd_ssh(args)
    elif args.cmd == "pull":
        cmd_pull(args)
    elif args.cmd == "down":
        cmd_down(args)
    elif args.cmd == "test-alert":
        cfg = load_config()
        send_alert(
            title="Colab A100 低负载预警 (测试)",
            message="实例 [ultra] (A100) GPU 利用率已连续低于 5% (当前 0.0%, 功耗 44W)。若继续无负载将在 15 分钟后自动拉闸关机！",
            cfg=cfg,
            voice_text="注意，Colab A100 机器负载过低，请检查任务状态。",
        )


if __name__ == "__main__":
    main()
