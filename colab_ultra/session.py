import argparse
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from rich.console import Console

from colab_ultra.alerts import log_breaker_event
from colab_ultra.backends.colab import (
    assignments,
    backup_vm_outputs_to_drive,
    colab_state,
    restart_vm_kernel,
    run_detached,
)
from colab_ultra.backends.jupyter import jupyter_exec_ws
from colab_ultra.backends.kaggle import (
    _load_kaggle_sessions,
    build_in_vm_sync_snippet,
    forget_kaggle_session,
    get_kaggle_quota,
    get_short_lived_drive_token,
    is_kaggle_gpu_spec,
    kaggle_active,
    load_kaggle_token,
    pull_live_tarball_to_local,
    shut_down_kaggle_session,
    spin_up_kaggle_session,
    sync_project_to_kaggle,
)
from colab_ultra.config import (
    BASE_DIR,
    PROBE_LOCAL_PATH,
    load_config,
    resolve_colab_bin,
    save_config,
)
from colab_ultra.drive import (
    detect_local_drive_hub,
    initialize_drive,
)
from colab_ultra.staging import kaggle_runtime_code, pull_code
from colab_ultra.telemetry import (
    build_agent_status_payload,
    fetch_vm_telemetry,
    inspect_all_vms,
    render_dashboard,
    start_remote_probe_via_runtime,
    sync_dashboard_json,
    upload_file_to_vm,
)

console = Console()


@contextmanager
def _billing_records():
    """Serialize local billing updates across concurrent CLI processes."""
    path = Path.home() / ".config/colab-cli/session_tracker.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = json.loads(path.read_text()) if path.exists() else {}
        yield records
        save_config(records, path)


def _record_starts(names, vms, ccu, gpu):
    """Remember each start balance and fallback hourly rate for settlement."""
    rates = {"CPU": 0.08, "NONE": 0.08, "T4": 1.19, "T4X2": 0.0, "L4": 1.71,
             "G4": 10.05, "A100": 8.71, "H100": 18.05}
    by_name = {vm["name"]: vm for vm in vms}
    with _billing_records() as records:
        for name in names:
            vm = by_name.get(name, {})
            accelerator = (vm.get("accelerator") or gpu or "CPU").upper()
            provider = vm.get("provider", "colab")
            rate = 0 if provider == "kaggle" else rates.get(accelerator, ccu.consumption_rate_hourly)
            records.setdefault(name, {"session": name, "gpu": accelerator, "provider": provider,
                                      "rate": rate, "start_time": time.time(),
                                      "start_balance": ccu.paid_compute_units_balance})


def _settle(released, end_balance):
    """Attach duration and estimated cost only for verified releases."""
    with _billing_records() as records:
        for vm in released:
            start = records.pop(vm["name"], None)
            if start is None:
                continue
            seconds = max(0, time.time() - float(start["start_time"]))
            spent = max(0, float(start["start_balance"]) - end_balance)
            estimate = seconds * float(start["rate"]) / 3600
            vm.update(duration_sec=round(seconds, 1),
                      duration_str=f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒",
                      estimated_cost_ccu=round(spent if spent > 0.005 else estimate, 3))


def _spin_up_single_session(colab_bin: str, session_name: str, args):
    """Create a single Colab session, mount Drive, and initialize workspace."""
    tpu_choice = (getattr(args, "tpu", None) or "").strip().lower()
    if getattr(args, "cpu", False):
        gpu_choice = "CPU"
    else:
        gpu_choice = (args.gpu or "T4").upper()

    is_cpu = (gpu_choice in ("CPU", "NONE", "OFF")) and not tpu_choice
    # For CPU data processing, default to 8-vCPU / 50GiB High-RAM (0.26 CU/h) unless --std-mem is requested
    use_high_mem = bool(args.high_mem or (is_cpu and not getattr(args, "std_mem", False)))

    cmd = [colab_bin, "--auth", "adc", "new", "-s", session_name]
    if tpu_choice:
        cmd.extend(["--tpu", tpu_choice])
        hw_label = f"TPU-{tpu_choice.upper()} (超高核CPU宿主机)"
    elif not is_cpu:
        cmd.extend(["--gpu", gpu_choice])
        hw_label = f"{gpu_choice}{' High-RAM' if use_high_mem else ''}"
    else:
        hw_label = "CPU High-RAM (8核 50GB, 0.26 CU/h)" if use_high_mem else "CPU Standard (2核 12GB, 0.08 CU/h)"

    # G4 only supports STANDARD shape on Colab; passing --high-mem fails allocation
    if use_high_mem and not tpu_choice and gpu_choice != "G4":
        cmd.append("--high-mem")

    if not getattr(args, "json", False):
        console.print(f"⚡ [bold cyan]正在合闸启动 Colab {hw_label} 实例 ({session_name})...[/bold cyan]")
    subprocess.check_call(
        cmd,
        stdout=subprocess.DEVNULL if getattr(args, "json", False) else None,
        stderr=subprocess.DEVNULL if getattr(args, "json", False) else None,
    )

    if not getattr(args, "no_drive", False):
        if not getattr(args, "json", False):
            console.print(f"📦 [cyan]正在默认自动挂载 20TB Google Drive 并初始化永久工作区 ({session_name})...[/cyan]")
        try:
            initialize_drive(colab_bin, session_name, quiet=args.json)
        except Exception as e:
            if not getattr(args, "json", False):
                console.print(f"[yellow]⚠️ 自动挂载云盘遇到提示 ({session_name}): {e}[/yellow]")


def _has_free_kaggle_gpu_quota(quota) -> bool:
    if not isinstance(quota, dict):
        return False
    remaining = quota.get("gpu_remaining_hours") or 0.0
    return (
        isinstance(remaining, (int, float))
        and remaining >= 0.5
        and (quota.get("gpu_reserved_seconds") or 0.0) == 0.0
    )


def choose_kaggle_for_t4(
    gpu_raw, *, force_colab, force_kaggle, is_cpu_or_tpu, high_mem, workers,
    cfg, token_present, quota, now=None,
) -> tuple[bool, str | None]:
    """Decide whether to use Kaggle and return any routing message."""
    if is_cpu_or_tpu:
        return False, None
    if force_kaggle or (not force_colab and is_kaggle_gpu_spec(gpu_raw)):
        return True, None

    is_t4 = gpu_raw.strip().upper() == "T4"
    if force_colab:
        if is_t4 and token_present and _has_free_kaggle_gpu_quota(quota) and not quota.get("error"):
            return False, (
                f"Kaggle 免费额度剩 {quota['gpu_remaining_hours']} 小时，"
                "已按 --colab 强制使用付费 Colab T4（约 1.19 CU/h）"
            )
        return False, None

    if not (is_t4 and not high_mem and workers == 1 and cfg.get("prefer_kaggle_for_t4", True) and token_present):
        return False, None

    if isinstance(quota, dict) and quota.get("error"):
        cache = quota.get("cache")
        if isinstance(cache, dict):
            past_refresh = False
            try:
                now = now if now is not None else datetime.now(timezone.utc)
                if now.tzinfo is None:
                    now = now.replace(tzinfo=timezone.utc)
                refresh = datetime.fromisoformat((cache.get("refresh_time") or "").replace("Z", "+00:00"))
                if refresh.tzinfo is None:
                    refresh = refresh.replace(tzinfo=timezone.utc)
                past_refresh = now > refresh
            except (AttributeError, TypeError, ValueError):
                pass
            if past_refresh or _has_free_kaggle_gpu_quota(cache):
                fetched_at = cache.get("fetched_at") or "未知"
                return True, f"Kaggle 额度接口失败，按本地缓存（{fetched_at}）尝试 Kaggle"
        return False, f"Kaggle 额度接口失败（{quota['error']}），改用 Colab T4"

    return _has_free_kaggle_gpu_quota(quota), None


def cmd_up(args):
    """Spin up one or more sessions (Colab GPU/CPU/TPU or Free Kaggle Dual-T4), deploy probe, and initialize Drive workspace."""
    cfg = load_config()
    colab_bin = resolve_colab_bin()
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    session_names = [args.session] if workers == 1 else [f"{args.session}-{i}" for i in range(1, workers + 1)]

    force_colab = bool(getattr(args, "colab", False))
    force_kaggle = bool(getattr(args, "kaggle", False))
    gpu_raw = (getattr(args, "gpu", None) or "T4").strip()
    is_cpu_or_tpu = bool(getattr(args, "cpu", False) or getattr(args, "tpu", None) or gpu_raw.upper() in ("CPU", "NONE", "OFF"))
    high_mem = getattr(args, "high_mem", False)
    kaggle_gpu_spec = is_kaggle_gpu_spec(gpu_raw)

    quota = None
    token_present = False
    if (
        not is_cpu_or_tpu
        and not force_kaggle
        and not kaggle_gpu_spec
        and gpu_raw.upper() == "T4"
        and (force_colab or (not high_mem and workers == 1 and cfg.get("prefer_kaggle_for_t4", True)))
    ):
        token_present = bool(load_kaggle_token())
        if token_present:
            quota = get_kaggle_quota()

    use_kaggle, route_msg = choose_kaggle_for_t4(
        gpu_raw, force_colab=force_colab, force_kaggle=force_kaggle,
        is_cpu_or_tpu=is_cpu_or_tpu, high_mem=high_mem, workers=workers,
        cfg=cfg, token_present=token_present, quota=quota,
    )
    if route_msg:
        warning_console = Console(stderr=True) if getattr(args, "json", False) else console
        warning_console.print(f"[yellow]⚠️ {route_msg}[/yellow]")

    started_sessions = []
    if use_kaggle:
        for s_name in session_names:
            try:
                spin_up_kaggle_session(
                    session_name=s_name,
                    gpu_choice=gpu_raw if kaggle_gpu_spec else "T4x2",
                    cfg=cfg,
                    probe_local_path=PROBE_LOCAL_PATH,
                    quiet=bool(getattr(args, "json", False)),
                    console=console,
                )
                started_sessions.append(s_name)
            except Exception as e:
                if not force_kaggle and not kaggle_gpu_spec and gpu_raw.upper() == "T4":
                    if not getattr(args, "json", False):
                        console.print(f"[yellow]⚠️ Kaggle 免费双卡暂不可用 ({e})，自动回退至 Colab T4...[/yellow]")
                    use_kaggle = False
                    break
                raise SystemExit(f"❌ Kaggle 实例启动失败: {e}")

    if not use_kaggle:
        pre_endpoints = set()
        try:
            _, pre_assigns = colab_state.sync_sessions()
            pre_endpoints = {a.endpoint for a in pre_assigns if getattr(a, "endpoint", None)}
        except Exception:
            pass
        for s_name in session_names:
            try:
                _spin_up_single_session(colab_bin, s_name, args)
                started_sessions.append(s_name)
            except KeyboardInterrupt:
                try:
                    colab_state._sessions = None
                    tracked, post_assigns = colab_state.sync_sessions()
                    tracked_eps = {s.endpoint for s in tracked.values() if getattr(s, "endpoint", None)}
                    for a in post_assigns:
                        ep = getattr(a, "endpoint", None)
                        if ep and ep not in pre_endpoints and ep not in tracked_eps:
                            colab_state.client.unassign(ep)
                except Exception:
                    pass
                raise
            except subprocess.CalledProcessError as e:
                if not started_sessions:
                    raise SystemExit(f"Unable to start {s_name}. Run colab-ultra status to check assignments.") from e
                if not getattr(args, "json", False):
                    console.print(f"[yellow]⚠️ 节点 [{s_name}] 启动受限 (已开启 {len(started_sessions)} 台): {e}[/yellow]")
                break

    colab_state._sessions = None  # new ran in a separate process
    # Deploy telemetry probe and refresh dashboard (force fresh read after Drive mount)
    ccu, vms = inspect_all_vms()
    for vm in vms:
        if vm["name"] in started_sessions and vm.get("provider") != "kaggle":
            try:
                if os.path.exists(PROBE_LOCAL_PATH):
                    upload_file_to_vm(vm["proxy_url"], vm["proxy_token"], PROBE_LOCAL_PATH, "content/.colab_probe.py")
                    start_remote_probe_via_runtime(vm["proxy_url"], vm["proxy_token"])
                    vm["telemetry"] = fetch_vm_telemetry(vm["proxy_url"], vm["proxy_token"], auto_deploy=False)
            except Exception:
                pass
            init_script = os.path.join(BASE_DIR, "workflows", "init_20tb_drive.py")
            if os.path.exists(init_script):
                try:
                    upload_file_to_vm(vm["proxy_url"], vm["proxy_token"], init_script, "content/init_20tb_drive.py")
                except Exception:
                    pass
    try:
        _record_starts(started_sessions, vms, ccu, getattr(args, "gpu", None))
    except (OSError, ValueError) as exc:
        print(f"Unable to record session billing: {exc}", file=sys.stderr)

    if getattr(args, "json", False):
        sync_dashboard_json(ccu, vms)
        payload = build_agent_status_payload(ccu, vms)
        payload["started_sessions"] = started_sessions
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        render_dashboard(ccu, vms)
        console.print(f"[green]✅ 实例 {started_sessions} 已就绪（硬件探针 + 统一云盘工作区 + 计费打点已激活）！[/green]")


def _resolve_exec_source(args, project_name: str | None) -> tuple[str, str]:
    """Load a project file or the inline code supplied by the caller."""
    if args.file:
        src_path = os.path.expanduser(args.file)
        if not os.path.isabs(src_path) and project_name:
            cand = os.path.expanduser(f"~/ColabUltra/{project_name}/code/{args.file}")
            if os.path.exists(cand):
                src_path = cand
        real_src = os.path.realpath(src_path)
        if real_src.startswith(("/tmp/", "/private/tmp/", "/var/folders/")):
            raise SystemExit(
                f"❌ [colab-ultra] 严禁传入 /tmp 临时脚本 ({args.file})！"
                f"请将脚本保存在 ~/ColabUltra/{project_name or '<project>'}/code/ 云盘目录中再执行。"
            )
        with open(src_path, "r", encoding="utf-8") as src_f:
            user_code = src_f.read()
        script_label = os.path.basename(args.file)
    elif args.code:
        user_code = args.code
        script_label = "<colab_ultra_exec>"
    else:
        raise SystemExit("Provide either inline code or -f <script.py>")
    return user_code, script_label


def _job_source(user_code, script_label, task_label, prelude, pack=False):
    """Wrap a script with an execution heartbeat and optional Kaggle output sync."""
    return f"""import gc, json, os, subprocess, sys, threading, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
{prelude}
_scripts = "/content/drive/MyDrive/ColabUltra/scripts"
if os.path.isfile(os.path.join(_scripts, "cu_remote", "__init__.py")):
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from cu_remote import stage, vram_policy
    sys.modules.setdefault("fast_stage_nvme", stage)
    sys.modules.setdefault("vram_policy", vram_policy)

_hb_stop = threading.Event()
_hb_file = "/content/.task_heartbeat.json"
def _hb_loop():
    probe = "/content/.colab_probe.py"
    telemetry = "/content/.colab_telemetry.json"
    heartbeat = {{"active": True, "task": {task_label!r}, "local_pid": {os.getpid()}, "started_at": time.time()}}
    while not _hb_stop.is_set():
        try:
            with open(_hb_file, "w") as handle:
                json.dump(heartbeat, handle)
            if os.path.exists(probe) and (not os.path.exists(telemetry) or time.time() - os.path.getmtime(telemetry) > 45):
                subprocess.Popen(["python3", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            if {pack!r}:
                _cu_pack_runs(final=False)
        except Exception:
            pass
        _hb_stop.wait(15)
threading.Thread(target=_hb_loop, daemon=True).start()
try:
    exec(compile({user_code!r}, {script_label!r}, "exec"), globals())
finally:
    _hb_stop.set()
    try:
        if {pack!r}:
            _cu_pack_runs(final=True)
    finally:
        if os.path.exists(_hb_file):
            os.remove(_hb_file)
"""


def cmd_exec(args):
    """Agent-First remote code/script execution with automatic post-run hardware telemetry (Colab & Kaggle unified)."""
    project_name = getattr(args, "project", None)
    timeout_sec = str(getattr(args, "timeout", 3600.0) or 3600.0)
    user_code, script_label = _resolve_exec_source(args, project_name)
    task_label = f"{project_name}/{script_label}" if project_name else script_label
    proj_prelude = ""
    if project_name:
        proj_prelude = (
            f"_proj_root = {os.path.join('/content/drive/MyDrive/ColabUltra', project_name)!r}\n"
            "_code_dir = f'{_proj_root}/code'\n"
            "for _sub in ('code', 'models', 'data', 'runs'):\n"
            "    os.makedirs(f'{_proj_root}/{_sub}', exist_ok=True)\n"
            "os.chdir(_code_dir)\n"
            "if _code_dir not in sys.path:\n"
            "    sys.path.insert(0, _code_dir)\n"
        )

    # Check if target session is an active Kaggle Free Dual-T4 session
    k_sessions = _load_kaggle_sessions()
    k_sess = k_sessions.get(args.session)
    if k_sess:
        if not k_sess.get("proxy_url") or not kaggle_active(k_sess):
            raise SystemExit(f"Lost Kaggle session {args.session}. Run colab-ultra status, then "
                             f"colab-ultra up -s {args.session} --kaggle.")

        p_url = k_sess["proxy_url"]
        local_hub = detect_local_drive_hub()
        if getattr(args, "restart_kernel", False):
            restart_vm_kernel(p_url, "")
        if project_name:
            sync_project_to_kaggle(p_url, project_name, local_hub)

        drive_tok = get_short_lived_drive_token()
        kaggle_sync_defs = build_in_vm_sync_snippet(project_name, drive_tok)
        prelude = (
            "for _a in ('last_traceback', 'last_value', 'last_type', 'last_exc'):\n"
            "    if hasattr(sys, _a):\n"
            "        try: delattr(sys, _a)\n"
            "        except Exception: pass\n"
            "gc.collect()\n"
            "if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():\n"
            "    try: sys.modules['torch'].cuda.empty_cache()\n"
            "    except Exception: pass\n"
        ) + proj_prelude + kaggle_runtime_code() + kaggle_sync_defs
        wrapped_kaggle_code = _job_source(user_code, script_label, task_label, prelude, pack=True)

        proj_local_dir = os.path.join(local_hub, project_name) if (local_hub and project_name) else None

        def _live_sync_tick():
            if proj_local_dir:
                pull_live_tarball_to_local(p_url, "/content/_cu_live_runs.tar.gz", proj_local_dir)

        t0 = time.time()
        ret_code, out_str, err_str = jupyter_exec_ws(
            p_url,
            wrapped_kaggle_code,
            timeout=float(timeout_sec),
            stream_stdout=not getattr(args, "json", False),
            stream_to_stderr=bool(getattr(args, "json", False)),
            on_periodic_tick=_live_sync_tick if proj_local_dir else None,
            tick_interval_sec=20.0,
        )
        if proj_local_dir:
            pull_live_tarball_to_local(p_url, "/content/_cu_final_runs.tar.gz", proj_local_dir)
        _exec_result(args, t0, ret_code, out_str, err_str)
        return

    wrapped_code = _job_source(user_code, script_label, task_label, proj_prelude)
    t0 = time.time()
    ret_code, out_str, err_str = run_detached(args, wrapped_code)
    _exec_result(args, t0, ret_code, out_str, err_str)


def _exec_result(args, t0, ret_code, out_str, err_str):
    """Emit the common exec result in the selected CLI format."""
    elapsed = round(time.time() - t0, 2)

    if getattr(args, "json", False):
        ccu, vms = inspect_all_vms()
        sync_dashboard_json(ccu, vms)
        payload = build_agent_status_payload(ccu, vms)
        payload.update({
            "ok": ret_code == 0,
            "exit_code": ret_code,
            "elapsed_sec": elapsed,
            "stdout": out_str,
            "stderr": err_str,
        })
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif ret_code != 0:
        sys.exit(ret_code)


def teardown_verified(vm, local_hub, backup=True):
    """Verify by endpoint after each stop, preserving local recovery state on failure."""
    from colab_ultra.retry import retry, transient

    kaggle = vm.get("provider") == "kaggle"
    for attempt in range(2):
        try:
            if kaggle:
                shut_down_kaggle_session(vm, local_hub=local_hub, flush_outputs=backup and not attempt)
            else:
                if backup and not attempt:
                    backup_vm_outputs_to_drive(vm["proxy_url"], vm["proxy_token"])
                colab_state.client.unassign(vm["endpoint"])
        except Exception as exc:
            # A lost response may still have released the assignment.
            if not transient(exc) and getattr(getattr(exc, "response", None), "status_code", None) != 404:
                raise
        time.sleep(1)
        active = retry(lambda: kaggle_active(vm) if kaggle else
                       any(a.endpoint == vm["endpoint"] for a in colab_state.client.list_assignments()))
        if not active:
            if kaggle:
                forget_kaggle_session(vm)
            elif vm["name"] != "web-session":
                local = colab_state.store.list().get(vm["name"])
                if local and local.endpoint == vm["endpoint"]:
                    colab_state.store.remove(vm["name"])
            return
    raise RuntimeError(f"Session {vm['name']} ({vm['endpoint']}) still active after two teardown attempts. "
                     "Run colab-ultra status.")


def cmd_down(args):
    """Bound the entire CLI operation, including backup and status RPCs."""
    import signal

    class DeadlineExpired(BaseException):
        """Escape backend exception and auth handlers on the CLI thread."""

    def expired(*_):
        raise DeadlineExpired

    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        _down(args)
    except DeadlineExpired:
        raise SystemExit("Teardown exceeded 120s. Run colab-ultra status to verify the session.") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _down(args):
    """Release selected sessions, then settle their billing records."""
    cfg = load_config()
    vms = assignments() + list(_load_kaggle_sessions().values())
    if not vms:
        if args.json:
            print(json.dumps({"ok": True, "unassigned": [], "active_vms_count": 0}))
        else:
            console.print("当前没有运行中的云端实例。")
        return
    targets = vms
    if not args.all:
        if args.session:
            targets = [v for v in vms if v["name"] == args.session or v["endpoint"].startswith(args.session)]
        if len(targets) != 1:
            raise SystemExit("Select one active session with colab-ultra down -s NAME. Run colab-ultra status.")
    local_hub = detect_local_drive_hub()
    unassigned = []
    failures = []
    for vm in targets:
        if not args.json:
            console.print(f"Releasing {vm['name']} ({vm['endpoint']})...")
        try:
            teardown_verified(vm, local_hub, cfg.get("auto_backup_to_drive_before_stop", True))
        except Exception as exc:
            failures.append(f"{vm['name']}: {exc}")
            print(f"Release unverified for {failures[-1]}", file=sys.stderr)
            continue
        record = {key: vm[key] for key in ("name", "provider", "endpoint", "accelerator")}
        log_breaker_event("MANUAL_DOWN", record)
        unassigned.append(record)
        if not args.json:
            console.print(f"Verified release of {vm['name']} ({vm['endpoint']})")
    ccu_after, vms_after = inspect_all_vms()
    sync_dashboard_json(ccu_after, vms_after, cfg=cfg)

    end_balance = float(ccu_after.paid_compute_units_balance or 0)
    try:
        _settle(unassigned, end_balance)
    except (OSError, KeyError, ValueError) as exc:
        print(f"Unable to settle local billing records: {exc}", file=sys.stderr)

    if getattr(args, "json", False):
        payload = build_agent_status_payload(ccu_after, vms_after, cfg=cfg)
        payload["unassigned"] = unassigned
        if failures:
            payload.update(ok=False, errors=failures)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _rem_rate = float(ccu_after.consumption_rate_hourly or 0.0)
        console.print("\n[bold green]✅ 算力释放与计费核验报告：[/bold green]")
        console.print(f"• 实例释放状态: Active assignments: {len(vms_after)}")
        console.print(f"• 实时消耗速率: {_rem_rate:.2f} CCU/h")
        for _u in unassigned:
            _dur = _u.get("duration_str", "未知")
            _cost = _u.get("estimated_cost_ccu", 0.0)
            console.print(f"• 已释放会话: [{_u['name']}] {_u['accelerator']} ({_u['provider'].upper()}) | 运行时长: {_dur} | 本次消耗: {_cost:.2f} CCU")
        console.print(f"• 当前账户余额: {end_balance:.2f} CCU")
    if failures:
        raise SystemExit("Release unverified: " + ", ".join(failures))



def cmd_pull(args):
    """Spin up a cheap/free CPU instance, download large files/repos directly into Google Drive via Google Cloud network, and safely down immediately."""
    session = getattr(args, "session", "pull-worker") or "pull-worker"
    project = args.project
    source = args.source
    dest_dir = getattr(args, "dest", "models") or "models"
    filename = getattr(args, "filename", None)

    console.print(f"🚀 [cyan]正在启动云端 CPU 节点 ({session})，走 Google Cloud 内网高速下载至 Drive，不消耗本地带宽...[/cyan]")
    up_args = argparse.Namespace(
        session=session,
        gpu="CPU",
        cpu=True,
        std_mem=True,
        high_mem=False,
        tpu=None,
        workers=1,
        no_drive=False,
        colab=True,
        kaggle=False,
        json=False,
    )
    cmd_up(up_args)

    dl_code = pull_code(source, project, dest_dir, filename)
    exec_args = argparse.Namespace(
        session=session,
        project=project,
        file=None,
        timeout=3600.0,
        restart_kernel=False,
        code=dl_code,
        json=getattr(args, "json", False),
    )
    try:
        cmd_exec(exec_args)
    finally:
        console.print(f"🔌 [cyan]下载完成，立即释放云端 CPU 节点 ({session})，避免闲置算力扣费...[/cyan]")
        down_args = argparse.Namespace(
            session=session,
            all=False,
            json=getattr(args, "json", False),
        )
        cmd_down(down_args)
