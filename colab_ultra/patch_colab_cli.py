#!/usr/bin/env python3
"""
Idempotent upstream patcher for Google's official `google-colab-cli` (v0.7.x):
1. Fixes the upstream `colab drivemount` HTTP 400 Bad Request bug in `commands/automation.py`
   (removes malformed `files={"file_id": (None, "empty.ipynb")}` multipart payload, sets
   `record="false"`, adds `X-Colab-Client-Agent: colab-cli`, and auto-opens browser URI).
2. Accelerates CLI startup by ~8x (from ~7s to ~0.8s) by loading
   `~/.config/gcloud/application_default_credentials.json` directly in `auth.py`
   without probing slow GCE metadata server timeouts on non-GCE machines.
3. Defaults `AuthProvider` to `ADC` in `common.py` and `cli.py`.
"""

import os


def apply_upstream_patches() -> list[str]:
    applied = []
    try:
        import colab_cli
    except ImportError:
        return ["跳过补丁：未找到 colab_cli 模块"]

    pkg_dir = os.path.dirname(os.path.realpath(colab_cli.__file__))

    # 1. Patch common.py: default AuthProvider.OAUTH2 -> AuthProvider.ADC
    common_py = os.path.join(pkg_dir, "common.py")
    if os.path.exists(common_py):
        with open(common_py, "r", encoding="utf-8") as f:
            src = f.read()
        if "self.auth_provider = AuthProvider.OAUTH2" in src:
            src = src.replace(
                "self.auth_provider = AuthProvider.OAUTH2",
                "self.auth_provider = AuthProvider.ADC",
            )
            with open(common_py, "w", encoding="utf-8") as f:
                f.write(src)
            applied.append("common.py (默认启用 ADC 认证)")

    # 2. Patch cli.py: default AuthProvider.OAUTH2 -> AuthProvider.ADC
    cli_py = os.path.join(pkg_dir, "cli.py")
    if os.path.exists(cli_py):
        with open(cli_py, "r", encoding="utf-8") as f:
            src = f.read()
        if "auth: AuthProvider = typer.Option(\n        AuthProvider.OAUTH2," in src:
            src = src.replace(
                "auth: AuthProvider = typer.Option(\n        AuthProvider.OAUTH2,",
                "auth: AuthProvider = typer.Option(\n        AuthProvider.ADC,",
            )
            with open(cli_py, "w", encoding="utf-8") as f:
                f.write(src)
            applied.append("cli.py (CLI 默认参数切换为 ADC)")

    # 3. Patch auth.py: fast direct ADC file loading (0.8s instead of 7s GCE metadata timeout)
    auth_py = os.path.join(pkg_dir, "auth.py")
    if os.path.exists(auth_py):
        with open(auth_py, "r", encoding="utf-8") as f:
            src = f.read()
        fast_adc_snippet = (
            '    adc_file = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")\n'
            '    creds = None\n'
            '    if os.path.exists(adc_file):\n'
            '        try:\n'
            '            creds = Credentials.from_authorized_user_file(adc_file)\n'
            '            if not creds.valid:\n'
            '                creds.refresh(Request())\n'
            '            return creds\n'
            '        except Exception as e:\n'
            '            logger.debug(f"Direct ADC load failed: {e}")\n'
        )
        target_anchor = "    with warnings.catch_warnings():"
        if "Credentials.from_authorized_user_file(adc_file)" not in src and target_anchor in src:
            src = src.replace(target_anchor, fast_adc_snippet + target_anchor, 1)
            with open(auth_py, "w", encoding="utf-8") as f:
                f.write(src)
            applied.append("auth.py (ADC 极速加载补丁: 7s -> 0.8s)")

    # 4. Patch commands/automation.py: fix HTTP 400 in `colab drivemount`
    auto_py = os.path.join(pkg_dir, "commands", "automation.py")
    if os.path.exists(auto_py):
        with open(auto_py, "r", encoding="utf-8") as f:
            src = f.read()
        if '"record": "true"' in src or 'files={"file_id": (None, "empty.ipynb")}' in src:
            old_block_start = '            url = f"{state.client.colab_domain}/tun/m/credentials-propagation/{s.endpoint}"'
            old_block_end = '            if get_status_code(resp) == 200:'
            if old_block_start in src and old_block_end in src:
                pre, rest = src.split(old_block_start, 1)
                _, post = rest.split(old_block_end, 1)
                fixed_block = '''            url = f"{state.client.colab_domain}/tun/m/credentials-propagation/{s.endpoint}"
            params = {
                "authuser": "0",
                "authtype": "dfs_ephemeral",
                "version": "2",
                "dryrun": "true",
                "propagate": "true",
                "record": "false",
            }
            typer.echo(
                f"\\n[colab] Intercepted Drive Auth Request. Connecting to {state.client.colab_domain}..."
            )

            creds = state.client.session
            base_headers = {"X-Colab-Client-Agent": "colab-cli", "Accept": "application/json"}
            resp = creds.request("GET", url, params=params, headers=base_headers)
            token = (
                json.loads(resp.text.split("\\n", 1)[-1]).get("token")
                if get_status_code(resp) == 200
                else None
            )

            headers = {**base_headers, "x-goog-colab-token": token}
            resp = creds.request(
                "POST",
                url,
                params=params,
                headers=headers,
            )
            data = json.loads(resp.text.split("\\n", 1)[-1])

            if not data.get("success"):
                uri = data.get("unauthorized_redirect_uri")
                typer.echo(
                    f"\\n[colab] REQUIRED: Google Drive Authorization needed.\\nPlease visit:\\n\\n{uri}\\n"
                )
                try:
                    import webbrowser
                    webbrowser.open(uri)
                except Exception:
                    pass
                state.history.log_event(s.name, "drive_auth_needed", {"uri": uri})
                sys.stdout.write("Press Enter after you have granted access... ")
                sys.stdout.flush()
                try:
                    with open("/dev/tty") as tty:
                        tty.readline()
                except (FileNotFoundError, OSError, IOError):
                    try:
                        sys.stdin.readline()
                    except Exception:
                        try:
                            input()
                        except Exception:
                            pass

            typer.echo("[colab] Authorizing VM...")
            params["dryrun"] = "false"
            resp = creds.request(
                "POST",
                url,
                params=params,
                headers=headers,
            )
'''
                src = pre + fixed_block + old_block_end + post
                with open(auto_py, "w", encoding="utf-8") as f:
                    f.write(src)
                applied.append("commands/automation.py (修复官方 drivemount HTTP 400 Bug)")

    return applied
