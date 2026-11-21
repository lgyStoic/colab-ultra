---
name: colab-ultra
description: Agent-First Google Colab GPU & Multi-Core CPU orchestrator with physical GPU wattage (W) smart circuit breaker, 8-to-44 vCPU / 5-node CPU cluster scheduling, dual-ended Google Drive project workspace (local ~/ColabUltra <-> remote /content/drive/MyDrive/ColabUltra), and 24/7 autonomous VPS sentinel protection. Use whenever an AI Agent needs to run GPU compute, pure-CPU data preprocessing/ETL, batch diffusion image/video inference (e.g. MiniMax-H3-Image), or DiT/LoRA/PDD training on Google Colab without risking runaway Compute Unit (CCU) billing.
---

# Colab Ultra — Agent-First GPU Orchestrator & Smart Wattage Breaker

Always use `colab-ultra` instead of raw `colab` or `python3 breaker.py` when managing Google Colab & Kaggle runtimes. Pass `--json` only on `status` when you need to parse it. Run `up` / `exec` / `down` WITHOUT `--json`, WITHOUT pipes (`| grep/tail/head`), each as its own command (not chained in one subshell), so progress streams live into the task log.

## 1. Per-Project Dual-Ended Google Drive Workspace (`<project>/`)

Every project is isolated under its own directory on Google Drive, shared seamlessly between the local Mac (`0 CU/h`) and remote Colab VMs:
- **Local Mac Path** (via Google Drive for Desktop in streaming mode):
  `~/ColabUltra/<project>/` (symlinked to `~/Library/CloudStorage/GoogleDrive-*/My Drive/ColabUltra/<project>/`)
- **Remote Colab Path**:
  `/content/drive/MyDrive/ColabUltra/<project>/`

Standard layout inside `<project>/`:
- `code/` — Project source code & entry scripts. Read and edit files directly in `~/ColabUltra/<project>/code/` locally, and run directly with `/content/drive/MyDrive/ColabUltra/<project>/code` as `cwd` on Colab via `-p <project>`. **NEVER write one-off or temporary scripts to `/tmp/` — `colab-ultra exec -f` hard-rejects `/tmp/` paths; all scripts must live inside `~/ColabUltra/<project>/code/`.**
- `models/` — Pre-downloaded persistent model weights for this project.
  - **Persistent Download Rule**: All large weights and datasets MUST be downloaded directly on Colab CPU instances (`--cpu` or `colab-ultra pull`) into `/content/drive/MyDrive/...` at Google Cloud wire speed; **NEVER** download multi-GB model files on the local Mac to sync up via Google Drive Desktop, and **NEVER** spin up expensive GPU VMs (`G4`/`A100`/`H100`) just to download weights from the internet. When using multi-connection downloaders (`aria2c -x16 -s16`), download to local NVMe (`/content/...`) first and then sequentially `cp` into `/content/drive/MyDrive/...` (avoids FUSE multi-thread random-write throttling to ~30 MB/s).
  - **Fast NVMe Staging (Bypass Slow Drive FUSE)**: On Colab GPU VMs, **NEVER** use `cp -r` or `shutil.copytree` over `/content/drive/MyDrive/...` (`~25–39 MB/s` FUSE bottleneck, which also saturates 2-core T4 CPUs). Instead, stage weights onto local NVMe `/content/models` via **Drive v3 API + `aria2c`** (for private Drive weights) and **HF/ModelScope CDN `aria2c`/`hf_transfer`** (for public weights), overlapped with `pip`/`uv` setup using `/content/drive/MyDrive/ColabUltra/scripts/fast_stage_nvme.py` — see [references/fast-nvme-staging.md](references/fast-nvme-staging.md).
- `data/` — Datasets and pre-extracted feature caches (e.g., `text_cache/`, `.jsonl`).
- `runs/<run_name>/` — Checkpoints, exported weights, sample images, `log.jsonl`, and TensorBoard logs.

## 2. Hardware Tier Matrix, VRAM Sizing & Cost Guardrails

Quick VRAM estimation (including activation / KV cache headroom):
- **BF16 / FP16**: `~2.4 × Params(B) GB` | **FP8 / INT8**: `~1.2 × Params(B) GB` | **4-bit / NVFP4 / 3-bit**: `~0.7 × Params(B) GB`

| Task Type | Recommended Tier | Hardware Spec | Cost | Guardrails & Selection Rules |
| :--- | :--- | :--- | :--- | :--- |
| **Model Download / Unit Tests** | `--cpu --std-mem` / `pull` | `2 vCPU + 12 GB RAM` | `0.08 CU/h` | **Hard Gate**: If `burn_rate_hourly >= 5.0` and weights aren't in Drive yet, stop and download on CPU first. |
| **Data ETL / Parquet** | `--cpu` / `--cpu --workers 5` / `--tpu v6e1` | `8–44 vCPU, 51–255 GB RAM` | `0.26–5.09 CU/h` | Pure-CPU preprocessing; never burn GPU quota for ETL. |
| **Free Dual-GPU / Light Jobs** | `--gpu T4x2` (default `--gpu T4`) | `2 × Tesla T4 (30 GiB VRAM)` | **`0.00 CU/h`** (30h/wk) | Auto-prefers free Kaggle `T4x2` whenever ≥0.5h free quota remains; pass `--colab` to force paid Colab T4. |
| **Single-GPU ≤22 GiB / 14B–32B Q4** | `--gpu L4` | `22.5 GiB VRAM` (Ada `sm_89`), `53 GB RAM` | `0.48–1.71 CU/h` | Native FP8; use `cpu` offload or EntroPack 3-bit for >20 GB models. |
| **Heavy DiT / 70B+ / Full BF16** | **`--gpu G4`** | **`95.6 GiB VRAM`** (Blackwell `sm_120`), `177 GB RAM` | `~8.7–10.05 CU/h` | **Prefer `G4` over `A100`/`H100`** for heavy DiT (MiniMax-H3, Wan-14B, 70B+ LLM); load full model resident in VRAM with zero disk offload. Ask user confirmation before launching multi-hour expensive runs (e.g., full 2K video refine). |

## 3. Core Agent Workflow (5-Step Loop)

1. **Zero-Burn Prep Locally & Check Active VMs (`0 CU/h`)**:
   - **Prepare everything locally BEFORE spinning up paid GPUs (`G4`/`L4`)**: write/verify scripts, prompts, configs, and reference inputs in `~/ColabUltra/<project>/` first (`0 CU/h`), so GPU VMs ignite and run immediately without idle thinking time.
   ```bash
   colab-ultra status --json
   ```
   - Inspect `paid_balance`, `burn_rate_hourly`, `active_vms_count`, `vms` (including each VM's `active_task: {task, local_pid, elapsed_sec}` to see which project/script owns it!), and `local_drive_mounted`.

2. **Spin Up Lowest-Cost Hardware with Auto-Mounted Google Drive**:
   ```bash
   colab-ultra up -s ultra --gpu T4x2
   colab-ultra up --cpu --std-mem -s prep
   colab-ultra up --cpu -s data
   colab-ultra up -s l4 --gpu L4
   colab-ultra up -s g4 --gpu G4
   ```
   - Allocates the VM, mounts Google Drive at `/content/drive/MyDrive` via `dfs_persistent` zero-click mount, and injects the hardware wattage & I/O telemetry probe (`/content/.colab_probe.py`).

3. **Execute Directly Inside the Drive Project Directory (`-p <project>`, Live Streaming)**:
   ```bash
   colab-ultra exec -s <session> -p <project> -f ~/ColabUltra/<project>/code/run.py --timeout 7200
   ```
   - Omit `--json` on long jobs for clean real-time stdout streaming into task logs (even if `--json` is passed, `colab-ultra` mirrors live output lines to `stderr`). Pass `--restart-kernel` when re-running after an OOM or exception on the same VM to release leaked GPU tensors.
   - `colab-ultra exec` uses a detached `nohup` + short-lived WebSocket incremental log-polling engine (with `/tmp/_cu_job_<id>.rc` exit-code sentinel): it streams live stdout, keeps the Jupyter kernel alive on every poll (preventing Colab's 30-min idle reclaim), auto-refreshes `colab-runtime-proxy-token` every 40 min / on error, writes `/content/.task_heartbeat.json`, sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and **exits immediately with the exact remote exit code** once finished (no `wsclient.py` hangs, no polling timers needed — run `colab-ultra exec ...` then `colab-ultra down -s <session>` as separate commands for zero-idle teardown).
   - **CRITICAL TUNNEL & ANTI-TIMEOUT RULE**:
     1. **Long Batch Jobs (>20 min)**: Always run via `colab-ultra exec ... --timeout 36000` (never raw `colab exec`, which suffers from 10s handshake races, 85-min token expiry, and post-exit hangs).
     2. **Tunnel / Web Service Keepalive**: When serving a WebUI/API over Cloudflare Tunnel (e.g., ComfyUI where HTTP requests bypass the Jupyter kernel), keep a foreground `colab-ultra exec` wrapper alive so its periodic short polls keep the Jupyter kernel active and prevent Colab idle reclamation.

4. **Track Training via Standalone TensorBoard Dashboard (`http://35.212.142.112/tb/`)**:
   - Log scalar curves (`loss`, `gnorm`, `s_per_it`, `mem_gib`) and step evaluation images to `log.jsonl` + `samples/` (and `SummaryWriter(out_dir / "tb")`).
   - The VPS sentinel (`vps_sentinel.py`) syncs `log.jsonl` and `samples/*.jpg` to `http://35.212.142.112/tb/`.

5. **Multi-Session Safe Shutdown & Mandatory Billing Verification**:
   - Before tearing down after writing outputs/checkpoints to Drive, call `os.sync()` and verify target files exist with `size > 0`.
   - **Default to single-session teardown** so parallel user/agent sessions are NEVER killed accidentally:
     ```bash
     colab-ultra down -s <session>
     ```
   - Only pass `colab-ultra down --all` after confirming `vms` in `status` contains no other active tasks you want to keep.
   - Always report a concise release & dual-track CCU billing summary from the `down --json` output (auto-recorded by `up` and settled by `down` via `~/.config/colab-cli/session_tracker.json`):
     - Released session (`unassigned`, including `duration_str` & `estimated_cost_ccu`), remaining active `vms`, `burn_rate_hourly`, and updated `paid_balance` / `kaggle_free_gpu_hours_left`.

## 4. Physical GPU Wattage (`power_draw_w`) & I/O Circuit Breaker
- Standard `utilization.gpu` (`%`) can report `100%` during NCCL spin-wait deadlocks while doing zero actual math.
- `colab-ultra` evaluates physical GPU board power draw (`nvidia-smi --query-gpu=power.draw`), active disk/Drive I/O (`io_mb_s >= 2.0 MB/s`), and execution heartbeats:
  - **A100 / H100 / G4**: `< 75W` (A100/H100) = idle or deadlocked.
  - **L4 / T4**: `< 28W` = idle or deadlocked.
- If a session disconnects or hangs at low wattage with no active I/O or heartbeat for **10 minutes**, the local daemon (`colab-ultra watch`) or 24/7 VPS sentinel (`vps_sentinel.py`) automatically rescues files to Google Drive and terminates the VM.

## 5. Model Serving, Shared Scripts & Optimization Recipes (Progressive References)
- **Zero-Code `MiniMax-H3-Image` Batch Generator (`workflows/minimax_h3_batch.py`)**:
  For any project needing batch `MiniMax-H3-Image` generation on L4/G4, **do NOT write a new script** — only create `~/ColabUltra/<project>/code/prompts.json` (`{run_name, width, height, dit_variant, prompts: [{slug, prompt}]}`) and run the single canonical workflow directly:
  `colab-ultra exec -s l4 -p <project> -f ~/.gemini/config/skills/colab-ultra/workflows/minimax_h3_batch.py && colab-ultra down -s l4`
  (Built-in: HF CDN NVMe staging + `vram_policy.py` + batched TextEncoder + `dit_4p5bit_int8` + `pipe.compile` + batched full-frame VAE decode; L4 verified ~`24.5s/img` steady-state, ~`0.028 CU/img` for 2048×2048).
- **Auto GPU VRAM & Offload Policy (`scripts/vram_policy.py`)**:
  Synced to `/content/drive/MyDrive/ColabUltra/scripts/vram_policy.py` (`~/ColabUltra/scripts/vram_policy.py`). Import `build_vram_plan(component_sizes_gb)` to automatically choose `full_resident` (G4/H100), `selective_offload` (L4/A100-40G), or `tight_offload` (T4) plus `vae_tiling` and `staged_batch` flags from live GPU VRAM instead of hand-coding offload configs per script.
- **Fast NVMe Weight Staging & Small-File Bundles (Bypassing Drive FUSE via Drive v3 API + HF CDN `aria2c`)**:
  Read [references/fast-nvme-staging.md](references/fast-nvme-staging.md) whenever cold-starting multi-GB models or thousand-file plugin trees (`custom_nodes`) on Colab GPU VMs.
