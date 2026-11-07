---
name: colab-ultra
description: Agent-First Google Colab GPU orchestrator with physical GPU wattage (W) smart circuit breaker, automatic 20TB Google Drive workspace persistence, and 24/7 autonomous VPS sentinel protection. Use whenever an AI Agent needs to run GPU compute, LLM inference, or DiT/LoRA training on Google Colab without risking runaway Compute Unit (CCU) billing.
---

# Colab Ultra — Agent-First GPU Orchestrator & Smart Wattage Breaker

Always use `colab-ultra` (with `--json` for structured Agent parsing) instead of raw `colab` when managing Google Colab GPU runtimes.

## 1. Core Agent Workflow (4-Step Loop)

1. **Check Account Balance & Active VMs First**:
   ```bash
   colab-ultra status --json
   ```
   - Inspect `paid_balance`, `burn_rate_hourly`, `active_vms_count`, and each VM's `power_draw_w`, `vram_used_mib`, `drive_mounted`, and `is_low_load`.

2. **Spin Up GPU with Auto-Mounted Google Drive Workspace**:
   - **Rule of Thumb**: Default to `T4` for environment setup, debugging, or lightweight inference; escalate to `A100` or `H100` only when executing heavy training or benchmark runs.
   ```bash
   colab-ultra up -s ultra --gpu T4 --json
   # For heavy DiT / LoRA training or 80GB VRAM workloads:
   colab-ultra up -s ultra --gpu A100 --json
   ```
   - This automatically:
     - Allocates the GPU runtime;
     - Mounts Google Drive via Colab's first-party `dfs_ephemeral` channel;
     - Redirects `HF_HOME`, `TORCH_HOME`, `/content/workspace`, and `/content/outputs` to `/content/drive/MyDrive/ColabUltra/`;
     - Restores any files rescued from previous auto-breaker shutdowns;
     - Injects the out-of-band hardware wattage probe (`/content/.colab_probe.py`).

3. **Execute Code or Scripts with Real-Time Hardware Telemetry**:
   ```bash
   colab-ultra exec -s ultra "import torch; print(torch.cuda.get_device_name(0))" --json
   colab-ultra exec -s ultra -f workflows/ai_image_gen.py --json
   ```
   - Always save checkpoints, generated images, and logs under `/content/outputs` or `/content/workspace` so they persist directly to Google Drive.
   - Every `colab-ultra exec --json` call returns `stdout`, `stderr`, `exit_code`, `elapsed_sec`, **and** live post-run `power_draw_w`, `vram_used_mib`, and `paid_balance`.

4. **Immediate Safe Shutdown After Task Completion**:
   ```bash
   colab-ultra down --all --json
   ```
   - Automatically backs up `/content/outputs` and loose root artifacts to Google Drive (`ColabUltra/emergency_backups/`), unassigns the VM, and verifies `burn_rate_hourly == 0.0`.

## 2. Why Physical GPU Wattage (`power_draw_w`) Matters for Agents
- Standard `utilization.gpu` (`%`) can report `100%` during NCCL spin-wait deadlocks while doing zero actual math.
- `colab-ultra` evaluates physical GPU board power draw (`nvidia-smi --query-gpu=power.draw`):
  - **A100 / H100** (TDP 350–700W): `< 75W` = idle or deadlocked.
  - **L4 / T4 / G4** (TDP 70–150W): `< 28W` = idle or deadlocked.
- If an Agent session disconnects or a training script hangs at low wattage for **10 minutes**, the local daemon (`colab-ultra watch`) or 24/7 VPS sentinel (`vps_sentinel.py`) automatically rescues files to Google Drive and terminates the VM.
