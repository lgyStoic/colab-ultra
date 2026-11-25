---
name: colab-ultra
description: Manage Google Colab and Kaggle sessions, stage weights in Google Drive, run project recipes, and monitor hardware load and compute-unit billing.
---

# Colab Ultra

## Command Usage

Run each command separately so the task log shows live progress.

| Command | Use |
| --- | --- |
| `colab-ultra init` | Configure ADC, alerts and the optional VPS sentinel. Sync the shared VM library when local Drive is available. |
| `colab-ultra sync-remote` | Deploy the current checkout's runtime to `~/ColabUltra/scripts/cu_remote/`. |
| `colab-ultra status --json` | Read active sessions, task heartbeats, GPU load, Drive status, paid balance and burn rate. |
| `colab-ultra specs` | Compare hardware, native precisions and peak throughput per CU. See the [hardware specifications](references/hardware-specs.md). |
| `colab-ultra up -s prep --cpu --std-mem` | Allocate a low-cost CPU worker for preparation. Use `--cpu` for high-memory ETL or `--workers N` for multiple nodes. |
| `colab-ultra up -s l4 --gpu L4` | Allocate a GPU. T4 can route to Kaggle's free dual-T4 when quota permits. Use `--colab` or `--kaggle` to select a provider. |
| `colab-ultra exec -s l4 -p tiles -f recipes/minimax_h3_diffsynth.py --timeout 7200` | Run the local script in the project's remote `code/` directory and stream output. |
| `colab-ultra logs -s NAME -n 100` | Read the latest exec log, including while the kernel is busy. Use `-n 0` for all lines. |
| `colab-ultra models ls` | List local Drive variants, payload sizes and manifest status. |
| `colab-ultra models verify MODEL/VARIANT` | Check filenames and sizes. Omit the target to check the entire library. |
| `colab-ultra models rm MODEL/VARIANT` | Preview removal of one variant. Add `--yes` to delete it. |
| `colab-ultra pull org/repo --dest models/my-model/gguf-q4 --filename weights.gguf` | Stage a repository, matching files or a URL through a temporary CPU worker. |
| `colab-ultra down -s l4` | Back up outputs when configured, verify release and report remaining assignments and billing. |
| `colab-ultra watch` | Monitor watts, GPU/CPU activity, disk I/O and task heartbeats. Defaults are 5 minutes to alert and 10 minutes to stop. |
| `colab-ultra test-alert` | Exercise the configured notification, voice and webhook channels. |

## Directory Layout

- `~/ColabUltra/models/<model>/<variant>/` holds shared weights, sidecars and a per-variant manifest. Recipes use `cu_remote.models.ensure()` to stage a variant on VM-local storage. See the [model library guide](references/model-library.md).
- Project files under `~/ColabUltra/<project>/` map to `/content/drive/MyDrive/ColabUltra/<project>/` on Colab. Use `code/`, `data/` and `runs/` for scripts, inputs and results. Project-scoped pulls still support `models/`. Kaggle mirrors the project and syncs outputs.
- `recipes/` contains standalone inference scripts. Prepare `code/prompts.json` and validate it with a local dry run before allocating a GPU. See the [recipe contract](recipes/README.md).
- `colab_ultra/remote/` is the source for the VM package `cu_remote`. It contains staging, binary caching, image encoding and VRAM policy. The [staging guide](references/fast-nvme-staging.md) includes a benchmark example.
- `colab_ultra/telemetry_probe.py` runs on the VM. `colab_ultra/vps_sentinel.py` runs independently on a configured VPS. Workstation configuration lives in `~/.config/colab-ultra/config.json`.

## Hard Rules

- Do not pass `--colab` for T4 jobs unless Kaggle is unavailable or out of quota, multiple workers or high-RAM are needed, or another Kaggle session is already running. Write outputs under `/content/drive/MyDrive/ColabUltra/<project>/runs/` and run via `exec -p <project>` so Kaggle mirrors the project and syncs `runs/` back to `~/ColabUltra/<project>/runs/`; inputs over 80 MB require `init --with-drive-scope` or an HF/URL download.
- Run `up`, `exec` and `down` with live terminal output. Omit `--json`, pipes such as `| grep`, and shell chains. Use `--json` only on `status` when parsing its output.
- Never poll `colab-ultra exec` progress with timers or `sleep` loops. Wait for the background command exit notification; only inspect on demand with `colab-ultra logs -s NAME -n 50` or `colab-ultra status --json` if the user asks for progress or the WebSocket disconnects.
- Land multi-connection downloads (default `hf_transfer` c=64, or `aria2c` c=16) on local NVMe under `/content/` first. Then copy sequentially to `/content/drive/MyDrive/` with `cp` or `shutil.copy2`. Reject incomplete transfers and `.aria2` sidecars.
- Copy Drive-cached binaries into `/tmp` and run `chmod +x` there. Drive FUSE does not preserve the execute bit.
- Confirm outputs have reached Drive before releasing the session. Use `down -s NAME` for one session. Check all active tasks before using `down --all`.

## Pitfalls Learned

- A prior 34.16 GB download wrote directly to Drive FUSE at about **32.6 MB/s** and took **1048.7s**. Sequential writes from local storage measured about **440–495 MB/s**. Include the final Drive copy when comparing backends.
- On Colab High-RAM CPU (`smoke`), staging the 3 MiniMax-H3 HF files (`34,162,458,528` bytes, `workers=3`) to NVMe took **99.46s** (**343.47 MB/s**) with `hf_transfer` (`c=64`, now HF/HTTP default) vs **119.45s** (**285.99 MB/s**) with `aria2` (`c=16`), a **20.1%** speedup.
- On L4, an Nsight Systems (`nsys`) profile measured **~89.1% active SM utilization**. The workload was compute-bound: `mul_mat_q` accounted for **53.8%**, SageAttention for **~17.8%**, and PCIe memcpy for **~1.3%**.
- A closed WebSocket can leave a remote task running. After a lost acknowledgement, check `colab-ultra status` and the project's outputs before submitting again. For a missing assignment, recreate the named session with its original hardware flags.
