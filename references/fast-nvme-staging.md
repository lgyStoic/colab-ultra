# Fast NVMe Weight Staging on Colab (Bypassing Drive FUSE)

On Google Colab GPU VMs, reading multi-GB `.safetensors` files via Drive FUSE (`/content/drive/MyDrive/...`) using `cp -r`, `shutil.copytree`, or random `DiskMap`/`mmap` reads is bottlenecked at `~25–39 MB/s` (and saturates 2-core T4 CPUs).

When staging weights into ephemeral local NVMe (`/content/models` or `/content/fast_models`) for fast cold start, use **Dual-Channel Concurrent Staging** overlapped with environment setup (**Tri-Parallel Overlap**), routing each file to its fastest channel:

1. **Channel B — Public Large Weights (Fastest: `~350–450 MB/s`)**: Pull public models **directly from HuggingFace GCP CDN (`us.gcp.cdn.hf.co`)** into local NVMe using `aria2c` (`-x 16 -s 16 -k 16M`). On Colab (GCP internal network), HF GCP CDN is **2–3× faster than Google Drive** — do NOT route public HF weights through Drive when a direct HF URL exists.
2. **Channel A — Private / Custom / Non-HF Weights (`~150 MB/s`, 4–6× faster than Drive FUSE)**: For private checkpoints, trained LoRAs, or weights missing from HuggingFace (e.g. ModelScope-only files pre-staged to Drive on `--cpu`), auto-discover Colab's running DriveFS auth server (`--metadata_server_auth_uri` from `/proc/*/cmdline`, auto-prefixing `http://`, with `127.0.0.1:8012` & `DRIVE_ACCESS_TOKEN` fallbacks), resolve exact MyDrive-relative paths to unique `file_id`s, and pull via **Google Drive v3 HTTPS API + `aria2c`** — bypassing FUSE completely and avoiding duplicate filename collisions (`diffusion_pytorch_model.safetensors`).
3. **Combined Dual-Channel + Tri-Parallel Overlap (`~460+ MB/s` aggregate)**: Run Channel A + Channel B simultaneously in a background thread while the main thread runs `git clone` / `uv pip install --system`, then `join()` before loading the model.

> **Scope Note**: Persistent downloads of new multi-GB weights into Google Drive still belong on `--cpu` `prep` VMs (`colab-ultra pull`). This recipe is specifically for staging weights onto local NVMe on Colab GPU VMs at runtime.
>
> **Reusable Script**: Available locally at [scripts/fast_stage_nvme.py](../scripts/fast_stage_nvme.py) and synced to Google Drive at `/content/drive/MyDrive/ColabUltra/scripts/fast_stage_nvme.py` (`~/ColabUltra/scripts/fast_stage_nvme.py`).

---

## 1. Using `fast_stage_nvme.py` (CLI or Python Import)

### Option A: Import Directly & Overlap with `pip` / `uv` Setup (Recommended)

```python
import sys
import threading
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
from fast_stage_nvme import stage_dual_channel

staging_err = []

def _bg_stage():
    try:
        stage_dual_channel(
            dest_root=Path("/content/models"),
            # Pass exact MyDrive-relative paths so duplicate filenames (e.g. vae/diffusion_pytorch_model.safetensors) never collide!
            drive_files=[
                ("MiniMax/MiniMax-H3-Image/vae", "ColabUltra/huizhou-h3/models/MiniMax/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors"),
            ],
            hf_urls=[
                (
                    "Comfy-Org/MiniMax-H3-Image/diffusion_models",
                    "minimax_h3_image_pruned_int8_convrot.safetensors",
                    "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
                ),
            ],
        )
    except Exception as exc:
        staging_err.append(exc)

t_stage = threading.Thread(target=_bg_stage, daemon=True)
t_stage.start()

# Main thread concurrently installs Python dependencies / clones repos
setup_python_env()

t_stage.join()
if staging_err:
    raise staging_err[0]
```

### Option B: Standalone CLI Call

```bash
python3 /content/drive/MyDrive/ColabUltra/scripts/fast_stage_nvme.py \
  --dest-root /content/models \
  --drive "MiniMax/MiniMax-H3-Image/vae|ColabUltra/huizhou-h3/models/MiniMax/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors" \
  --hf "Comfy-Org/MiniMax-H3-Image/diffusion_models|minimax_h3_image_pruned_int8_convrot.safetensors|https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"
```

### Built-in Safety Guarantees in `fast_stage_nvme.py`
- **1-Second `aria2` Install (No `r2u` Apt Stall)**: Checks Drive-cached Ubuntu `.deb` bundle (`/content/drive/MyDrive/ColabUltra/bin/aria2_debs/*.deb`) first (`dpkg -i` in ~1s). On first run, skips `/etc/apt/sources.list.d/` (`-o Dir::Etc::sourceparts=-`) to avoid 6-minute `r2u.stat.illinois.edu` timeouts, installs from official Ubuntu repos in ~3–5s, and populates the Drive cache automatically.
- **Native Colab Auth Auto-Discovery**: Reads `--metadata_server_auth_uri` from Colab's running `/opt/google/drive/drive` process (auto-prefixing `http://` when omitted; works out-of-the-box with standard `colab-ultra up` / `colab drivemount`, no third-party `colab-dfs-ext` required).
- **Zero Filename Collisions**: Walks Drive folder IDs segment-by-segment from `'root'` when given a path, and errors out on ambiguous bare filenames.
- **Resume-Safe Partial Check**: Checks `not Path(f"{dest}.aria2").exists()` (plus exact Drive byte size) before skipping existing files, so interrupted downloads are resumed rather than mistaken as complete.
- **Live Progress & Token Hygiene**: Prints `aria2c` progress summaries every 10s (`--summary-interval=10`), stores `Bearer` token in a `0o600` temp file, and unlinks it in `finally`.
- **Pair with `vram_policy.py`**: After staging weights onto `/content/models`, import `vram_policy.build_vram_plan(component_sizes_gb)` from `/content/drive/MyDrive/ColabUltra/scripts` to automatically choose `full_resident` / `selective_offload` / `tight_offload` and `vae_tiling` from actual GPU VRAM.

---

## 2. Small-File Bundle Pattern (Avoiding FUSE `stat()` Storms)

When staging directories with thousands of small `.py` files (such as ComfyUI `custom_nodes` or large Python plugin trees), copying or importing directly from `/content/drive/MyDrive/...` triggers a severe FUSE `stat()` storm (often 20–60s+ just to scan/import).

Instead:
1. Pre-package the directory once into a single `.tar.gz` archive on Drive (e.g., `custom_nodes_bundle.tar.gz`).
2. Pull the single archive via `fast_stage_nvme.py` (`--drive`) onto `/content/` and unpack locally with `tar -xzf`:
   ```bash
   tar -xzf /content/custom_nodes_bundle.tar.gz -C /content/ComfyUI/
   ```
   Unpacking a single local archive takes ~1.5 seconds vs ~60 seconds over FUSE `rsync`.
