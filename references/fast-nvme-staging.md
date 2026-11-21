# Staging weights on a VM

Use `cu_remote.models.ensure(model, variant, files, to=nvme_dir)` for weights shared across projects. It checks the Drive manifest before downloading and stages the requested files on local storage. See the [model library guide](model-library.md).

For individual transfers and benchmarks, import `cu_remote.stage` from the synced Drive scripts directory. A plan resolves one source file, its destination, concurrency and minimum valid size. `run_many` stages files concurrently, propagates worker errors and serializes final copies to Drive. The default floor is 1,000,000 bytes. Use `min_bytes=0` for small, nonempty metadata files.

```python
import sys
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
from cu_remote import stage

models = Path("/content/models")
plans = [
    stage.plan("hf://unsloth/MiniMax-H3-Image-GGUF/minimax_h3_image_pruned-UD-Q4_K_XL.gguf",
               models / "dit.gguf"),
    stage.plan("drive://ColabUltra/models/moss-image-sft/lora-lodestones-8step/lora.safetensors", models / "adapter.safetensors"),
]
stage.run_many(plans)
```

HF and HTTP(S) downloads default to `hf_transfer` with 64 connections (auto-installed on first use if missing); `drive://` paths stay on `copy` (or Drive API when `DRIVE_ACCESS_TOKEN` is set). `stage.repo_plans(repo, pattern, directory, revision="main")` uses `huggingface_hub` to enumerate matching files and preserves subdirectories. Use `HF_TOKEN` for authenticated HF downloads. With an explicit `DRIVE_ACCESS_TOKEN`, `drive://` uses the Drive v3 media API and verifies its reported byte count.

## Comparing downloaders

Provision aria2 with `stage.ensure_aria2()`; `hf_transfer` is installed automatically if absent. Give both trials the same file, distinct fresh destinations on the same filesystem, and identical file-level concurrency. Compare returned bytes, elapsed seconds and decimal MB/s. For a Drive destination, elapsed time includes the sequential copy from NVMe.

```python
for backend, connections in (("aria2", 16), ("hf_transfer", 64)):
    item = stage.plan(plans[0].source, Path("/content/bench") / backend / "dit.gguf",
                      backend=backend, concurrency=connections)
    print(backend, stage.benchmark([item]))
```

Measured on Colab High-RAM CPU (`smoke`, 3 MiniMax-H3 HuggingFace files totaling `34,162,458,528` bytes / ~34.16 GB, `workers=3` to NVMe):

- `aria2` (`concurrency=16`): `119.45s` (**285.99 MB/s**)
- `hf_transfer` (`concurrency=64`): `99.46s` (**343.47 MB/s**, **+20.1% faster**, set as HF/HTTP default)

aria2 caps splits and connections per file at 16. The HF worker sets `HF_HUB_ENABLE_HF_TRANSFER=1` and passes the requested parallelism directly to `hf_transfer.download`. The runner rejects cached benchmark destinations. For trees containing thousands of small files, stage a single archive and unpack it on NVMe to avoid repeated Drive FUSE metadata requests.
