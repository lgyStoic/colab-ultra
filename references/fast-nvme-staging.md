# Staging weights on a VM

Import `cu_remote.stage` from the synced Drive scripts directory. A plan resolves one source file, its destination, concurrency and minimum valid size. `run_many` stages files concurrently, propagates worker errors and serializes final copies to Drive. The default floor is 1,000,000 bytes. Use `min_bytes=0` for small, nonempty metadata files.

```python
import sys
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
from cu_remote import stage

models = Path("/content/models")
plans = [
    stage.plan("hf://unsloth/MiniMax-H3-Image-GGUF/minimax_h3_image_pruned-UD-Q4_K_XL.gguf",
               models / "dit.gguf", concurrency=16),
    stage.plan("drive://ColabUltra/tiles/models/adapter.safetensors", models / "adapter.safetensors"),
]
stage.run_many(plans)
```

HF resolve URLs and ordinary HTTP(S) URLs also work. `stage.repo_plans(repo, pattern, directory, revision="main")` uses `huggingface_hub` to enumerate matching files and preserves subdirectories. Use `HF_TOKEN` for authenticated HF downloads. Drive paths use the mounted filesystem. With an explicit `DRIVE_ACCESS_TOKEN`, `drive://` uses the Drive v3 media API and verifies its reported byte count.

## Comparing downloaders

Install `hf_transfer` on the benchmark VM before timing. Provision aria2 with `stage.ensure_aria2()`. Give both trials the same file, distinct fresh destinations on the same filesystem, and identical file-level concurrency. Compare returned bytes, elapsed seconds and decimal MB/s. For a Drive destination, elapsed time includes the sequential copy from NVMe.

```python
for backend, connections in (("aria2", 16), ("hf_transfer", 64)):
    item = stage.plan(plans[0].source, Path("/content/bench") / backend / "dit.gguf",
                      backend=backend, concurrency=connections)
    print(backend, stage.benchmark([item]))
```

aria2 caps splits and connections per file at 16. The HF worker sets `HF_HUB_ENABLE_HF_TRANSFER=1` and passes the requested parallelism directly to `hf_transfer.download`. The runner rejects cached benchmark destinations. For trees containing thousands of small files, stage a single archive and unpack it on NVMe to avoid repeated Drive FUSE metadata requests.
