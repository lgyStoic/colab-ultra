# Shared model library

All Colab projects on the same mounted Drive can use the files in `ColabUltra/models/<model>/<variant>/`. Use lowercase model and variant names matching `^[a-z0-9][a-z0-9._-]*$`. Keep the original filenames and subdirectories within a variant. The two directory names identify the release and representation to reuse.

```text
ColabUltra/models/
  minimax-h3-image/
    bf16/
      manifest.json
      processor/tokenizer.json
      vae/diffusion_pytorch_model.safetensors
    gguf-ud-q4_k_xl/
      manifest.json
      minimax_h3_image_pruned-UD-Q4_K_XL.gguf
```

## Load a variant in a recipe

Run `colab-ultra sync-remote` to deploy the helper. Recipes import it from the VM's shared scripts directory.

```python
import sys
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
from cu_remote import models

name = "vae/diffusion_pytorch_model.safetensors"
files = {name: f"hf://MiniMaxAI/MiniMax-H3-Image/{name}"}
local = Path("/content/models/minimax-h3-image/bf16")
models.describe("minimax-h3-image", "bf16", files, to=local)
paths = models.ensure("minimax-h3-image", "bf16", files, to=local)
vae_path = paths[name]
```

`describe` prints the transfer plan offline. `ensure` takes HF or HTTP(S) file URLs and returns a dictionary keyed by relative filename. Omitting `to` returns the library paths. A missing file or size mismatch triggers a download. Reuse requires a nonempty file matching its manifest byte count. Choose a new variant when changing the underlying weights. The optional `revision` selects the HF revision for `hf://` URLs. A resolve URL carries its own revision.

Each variant has one `manifest.json`. This example shows the observed MiniMax release VAE size after migration.

```json
{
  "model": "minimax-h3-image",
  "variant": "bf16",
  "files": {
    "vae/diffusion_pytorch_model.safetensors": {
      "source": null,
      "bytes": 2336964672,
      "sha256": null
    }
  }
}
```

Migrated files acquire a source URL when a recipe first requests them. When available, the SHA-256 comes from Hugging Face LFS metadata. An unavailable digest stays `null`. A size match does not verify the file contents.

## Prefetch from the CLI

```bash
colab-ultra pull hf://MiniMaxAI/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors --dest models/minimax-h3-image/bf16/vae
```

The temporary CPU worker saves this file and its manifest in the global library. A repository ID and `--filename` glob preserve matching subdirectories. For a project-specific download, `colab-ultra pull org/repo -p tiles --dest models` still writes under `ColabUltra/tiles/models/`.

## Migrate an existing hub

```bash
python3 tools/migrate_models.py --hub ~/ColabUltra --report /tmp/model-inventory.json
```

The default prints a dry-run inventory. An explicit `--report` also saves the rows and totals as JSON. Review every `CONFLICT` and `UNMAPPED` row. The mapping covers the listed MiniMax and MOSS legacy directories. Unmapped files and skipped cache or system entries stay in place.

With `--apply`, any size conflict stops the migration before changes begin. A conflict-free run renames files within the hub, unlinks a duplicate when its destination has the same byte count, merges variant manifests and removes empty source directories. Duplicate detection uses size alone. Migration reads directory entries, file sizes and existing variant manifests without opening model payloads.
