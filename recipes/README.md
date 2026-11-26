# Inference recipes

Each recipe is a standalone Python script that can run through `colab-ultra exec -f`. The launcher sends the script's contents to the VM. Shared helpers belong in `colab_ultra/remote/`. Run `colab-ultra sync-remote` after changing them.

With `-p tiles`, the VM starts in `/content/drive/MyDrive/ColabUltra/tiles/code`. It reads `prompts.json` there and publishes images under `../runs/<run_name>/`. The scripts add `/content/drive/MyDrive/ColabUltra/scripts` to `sys.path` and import `cu_remote`. Locally, they can import the same sources from a checkout.

```json
{
  "run_name": "first_batch",
  "width": 1024,
  "height": 1024,
  "prompts": [{"slug": "tile_01", "prompt": "A fishing boat in morning fog"}]
}
```

Run from the checkout to validate the JSON and see the download and inference plan. The MiniMax-H3 recipe and template accept `--dry-run` or `CU_DRY_RUN=1`. A dry run needs only Python and leaves the output directory untouched.

```bash
H3_PROMPTS_JSON=~/ColabUltra/tiles/code/prompts.json python3 recipes/minimax_h3_infer.py --dry-run
H3_PROMPTS_JSON=~/ColabUltra/tiles/code/prompts.json CU_DRY_RUN=1 python3 recipes/minimax_h3_infer.py
```

- **Heterogeneous H3 pipeline** (`h3_pipeline_worker.py`) runs `embed`, `dit`, `vae`, or dual-GPU concurrent `embed_vae` workers over Redis Streams or Upstash REST (`code/pipeline_worker.json`, `--dry-run` supported). On 2×T4 it offloads the tail 14 TE layers to pinned CPU RAM so `embed` and `float32` `vae` run concurrently, while `dit` runs `int8_convrot` or `bf16` on L4/A100 with versioned Triton/Inductor compile cache persistence.
- **Standalone H3** (`minimax_h3_infer.py`) uses the `h3_infer` package published by `colab-ultra sync-remote`. Sync publishes the bundled `engines/h3_infer` package or uses `H3_INFER_SOURCE` pointing to a custom `h3_infer` directory. The recipe caches Stage 1 Qwen3-VL embeddings under `data/h3_embeds_cache/`, supports `edit_image` with one or more references, keeps the DiT resident across the batch, and batches up to 64 tiles per 2K VAE decode. It defaults to PNG, `skip_existing: true`, and `attention_backend: "auto"`. On GPUs with `sm >= 80`, auto installs `sageattention==1.0.6` if missing and prefers SageAttention, then FlashAttention, then SDPA. Older GPUs use SDPA.
  See the [H3 inference benchmarks](../references/h3-infer-benchmarks.md) for measured performance.
- **Custom inference** starts with `_template.py`, which calls `models.ensure` to fetch missing weights into the shared Drive library and copy them to local storage before inference. Keep config validation and dry-run handling before package installation, downloads, GPU imports or output writes:
  1. Edit `MODELS` and implement `infer(item, out_path)` in `_template.py`.
  2. Dry-run locally (`python3 recipes/_template.py --dry-run`), then run remotely (`colab-ultra exec -s NAME -p PROJECT -f recipes/_template.py`).

```bash
colab-ultra exec -s l4 -p tiles -f recipes/minimax_h3_infer.py --timeout 7200
```
