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

Run from the checkout to validate the JSON and see the download and inference plan. The DiffSynth recipe and template accept `--dry-run` or `CU_DRY_RUN=1`. A dry run needs only Python and leaves the output directory untouched.

```bash
H3_PROMPTS_JSON=~/ColabUltra/tiles/code/prompts.json python3 recipes/minimax_h3_diffsynth.py --dry-run
H3_PROMPTS_JSON=~/ColabUltra/tiles/code/prompts.json CU_DRY_RUN=1 python3 recipes/minimax_h3_diffsynth.py
```

- **DiffSynth** supports per-item `edit_image` (one path or several), `seed`, dimensions, and JPEG/PNG/lossless WebP. Set `vae_fp32` for the full-precision VAE or `dit_nf4` for NF4 DiT weights. It chooses model placement with `cu_remote.vram_policy`. This replaces `workflows/minimax_h3_batch.py`.
- **Custom inference** starts with `_template.py`, which calls `models.ensure` to fetch missing weights into the shared Drive library and copy them to local storage before inference. Keep config validation and dry-run handling before package installation, downloads, GPU imports or output writes:
  1. Edit `MODELS` and implement `infer(item, out_path)` in `_template.py`.
  2. Dry-run locally (`python3 recipes/_template.py --dry-run`), then run remotely (`colab-ultra exec -s NAME -p PROJECT -f recipes/_template.py`).

```bash
colab-ultra exec -s l4 -p tiles -f recipes/minimax_h3_diffsynth.py --timeout 7200
```
