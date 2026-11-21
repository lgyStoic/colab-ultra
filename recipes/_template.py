"""Copy this file into a project, list weights in MODELS, and implement infer()."""
import sys
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
try:
    from cu_remote import recipe, stage
except ModuleNotFoundError as exc:
    if exc.name != "cu_remote":
        raise
    # Source checkout for offline dry runs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from colab_ultra.remote import recipe, stage

MODEL_DIR = Path("/content/models/custom")
MODELS = [
    # ("hf://org/repo/model.safetensors", "model.safetensors"),
]


def infer(item, out_path):
    """Run model inference for a single prompt item and write out_path."""
    raise NotImplementedError("write your inference here")


def main():
    """Validate prompts, stage configured weights, and run inference per tile."""
    cfg, code, root = recipe.load_project()
    run_name = cfg.get("run_name", root.name)
    plans = [stage.plan(src, MODEL_DIR / dest) for src, dest in MODELS]
    if recipe.dry_run():
        recipe.describe(cfg, run_name)
        for plan in plans:
            print(f"stage: {plan.source} -> {plan.dest} ({plan.backend}, c={plan.concurrency})")
        return
    if plans:
        stage.run_many(plans)
    out = root / "runs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(cfg["prompts"], 1):
        slug = item.get("slug", f"img_{idx:02d}")
        fmt = item.get("output_format", cfg.get("output_format", "jpg")).lower().lstrip(".")
        infer(item, out / f"{slug}.{fmt}")


if __name__ == "__main__":
    main()
