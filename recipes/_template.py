"""Copy this file into a project and supply the model-specific inference step."""
import sys
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
try:
    from cu_remote import recipe
except ModuleNotFoundError as exc:
    if exc.name != "cu_remote":
        raise
    # Source checkout for offline dry runs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from colab_ultra.remote import recipe


def main():
    """Load the project and visit each configured tile."""
    cfg, code, root = recipe.load_project()
    if recipe.dry_run():
        recipe.describe(cfg, cfg.get("run_name", root.name))
        return
    for item in cfg["prompts"]:
        # Replace this line with the model-specific inference call.
        print(item["prompt"])


if __name__ == "__main__":
    main()
