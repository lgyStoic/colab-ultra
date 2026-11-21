"""Read a project's prompts and validate a recipe before allocating hardware."""
import json
import os
import sys
from pathlib import Path


def dry_run():
    """Accept a script flag locally or an environment variable inside exec."""
    return "--dry-run" in sys.argv[1:] or os.environ.get("CU_DRY_RUN") == "1"


def load_project(formats=("jpg", "jpeg", "png", "webp")):
    """Read prompts from a project root, code directory, or explicit JSON path."""
    path = Path(os.environ.get("H3_PROMPTS_JSON", "prompts.json"))
    if not path.is_file() and path == Path("prompts.json"):
        path = Path("code") / path
    path = path.resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg.get("prompts"), list) or not cfg["prompts"]:
        raise ValueError(f"{path}: prompts must be a nonempty list")
    name = cfg.get("run_name")
    if name is not None and (not name or Path(name).name != name or name == ".."):
        raise ValueError(f"{path}: run_name must be a directory name")
    for key in ("sage_attn", "compile_dit", "dit_nf4", "vae_fp32"):
        if key in cfg and not isinstance(cfg[key], bool):
            raise ValueError(f"{path}: {key} must be a JSON boolean")
    seen = set()
    for idx, item in enumerate(cfg["prompts"], 1):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"Item {idx}: expected a prompt string")
        slug = item.get("slug", f"img_{idx:02d}")
        if slug in seen or Path(slug).name != slug or slug in ("", ".", ".."):
            raise ValueError(f"Item {idx}: invalid or duplicate slug {slug!r}")
        seen.add(slug)
        fmt = item.get("output_format", cfg.get("output_format", formats[0])).lower().lstrip(".")
        if fmt not in formats:
            raise ValueError(f"Item {idx}: unsupported output_format {fmt}")
        for key in ("width", "height"):
            if int(item.get(key, cfg.get(key, 2048))) <= 0:
                raise ValueError(f"Item {idx}: {key} must be positive")
        int(item.get("seed", cfg.get("seed_base", 100)))
    root = path.parent.parent if path.parent.name == "code" else path.parent
    return cfg, path.parent, root


def resolve_image(spec, code, root):
    """Resolve reference inputs relative to the project or its standard folders."""
    path = Path(spec).expanduser()
    for folder in (code, root, root / "data", root / "runs"):
        candidate = folder / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Reference image {spec!r} missing under {root}")


def describe(cfg, run_name):
    """Show which prompt, options and reference images each tile will use."""
    print(f"Dry run: {run_name} ({len(cfg['prompts'])} tiles)")
    for i, item in enumerate(cfg["prompts"], 1):
        print(item.get("slug", f"img_{i:02d}"), json.dumps(item, ensure_ascii=False))
