#!/usr/bin/env python3
"""Inventory legacy model trees by size and optionally move them within a Drive hub."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_ultra.remote.models import read_manifest, variant_dir, write_manifest


# Trailing /** matches a complete tree and preserves every relative filename.
MAPPING = {
    "models/MiniMaxAI/MiniMax-H3-Image/**": "models/minimax-h3-image/bf16",
    "models/h3gguf/minimax_h3_image_pruned-UD-Q4_K_XL.gguf":
        "models/minimax-h3-image/gguf-ud-q4_k_xl/minimax_h3_image_pruned-UD-Q4_K_XL.gguf",
    "models/h3gguf/qwen3vl_32b_minimax_h3-Q4_K_M.gguf":
        "models/minimax-h3-qwen3vl-32b/gguf-q4_k_m/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    "models/h3gguf/diffusion_pytorch_model.safetensors":
        "models/minimax-h3-image/bf16/vae/diffusion_pytorch_model.safetensors",
    "huizhou-h3/models/MiniMax/MiniMax-H3-Image/**": "models/minimax-h3-image/bf16",
    "huizhou-h3/models/Comfy-Org/MiniMax-H3-Image/**": "models/minimax-h3-image/comfy-int8-convrot",
    "moss-distill/models/MOSS-Image-SFT/**": "models/moss-image-sft/bf16",
    "moss-distill/models/lodestones_8step/**": "models/moss-image-sft/lora-lodestones-8step",
}
VARIANTS = {Path(*Path(dst).parts[:3]) for dst in MAPPING.values()}
TYPE_DIRS = {"llm", "checkpoints", "loras", "vae_clip"}
ACTIONS = ("MOVE", "DUPLICATE", "CONFLICT", "UNMAPPED", "REMOVE-EMPTY-DIR")


@dataclass(frozen=True)
class Item:
    action: str
    src: str
    dst: str | None
    bytes: int


def destination(rel):
    for pattern, dest in MAPPING.items():
        if pattern.endswith("/**"):
            base = Path(pattern[:-3])
            if base in rel.parents:
                return Path(dest) / rel.relative_to(base)
        elif rel == Path(pattern):
            return Path(dest)
    return None


def ignored(name):
    return name in {".cache", "__pycache__", ".DS_Store"} or name.startswith("._")


def canonical(rel, files):
    if rel in VARIANTS:
        return True
    if len(rel.parts) == 3 and rel.parts[0] == "models" and "manifest.json" in files:
        try:
            variant_dir(*rel.parts[1:])
        except ValueError:
            return False
        return True
    return False


def classify(hub, dst, size, planned, planned_dirs):
    target = hub / dst
    if dst in planned_dirs or dst.parts[3] == "manifest.json":
        return "CONFLICT"
    for parent in dst.parents:
        folder = hub / parent
        if parent in planned or folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            return "CONFLICT"
    if target.is_symlink() or (target.exists() and not target.is_file()):
        return "CONFLICT"
    existing = target.stat().st_size if target.is_file() else planned.get(dst)
    if existing is not None:
        return "DUPLICATE" if size == existing else "CONFLICT"
    planned[dst] = size
    planned_dirs.update(dst.parents)
    return "MOVE"


def walk_error(error):
    raise error


def scan(hub):
    """Account for pending destinations as well as files already present on disk."""
    if not hub.is_dir():
        raise FileNotFoundError(f"Drive hub unavailable: {hub}")
    roots = [hub / "models"] + [p / "models" for p in sorted(hub.iterdir())
                                if p.name != "models" and not ignored(p.name) and not p.is_symlink() and p.is_dir()]
    items = []
    planned = {}
    planned_dirs = set()
    removable = set()
    candidates = {}
    source_dirs = {Path(p.removesuffix("/**")) for p in MAPPING}
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            continue
        for folder, dirs, files in os.walk(root, onerror=walk_error):
            path = Path(folder)
            rel_dir = path.relative_to(hub)
            children = dirs + files
            dirs[:] = sorted(d for d in dirs if not ignored(d) and not (path / d).is_symlink())
            if canonical(rel_dir, files):
                dirs.clear()
                continue
            if path != hub / "models" and (
                path == root or rel_dir in source_dirs or any(base in rel_dir.parents or rel_dir in base.parents
                                                             for base in source_dirs)
                or (path.parent == root and path.name in TYPE_DIRS)
            ):
                candidates[path] = children
            for name in sorted(files):
                if ignored(name):
                    continue
                src = path / name
                rel = src.relative_to(hub)
                dst = destination(rel)
                size = src.lstat().st_size
                if dst is None or src.is_symlink() or not src.is_file():
                    items.append(Item("UNMAPPED", str(rel), None, size))
                    continue
                action = classify(hub, dst, size, planned, planned_dirs)
                items.append(Item(action, str(rel), str(dst), size))
                if action in {"MOVE", "DUPLICATE"}:
                    removable.add(src)
    # A skipped cache or an unmapped file still occupies its parent.
    for path in sorted(candidates, key=lambda p: (-len(p.parts), p)):
        if all(path / name in removable for name in candidates[path]):
            items.append(Item("REMOVE-EMPTY-DIR", str(path.relative_to(hub)), None, 0))
            removable.add(path)
    return items


def apply(hub, items):
    if any(item.action == "CONFLICT" for item in items):
        raise ValueError("Conflicting destinations. No migration changes made.")
    # Load existing manifests before moving files. Include canonical files so an
    # interrupted move can be recovered by running the same migration again.
    manifests = {}
    for rel in sorted(VARIANTS):
        vdir = hub / rel
        if any(p.is_symlink() for p in (vdir, *vdir.parents) if p.is_relative_to(hub)):
            raise ValueError(f"Unexpected symlink in {vdir}")
        if (vdir / "manifest.json").is_symlink():
            raise ValueError(f"Unexpected symlink in {vdir}/manifest.json")
        manifests[vdir] = read_manifest(vdir)
    moved = set()
    for item in items:
        if item.action not in {"MOVE", "DUPLICATE"}:
            continue
        src, dst = hub / item.src, hub / item.dst
        if src.is_symlink() or src.stat().st_size != item.bytes:
            raise ValueError(f"Source changed after inventory: {src}")
        if item.action == "MOVE":
            dst.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(dst):
                raise FileExistsError(dst)
            os.rename(src, dst)
            moved.add(dst)
        elif dst.is_file() and not dst.is_symlink() and dst.stat().st_size == item.bytes:
            src.unlink()
        else:
            raise ValueError(f"Duplicate target changed after inventory: {dst}")
    for vdir, manifest in manifests.items():
        if not vdir.is_dir():
            continue
        entries = manifest["files"]
        changed = False
        for folder, dirs, files in os.walk(vdir, onerror=walk_error):
            path = Path(folder)
            dirs[:] = [d for d in dirs if not ignored(d) and not (path / d).is_symlink()]
            for name in files:
                target = path / name
                key = target.relative_to(vdir).as_posix()
                if key == "manifest.json" or ignored(name) or target.is_symlink() or not target.is_file():
                    continue
                size = target.stat().st_size
                if target in moved or entries.get(key, {}).get("bytes") != size:
                    entries[key] = {"source": None, "bytes": size, "sha256": None}
                    changed = True
        if changed or not (vdir / "manifest.json").exists():
            write_manifest(vdir, manifest)
    for item in items:
        if item.action == "REMOVE-EMPTY-DIR":
            (hub / item.src).rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, help="Drive hub (default: detected hub or ~/ColabUltra)")
    parser.add_argument("--report", type=Path, help="write the inventory as JSON")
    parser.add_argument("--apply", action="store_true", help="execute a conflict-free inventory")
    args = parser.parse_args(argv)
    hub = args.hub
    if hub is None:
        try:
            from colab_ultra.drive import detect_local_drive_hub
        except ImportError:
            detected = None
        else:
            detected = detect_local_drive_hub(create=False)
        hub = Path(detected or "~/ColabUltra")
    hub = hub.expanduser().resolve()
    try:
        items = scan(hub)
        counts = Counter(item.action for item in items)
        sizes = defaultdict(int)
        print("action\tsrc\tdst\tbytes")
        for item in items:
            print(item.action, item.src, item.dst or "-", item.bytes, sep="\t")
            sizes[item.action] += item.bytes
        totals = {}
        for action in ACTIONS:
            totals[action] = {"count": counts[action], "bytes": sizes[action]}
            print(f"{action:17} {counts[action]:5}  {sizes[action] / 1e9:10.3f} GB")
        if args.apply and counts["CONFLICT"]:
            raise ValueError("Conflicting destinations. No migration changes made.")
        if args.report:
            path = args.report.resolve()
            if path.is_relative_to(hub):
                rel = path.relative_to(hub)
                in_models = rel.parts[:1] == ("models",) or rel.parts[1:2] == ("models",)
                if in_models and (path.exists() or any(i.dst == str(rel) or
                        (i.action == "REMOVE-EMPTY-DIR" and Path(i.src) in rel.parents) for i in items)
                        or path.name == "manifest.json"):
                    raise ValueError(f"Report path conflicts with the model inventory: {path}")
            report = {"hub": str(hub), "items": [asdict(i) for i in items], "totals": totals}
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.apply:
            apply(hub, items)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
