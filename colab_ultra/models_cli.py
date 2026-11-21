"""Inspect the local Drive model library using manifests and file metadata."""
import errno
import json
import os
import shutil
import stat
from pathlib import Path

from rich.console import Console
from rich.table import Table

from colab_ultra.drive import detect_local_drive_hub
from colab_ultra.remote.models import check_relative, read_manifest, variant_dir


def add_models_parser(sub):
    parser = sub.add_parser("models", help="管理本地 Drive 共享模型库")
    actions = parser.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("ls", help="列出模型版本、大小与清单状态")
    verify = actions.add_parser("verify", help="按清单核对文件路径与大小")
    remove = actions.add_parser("rm", help="预览或删除一个模型版本")
    for action in (listing, verify, remove):
        action.add_argument("--hub", type=Path, help="ColabUltra 根目录")
    for action in (listing, verify):
        action.add_argument("--json", action="store_true", help="输出 JSON")
    verify.add_argument("target", nargs="?", metavar="MODEL/VARIANT")
    remove.add_argument("target", metavar="MODEL/VARIANT")
    remove.add_argument("--yes", action="store_true", help="确认删除")


def _ignored(name):
    return name in {".cache", "__pycache__", ".DS_Store"} or name.startswith("._")


def _target(root, target):
    parts = target.split("/")
    if len(parts) != 2:
        raise ValueError(f"Expected MODEL/VARIANT, got {target!r}")
    model, variant = parts
    variant_dir(model, variant)
    path = root / model / variant
    if path.parent.is_symlink() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"Model variant unavailable: {path}")
    return path


def _variants(root):
    if not root.exists():
        return
    for model in sorted(root.iterdir()):
        if model.is_symlink() or not model.is_dir() or _ignored(model.name):
            continue
        try:
            variant_dir(model.name, "default")
        except ValueError:
            continue
        for path in sorted(model.iterdir()):
            if path.is_symlink() or not path.is_dir() or _ignored(path.name):
                continue
            try:
                variant_dir(model.name, path.name)
            except ValueError:
                continue
            yield path


def _payload_sizes(vdir):
    sizes = {}
    pending = [vdir]
    while pending:
        for path in pending.pop().iterdir():
            if _ignored(path.name) or path == vdir / "manifest.json":
                continue
            info = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                sizes[path.relative_to(vdir).as_posix()] = info.st_size
    return sizes


def _inspect(vdir):
    sizes = _payload_sizes(vdir)
    issues = []
    row = {"model": vdir.parent.name, "variant": vdir.name, "files": len(sizes),
           "bytes": sum(sizes.values()), "issues": issues}
    path = vdir / "manifest.json"
    try:
        if path.is_symlink():
            raise ValueError("manifest.json is a symbolic link")
        if not path.exists():
            raise FileNotFoundError("manifest.json is missing")
        if not path.is_file():
            raise ValueError("manifest.json is not a regular file")
        expected = {}
        for rel, entry in read_manifest(vdir)["files"].items():
            check_relative(rel)
            if any(_ignored(name) for name in rel.split("/")):
                continue
            size = entry.get("bytes") if isinstance(entry, dict) else None
            if type(size) is not int or size < 0 or rel.split("/")[0] == "manifest.json":
                raise ValueError(f"Invalid manifest entry: {rel}")
            expected[rel] = size
    except (OSError, ValueError) as exc:
        issues.append({"kind": "invalid_manifest", "detail": str(exc)})
        row["manifest"] = "MISSING" if isinstance(exc, FileNotFoundError) else "MISMATCH"
        return row
    for rel in sorted(sizes.keys() | expected.keys()):
        if rel not in sizes:
            issues.append({"kind": "missing", "path": rel})
        elif rel not in expected:
            issues.append({"kind": "untracked", "path": rel})
        elif sizes[rel] != expected[rel]:
            issues.append({"kind": "size_mismatch", "path": rel,
                           "expected": expected[rel], "actual": sizes[rel]})
    row["manifest"] = "MISMATCH" if issues else "OK"
    return row


def _remove(vdir, confirmed):
    sizes = _payload_sizes(vdir)
    label = f"{vdir.parent.name}/{vdir.name} ({len(sizes)} files, {sum(sizes.values()) / 1e9:.3f} GB)"
    if not confirmed:
        print(f"Dry run: would remove {label}. Use --yes to delete.")
        return
    # Anchor deletion to real directories even if a Drive sync replaces a parent.
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(vdir.parent.parent, flags)
    try:
        model_fd = os.open(vdir.parent.name, flags, dir_fd=root_fd)
        try:
            shutil.rmtree(vdir.name, dir_fd=model_fd)
        finally:
            os.close(model_fd)
        try:
            os.rmdir(vdir.parent.name, dir_fd=root_fd)
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
    finally:
        os.close(root_fd)
    print(f"Removed {label}.")


def cmd_models(args):
    hub = args.hub or detect_local_drive_hub(create=False) or Path.home() / "ColabUltra"
    root = Path(hub).expanduser() / "models"
    try:
        if root.is_symlink():
            raise ValueError(f"Model library is a symbolic link: {root}")
        target = getattr(args, "target", None)
        if args.action == "rm":
            _remove(_target(root, target), args.yes)
            return
        paths = [_target(root, target)] if target is not None else _variants(root)
        rows = [_inspect(path) for path in paths]
    except (ValueError, OSError) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            raise SystemExit(1) from exc
        raise SystemExit(str(exc)) from exc

    totals = {"variants": len(rows), "files": sum(r["files"] for r in rows),
              "bytes": sum(r["bytes"] for r in rows)}
    clean = all(not row["issues"] for row in rows)
    if args.json:
        print(json.dumps({"variants": rows, "totals": totals}, ensure_ascii=False))
    elif args.action == "ls":
        table = Table("model/variant", "files", "size (GB)", "manifest")
        for row in rows:
            table.add_row(f"{row['model']}/{row['variant']}", str(row["files"]),
                          f"{row['bytes'] / 1e9:.3f}", row["manifest"])
        table.add_section()
        table.add_row("Total", str(totals["files"]), f"{totals['bytes'] / 1e9:.3f}", "")
        Console().print(table)
    else:
        for row in rows:
            print(f"{row['model']}/{row['variant']}  {row['manifest']}")
            for issue in row["issues"]:
                detail = issue.get("path") or issue["detail"]
                if issue["kind"] == "size_mismatch":
                    detail += f" (expected {issue['expected']} bytes, found {issue['actual']})"
                print(f"  {issue['kind']}: {detail}")
        print(f"Verified {len(rows)} variants, {sum(len(r['issues']) for r in rows)} issues.")
    if args.action == "verify" and not clean:
        raise SystemExit(1)
