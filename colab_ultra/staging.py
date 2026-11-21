"""Deploy the VM library and generate source for a remote pull."""
import filecmp
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from .drive import scripts_target
from .remote import models


RUNTIME = Path(__file__).parent / "remote"


def runtime_files():
    """Select exactly the Python sources needed by a VM."""
    return sorted(p for p in RUNTIME.rglob("*.py") if not p.name.startswith("._") and "__pycache__" not in p.parts)


def kaggle_runtime_code():
    """Materialize the same runtime in Kaggle's emulated Drive before execution."""
    sources = {str(p.relative_to(RUNTIME)): p.read_bytes() for p in runtime_files()}
    return f"""
from pathlib import Path
_cu_root = Path("/content/drive/MyDrive/ColabUltra/scripts/cu_remote")
for _name, _content in {sources!r}.items():
    _path = _cu_root / _name
    _path.parent.mkdir(parents=True, exist_ok=True)
    if not _path.exists() or _path.read_bytes() != _content:
        _path.write_bytes(_content)
"""


def sync_remote(hub=None):
    """Publish and verify the library before retiring either legacy script."""
    source = RUNTIME
    scripts = scripts_target(hub)
    target = scripts / "cu_remote"
    target.mkdir(parents=True, exist_ok=True)
    files = runtime_files()
    required = {"__init__.py", "stage.py", "vram_policy.py", "models.py"}
    if not required.issubset(p.relative_to(source).as_posix() for p in files):
        raise FileNotFoundError(f"Incomplete runtime in {source}")
    with tempfile.TemporaryDirectory(dir=scripts) as temp:
        for src in files:
            dest = target / src.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and filecmp.cmp(src, dest, shallow=False):
                continue
            staged = Path(temp) / src.name
            shutil.copyfile(src, staged)
            staged.replace(dest)
    for src in files:
        dest = target / src.relative_to(source)
        if not filecmp.cmp(src, dest, shallow=False):
            raise IOError(f"Drive sync verification failed: {dest}")
    for name in ("fast_stage_nvme.py", "vram_policy.py"):
        (scripts / name).unlink(missing_ok=True)
    return target


def pull_code(source, project=None, dest="models", filename=None):
    """Resolve the destination before allocating a worker, then generate its transfer code."""
    if project is not None:
        models.check_relative(project)
        # Legacy project pulls also accept directory spellings such as "." and "models/".
        if Path(dest).is_absolute() or ".." in Path(dest).parts:
            raise ValueError(f"Expected a project-relative destination: {dest!r}")
    else:
        models.check_relative(dest)
    if filename is not None:
        models.check_relative(filename)

    parts = Path(dest).parts
    shared = project is None and parts[:1] == ("models",)
    if shared:
        if len(parts) < 3:
            raise ValueError("--dest must include models/<model>/<variant> for a shared model")
        models.variant_dir(parts[1], parts[2])
    target = Path("/content/drive/MyDrive/ColabUltra") / (project or "") / dest
    if "://" in source or source.startswith("/content/drive/"):
        name = filename or Path(urlsplit(source).path).name
        models.check_relative(name)
        if shared and urlsplit(source).scheme not in {"hf", "http", "https"}:
            raise ValueError("Shared models require an HF or HTTP(S) file URL")
        transfer = f"plans = [stage.plan(source, target / {name!r}, min_bytes=0)]"
    else:
        transfer = """
try:
    import huggingface_hub
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
plans = stage.repo_plans(source, filename or "*", target, min_bytes=0)
"""
    publish = "stage.run_many(plans)"
    if shared:
        publish = f"""
from cu_remote import models
vdir = models.variant_dir({parts[1]!r}, {parts[2]!r})
files = {{p.dest.relative_to(vdir.resolve()).as_posix(): p.source for p in plans}}
models.ensure({parts[1]!r}, {parts[2]!r}, files)
"""
    return f"""
import os
import sys
from pathlib import Path
sys.path.insert(0, "/content/drive/MyDrive/ColabUltra/scripts")
from cu_remote import stage
target = Path({str(target)!r})
source = {source!r}
filename = {filename!r}
{transfer}
{publish}
import time
os.sync()
def _tx_bytes():
    for ln in Path("/proc/net/dev").read_text().splitlines():
        if "eth0:" in ln:
            return int(ln.split("eth0:", 1)[1].split()[8])
    return 0
print("Flushing Drive FUSE write-back cache to Google Drive...", flush=True)
low_cnt = 0
for _ in range(30):
    b0 = _tx_bytes()
    time.sleep(3.0)
    os.sync()
    mb_s = max(0, _tx_bytes() - b0) / (3.0 * 1024 * 1024)
    if mb_s < 2.0:
        low_cnt += 1
        if low_cnt >= 2:
            break
    else:
        low_cnt = 0
        print(f"  [Drive upload] {{mb_s:.1f}} MB/s...", flush=True)
print("✅ Download and Drive flush complete!", flush=True)
"""
