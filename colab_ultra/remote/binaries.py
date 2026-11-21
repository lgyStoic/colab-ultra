"""Materialize executables on a filesystem that supports the Unix execute bit."""
import shutil
import tempfile
from pathlib import Path


def _publish(source, target, executable=False):
    """Copy beside the target and replace it only after a complete copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as work:
        temp = Path(work) / target.name
        shutil.copy2(source, temp)
        if executable:
            temp.chmod(0o755)
        temp.replace(target)


def ensure_binary(name, cache, build):
    """Call build() on cache miss. It returns {binary_name: built_path}."""
    cached = Path(cache) / name
    if not cached.is_file() or not cached.stat().st_size:
        for target, source in build().items():
            _publish(source, Path(cache) / target)
    local = Path("/tmp") / name
    _publish(cached, local, executable=True)
    return local
