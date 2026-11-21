"""Background lossless WebP encoding and atomic tile publication."""
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path


def encode(png, dest):
    """Keep the input PNG until the complete output has been published."""
    with tempfile.TemporaryDirectory(dir=dest.parent) as work:
        temp = Path(work) / dest.name
        if dest.suffix.lower() == ".webp":
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(png), "-c:v", "libwebp",
                            "-lossless", "1", "-compression_level", "1", str(temp)], check=True)
        elif dest.suffix.lower() == ".png":
            shutil.copy2(png, temp)
        else:
            raise ValueError(f"Expected .webp or .png: {dest}")
        temp.replace(dest)
    png.unlink()


@contextmanager
def encoders(workers=2):
    """Yield an enqueue function and surface encoding errors when draining."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []

        def enqueue(png, dest):
            pending.append(pool.submit(encode, png, dest))

        yield enqueue
        for future in pending:
            future.result()
