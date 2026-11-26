"""Top-level and runtime imports avoid heavy inference dependencies."""

import os
import subprocess
import sys
from pathlib import Path


def test_clean_imports():
    script = """
import sys
import h3_infer
assert 'torch' not in sys.modules
for name in h3_infer.__all__:
    getattr(h3_infer, name)
bad = {'diffusers', 'transformers', 'accelerate', 'einops', 'torchvision'} & sys.modules.keys()
assert not bad, bad
"""
    env = os.environ.copy()
    engines_root = str(Path(__file__).resolve().parents[2] / "engines")
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = engines_root + os.pathsep + pythonpath if pythonpath else engines_root
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
