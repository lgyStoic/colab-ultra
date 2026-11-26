"""Load the DiffSynth reference modules without its package-wide imports."""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None

if not HAS_TORCH:
    collect_ignore_glob = ["test_*.py"]
else:
    import torch

    engines_root = str(Path(__file__).resolve().parents[2] / "engines")
    if engines_root not in sys.path:
        sys.path.insert(0, engines_root)

    @pytest.fixture(scope="session", autouse=True)
    def cpu_threads():
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(previous)


@pytest.fixture
def diffsynth_reference(monkeypatch):
    default = Path(__file__).resolve().parents[3] / "diffsynth"
    root = Path(os.environ.get("DIFFSYNTH_ROOT", default)) / "diffsynth"
    if not root.is_dir():
        pytest.skip(f"DiffSynth source missing at {root}. Set DIFFSYNTH_ROOT to the checkout.")

    package = "_h3_diffsynth_reference"
    monkeypatch.setenv("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    for suffix in ("", ".models", ".diffusion", ".core"):
        name = package + suffix
        directory = root.joinpath(*suffix.removeprefix(".").split("."))
        module = ModuleType(name)
        module.__path__ = [str(directory)]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        monkeypatch.setitem(sys.modules, name, module)

    def load(name):
        return importlib.import_module(f"{package}.{name}")

    yield root, load
    for name in tuple(sys.modules):
        if name.startswith(package + ".") and name not in (package + ".models", package + ".diffusion", package + ".core"):
            sys.modules.pop(name)
