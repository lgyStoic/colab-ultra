"""Compile the DiT as a whole or share one compiled region across its blocks."""

from typing import Literal

import torch

CompileMode = Literal["none", "regional", "full"]


def run_block(block, hidden, *args):
    return block(hidden, *args)


def compile_dit(dit, mode: CompileMode = "regional", **kwargs):
    if mode == "none":
        dit._block_runner = run_block
    elif mode == "regional":
        kwargs.setdefault("dynamic", True)
        dit._block_runner = torch.compile(run_block, **kwargs)
    elif mode == "full":
        if dit.offload:
            raise ValueError("Full compilation requires offload=False")
        dit._block_runner = run_block
        return torch.compile(dit, **kwargs)
    else:
        raise ValueError(f"Unknown compile mode {mode!r}")
    return dit
