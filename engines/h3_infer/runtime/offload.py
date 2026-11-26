"""Stream CPU-backed inference blocks through two fixed-address CUDA slots."""

from __future__ import annotations

import gc
from copy import deepcopy
from threading import RLock

import torch
from torch import nn

from ..ops.quant import Int8ConvRotLinear


def _tensors(module):
    yield from module.named_parameters(remove_duplicate=False)
    yield from module.named_buffers(remove_duplicate=False)


def _device_tensor(tensor, data):
    if isinstance(tensor, nn.Parameter):
        return nn.Parameter(data, requires_grad=tensor.requires_grad)
    return data


def _bind(module, tensors):
    for name, tensor in tensors.items():
        path, _, leaf = name.rpartition(".")
        setattr(module.get_submodule(path), leaf, tensor)


def _block_bytes(block):
    tensors = {id(t): t for _, t in _tensors(block)}
    return sum(t.numel() * t.element_size() for t in tensors.values())


def _block_layout(block):
    tensors = tuple((name, tensor.shape, tensor.dtype) for name, tensor in _tensors(block))
    rotations = tuple(
        (name, layer.convrot, layer.convrot_groupsize)
        for name, layer in block.named_modules()
        if isinstance(layer, Int8ConvRotLinear)
    )
    return tensors, rotations


class AsyncBlockOffloader:
    """Keep a resident prefix and overlap the remaining transfers with computation."""

    def __init__(self, blocks: nn.ModuleList, device: str | torch.device,
                 resident_blocks: int | str = "auto", pin_staging: bool = True,
                 headroom_bytes: int = 6 * 1024**3):
        if resident_blocks != "auto" and not isinstance(resident_blocks, int):
            raise ValueError("resident_blocks must be an integer or 'auto'")
        if headroom_bytes < 0:
            raise ValueError("headroom_bytes must be nonnegative")
        self.blocks = blocks
        self.device = torch.device(device)
        self.resident_blocks = resident_blocks
        self.pin_staging = pin_staging
        self.headroom_bytes = headroom_bytes
        self.block_bytes = max((_block_bytes(block) for block in blocks), default=0)
        if self.device.type != "cuda" or not self.block_bytes:
            self.num_resident = len(blocks)
        else:
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if resident_blocks == "auto":
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                budget = max(0, free_bytes - 2 * self.block_bytes - headroom_bytes)
                resident_blocks = budget // self.block_bytes
            self.num_resident = min(len(blocks), max(0, resident_blocks))

        self.slot_modules = []
        self.staging = []
        self.copy_stream = None
        self.compute_stream = None
        self.ready_events = []
        self.done_events = []
        self.prepared = False
        self._backing = []
        self._slot_tensors = []
        self._ready = [False, False]
        self._done = [False, False]
        self._lock = RLock()

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare_gpu(self):
        with self._lock:
            if self.prepared:
                return
            if self.device.type != "cuda":
                self.prepared = True
                return

            self.compute_stream = torch.cuda.current_stream(self.device)
            try:
                for index in range(self.num_resident):
                    block = self.blocks[index]
                    backing = dict(_tensors(block))
                    if any(t.device != self.device for t in backing.values()):
                        moved = {
                            id(t): _device_tensor(t, t.detach().to(self.device))
                            for t in backing.values()
                        }
                        self._backing.append((block, backing))
                        _bind(block, {name: moved[id(t)] for name, t in backing.items()})

                if self.num_resident < len(self.blocks):
                    template = self.blocks[self.num_resident]
                    layout = _block_layout(template)
                    if all(_block_layout(self.blocks[i]) == layout
                           for i in range(self.num_resident + 1, len(self.blocks))):
                        self._prepare_slots(template)
                self.prepared = True
            except BaseException:
                self.release_gpu()
                raise

    def _prepare_slots(self, template):
        weights = dict(_tensors(template))
        for _ in range(2):
            memo = {
                id(t): _device_tensor(t, torch.empty_like(t, device=self.device))
                for t in weights.values()
            }
            slot = deepcopy(template, memo)
            self.slot_modules.append(slot)
            self._slot_tensors.append(dict(_tensors(slot)))
            if self.pin_staging:
                self.staging.append({
                    name: torch.empty_like(t, device="cpu", pin_memory=True)
                    for name, t in weights.items()
                })
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_stream.wait_stream(self.compute_stream)
        self.ready_events = [torch.cuda.Event() for _ in range(2)]
        self.done_events = [torch.cuda.Event() for _ in range(2)]

    def release_gpu(self):
        with self._lock:
            try:
                if self.copy_stream is not None:
                    self.copy_stream.synchronize()
                if self.compute_stream is not None:
                    self.compute_stream.synchronize()
            finally:
                for block, backing in self._backing:
                    _bind(block, backing)
                self._backing.clear()
                self.slot_modules.clear()
                self._slot_tensors.clear()
                self.staging.clear()
                self.ready_events.clear()
                self.done_events.clear()
                self.copy_stream = self.compute_stream = None
                self._ready = [False, False]
                self._done = [False, False]
                self.prepared = False
                gc.collect()
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()

    @torch.no_grad()
    def _prefetch(self, index, slot):
        source = dict(_tensors(self.blocks[index]))
        if self.pin_staging:
            # The previous H2D read must finish before the host overwrites these pages.
            if self._ready[slot]:
                self.ready_events[slot].synchronize()
            staging = self.staging[slot]
            for name, tensor in source.items():
                staging[name].copy_(tensor)
            source = staging
        with torch.cuda.stream(self.copy_stream):
            if self._done[slot]:
                self.copy_stream.wait_event(self.done_events[slot])
            for name, target in self._slot_tensors[slot].items():
                target.copy_(source[name], non_blocking=True)
            self.ready_events[slot].record(self.copy_stream)
        self._ready[slot] = True

    def run_blocks(self, block_runner, hidden, *step_args):
        with self._lock:
            self.prepare_gpu()
            if self.device.type != "cuda":
                for block in self.blocks:
                    hidden = block_runner(block, hidden, *step_args)
                return hidden

            stream = torch.cuda.current_stream(self.device)
            if stream != self.compute_stream:
                stream.wait_stream(self.compute_stream)
                self.compute_stream = stream

            first = self.num_resident
            if self.slot_modules:
                self._prefetch(first, 0)
            for index in range(first):
                hidden = block_runner(self.blocks[index], hidden, *step_args)
            if not self.slot_modules:
                # Mixed quantization layouts require the original block's structure.
                for index in range(first, len(self.blocks)):
                    block = self.blocks[index]
                    memo = {
                        id(t): _device_tensor(t, t.detach().to(self.device))
                        for _, t in _tensors(block)
                    }
                    hidden = block_runner(deepcopy(block, memo), hidden, *step_args)
                return hidden
            for index in range(first, len(self.blocks)):
                slot = (index - first) % 2
                if index + 1 < len(self.blocks):
                    self._prefetch(index + 1, slot ^ 1)
                stream.wait_event(self.ready_events[slot])
                try:
                    hidden = block_runner(self.slot_modules[slot], hidden, *step_args)
                finally:
                    self.done_events[slot].record(stream)
                    self._done[slot] = True
            return hidden
