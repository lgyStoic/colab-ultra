"""Non-causal self-attention on dense or packed MiniMax-H3 rows."""

from functools import lru_cache
from importlib import import_module
from typing import Literal

import torch
from torch.nn import functional as F

AttentionBackend = Literal["auto", "flash_attn", "sageattention", "sdpa"]
BACKENDS = ("auto", "flash_attn", "sageattention", "sdpa")


@lru_cache
def _optional_kernel(backend, packed):
    names = {
        "flash_attn": ("flash_attn_func", "flash_attn_varlen_func"),
        "sageattention": ("sageattn", "sageattn_varlen"),
    }
    try:
        return getattr(import_module(backend), names[backend][packed], None)
    except (ImportError, OSError, RuntimeError):
        # An unavailable CUDA extension can fail at import or shared-library load.
        return None


@torch.compiler.assume_constant_result
def _has_kernel(backend, packed) -> bool:
    # Check optional imports while tracing without letting Dynamo guard on a Python function.
    return _optional_kernel(backend, packed) is not None


@torch.library.custom_op("h3_infer::sageattn_nhd", mutates_args=())
def _sageattn_nhd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float | None = None,
) -> torch.Tensor:
    kernel = _optional_kernel("sageattention", False)
    if kernel is None:
        raise RuntimeError("sageattention is unavailable")
    output = kernel(q.contiguous(), k.contiguous(), v.contiguous(),
                    tensor_layout="NHD", is_causal=False, sm_scale=sm_scale)
    return output.to(v.dtype).contiguous()


@_sageattn_nhd.register_fake
def _sageattn_nhd_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float | None = None,
) -> torch.Tensor:
    return torch.empty_like(v, memory_format=torch.contiguous_format)


@torch.library.custom_op("h3_infer::sageattn_varlen", mutates_args=())
def _sageattn_varlen(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 0, max_seqlen_k: int = 0, sm_scale: float | None = None,
) -> torch.Tensor:
    kernel = _optional_kernel("sageattention", True)
    if kernel is None:
        raise RuntimeError("sageattention varlen kernel is unavailable")
    seqlen_q = max_seqlen_q or int(cu_seqlens_q.diff().max())
    seqlen_k = max_seqlen_k or int(cu_seqlens_k.diff().max())
    output = kernel(
        q.contiguous(), k.contiguous(), v.contiguous(),
        cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous(),
        seqlen_q, seqlen_k, is_causal=False, sm_scale=sm_scale,
    )
    return output.to(v.dtype).contiguous()


@_sageattn_varlen.register_fake
def _sageattn_varlen_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 0, max_seqlen_k: int = 0, sm_scale: float | None = None,
) -> torch.Tensor:
    return torch.empty_like(v, memory_format=torch.contiguous_format)


def _cuda_kernel(backend, major, dtype, head_dim, packed):
    if major < 8 or dtype not in (torch.float16, torch.bfloat16):
        return "sdpa", False
    choices = ("sageattention", "flash_attn") if backend == "auto" else (backend,)
    for name in choices:
        if name == "flash_attn" and (major not in (8, 9) or head_dim > 256 or head_dim % 8):
            continue
        if name == "sageattention" and head_dim not in (64, 128):
            continue
        if _has_kernel(name, packed):
            return name, packed
        if name == "sageattention" and packed and _has_kernel(name, False):
            return name, False
    return "sdpa", False


def recommended_attention_backend(major: int) -> str:
    """Prefer SageAttention, then FlashAttention or SDPA for packed inference."""
    return _cuda_kernel("auto", major, torch.bfloat16, 128, packed=True)[0]


def _sdpa(q, k, v, scale):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale,
    )
    return out.transpose(1, 2)


def _packed_attention(kernel, q, k, v, lengths, scale):
    # A single sequence is already contiguous in row order.
    if lengths.numel() == 2:
        return kernel(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale).squeeze(0)
    bounds = lengths.tolist()
    output = torch.empty_like(v)
    for start, stop in zip(bounds, bounds[1:]):
        if start < stop:
            output[start:stop] = kernel(
                q[None, start:stop], k[None, start:stop], v[None, start:stop], scale,
            )[0]
    return output


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    softmax_scale: float | None = None,
    backend: AttentionBackend = "auto",
) -> torch.Tensor:
    """Attend over [B, S, heads, dim] or [total, heads, dim] with packed boundaries.

    Packed Q, K, and V share the same sequence boundaries. Provide cumulative
    lengths starting at zero and ending at total. SDPA runs on every device.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown attention backend {backend!r}")
    packed = q.ndim == 3
    if q.ndim not in (3, 4) or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Expected matching rank-3 packed or rank-4 dense Q, K, V")
    if packed and (cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2):
        raise ValueError("Packed attention requires one-dimensional cu_seqlens")
    if not packed and (cu_seqlens is not None or max_seqlen is not None):
        raise ValueError("Dense attention does not take packed sequence lengths")
    if q.numel() == 0:
        return torch.empty_like(v)

    major = torch.cuda.get_device_capability(q.device)[0] if q.is_cuda else 0
    output_dtype = q.dtype
    if major and major < 8 and output_dtype == torch.bfloat16:
        limit = torch.finfo(torch.float16).max
        # Clamp in fp32 so 65504 is not rounded up to 65536 in bf16.
        q, k, v = (x.float().clamp(-limit, limit).half() for x in (q, k, v))

    name, use_varlen = ("sdpa", False) if backend == "sdpa" else _cuda_kernel(
        backend, major, q.dtype, q.shape[-1], packed,
    )
    if name == "sageattention" and not use_varlen:
        if packed:
            output = _packed_attention(_sageattn_nhd, q, k, v, cu_seqlens, softmax_scale)
        else:
            output = _sageattn_nhd(q, k, v, softmax_scale)
    elif name == "sageattention":
        offsets = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        seq_len = 0 if max_seqlen is None else max_seqlen
        output = _sageattn_varlen(q, k, v, offsets, offsets, seq_len, seq_len, softmax_scale)
    elif name != "sdpa":
        kernel = _optional_kernel(name, use_varlen)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        kwargs = {"sm_scale" if name == "sageattention" else "softmax_scale": softmax_scale,
                  "is_causal" if name == "sageattention" else "causal": False}
        if use_varlen:
            offsets = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
            max_len = max_seqlen if max_seqlen is not None else int(offsets.diff().max())
            output = kernel(q, k, v, offsets, offsets, max_len, max_len, **kwargs)
        else:
            output = kernel(q, k, v, **kwargs)
    elif packed:
        output = _packed_attention(_sdpa, q, k, v, cu_seqlens, softmax_scale)
    else:
        output = _sdpa(q, k, v, softmax_scale)
    return output.to(output_dtype)


def apply_rotary(x, cos, sin):
    """Rotate the leading head channels using broadcastable sine and cosine."""
    size = cos.shape[-1]
    rotating, passthrough = x[..., :size], x[..., size:]
    first, second = rotating.chunk(2, dim=-1)
    swapped = torch.cat((-second, first), dim=-1)
    rotated = (rotating * cos.to(x.dtype)) + (swapped * sin.to(x.dtype))
    return torch.cat((rotated, passthrough), dim=-1)
