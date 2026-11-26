"""INT8 linear projections with optional grouped Hadamard rotation."""

from functools import cache
from math import prod
from threading import Lock

import torch
from torch import nn
from torch.nn import functional as F

_hadamard_cache = {}
_hadamard_lock = Lock()
_triton_matmul_lock = Lock()
_triton_float_dtypes = (torch.float16, torch.bfloat16, torch.float32)


def _validate_hadamard_size(size: int) -> None:
    if size < 4 or (size & (size - 1)) or (size.bit_length() - 1) % 2:
        raise ValueError("Hadamard size must be a power of 4 and at least 4")


def build_regular_hadamard(size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    _validate_hadamard_size(size)
    h4 = torch.tensor(
        [[1, 1, 1, -1],
         [1, 1, -1, 1],
         [1, -1, 1, 1],
         [-1, 1, 1, 1]],
        dtype=dtype, device=device,
    )
    h = h4
    for _ in range((size.bit_length() - 1) // 2 - 1):
        h = torch.kron(h, h4)
    return h / size ** 0.5


def _get_cached_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    if stream is not None:
        device = stream.device
    key = (size, str(device), dtype)
    with _hadamard_lock:
        entry = _hadamard_cache.get(key)
        if entry is None:
            with torch.inference_mode(False):
                h = build_regular_hadamard(size, device, dtype)
            ready = stream.record_event() if stream is not None else None
            _hadamard_cache[key] = (h, stream, ready)
        else:
            h, build_stream, ready = entry
            if ready is not None and stream != build_stream:
                stream.wait_event(ready)
        return h


def rotate_activation(x: torch.Tensor, group_size: int = 256,
                      hadamard: torch.Tensor | None = None) -> torch.Tensor:
    features = x.shape[-1]
    if group_size <= 0 or features % group_size:
        raise ValueError("Activation features must be divisible by the ConvRot group size")
    h = build_regular_hadamard(group_size, x.device, x.dtype) if hadamard is None else hadamard
    grouped = x.reshape(-1, features // group_size, group_size)
    return torch.matmul(grouped, h.to(device=x.device, dtype=x.dtype)).reshape(x.shape)


def rotate_weight(weight: torch.Tensor, group_size: int = 256,
                  hadamard: torch.Tensor | None = None) -> torch.Tensor:
    out_features, in_features = weight.shape
    if group_size <= 0 or in_features % group_size:
        raise ValueError("Weight input features must be divisible by the ConvRot group size")
    h = build_regular_hadamard(group_size, weight.device, weight.dtype) if hadamard is None else hadamard
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    return torch.matmul(grouped, h.T.to(device=weight.device, dtype=weight.dtype)).reshape(weight.shape)


def quantize_int8_rowwise(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_scale = x_2d.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) / 127.0
    x_i8 = (x_2d.float() / x_scale).round().clamp(-127, 127).to(torch.int8)
    return x_i8, x_scale


@cache
def _get_triton_kernels():
    # Called only from the CUDA path so Triton remains an optional dependency.
    global tl  # Triton resolves language symbols through the JIT function's globals.
    try:
        import triton
        import triton.language as tl
    except (ImportError, OSError):
        return None

    @triton.jit
    def _quantize_rowwise_kernel(x_ptr, y_ptr, s_ptr, n_cols, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        max_abs = tl.full((BLOCK_K,), 0.0, tl.float32)
        for start in tl.range(0, n_cols, BLOCK_K):
            cols = start + offsets
            values = tl.load(x_ptr + row * n_cols + cols, cols < n_cols, other=0).to(tl.float32)
            max_abs = tl.maximum(max_abs, tl.abs(values))
        scale = tl.maximum(tl.max(max_abs, axis=0) / 127.0, 1e-12)

        for start in tl.range(0, n_cols, BLOCK_K):
            cols = start + offsets
            values = tl.load(x_ptr + row * n_cols + cols, cols < n_cols, other=0).to(tl.float32)
            scaled = tl.div_rn(values, scale)
            lower = tl.floor(scaled)
            fraction = scaled - lower
            # Match torch.round for positive and negative half-integers.
            round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
            quantized = tl.clamp(tl.where(round_up, lower + 1.0, lower), -127.0, 127.0)
            tl.store(y_ptr + row * n_cols + cols, quantized.to(tl.int8), cols < n_cols)
        tl.store(s_ptr + row, scale)

    @triton.autotune(configs=[
        triton.Config({"block_m": 128, "block_n": 256, "block_k": 64, "group_size_m": 8},
                      num_stages=3, num_warps=8),
        triton.Config({"block_m": 128, "block_n": 128, "block_k": 64, "group_size_m": 8},
                      num_stages=4, num_warps=4),
        triton.Config({"block_m": 64, "block_n": 128, "block_k": 64, "group_size_m": 8},
                      num_stages=4, num_warps=4),
        triton.Config({"block_m": 64, "block_n": 64, "block_k": 64, "group_size_m": 8},
                      num_stages=4, num_warps=4),
    ], key=["n", "k"])
    @triton.jit(do_not_specialize=["m"])
    def _int8_matmul_dequant_kernel(
        a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, bias_ptr, c_ptr, m, n, k, stride_bk, stride_bn,
        has_bias: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr,
        block_k: tl.constexpr, group_size_m: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(m, block_m)
        num_pid_n = tl.cdiv(n, block_n)
        pids_per_group = group_size_m * num_pid_n
        first_pid_m = (pid // pids_per_group) * group_size_m
        group_m = tl.minimum(num_pid_m - first_pid_m, group_size_m)
        pid_in_group = pid % pids_per_group
        pid_m = first_pid_m + pid_in_group % group_m
        pid_n = pid_in_group // group_m

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        acc = tl.full((block_m, block_n), 0, tl.int32)
        for k_start in tl.range(0, k, block_k):
            cols_k = k_start + offs_k
            a = tl.load(a_ptr + offs_m[:, None] * k + cols_k[None, :],
                        (offs_m[:, None] < m) & (offs_k[None, :] < k - k_start), other=0)
            b = tl.load(b_ptr + cols_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        (offs_k[:, None] < k - k_start) & (offs_n[None, :] < n), other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)

        scale_a = tl.load(a_scale_ptr + offs_m, offs_m < m, other=0.0)
        scale_b = tl.load(b_scale_ptr + offs_n, offs_n < n, other=0.0).to(tl.float32)
        c = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
        if has_bias:
            bias = tl.load(bias_ptr + offs_n, offs_n < n, other=0.0).to(tl.float32)
            c = c + bias[None, :]
        tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], c.to(c_ptr.dtype.element_ty),
                 (offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def _dequant_int32_kernel(
        c_ptr, a_scale_ptr, b_scale_ptr, bias_ptr, out_ptr, n, stride_cm,
        has_bias: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < n
        scale_a = tl.load(a_scale_ptr + row).to(tl.float32)
        c_i32 = tl.load(c_ptr + row * stride_cm + cols, mask=mask, other=0)
        scale_b = tl.load(b_scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        c = c_i32.to(tl.float32) * scale_a * scale_b
        if has_bias:
            c = c + tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * n + cols, c.to(out_ptr.dtype.element_ty), mask=mask)

    return triton, _quantize_rowwise_kernel, _int8_matmul_dequant_kernel, _dequant_int32_kernel


def _triton_quantize_rowwise(x_2d):
    _, kernel, _, _ = _get_triton_kernels()
    m, k = x_2d.shape
    quantized = torch.empty_like(x_2d, dtype=torch.int8)
    scale = x_2d.new_empty((m,), dtype=torch.float32)
    kernel[(m,)](x_2d, quantized, scale, k, BLOCK_K=1024)
    return quantized, scale


def _triton_int8_matmul_dequant(a, b, a_scale, b_scale, bias, *, out_dtype):
    triton, _, kernel, _ = _get_triton_kernels()
    m, k = a.shape
    n = b.shape[0]
    out = a.new_empty((m, n), dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),)

    # The autotuner shares mutable launch state across calls.
    with _triton_matmul_lock:
        kernel[grid](a, b, a_scale, b_scale, a_scale if bias is None else bias, out,
                     m, n, k, b.stride(1), b.stride(0), has_bias=bias is not None,
                     enable_fp_fusion=False)
    return out


def _triton_dequant_int32(out_i32, a_scale, b_scale, bias, *, m, out_dtype):
    triton, _, _, kernel = _get_triton_kernels()
    n = out_i32.shape[1]
    out = out_i32.new_empty((m, n), dtype=out_dtype)
    kernel[(m, triton.cdiv(n, 1024))](
        out_i32, a_scale, b_scale, a_scale if bias is None else bias, out,
        n, out_i32.stride(0), has_bias=bias is not None, BLOCK_N=1024, num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def _cublas_int8_matmul_triton_dequant(a, b, a_scale, b_scale, bias, *, out_dtype):
    m = a.shape[0]
    # torch._int_mm requires aligned dimensions and more than 16 input rows.
    padded_m = max(24, ((m + 7) // 8) * 8)
    a_mm = F.pad(a, (0, 0, 0, padded_m - m)) if padded_m != m else a
    out_i32 = torch._int_mm(a_mm, b.t())
    return _triton_dequant_int32(out_i32, a_scale, b_scale, bias, m=m, out_dtype=out_dtype)


def _validate_int8_convrot_inputs(x, weight, weight_scale, bias, convrot, convrot_groupsize):
    if not x.ndim or not x.is_floating_point():
        raise ValueError("Expected a floating-point input with a feature dimension")
    if weight.ndim != 2 or weight.dtype != torch.int8 or weight.shape[1] != x.shape[-1] or not weight.shape[1]:
        raise ValueError("Expected int8 weight [out_features, in_features] matching the input")
    n, k = weight.shape
    if not weight_scale.is_floating_point() or weight_scale.numel() not in (1, n):
        raise ValueError("Expected floating-point weight scales (one per output feature or one shared scale)")
    if bias is not None and (
        not bias.is_floating_point() or bias.ndim > 2
        or (bias.ndim and bias.shape[-1] not in (1, n))
        or (bias.ndim == 2 and bias.shape[0] not in (1, prod(x.shape[:-1])))
    ):
        raise ValueError("Expected floating-point bias broadcastable to the flattened output")
    if weight.device != x.device or weight_scale.device != x.device or (bias is not None and bias.device != x.device):
        raise ValueError("Expected input, weight, scales, and bias on the same device")
    if convrot and (convrot_groupsize <= 0 or k % convrot_groupsize):
        raise ValueError("Activation features must be divisible by the ConvRot group size")


def _int8_convrot_linear_eager(x, weight, weight_scale, bias, convrot, convrot_groupsize, hadamard):
    n, k = weight.shape
    out_dtype = x.dtype
    if convrot:
        x = rotate_activation(x, convrot_groupsize, hadamard)
    x_i8, x_scale = quantize_int8_rowwise(x.reshape(-1, k))
    m = x_i8.shape[0]
    if x.is_cuda and k % 8 == 0 and n % 8 == 0:
        # The kernel requires aligned dimensions and more than 16 input rows.
        padded_m = max(24, ((m + 7) // 8) * 8)
        if padded_m != m:
            x_i8 = F.pad(x_i8, (0, 0, 0, padded_m - m))
        out_i32 = torch._int_mm(x_i8.contiguous(), weight.contiguous().t())[:m]
    else:
        # CUDA addmm lacks an int32 kernel for the unaligned fallback.
        accumulation_device = "cpu" if x.is_cuda else x.device
        lhs = x_i8.to(device=accumulation_device, dtype=torch.int32)
        rhs = weight.to(device=accumulation_device, dtype=torch.int32).t()
        out_i32 = torch.matmul(lhs, rhs).to(x.device)

    scale = weight_scale.reshape(1, -1)

    def dequantize(values, row_scale, bias):
        result = values.float() * row_scale * scale
        if bias is not None:
            result = result + bias
        return result.to(out_dtype)

    rows_per_chunk = max(1, (64 * 1024**2) // max(1, n))
    if m <= rows_per_chunk:
        out = dequantize(out_i32, x_scale, bias)
    else:
        out = x.new_empty((m, n), dtype=out_dtype)
        for start in range(0, m, rows_per_chunk):
            stop = start + rows_per_chunk
            row_bias = bias[start:stop] if bias is not None and bias.ndim == 2 and bias.shape[0] == m else bias
            out[start:stop] = dequantize(out_i32[start:stop], x_scale[start:stop], row_bias)
    return out.reshape(*x.shape[:-1], n)


@torch.library.custom_op("h3_infer::int8_convrot_linear", mutates_args=())
def _int8_convrot_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None,
    convrot: bool, convrot_groupsize: int,
) -> torch.Tensor:
    _validate_int8_convrot_inputs(x, weight, weight_scale, bias, convrot, convrot_groupsize)
    n, k = weight.shape
    h = _get_cached_hadamard(convrot_groupsize, x.device, x.dtype) if convrot else None
    if not x.numel() or not n:
        return x.new_empty((*x.shape[:-1], n))

    if (x.is_cuda and x.dtype in _triton_float_dtypes and weight_scale.dtype in _triton_float_dtypes
            and (bias is None or (bias.dtype in _triton_float_dtypes
                                 and (bias.ndim < 2 or bias.shape[0] == 1)))
            and _get_triton_kernels() is not None):
        out_shape = (*x.shape[:-1], n)
        out_dtype = x.dtype
        with torch.cuda.device(x.device):
            if convrot:
                x = rotate_activation(x, convrot_groupsize, h)
            x_2d = x.reshape(-1, k).contiguous()
            x_i8, x_scale = _triton_quantize_rowwise(x_2d)
            b_scale = weight_scale.reshape(-1).expand(n).contiguous()
            if bias is not None:
                bias = bias.reshape(-1).expand(n).contiguous()
            if k % 8 == 0 and n % 8 == 0:
                out = _cublas_int8_matmul_triton_dequant(
                    x_i8, weight.contiguous(), x_scale, b_scale, bias, out_dtype=out_dtype,
                )
            else:
                out = _triton_int8_matmul_dequant(
                    x_i8, weight.contiguous(), x_scale, b_scale, bias, out_dtype=out_dtype,
                )
        return out.reshape(out_shape)
    return _int8_convrot_linear_eager(x, weight, weight_scale, bias, convrot, convrot_groupsize, h)


@_int8_convrot_linear.register_fake
def _int8_convrot_linear_fake(x, weight, weight_scale, bias, convrot, convrot_groupsize):
    _validate_int8_convrot_inputs(x, weight, weight_scale, bias, convrot, convrot_groupsize)
    if convrot:
        _validate_hadamard_size(convrot_groupsize)
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def int8_convrot_linear(x, weight, weight_scale, bias=None, *,
                       convrot=True, convrot_groupsize=256, hadamard=None):
    if hadamard is None:
        return torch.ops.h3_infer.int8_convrot_linear(x, weight, weight_scale, bias, convrot, convrot_groupsize)
    _validate_int8_convrot_inputs(x, weight, weight_scale, bias, convrot, convrot_groupsize)
    return _int8_convrot_linear_eager(x, weight, weight_scale, bias, convrot, convrot_groupsize, hadamard)


class Int8ConvRotLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, convrot: bool = True,
                 convrot_groupsize: int = 256, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty((out_features, 1), dtype=torch.float32), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32), requires_grad=False) if bias else None

    def forward(self, x):
        return torch.ops.h3_infer.int8_convrot_linear(
            x, self.weight, self.weight_scale, self.bias, self.convrot, self.convrot_groupsize,
        )
