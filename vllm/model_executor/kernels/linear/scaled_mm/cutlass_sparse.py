# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""2:4 sparse linear kernels: FP8 on SM120 and INT8 on SM86.

Weights must hold at most two nonzero values in each group of four inputs. Build
the kernel with `tools/build_sparse_fp8_sm120.sh` or
`tools/build_sparse_int8_sm80.sh`, then select it with
`--linear-backend cutlass_sparse`.
"""

import ctypes
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

_LIB_PATH = Path(__file__).resolve().parents[4] / "_sparse_fp8_sm120.so"
_INT8_LIB_PATH = Path(__file__).resolve().parents[4] / "_sparse_int8_sm80.so"
_lib: ctypes.CDLL | None = None
_int8_lib: ctypes.CDLL | None = None

# Hardware metadata code for each 4-bit mask of kept positions in a group.
_ORDERED_METADATA = {
    0b0011: 0x4,
    0b0101: 0x8,
    0b1001: 0xC,
    0b0110: 0x9,
    0b1010: 0xD,
    0b1100: 0xE,
}


def _sparse_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(_LIB_PATH))
        _lib.vllm_sparse_fp8_mm_sm120.restype = ctypes.c_int
        _lib.vllm_sparse_fp8_mm_sm120.argtypes = [ctypes.c_size_t] * 7 + [
            ctypes.c_int
        ] * 4
    return _lib


def _sparse_int8_lib() -> ctypes.CDLL:
    global _int8_lib
    if _int8_lib is None:
        _int8_lib = ctypes.CDLL(str(_INT8_LIB_PATH))
        _int8_lib.vllm_sparse_int8_mm_sm80.restype = ctypes.c_int
        _int8_lib.vllm_sparse_int8_mm_sm80.argtypes = [ctypes.c_size_t] * 7 + [
            ctypes.c_int
        ] * 4
    return _int8_lib


def _select_24(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the `[M, K/2]` kept values and the `[M, K/4]` metadata codes.

    Raises:
        ValueError: If a group of four holds more than two nonzero values.
    """
    M, K = weight.shape
    device = weight.device
    nonzero = (weight != 0).reshape(M, K // 4, 4)
    if nonzero.sum(dim=2).max() > 2:
        raise ValueError("cutlass_sparse requires a 2:4 sparse weight.")
    rank = nonzero.int() * 4 - torch.arange(4, device=device)
    kept = rank.topk(2, dim=2).indices.sort(dim=2).values

    groups = torch.arange(K // 4, device=device)[None, :, None] * 4
    values = weight.gather(1, (kept + groups).reshape(M, K // 2))

    table = torch.zeros(16, dtype=torch.int32, device=device)
    for mask, code in _ORDERED_METADATA.items():
        table[mask] = code
    return values, table[(1 << kept[:, :, 0]) | (1 << kept[:, :, 1])]


def compress_24(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress an `[M, K]` FP8 weight into `[M, K/2]` values and metadata."""
    M, K = weight.shape
    device = weight.device
    values, codes = _select_24(weight)

    aligned_m = (M + 127) // 128 * 128
    rows = torch.arange(M, device=device)[:, None]
    starts = torch.arange(K // 4, device=device)[None, :] * 4
    atoms = (starts // 128) * (aligned_m // 128) + rows // 128
    offsets = atoms * 2048 + (rows % 128) * 16 + (starts % 128) // 8
    shifts = ((starts % 128) // 4 % 2) * 4
    metadata = torch.zeros(aligned_m * K // 8, dtype=torch.int64, device=device)
    metadata.index_add_(0, offsets.reshape(-1), (codes << shifts).reshape(-1))
    return values, metadata.to(torch.uint8)


def compress_24_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress an `[M, K]` s8 weight into `[M, K/2]` values and SM80 metadata.

    Each uint32 holds the 4-bit codes of eight groups, lowest first. Words are
    laid out as CUTLASS's `ColumnMajorInterleaved<2>`, with the rows of every 16
    reordered: within a block of 32 words, rows 0-7 take the even pairs and rows
    8-15 the odd pairs. CUTLASS only mentions this reordering; the mapping was
    measured on an RTX 3090.
    """
    M, K = weight.shape
    device = weight.device
    values, codes = _select_24(weight)
    groups = torch.arange(K // 4, device=device)
    words = groups // 8
    rows = torch.arange(M, device=device)[:, None]
    row_offsets = (rows // 16) * 32 + (rows % 8) * 4 + (rows % 16) // 8
    offsets = (words // 2)[None, :] * (2 * M) + row_offsets + (words % 2)[None, :] * 2
    shifts = (groups % 8)[None, :] * 4
    metadata = torch.zeros(M * K // 32, dtype=torch.int64, device=device)
    metadata.index_add_(0, offsets.reshape(-1), (codes.long() << shifts).reshape(-1))
    return values, metadata.to(torch.uint32)


def _sparse_int8_mm_func(
    x: torch.Tensor,
    weight: torch.Tensor,
    metadata: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    assert x.is_contiguous() and x_scale.is_contiguous()
    N, K = x.shape
    M = weight.shape[0]
    # The kernel faults unless N is aligned to the 16-byte accesses of operand B.
    padding = -N % 16
    if padding:
        x = torch.nn.functional.pad(x, (0, 0, 0, padding))
        if x_scale.numel() != 1:
            x_scale = torch.nn.functional.pad(x_scale.reshape(-1), (0, padding))
    out = torch.empty((M, N + padding), dtype=torch.bfloat16, device=x.device)
    status = _sparse_int8_lib().vllm_sparse_int8_mm_sm80(
        torch.cuda.current_stream().cuda_stream,
        out.data_ptr(),
        weight.data_ptr(),
        metadata.data_ptr(),
        x.data_ptr(),
        weight_scale.data_ptr(),
        x_scale.data_ptr(),
        int(x_scale.numel() != 1),
        M,
        N + padding,
        K,
    )
    if status != 0:
        raise RuntimeError(f"vllm_sparse_int8_mm_sm80 failed with status {status}.")
    return out[:, :N].t().contiguous()


def _sparse_int8_mm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    metadata: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.size(0), weight.size(0)), dtype=torch.bfloat16, device=x.device
    )


direct_register_custom_op(
    "sparse_int8_mm_func", _sparse_int8_mm_func, fake_impl=_sparse_int8_mm_fake
)


def _sparse_fp8_mm_func(
    x: torch.Tensor,
    weight: torch.Tensor,
    metadata: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    assert x.is_contiguous() and x_scale.is_contiguous()
    N, K = x.shape
    M = weight.shape[0]
    out = torch.empty((N, M), dtype=torch.bfloat16, device=x.device)
    status = _sparse_lib().vllm_sparse_fp8_mm_sm120(
        torch.cuda.current_stream().cuda_stream,
        out.data_ptr(),
        weight.data_ptr(),
        metadata.data_ptr(),
        x.data_ptr(),
        weight_scale.data_ptr(),
        x_scale.data_ptr(),
        int(x_scale.numel() != 1),
        M,
        N,
        K,
    )
    if status != 0:
        raise RuntimeError(f"vllm_sparse_fp8_mm_sm120 failed with status {status}.")
    return out


def _sparse_fp8_mm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    metadata: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.size(0), weight.size(0)), dtype=torch.bfloat16, device=x.device
    )


direct_register_custom_op(
    "sparse_fp8_mm_func", _sparse_fp8_mm_func, fake_impl=_sparse_fp8_mm_fake
)


class CutlassSparseFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_device_capability(120):
            return False, "requires SM120."
        if not _LIB_PATH.exists():
            return False, "run tools/build_sparse_fp8_sm120.sh first."
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        N, K = c.weight_shape
        if K % 128 != 0 or N < 128:
            return False, f"requires K divisible by 128 and N >= 128, got {N=}, {K=}."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_name, weight_scale_name, _, _ = self.layer_param_names
        # Scheme processing leaves the weight as [K, N].
        values, self.metadata = compress_24(
            getattr(layer, weight_name).t().contiguous()
        )
        replace_parameter(layer, weight_name, values)
        replace_parameter(
            layer,
            weight_scale_name,
            getattr(layer, weight_scale_name).reshape(-1).contiguous(),
        )

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        assert bias is None and out_dtype == torch.bfloat16
        output = torch.ops.vllm.sparse_fp8_mm_func(
            A.contiguous(), B, self.metadata, As.contiguous(), Bs
        )
        return output.view(*output_shape[:-1], B.shape[0])


class CutlassSparseInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_device_capability(86):
            return False, "requires SM86."
        if not _INT8_LIB_PATH.exists():
            return False, "run tools/build_sparse_int8_sm80.sh first."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.is_static_input_scheme or not c.input_symmetric:
            return False, "requires dynamic symmetric activations."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, *_ = self.layer_param_names
        weight = getattr(layer, w_q_name)
        M, K = weight.shape
        if M % 16 or K % 128:
            raise ValueError(
                f"cutlass_sparse requires M divisible by 16 and K by 128, got {M=}, "
                f"{K=}."
            )
        values, self.metadata = compress_24_int8(weight.data)
        replace_parameter(layer, w_q_name, values)

        weight_scale = getattr(layer, w_s_name)
        if not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(layer, w_s_name, weight_scale.reshape(-1).contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None and x.dtype == torch.bfloat16
        w_q, w_s, i_s, i_zp, _ = self._get_layer_params(layer)
        x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), i_s, i_zp, symmetric=True)
        return torch.ops.vllm.sparse_int8_mm_func(
            x_q, w_q, self.metadata, x_s.contiguous(), w_s
        )
