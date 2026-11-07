# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""2:4 sparse FP8 linear kernel for SM120.

Weights must hold at most two nonzero values in each group of four inputs. Build
the kernel with `tools/build_sparse_fp8_sm120.sh`, then select it with
`--linear-backend cutlass_sparse`.
"""

import ctypes
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

_LIB_PATH = Path(__file__).resolve().parents[4] / "_sparse_fp8_sm120.so"
_lib: ctypes.CDLL | None = None

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


def compress_24(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress an `[M, K]` FP8 weight into `[M, K/2]` values and metadata.

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
    codes = table[(1 << kept[:, :, 0]) | (1 << kept[:, :, 1])]

    aligned_m = (M + 127) // 128 * 128
    rows = torch.arange(M, device=device)[:, None]
    starts = torch.arange(K // 4, device=device)[None, :] * 4
    atoms = (starts // 128) * (aligned_m // 128) + rows // 128
    offsets = atoms * 2048 + (rows % 128) * 16 + (starts % 128) // 8
    shifts = ((starts % 128) // 4 % 2) * 4
    metadata = torch.zeros(aligned_m * K // 8, dtype=torch.int64, device=device)
    metadata.index_add_(0, offsets.reshape(-1), (codes << shifts).reshape(-1))
    return values, metadata.to(torch.uint8)


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
