# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVTX ranges labelled with the phase of each step, for VLLM_NVTX_ENABLE=1.

Emits `Step/<phase>` around each model forward and, in eager mode only,
`<phase>/Layer_NN/Attention` and `<phase>/Layer_NN/FFN` around the decoder layers,
where phase is Prefill, Decode or Mixed. With the variable unset nothing is
installed, so the forward path is unchanged.

Layer ranges need Python to run inside the model: Dynamo rejects NVTX calls under
the `fullgraph=True` that vLLM compiles with, and CUDA graph replay skips Python.
They are added by wrapping `forward` on eager models, not by editing model code.
"""

import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import wraps
from typing import TYPE_CHECKING, Any

import numpy as np
import torch.cuda.nvtx as nvtx
from torch import nn

import vllm.envs as envs
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import BatchReqState

logger = init_logger(__name__)

NVTX_ENABLED: bool = envs.VLLM_NVTX_ENABLE

_PHASES = ("Prefill", "Decode", "Mixed")
_RANGE_NAMES = {"self_attn": "Attention", "mlp": "FFN"}
_DECODER_LAYER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)$")
_NO_RANGE = nullcontext()


def classify_phase(is_prefilling: np.ndarray) -> str:
    """Name a step's phase from the prefill state of each scheduled request.

    A layer runs once over all tokens of the batch, so a batch with both kinds of
    request cannot be attributed to either and is reported as Mixed.
    """
    if is_prefilling.all():
        return "Prefill"
    return "Mixed" if is_prefilling.any() else "Decode"


def no_step_range(batch_req_state: "BatchReqState | None") -> AbstractContextManager:
    return _NO_RANGE


class _PhaseRanges:
    """Holds the running step's phase for the wrapped layers of one model.

    Only the thread running execute_model writes it. Layers read it from that
    thread or from the microbatch threads, which finish before the step does.
    It is None outside a real step, so dummy runs emit no ranges.
    """

    def __init__(self) -> None:
        self.phase: str | None = None
        self.step_labels = {phase: f"Step/{phase}" for phase in _PHASES}

    @contextmanager
    def step(self, batch_req_state: "BatchReqState | None") -> Iterator[None]:
        if batch_req_state is None:
            yield
            return
        self.phase = classify_phase(batch_req_state.is_prefilling_np)
        nvtx.range_push(self.step_labels[self.phase])
        try:
            yield
        finally:
            nvtx.range_pop()
            self.phase = None

    def wrap_forward(self, module: nn.Module, label_suffix: str) -> None:
        labels = {phase: f"{phase}/{label_suffix}" for phase in _PHASES}
        forward = module.forward

        @wraps(forward)
        def forward_in_range(*args: Any, **kwargs: Any) -> Any:
            phase = self.phase
            if phase is None:
                return forward(*args, **kwargs)
            nvtx.range_push(labels[phase])
            try:
                return forward(*args, **kwargs)
            finally:
                nvtx.range_pop()

        module.forward = forward_in_range  # type: ignore[method-assign]


def install_nvtx_ranges(
    model: nn.Module, compilation_config: CompilationConfig
) -> Callable[["BatchReqState | None"], AbstractContextManager]:
    """Add layer ranges to an eager model. Call it once warmup is over.

    Returns:
        The factory of the context manager to run each step's forward in.
    """
    if not NVTX_ENABLED:
        return no_step_range

    ranges = _PhaseRanges()
    if (
        compilation_config.mode != CompilationMode.NONE
        or compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    ):
        logger.warning(
            "VLLM_NVTX_ENABLE=1: per-layer NVTX ranges need eager execution, so "
            "only Step/* ranges will be emitted. Use --enforce-eager for "
            "per-layer ranges."
        )
        return ranges.step

    num_wrapped = 0
    for name, layer in model.named_modules():
        match = _DECODER_LAYER_NAME.search(name)
        if match is None:
            continue
        for attr, range_name in _RANGE_NAMES.items():
            submodule = getattr(layer, attr, None)
            if isinstance(submodule, nn.Module):
                ranges.wrap_forward(
                    submodule, f"Layer_{int(match[1]):02d}/{range_name}"
                )
                num_wrapped += 1

    if num_wrapped:
        logger.info("VLLM_NVTX_ENABLE=1: added NVTX ranges to %d modules", num_wrapped)
    else:
        logger.warning(
            "VLLM_NVTX_ENABLE=1: no `layers.N.self_attn` or `layers.N.mlp` modules "
            "found in %s, so only Step/* ranges will be emitted.",
            type(model).__name__,
        )
    return ranges.step
