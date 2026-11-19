# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer activation dtype for the interleaved AITER MXFP4 MoE kernel.

`aiter.fused_moe` picks BF16 vs FP8 activations from the process-wide
`AITER_BF16_FP8_MOE_BOUND` unless the caller passes `quant_dtype_a`, which
overrides it. These tests run on real GFX950 hardware and cover what the env
var cannot express: two MoE layers in one process honouring different
`quantization_config` activation requests.
"""

import os

import pytest
import torch

from tests.quantization.utils import quant_config_args
from vllm.config.kernel import KernelConfig, MoEBackend
from vllm.config.model import ModelConfig
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    resolve_aiter_mxfp4_act_dtype,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform

ROCM_GFX950 = False
ROCM_AITER_SUPPORTED = False
AITER_HAS_QUANT_DTYPE_A = False

if current_platform.is_rocm():
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_SUPPORTED = is_aiter_found_and_supported()
    if ROCM_AITER_SUPPORTED:
        AITER_HAS_QUANT_DTYPE_A = rocm_aiter_ops.fused_moe_supports_quant_dtype_a()

requires_gfx950_aiter = pytest.mark.skipif(
    not (ROCM_GFX950 and ROCM_AITER_SUPPORTED),
    reason="Requires GFX950 (mi355x) with a supported AITER",
)
requires_quant_dtype_a = pytest.mark.skipif(
    not AITER_HAS_QUANT_DTYPE_A,
    reason="Requires an AITER build with fused_moe(quant_dtype_a=...)",
)

E, TOPK, H, INT = 8, 2, 1024, 1024
# Far above any token count these tests run, so AITER's own heuristic picks
# BF16 whenever the layer leaves `quant_dtype_a` unset.
BF16_BOUND = "2147483647"


def _make_moe_config(moe_backend: MoEBackend = "aiter") -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=E,
        experts_per_token=TOPK,
        hidden_dim=H,
        intermediate_size=INT,
        num_local_experts=E,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


@pytest.fixture
def clean_aiter_bound(monkeypatch):
    """Start from an unset bound so a leftover value cannot explain a result."""
    monkeypatch.delenv("AITER_BF16_FP8_MOE_BOUND", raising=False)
    yield


# -- The mechanism -----------------------------------------------------------


@requires_gfx950_aiter
@requires_quant_dtype_a
def test_quant_dtype_a_overrides_the_env_bound(
    default_vllm_config, dist_init, workspace_init, monkeypatch, clean_aiter_bound
):
    """`quant_dtype_a` wins over `AITER_BF16_FP8_MOE_BOUND`, so the activation
    dtype does not have to be a process-wide setting."""
    from aiter import dtypes

    from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

    layer = _build_layer(default_vllm_config, monkeypatch, activation=None, idx=0)
    x, topk_weights, topk_ids = _make_inputs(M=1)

    forced: dict = {}
    base_kwargs = rocm_aiter_moe.mxfp4_w4a16_extra_kwargs

    def patched(quant_config, moe_config):
        kwargs = base_kwargs(quant_config, moe_config)
        if forced["q_dtype_a"] is None:
            kwargs.pop("q_dtype_a", None)
        else:
            kwargs["q_dtype_a"] = forced["q_dtype_a"]
        return kwargs

    monkeypatch.setattr(rocm_aiter_moe, "mxfp4_w4a16_extra_kwargs", patched)

    def run(bound, q_dtype_a):
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = bound
        forced["q_dtype_a"] = q_dtype_a
        return layer.forward_modular(
            x.clone(), topk_weights.clone(), topk_ids.clone()
        ).clone()

    env_bf16 = run(BF16_BOUND, None)
    env_fp8 = run("0", None)
    override_bf16 = run("0", dtypes.bf16)
    override_fp8 = run(BF16_BOUND, dtypes.fp8)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / a.float().norm()).item()

    # Sanity: the env var is doing something, so the comparisons below mean
    # something. BF16 vs FP8 activations is a few percent on these shapes.
    env_effect = rel(env_bf16, env_fp8)
    assert env_effect > 1e-3, f"env bound had no effect ({env_effect})"

    assert torch.equal(env_fp8, override_fp8)
    # BF16 is not bitwise reproducible across the two paths: the kernel picks a
    # different tile config, which moves the last bits. That is about one BF16
    # ulp (~0.4%), well clear of the several-percent dtype change it must not be
    # confused with, but it is not noise -- keep the bound well under `env_effect`
    # so this still fails if `q_dtype_a=bf16` were ignored and FP8 ran instead.
    assert rel(env_bf16, override_bf16) < 0.25 * env_effect


# -- The capability a process-wide env var cannot provide ---------------------


@requires_gfx950_aiter
@requires_quant_dtype_a
@pytest.mark.parametrize("reverse", [False, True], ids=["in_order", "reversed"])
def test_two_layers_one_process_honour_different_activations(
    default_vllm_config,
    dist_init,
    workspace_init,
    monkeypatch,
    clean_aiter_bound,
    reverse,
):
    """Two MXFP4 MoE layers, one process, different `targets` activations.

    Built in both orders: a per-layer value cannot be clobbered by whichever
    layer loaded last, which is what a process-wide env var cannot promise."""
    from aiter import dtypes

    order = [1, 0] if reverse else [0, 1]
    layers = {}
    for idx in order:
        layers[idx] = _build_layer(
            default_vllm_config,
            monkeypatch,
            activation=("mxfp8" if idx == 1 else None),
            idx=idx,
            targets=True,
        )

    assert layers[0].quant_method.moe.aiter_mxfp4_act_dtype is dtypes.bf16
    assert layers[1].quant_method.moe.aiter_mxfp4_act_dtype is dtypes.fp8

    x, topk_weights, topk_ids = _make_inputs(M=1)
    out = {
        idx: layers[idx]
        .forward_modular(x.clone(), topk_weights.clone(), topk_ids.clone())
        .clone()
        for idx in (0, 1)
    }

    def rel(a, b):
        return ((a.float() - b.float()).norm() / a.float().norm()).item()

    # Same weights, same input: any difference is the activation dtype.
    layer_gap = rel(out[0], out[1])
    assert layer_gap > 1e-3, (
        "both layers computed the same activation dtype; the per-layer request "
        "did not reach the kernel"
    )

    # And each matches what the process-wide bound alone would have produced
    # for that dtype -- far closer to its own reference than the two layers are
    # to each other.
    for idx, bound in ((0, BF16_BOUND), (1, "0")):
        monkeypatch.setattr(layers[idx].quant_method.moe, "aiter_mxfp4_act_dtype", None)
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = bound
        reference = layers[idx].forward_modular(
            x.clone(), topk_weights.clone(), topk_ids.clone()
        )
        assert rel(out[idx], reference) < 0.25 * layer_gap


# -- Deferring to AITER when nobody asked -------------------------------------


@requires_gfx950_aiter
def test_no_request_leaves_the_bound_in_charge(default_vllm_config):
    """A layer whose converter never pinned the activation (gpt-oss) and whose
    activation nobody requested keeps deferring to AITER's own heuristic."""
    config = _make_moe_config()
    assert (
        resolve_aiter_mxfp4_act_dtype(Mxfp4MoeBackend.AITER_MXFP4_BF16, config) is None
    )


@requires_gfx950_aiter
def test_no_request_keeps_the_fp8_pin(default_vllm_config):
    """Layers converted by `convert_weight_to_mxfp4_moe_kernel_format` used to
    have the bound forced to 0; without a request they still run FP8."""
    from aiter import dtypes

    config = _make_moe_config()
    assert (
        resolve_aiter_mxfp4_act_dtype(
            Mxfp4MoeBackend.AITER_MXFP4_BF16, config, pins_fp8=True
        )
        is dtypes.fp8
    )


@requires_gfx950_aiter
def test_other_backends_are_untouched(default_vllm_config):
    config = _make_moe_config()
    for backend in (
        Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.EMULATION,
    ):
        assert resolve_aiter_mxfp4_act_dtype(backend, config, pins_fp8=True) is None, (
            backend
        )


@requires_gfx950_aiter
@pytest.mark.parametrize("layout", ["situ", "dsv4"])
def test_layouts_that_choose_their_own_activation_are_left_alone(
    default_vllm_config, monkeypatch, layout
):
    """SiTUv2 and DeepSeek-V4 a4w4 pass their own `q_dtype_a` and have their own
    tuned configs, so this resolver must not speak for them -- with or without a
    request."""
    config = _make_moe_config()
    if layout == "situ":
        config.activation = MoEActivation.SITU
    else:
        config.use_mxfp4_w4a4_dsv4 = True
    assert (
        resolve_aiter_mxfp4_act_dtype(
            Mxfp4MoeBackend.AITER_MXFP4_BF16, config, pins_fp8=True
        )
        is None
    )
    _request_activation(monkeypatch, kMxfp8Dynamic)
    assert (
        resolve_aiter_mxfp4_act_dtype(
            Mxfp4MoeBackend.AITER_MXFP4_BF16, config, kMxfp8Dynamic, pins_fp8=True
        )
        is None
    )


# -- Loud, not silent ---------------------------------------------------------


@requires_gfx950_aiter
def test_old_aiter_rejects_an_activation_request(default_vllm_config, monkeypatch):
    """An AITER without `quant_dtype_a` cannot honour the request, and says so
    instead of silently running a different activation dtype."""
    monkeypatch.setattr(
        rocm_aiter_ops, "fused_moe_supports_quant_dtype_a", lambda: False
    )
    config = _make_moe_config()
    with pytest.raises(ValueError, match="quant_dtype_a"):
        resolve_aiter_mxfp4_act_dtype(
            Mxfp4MoeBackend.AITER_MXFP4_BF16, config, kMxfp8Dynamic
        )


@requires_gfx950_aiter
@requires_quant_dtype_a
def test_fhmoe_shared_experts_reject_a_bf16_request(default_vllm_config, monkeypatch):
    """AITER's fused shared-expert dispatch only runs FP8 activations on this
    layout; it raises deep inside the kernel, so reject it at build time."""
    config = _make_moe_config()
    config.is_fused_shared_expert_requested = True
    _request_activation(monkeypatch, None)
    with pytest.raises(ValueError, match="shared-expert"):
        resolve_aiter_mxfp4_act_dtype(Mxfp4MoeBackend.AITER_MXFP4_BF16, config, None)
    # FP8 is what FHMoE runs anyway, so that request is fine.
    from aiter import dtypes

    _request_activation(monkeypatch, kMxfp8Dynamic)
    assert (
        resolve_aiter_mxfp4_act_dtype(
            Mxfp4MoeBackend.AITER_MXFP4_BF16, config, kMxfp8Dynamic
        )
        is dtypes.fp8
    )


@requires_gfx950_aiter
def test_unservable_request_is_not_substituted(default_vllm_config, monkeypatch):
    """Block-FP8 activations are not something this layout can execute. Backend
    selection reports that; the resolver must not quietly pick some other dtype,
    it must leave the layer exactly as it ran before the request existed."""
    from aiter import dtypes

    config = _make_moe_config()
    _request_activation(monkeypatch, kFp8Dynamic128Sym)
    assert (
        resolve_aiter_mxfp4_act_dtype(
            Mxfp4MoeBackend.AITER_MXFP4_BF16, config, kFp8Dynamic128Sym, pins_fp8=True
        )
        is dtypes.fp8
    )
    assert (
        resolve_aiter_mxfp4_act_dtype(
            Mxfp4MoeBackend.AITER_MXFP4_BF16, config, kFp8Dynamic128Sym
        )
        is None
    )


# -- helpers ------------------------------------------------------------------


def _request_activation(monkeypatch, key):
    """Stand in for `--quantization-config.moe.activation`, `null` included."""
    import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as oracle

    monkeypatch.setattr(oracle, "_user_moe_activation_override", lambda: key)
    monkeypatch.setattr(oracle, "_user_moe_activation_override_is_set", lambda: True)


def _build_layer(default_vllm_config, monkeypatch, *, activation, idx, targets=False):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    # `rocm_aiter_ops` caches both flags on first read, which may have happened
    # before the env vars above were set.
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    monkeypatch.setattr(rocm_aiter_ops, "_FMOE_ENABLED", True)
    prefix = f"model.layers.{idx}.mlp.experts"
    spec = {"weight": "mxfp4", "activation": activation}
    args = (
        quant_config_args(targets={f"*layers.{idx}.mlp.experts": spec})
        if targets
        else quant_config_args(moe=spec)
    )
    default_vllm_config.model_config = ModelConfig(dtype=torch.bfloat16)
    default_vllm_config.kernel_config = KernelConfig(moe_backend="aiter")
    with torch.device(current_platform.device_type):
        layer = FusedMoEFactory(
            num_experts=E,
            top_k=TOPK,
            hidden_size=H,
            intermediate_size=INT,
            params_dtype=torch.bfloat16,
            quant_config=OnlineQuantizationConfig(args),
            prefix=prefix,
        ).routed_experts
    layer.to_empty(device="cuda")
    # Same weights in every layer, so output differences are the activation.
    generator = torch.Generator(device="cuda").manual_seed(0)
    layer.w13_weight.data.copy_(
        torch.randn(
            E, 2 * INT, H, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        / H**0.5
    )
    layer.w2_weight.data.copy_(
        torch.randn(E, H, INT, dtype=torch.bfloat16, device="cuda", generator=generator)
        / INT**0.5
    )
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _make_inputs(M: int):
    generator = torch.Generator(device="cuda").manual_seed(1)
    x = (
        torch.randn(M, H, dtype=torch.bfloat16, device="cuda", generator=generator)
        / H**0.5
    )
    logits = torch.randn(M, E, device="cuda", generator=generator)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    return x, topk_weights.to(torch.float32), topk_ids.to(torch.int32)
