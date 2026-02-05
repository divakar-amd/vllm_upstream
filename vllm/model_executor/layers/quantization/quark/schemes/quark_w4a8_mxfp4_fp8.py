# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)

# Try to import AITER kernel
try:
    from aiter.ops.triton.gemm_a8wfp4 import gemm_a8wfp4
    AITER_AVAILABLE = True
    logger.info("AITER kernel for W4A8 is available.-------------------")
except (ImportError, AttributeError):
    AITER_AVAILABLE = False
    logger.warning(
        "AITER kernel for W4A8 not available. "
        "Will use emulation mode. Install aiter for better performance."
    )


__all__ = ["QuarkW4A8_MXFP4_FP8"]

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A8_MXFP4_FP8(QuarkScheme):
    """
    Quantization scheme for W4A8: MXFP4 weights + FP8 activations.
    
    - Weights: MXFP4 (4-bit microsca FP) with E8M0 scales per block of 32
    - Activations: FP8 E4M3 (static per-tensor quantization)
    
    This implementation uses the AITER Triton kernel for efficient computation
    on AMD GPUs, with fallback to emulation mode if AITER is not available.
    """
    
    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
    ):
        self.out_dtype = None # [DV todo] check on this. maybe bf16??.
        
        # Weight configuration (MXFP4)
        self.weight_dtype = "mxfp4"
        self.packed_factor: Fraction = Fraction(2, 1)  # 2 FP4 values per byte
        self.weight_block_size = OCP_MX_BLOCK_SIZE
        
        # Activation configuration (FP8)
        self.is_static_input_scheme = not input_quant_spec.get("is_dynamic")
        self.input_qscheme = input_quant_spec.get("qscheme")  # "per_tensor"
        
        if not self.is_static_input_scheme:
            raise NotImplementedError(
                "Dynamic FP8 activation quantization is not yet supported "
                "for W4A8. The current implementation expects static per-tensor "
                "FP8 scales stored in the checkpoint."
            )
        
        # Determine if we can use native AITER kernel
        self.use_native_kernel = (
            AITER_AVAILABLE 
            and current_platform.is_rocm()
            and self.is_static_input_scheme
        )
        
        if not self.use_native_kernel:
            logger.warning_once(
                "W4A8 MXFP4+FP8 will use emulation mode. "
                "For best performance on AMD GPUs, ensure AITER is installed "
                "and you're running on a ROCm platform."
            )
    
    @classmethod
    def get_min_capability(cls) -> int:
        # Requires MI300+ for FP8 support on AMD, or Ada Lovelace (89) on NVIDIA
        # For now, set to 89 (Ada) as baseline
        return 89
    
    def get_packed_dim(self, dim: int) -> int:
        """Calculate packed dimension for MXFP4 (2 values per byte)."""
        assert dim % 2 == 0, f"Dimension {dim} must be even for MXFP4 packing"
        return dim // 2
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        
        # MXFP4 WEIGHT (packed, 2 values per byte)
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                self.get_packed_dim(input_size_per_partition),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.packed_factor,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        
        # WEIGHT SCALE (E8M0 format, per block of 32)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.weight_block_size,
                dtype=torch.uint8,  # E8M0 format
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        
        # INPUT SCALE (FP8 per-tensor static scale)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(
                    len(output_partition_sizes),
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )
            # Initialize to avoid NaN
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Post-process weights after loading from checkpoint.
        Convert to format expected by the kernel.
        """
        # Weights and scales are already in the correct format from checkpoint
        # Just ensure they're non-trainable
        layer.weight = torch.nn.Parameter(
            layer.weight.data, requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )
        
        if self.is_static_input_scheme:
            # For static per-tensor, we have one scale per logical output partition
            # Convert to scalar if it's a single partition
            input_scale = layer.input_scale.data
            if input_scale.numel() == 1:
                input_scale = input_scale.item()
            else:
                # For fused modules (QKV), take the max scale for safety
                input_scale = input_scale.max().item()
            
            layer.input_scale = torch.nn.Parameter(
                torch.tensor(input_scale, dtype=torch.float32),
                requires_grad=False,
            )
    
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply W4A8 quantized linear layer.
        
        Args:
            layer: Linear layer with weight, weight_scale, input_scale
            x: Input activations in high precision (BF16/FP16)
            bias: Optional bias term
        
        Returns:
            Output in high precision (BF16/FP16)
        """
        if self.use_native_kernel:
            return self._apply_native_kernel(layer, x, bias)
        else:
            raise NotImplementedError("Emulation mode for W4A8 is not implemented.") # [DV todo] remove this. Do we need emulation mode??.
            # return self._apply_emulation(layer, x, bias)
    
    def _apply_native_kernel(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Use AITER Triton kernel for efficient W4A8 GEMM.
        
        AITER kernel signature:
        gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config)
        
        - x: FP8 E4M3 (M, K)
        - w: Packed FP4 (N, K//2), transposed internally
        - y: Output (M, N)
        - x_scales: Per-row FP32 scales (M, 1)
        - w_scales: E8M0 per-group scales (N, K//32)
        """
        M = x.shape[0]
        K = layer.input_size_per_partition
        N = layer.output_size_per_partition
        
        out_dtype = x.dtype if self.out_dtype is None else self.out_dtype

        # 1. Quantize activations to FP8
        # Since we have static per-tensor scale, we need to:
        # - Quantize: x_fp8 = round(x / input_scale).clamp(fp8_min, fp8_max)
        # - For kernel: we need per-token scales, so broadcast the per-tensor scale
        
        input_scale = layer.input_scale.item()
        x_fp8 = (x / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        
        # AITER kernel expects per-row scales (M, 1)
        # Since we have per-tensor, broadcast it
        x_scales = torch.full(
            (M, 1),
            input_scale,
            dtype=torch.float32,
            device=x.device,
        )
        
        # 2. Prepare output tensor
        y = torch.zeros(M, N, dtype=out_dtype, device=x.device)
        
        # 3. Call AITER kernel
        # Note: AITER expects w to be (N, K//2) and transposed internally
        # Our weight is (N, K//2) already, which matches!
        gemm_a8wfp4(
            x=x_fp8,
            w=layer.weight,
            y=y,
            x_scales=x_scales,
            w_scales=layer.weight_scale,
            dtype=out_dtype,
            config=None,  # Auto-tune
        )
        
        # 4. Add bias if present
        if bias is not None:
            y = y + bias
        
        return y
    
    def _apply_emulation(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Emulation mode: dequantize weights, quantize-dequantize activations,
        then use standard F.linear.
        """
        # Import dequantization utilities
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            dequant_mxfp4,
        )
        
        # 1. Dequantize MXFP4 weights to high precision
        weight_dq = dequant_mxfp4(
            layer.weight,
            layer.weight_scale,
            x.dtype,
        )
        
        # 2. Simulate FP8 quantization-dequantization on activations
        input_scale = layer.input_scale.item()
        x_fp8 = (x / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        x_dq = x_fp8.to(x.dtype) * input_scale
        
        # 3. Standard linear operation
        return F.linear(x_dq, weight_dq, bias)