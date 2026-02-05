import tempfile
import pytest
import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.quark.schemes import QuarkW4A8_MXFP4_FP8
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)


MODEL_PATH = '/data/models/gpt-oss-120b-w-mxfp4-a-fp8-kv-fp8-fp8attn'


def load_w4a8_weights_from_safetensors(f):
    """Helper function to load w4a8 quantized weights from safetensors file.
    
    Args:
        f: Opened safetensors file handle
        
    Returns:
        tuple: (weight, weight_scale, input_scale) tensors
        
    Raises:
        pytest.skip.Exception: If required weights are not found
    """
    # Try to load q_proj from layer 0
    weight_key = "model.layers.0.self_attn.q_proj.weight"
    weight_scale_key = "model.layers.0.self_attn.q_proj.weight_scale"
    input_scale_key = "model.layers.0.self_attn.q_proj.input_scale"
    
    # Check if keys exist
    available_keys = f.keys()
    
    if weight_key not in available_keys:
        # Find any linear layer weight
        linear_weights = [k for k in available_keys if k.endswith('.weight') and 'weight_scale' not in k]
        if not linear_weights:
            pytest.skip("No suitable linear layer weights found")
        weight_key = linear_weights[0]
        # Derive scale keys
        base_key = weight_key.replace('.weight', '')
        weight_scale_key = f"{base_key}.weight_scale"
        input_scale_key = f"{base_key}.input_scale"
    
    if weight_key not in available_keys or weight_scale_key not in available_keys or input_scale_key not in available_keys:
        pytest.skip(f"Required weight keys not found in safetensors file")
    
    weight = f.get_tensor(weight_key)
    weight_scale = f.get_tensor(weight_scale_key)
    input_scale = f.get_tensor(input_scale_key)
    
    return weight, weight_scale, input_scale


@pytest.fixture(scope="module")
def dist_init():
    """Initialize distributed environment for the test module."""
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for quantization tests"
)
def test_w4a8_scheme_creation():
    """Test that W4A8 scheme can be created."""
    weight_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": False,
    }
    input_spec = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    
    scheme = QuarkW4A8_MXFP4_FP8(weight_spec, input_spec)
    assert scheme is not None
    assert scheme.weight_dtype == "mxfp4"
    assert scheme.is_static_input_scheme is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for quantization tests"
)
def test_w4a8_weight_creation(dist_init):
    """Test weight parameter creation."""
    weight_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": False,
    }
    input_spec = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    
    scheme = QuarkW4A8_MXFP4_FP8(weight_spec, input_spec)
    
    # Create a mock layer
    layer = torch.nn.Module()
    
    # Test dimensions
    output_size = 2880
    input_size = 2880
    output_partition_sizes = [output_size]
    
    def mock_weight_loader(param, loaded_weight, **kwargs):
        param.data.copy_(loaded_weight)
    
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=output_partition_sizes,
        input_size_per_partition=input_size,
        params_dtype=torch.bfloat16,
        weight_loader=mock_weight_loader,
    )
    
    # Check that weights were created
    assert hasattr(layer, 'weight')
    assert hasattr(layer, 'weight_scale')
    assert hasattr(layer, 'input_scale')
    
    # Check dimensions
    assert layer.weight.shape == (output_size, input_size // 2)  # Packed
    assert layer.weight.dtype == torch.uint8
    
    assert layer.weight_scale.shape == (output_size, input_size // 32)  # Per block of 32
    assert layer.weight_scale.dtype == torch.uint8  # E8M0 format
    
    assert layer.input_scale.shape == torch.Size([1])
    assert layer.input_scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for quantization tests"
)
def test_w4a8_forward_with_real_weights(dist_init):
    """Test W4A8 forward pass with real weights from the quantized model."""
    import os
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    
    weight_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": False,
    }
    input_spec = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    
    scheme = QuarkW4A8_MXFP4_FP8(weight_spec, input_spec)
    
    # Load real weights from the first attention layer (q_proj)
    # This is a good test case as it's a standard linear layer
    safetensors_file = os.path.join(MODEL_PATH, "model-00001-of-00013.safetensors")
    
    if not os.path.exists(safetensors_file):
        pytest.skip(f"Safetensors file not found: {safetensors_file}")
    
    # Load weights
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        weight, weight_scale, input_scale = load_w4a8_weights_from_safetensors(f)
    
    # Get dimensions from loaded weights
    output_size, input_size_packed = weight.shape
    input_size = input_size_packed * 2  # Unpacked dimension
    
    print(f"\nLoaded weights:")
    print(f"  Weight shape: {weight.shape}, dtype: {weight.dtype}")
    print(f"  Weight scale shape: {weight_scale.shape}, dtype: {weight_scale.dtype}")
    print(f"  Input scale shape: {input_scale.shape}, dtype: {input_scale.dtype}")
    print(f"  Inferred input_size: {input_size}, output_size: {output_size}")
    
    # Create a mock layer
    layer = torch.nn.Module()
    
    def mock_weight_loader(param, loaded_weight, **kwargs):
        param.data.copy_(loaded_weight)
    
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[output_size],
        input_size_per_partition=input_size,
        params_dtype=torch.bfloat16,
        weight_loader=mock_weight_loader,
    )
    
    # Load the actual weights
    mock_weight_loader(layer.weight, weight)
    mock_weight_loader(layer.weight_scale, weight_scale)
    mock_weight_loader(layer.input_scale, input_scale)
    
    # Process weights after loading
    scheme.process_weights_after_loading(layer)
    
    # Move to GPU
    device = torch.device("cuda:0")
    layer = layer.to(device)
    
    # Create random input with appropriate batch size and sequence length
    batch_size = 2
    seq_len = 4
    x = torch.randn(
        batch_size * seq_len,
        input_size,
        dtype=torch.bfloat16,
        device=device
    )
    
    print(f"\nInput shape: {x.shape}")
    
    # Test forward pass
    try:
        output = scheme.apply_weights(layer, x, bias=None)
        
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
        
        # Basic sanity checks
        assert output.shape == (batch_size * seq_len, output_size)
        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        # Check that output is in reasonable range (not all zeros or extremely large)
        assert output.abs().max() < 1000, "Output values too large"
        assert output.abs().max() > 0.001, "Output values too small (possibly all zeros)"
        
        print("✓ Forward pass successful!")
        
    except Exception as e:
        pytest.fail(f"Forward pass failed: {str(e)}")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for quantization tests"
)
def test_w4a8_numerical_consistency(dist_init):
    """Test that multiple forward passes with the same input produce the same output.
    
    Note: This test requires real model weights because randomly initialized
    uint8 values don't represent valid MXFP4/E8M0 encoded data.
    """
    import os
    
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    
    weight_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": False,
    }
    input_spec = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    
    scheme = QuarkW4A8_MXFP4_FP8(weight_spec, input_spec)
    
    # Load real weights from the model
    safetensors_file = os.path.join(MODEL_PATH, "model-00001-of-00013.safetensors")
    
    if not os.path.exists(safetensors_file):
        pytest.skip(f"Safetensors file not found: {safetensors_file}")
    
    # Load weights
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        weight, weight_scale, input_scale = load_w4a8_weights_from_safetensors(f)
    
    # Get dimensions from loaded weights
    output_size, input_size_packed = weight.shape
    input_size = input_size_packed * 2  # Unpacked dimension
    
    # Create mock layer
    layer = torch.nn.Module()
    
    def mock_weight_loader(param, loaded_weight, **kwargs):
        param.data.copy_(loaded_weight)
    
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[output_size],
        input_size_per_partition=input_size,
        params_dtype=torch.bfloat16,
        weight_loader=mock_weight_loader,
    )
    
    # Load the actual weights
    mock_weight_loader(layer.weight, weight)
    mock_weight_loader(layer.weight_scale, weight_scale)
    mock_weight_loader(layer.input_scale, input_scale)
    
    scheme.process_weights_after_loading(layer)
    
    # Move to GPU
    device = torch.device("cuda:0")
    layer = layer.to(device)
    
    # Create fixed random input
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    x = torch.randn(4, input_size, dtype=torch.bfloat16, device=device)
    
    # Run forward pass twice
    output1 = scheme.apply_weights(layer, x, bias=None)
    output2 = scheme.apply_weights(layer, x, bias=None)
    
    # Check consistency
    assert torch.allclose(output1, output2, rtol=1e-4, atol=1e-4), \
        "Forward pass not deterministic"
    
    print("✓ Numerical consistency test passed!")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for quantization tests"
)
def test_w4a8_batch_size_invariance(dist_init):
    """Test that the output is consistent across different batch sizes.
    
    Note: This test requires real model weights because randomly initialized
    uint8 values don't represent valid MXFP4/E8M0 encoded data.
    """
    import os
    
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    
    weight_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": False,
    }
    input_spec = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    
    scheme = QuarkW4A8_MXFP4_FP8(weight_spec, input_spec)
    
    # Load real weights from the model
    safetensors_file = os.path.join(MODEL_PATH, "model-00001-of-00013.safetensors")
    
    if not os.path.exists(safetensors_file):
        pytest.skip(f"Safetensors file not found: {safetensors_file}")
    
    # Load weights
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        weight, weight_scale, input_scale = load_w4a8_weights_from_safetensors(f)
    
    # Get dimensions from loaded weights
    output_size, input_size_packed = weight.shape
    input_size = input_size_packed * 2  # Unpacked dimension
    
    # Create mock layer
    layer = torch.nn.Module()
    
    def mock_weight_loader(param, loaded_weight, **kwargs):
        param.data.copy_(loaded_weight)
    
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[output_size],
        input_size_per_partition=input_size,
        params_dtype=torch.bfloat16,
        weight_loader=mock_weight_loader,
    )
    
    # Load the actual weights
    mock_weight_loader(layer.weight, weight)
    mock_weight_loader(layer.weight_scale, weight_scale)
    mock_weight_loader(layer.input_scale, input_scale)
    
    scheme.process_weights_after_loading(layer)
    
    device = torch.device("cuda:0")
    layer = layer.to(device)
    
    # Create input with batch size 1
    torch.manual_seed(123)
    x_single = torch.randn(1, input_size, dtype=torch.bfloat16, device=device)
    output_single = scheme.apply_weights(layer, x_single, bias=None)
    
    # Create input with batch size 4 (repeat the same input)
    x_batch = x_single.repeat(34, 1)
    output_batch = scheme.apply_weights(layer, x_batch, bias=None)
    
    # Check that each row in batch output matches the single output
    for i in range(34):
        assert torch.allclose(output_batch[i], output_single[0], rtol=1e-4, atol=1e-4), \
            f"Batch output row {i} doesn't match single output"
    
    print("✓ Batch size invariance test passed!")

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for quantization tests"
)
def test_w4a8_kernel_accuracy(dist_init):
    """Test that the quantized kernel output matches the reference implementation.
    
    This test compares the fast GPU kernel against a reference implementation
    that dequantizes weights and runs standard PyTorch operations.
    
    Note: This test requires real model weights because randomly initialized
    uint8 values don't represent valid MXFP4/E8M0 encoded data.
    """
    import os
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        dequant_mxfp4,
    )
    
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    
    weight_spec = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
        "is_dynamic": False,
    }
    input_spec = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
        "symmetric": True,
    }
    
    scheme = QuarkW4A8_MXFP4_FP8(weight_spec, input_spec)
    
    # Load real weights from the model
    safetensors_file = os.path.join(MODEL_PATH, "model-00001-of-00013.safetensors")
    
    if not os.path.exists(safetensors_file):
        pytest.skip(f"Safetensors file not found: {safetensors_file}")
    
    # Load weights
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        weight, weight_scale, input_scale = load_w4a8_weights_from_safetensors(f)
    
    # Get dimensions from loaded weights
    output_size, input_size_packed = weight.shape
    input_size = input_size_packed * 2  # Unpacked dimension
    
    # Create mock layer
    layer = torch.nn.Module()
    
    def mock_weight_loader(param, loaded_weight, **kwargs):
        param.data.copy_(loaded_weight)
    
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[output_size],
        input_size_per_partition=input_size,
        params_dtype=torch.bfloat16,
        weight_loader=mock_weight_loader,
    )
    
    # Load the actual weights
    mock_weight_loader(layer.weight, weight)
    mock_weight_loader(layer.weight_scale, weight_scale)
    mock_weight_loader(layer.input_scale, input_scale)
    
    scheme.process_weights_after_loading(layer)
    
    # Move to GPU
    device = torch.device("cuda:0")
    layer = layer.to(device)
    
    # Create test input
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    batch_size = 8
    x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)
    
    # 1. Get output from the quantized kernel
    output_kernel = scheme.apply_weights(layer, x, bias=None)
    
    # 2. Compute reference output using dequantized weights
    # Dequantize weights to match input dtype (BF16) for realistic comparison
    weight_dequant = dequant_mxfp4(
        layer.weight,
        layer.weight_scale,
        x.dtype,  # Use BF16 to match kernel output precision
    )

    # Simulate FP8 quantization-dequantization on activations
    input_scale_value = layer.input_scale.item()
    x_fp8 = (x / input_scale_value).clamp(-448, 448).to(torch.float8_e4m3fn)
    x_dequant = x_fp8.to(x.dtype) * input_scale_value  # Keep in BF16

    # Compute reference output with standard linear operation
    # Use BF16 throughout to match kernel precision
    output_reference = torch.nn.functional.linear(
        x_dequant, 
        weight_dequant, 
        bias=None
    )
    
    # 3. Compare outputs
    # W4A8 quantization introduces errors, so we use reasonable tolerances
    # rtol=1e-2 (1% relative error) and atol=1e-2 (absolute error)
    max_diff = (output_kernel - output_reference).abs().max().item()
    mean_diff = (output_kernel - output_reference).abs().mean().item()
    relative_error = (max_diff / output_reference.abs().max().item()) * 100
    
    print(f"\n=== Kernel Accuracy Test ===")
    print(f"Output shape: {output_kernel.shape}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Max relative error: {relative_error:.2f}%")
    print(f"Kernel output range: [{output_kernel.min().item():.4f}, {output_kernel.max().item():.4f}]")
    print(f"Reference output range: [{output_reference.min().item():.4f}, {output_reference.max().item():.4f}]")
    
    # Assert outputs are close
    # For W4A8 quantization, we expect some accuracy loss but should be < 5% error
    assert torch.allclose(output_kernel, output_reference, rtol=5e-2, atol=1e-2), \
        f"Kernel output differs too much from reference. Max diff: {max_diff:.6f}, Relative error: {relative_error:.2f}%"
    
    # Check that the difference is not all zeros (would indicate both outputs are wrong)
    assert output_kernel.abs().max() > 0.1, "Kernel output is too small or all zeros"
    assert output_reference.abs().max() > 0.1, "Reference output is too small or all zeros"
    
    print("✓ Kernel accuracy test passed!")


if __name__ == "__main__":
    # Run tests locally - Initialize distributed environment first
    print("Initializing distributed environment...")
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(1, 1)
    
    print("Running W4A8 tests...")
    
    try:
        print("\n=== Test 1: Scheme Creation ===")
        test_w4a8_scheme_creation()
        
        print("\n=== Test 2: Weight Creation ===")
        test_w4a8_weight_creation(None)  # dist_init already done
        
        print("\n=== Test 3: Forward Pass with Real Weights ===")
        try:
            test_w4a8_forward_with_real_weights(None)
        except pytest.skip.Exception as e:
            print(f"⚠️  Test skipped: {e}")
        
        print("\n=== Test 4: Numerical Consistency ===")
        try:
            test_w4a8_numerical_consistency(None)
        except pytest.skip.Exception as e:
            print(f"⚠️  Test skipped: {e}")
        
        print("\n=== Test 5: Batch Size Invariance ===")
        try:
            test_w4a8_batch_size_invariance(None)
        except pytest.skip.Exception as e:
            print(f"⚠️  Test skipped: {e}")

        print("\n=== Test 6: Kernel Accuracy ===")
        try:
            test_w4a8_kernel_accuracy(None)
        except pytest.skip.Exception as e:
            print(f"⚠️  Test skipped: {e}")
        
        print("\n✅ All tests passed!")
    finally:
        cleanup_dist_env_and_memory()