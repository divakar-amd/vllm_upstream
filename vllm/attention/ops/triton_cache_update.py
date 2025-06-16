# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl

# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


# @triton.autotune(
#     # Choices of configs to auto-tune over
#     configs=[
#         triton.Config({'BLOCK_SIZE': 1,}),
#         triton.Config({'BLOCK_SIZE': 2,}),
#         triton.Config({'BLOCK_SIZE': 4,}),
#         triton.Config({'BLOCK_SIZE': 8,}),
#         # triton.Config({'BLOCK_SIZE': 16,}),
#         # triton.Config({'BLOCK_SIZE': 32,}),
#     ],
#     key=['padded_tokens'],
# )
@triton.jit
def triton_reshape_and_cache_flash_kernel(
        key, value, key_cache, value_cache, slot_mapping, num_tokens: tl.int64,
        padded_tokens: tl.int64, cache_block_size: tl.int64,
        stride_cache_w: tl.int64, stride_cache_x: tl.int64,
        stride_key: tl.int64, stride_value: tl.int64, num_heads: tl.constexpr,
        head_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < num_tokens

    dim = tl.arange(0, num_heads * head_dim)

    mask2d = (block_start + offsets[:, None]
              < num_tokens) & (dim[None, :] < (num_heads * head_dim))

    # 1. Load key and value
    key_load = tl.load(key + ((block_start + offsets)[:, None]) * stride_key +
                       dim[None, :],
                       mask=mask2d)
    value_load = tl.load(value +
                         ((block_start + offsets)[:, None]) * stride_value +
                         dim[None, :],
                         mask=mask2d)

    # 2. Load slot_mapping
    slot_values = tl.load(slot_mapping + block_start + offsets,
                          mask=mask,
                          other=0)  # [Block_size_m]

    # 3. Calculate cache block index.
    cache_block_idx = (slot_values // cache_block_size)
    cache_block_offset = (slot_values % cache_block_size)

    cache_update_idx = (cache_block_idx *
                        stride_cache_w) + (cache_block_offset * stride_cache_x)
    cache_update_idx_2d = cache_update_idx[:, None] + dim[None, :]

    # # 4. Update key_cache and value_cache
    tl.store(key_cache + cache_update_idx_2d, key_load, mask=mask2d)
    tl.store(value_cache + cache_update_idx_2d, value_load, mask=mask2d)


def triton_reshape_and_cache_flash(key,
                                   value,
                                   key_cache,
                                   value_cache,
                                   slot_mapping,
                                   BLOCK_SIZE_=None):
    # num_tokens = key.size(0)# <--- this is 8 but num_slots size is 6!!
    padded_tokens = key.size(0)
    num_tokens = slot_mapping.size(0)  # Number of tokens to process

    num_heads = key.size(1)
    head_dim = key.size(2)
    cache_block_size = key_cache.size(1)

    # [ToDo] Add assert statements
    # - check if tl.clamp required for fp8 support

    stride_w = cache_block_size * num_heads * head_dim
    stride_x = num_heads * head_dim
    stride_key = key.stride(0)
    stride_value = value.stride(0)

    grid = lambda meta: (triton.cdiv(padded_tokens, meta['BLOCK_SIZE']), )
    triton_reshape_and_cache_flash_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        padded_tokens,
        cache_block_size,
        stride_w,
        stride_x,
        stride_key,
        stride_value,
        num_heads,
        head_dim,
        BLOCK_SIZE=BLOCK_SIZE_ if BLOCK_SIZE_ is not None else 1,
    )
