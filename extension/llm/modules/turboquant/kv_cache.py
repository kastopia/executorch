# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TurboQuant KV cache compression for torch.export(strict=True).

Compresses KV cache to TQ4 nibble-packed format (3.8x memory savings)
using the TurboQuant algorithm (arXiv 2504.19874, ICLR 2026). The
codebook and rotation matrix are precomputed at init time using
turboquant-vllm; the forward path is pure PyTorch ops.

Paired with the fused ``triton::tq4_sdpa`` kernel, attention runs
directly on compressed data — the full decompressed cache is never
materialized.

Usage::

    from executorch.extension.llm.modules.turboquant import (
        TurboQuantKVCache,
        replace_kv_cache_with_turboquant,
    )

    # After model construction, before torch.export:
    replace_kv_cache_with_turboquant(model, kv_cache_class=KVCache)

Compatible with: turboquant-vllm 1.4.0
"""

import torch
import torch.nn as nn
from turboquant_vllm import solve_lloyd_max
from turboquant_vllm.quantizer import _generate_rotation_matrix


class TurboQuantKVCache(nn.Module):
    """KV cache with TQ4 compression.

    Stores K/V as nibble-packed uint8 indices (2 indices per byte) plus
    fp32 per-vector norms. The ``update()`` method compresses incoming
    K/V and returns the compressed cache buffers for use with the fused
    ``triton::tq4_sdpa`` kernel. A ``_decompress()`` method is provided
    for testing.

    Args:
        n_heads: Number of KV heads.
        head_dim: Dimension per head (must be even).
        max_seq_len: Maximum sequence length (cache is pre-allocated).
        bits: Quantization bits per coordinate (default 4).
        seed: Random seed for the rotation matrix.
    """

    def __init__(self, n_heads, head_dim, max_seq_len, bits=4, seed=42):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.half_dim = head_dim // 2

        centroids, boundaries = solve_lloyd_max(head_dim, bits)
        rotation = _generate_rotation_matrix(head_dim, seed=seed)

        self.register_buffer("centroids", centroids)
        self.register_buffer("boundaries", boundaries)
        self.register_buffer("rotation", rotation)

        # Compressed cache buffers
        self.register_buffer(
            "k_packed",
            torch.zeros(1, n_heads, max_seq_len, self.half_dim, dtype=torch.uint8),
        )
        self.register_buffer(
            "k_norms",
            torch.zeros(1, n_heads, max_seq_len, 1, dtype=torch.float32),
        )
        self.register_buffer(
            "v_packed",
            torch.zeros(1, n_heads, max_seq_len, self.half_dim, dtype=torch.uint8),
        )
        self.register_buffer(
            "v_norms",
            torch.zeros(1, n_heads, max_seq_len, 1, dtype=torch.float32),
        )

    def _compress(self, x):
        """Compress ``(B, H, T, D)`` tensor to nibble-packed uint8 + fp32 norms.

        All ops are torch.export-compatible: norm, matmul, bucketize, bitwise.
        """
        orig_shape = x.shape
        flat = x.reshape(-1, self.head_dim).float()

        norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
        normalized = flat / (norms + 1e-10)
        rotated = normalized @ self.rotation.float().T
        indices = torch.bucketize(rotated, self.boundaries.float())

        idx_u8 = indices.to(torch.uint8)
        packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]

        return (
            packed.reshape(*orig_shape[:-1], self.half_dim),
            norms.reshape(*orig_shape[:-1], 1),
        )

    def _decompress(self, packed, norms):
        """Decompress nibble-packed uint8 + fp32 norms back to float tensor.

        Provided for testing — the fused ``tq4_sdpa`` kernel decompresses
        per-tile in the attention inner loop, never calling this method.
        """
        orig_batch_shape = packed.shape[:-1]
        flat_packed = packed.reshape(-1, self.half_dim)
        flat_norms = norms.reshape(-1, 1)

        high = (flat_packed >> 4).long()
        low = (flat_packed & 0x0F).long()
        indices = torch.stack([high, low], dim=-1).reshape(-1, self.head_dim)

        reconstructed = self.centroids.float()[indices]
        unrotated = reconstructed @ self.rotation.float()
        scaled = unrotated * flat_norms

        return scaled.reshape(*orig_batch_shape, self.head_dim)

    def forward(self, input_pos, k_val, v_val):
        return self.update(input_pos, k_val, v_val)

    def update(self, input_pos, k_val, v_val):
        """Compress and store K/V, return compressed cache buffers.

        Args:
            input_pos: ``(T,)`` position indices.
            k_val: ``(B, H, T, D)`` key tensor.
            v_val: ``(B, H, T, D)`` value tensor.

        Returns:
            Tuple of ``(k_packed, k_norms, v_packed, v_norms)`` — the full
            compressed cache (all positions, not just the new tokens).
        """
        k_packed, k_norms = self._compress(k_val)
        v_packed, v_norms = self._compress(v_val)

        self.k_packed[:, :, input_pos] = k_packed
        self.k_norms[:, :, input_pos] = k_norms
        self.v_packed[:, :, input_pos] = v_packed
        self.v_norms[:, :, input_pos] = v_norms

        return self.k_packed, self.k_norms, self.v_packed, self.v_norms
