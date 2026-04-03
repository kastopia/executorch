# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TurboQuantKVCache module.

Verifies numerical equivalence with turboquant-vllm, nibble packing
correctness, bf16 tolerance, and torch.export compatibility.

No CUDA required — all tests run on CPU.

Usage:
    python -m pytest extension/llm/modules/test/test_turboquant_kv_cache.py -v
"""

import unittest

import torch

from executorch.extension.llm.modules.turboquant import TurboQuantKVCache
from torch.export import Dim, export

HEAD_DIM = 128
N_HEADS = 2
MAX_SEQ_LEN = 32
BITS = 4


class TestNumericalEquivalence(unittest.TestCase):
    """Our compress/decompress matches turboquant-vllm's TurboQuantMSE."""

    def test_compress_matches_library(self):
        from turboquant_vllm import TurboQuantMSE

        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        ref = TurboQuantMSE(HEAD_DIM, BITS, seed=42)

        x = torch.randn(1, N_HEADS, 5, HEAD_DIM)
        packed, norms = cache._compress(x)

        ref_indices, ref_norms = ref.quantize(x)

        # Unpack our nibbles
        flat = packed.reshape(-1, HEAD_DIM // 2)
        high = (flat >> 4).long()
        low = (flat & 0x0F).long()
        our_indices = torch.stack([high, low], dim=-1).reshape(-1, HEAD_DIM)

        self.assertTrue(
            torch.equal(our_indices, ref_indices.reshape(-1, HEAD_DIM)),
        )
        self.assertTrue(
            torch.allclose(norms.reshape(-1, 1), ref_norms.reshape(-1, 1)),
        )

    def test_decompress_matches_library(self):
        from turboquant_vllm import TurboQuantMSE

        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        ref = TurboQuantMSE(HEAD_DIM, BITS, seed=42)

        x = torch.randn(1, N_HEADS, 5, HEAD_DIM)

        ref_indices, ref_norms = ref.quantize(x)
        ref_recon = ref.dequantize(ref_indices, ref_norms)

        packed, norms = cache._compress(x)
        our_recon = cache._decompress(packed, norms)

        self.assertTrue(
            torch.allclose(our_recon, ref_recon.reshape(our_recon.shape), atol=1e-5),
        )

    def test_roundtrip_cosine_similarity(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        x = torch.randn(1, N_HEADS, 10, HEAD_DIM)
        packed, norms = cache._compress(x)
        recon = cache._decompress(packed, norms)

        cos = torch.nn.functional.cosine_similarity(
            x.reshape(-1, HEAD_DIM),
            recon.reshape(-1, HEAD_DIM),
        ).mean()
        self.assertGreater(cos.item(), 0.99)


class TestNibblePacking(unittest.TestCase):
    """uint8 nibble pack/unpack is bit-exact."""

    def test_roundtrip_all_index_pairs(self):
        all_pairs = torch.stack(
            torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij"),
            dim=-1,
        ).reshape(-1, 2)

        packed = (all_pairs[:, 0].to(torch.uint8) << 4) | all_pairs[:, 1].to(
            torch.uint8
        )

        high = (packed >> 4).long()
        low = (packed & 0x0F).long()

        self.assertTrue(torch.equal(high, all_pairs[:, 0].long()))
        self.assertTrue(torch.equal(low, all_pairs[:, 1].long()))


class TestBf16Tolerance(unittest.TestCase):
    """Codebook/rotation survive bf16 cast."""

    def test_bf16_cast_then_compress(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)

        for name, buf in cache.named_buffers():
            if buf.is_floating_point():
                cache.register_buffer(name, buf.to(torch.bfloat16))

        x = torch.randn(1, N_HEADS, 5, HEAD_DIM, dtype=torch.bfloat16)
        packed, norms = cache._compress(x)
        recon = cache._decompress(packed, norms)

        cos = torch.nn.functional.cosine_similarity(
            x.reshape(-1, HEAD_DIM).float(),
            recon.reshape(-1, HEAD_DIM).float(),
        ).mean()
        self.assertGreater(cos.item(), 0.99)


class TestTorchExport(unittest.TestCase):
    """TurboQuantKVCache survives torch.export(strict=True)."""

    def test_export_standalone(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        seq_dim = Dim("seq", min=1, max=MAX_SEQ_LEN - 1)

        with torch.no_grad():
            ep = export(
                cache,
                args=(
                    torch.arange(2),
                    torch.randn(1, N_HEADS, 2, HEAD_DIM),
                    torch.randn(1, N_HEADS, 2, HEAD_DIM),
                ),
                dynamic_shapes={
                    "input_pos": {0: seq_dim},
                    "k_val": {2: seq_dim},
                    "v_val": {2: seq_dim},
                },
                strict=True,
            )

        mod = ep.module()
        k = torch.randn(1, N_HEADS, 3, HEAD_DIM)
        v = torch.randn(1, N_HEADS, 3, HEAD_DIM)
        k_p, k_n, v_p, v_n = mod(torch.arange(3), k, v)

        self.assertEqual(k_p.shape, (1, N_HEADS, MAX_SEQ_LEN, HEAD_DIM // 2))
        self.assertEqual(k_n.shape, (1, N_HEADS, MAX_SEQ_LEN, 1))

    def test_state_accumulates(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        seq_dim = Dim("seq", min=1, max=MAX_SEQ_LEN - 1)

        with torch.no_grad():
            ep = export(
                cache,
                args=(
                    torch.arange(2),
                    torch.randn(1, N_HEADS, 2, HEAD_DIM),
                    torch.randn(1, N_HEADS, 2, HEAD_DIM),
                ),
                dynamic_shapes={
                    "input_pos": {0: seq_dim},
                    "k_val": {2: seq_dim},
                    "v_val": {2: seq_dim},
                },
                strict=True,
            )

        mod = ep.module()

        k0 = torch.randn(1, N_HEADS, 2, HEAD_DIM)
        mod(torch.arange(2), k0, torch.randn(1, N_HEADS, 2, HEAD_DIM))

        k1 = torch.randn(1, N_HEADS, 2, HEAD_DIM)
        k_p, k_n, _, _ = mod(
            torch.arange(2, 4), k1, torch.randn(1, N_HEADS, 2, HEAD_DIM)
        )

        # Positions 0-1 should still have k0's data
        recon_0 = cache._decompress(k_p[:, :, :2], k_n[:, :, :2])
        cos = torch.nn.functional.cosine_similarity(
            k0.reshape(-1, HEAD_DIM),
            recon_0.reshape(-1, HEAD_DIM),
        ).mean()
        self.assertGreater(cos.item(), 0.99)

        # Positions 4+ should be zero
        self.assertEqual(k_p[:, :, 4:].abs().max().item(), 0)


class TestEdgeCases(unittest.TestCase):

    def test_odd_head_dim_raises(self):
        with self.assertRaises(ValueError):
            TurboQuantKVCache(2, 127, 32)

    def test_head_dim_256(self):
        """Qwen 3.5 MoE config."""
        cache = TurboQuantKVCache(2, 256, 64)
        x = torch.randn(1, 2, 5, 256)
        packed, norms = cache._compress(x)
        self.assertEqual(packed.shape, (1, 2, 5, 128))
        self.assertEqual(packed.dtype, torch.uint8)

        recon = cache._decompress(packed, norms)
        cos = torch.nn.functional.cosine_similarity(
            x.reshape(-1, 256),
            recon.reshape(-1, 256),
        ).mean()
        self.assertGreater(cos.item(), 0.99)


if __name__ == "__main__":
    unittest.main()
