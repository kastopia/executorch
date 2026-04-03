# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# TQ4 fused Flash Attention kernel adapted from turboquant-vllm v1.4.0
# (Alberto-Codes/turboquant-vllm, Apache 2.0):
#   src/turboquant_vllm/triton/flash_attention_tq4_kv.py
#
# Compatible with: turboquant-vllm 1.4.0
#
# Reference: arXiv 2504.19874 — "TurboQuant: Online Vector Quantization
# with Near-optimal Distortion Rate" (ICLR 2026).
#
# Changes from upstream:
#   - tl.dot(p_cast, v, acc).to(tl.float32) for Triton 3.6 compat
#   - HAS_MASK support (bool mask [B, 1, L_Q, L_KV]) matching sdpa.py pattern
#   - Autotune configs trimmed for power-of-2 head_dim
#   - Wrapped as @triton_op for ExecuTorch CUDA backend

"""
Fused TQ4 SDPA: attention over nibble-packed compressed K/V cache.

Both K and V tiles are decompressed inline from uint8 nibble-packed indices
in the attention inner loop. The full decompressed cache is never materialized.
Q is pre-rotated by Pi^T, output is post-rotated by Pi outside the kernel.
GQA is handled internally (H_Q != H_KV).
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


# ---------------------------------------------------------------------------
# Autotune configs (trimmed for power-of-2 head_dim)
# ---------------------------------------------------------------------------

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [32, 64]
    for BN in [32, 64]
    for s in [2, 3]
    for w in [4, 8]
    if not (w == 8 and BM < 64)
]


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N_CTX_Q", "HEAD_DIM"])
@triton.jit
def _fwd_tq4_kv_kernel(
    Q_rot,
    K_packed,
    K_norms,
    V_packed,
    V_norms,
    Centroids,
    Mask,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kpz,
    stride_kph,
    stride_kpn,
    stride_kpd,
    stride_knz,
    stride_knh,
    stride_knn,
    stride_vpz,
    stride_vph,
    stride_vpn,
    stride_vpd,
    stride_vnz,
    stride_vnh,
    stride_vnn,
    stride_mb,
    stride_mq,
    stride_mk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    H_Q,
    H_KV,
    N_CTX_Q,
    N_CTX_KV,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    HALF_D_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    HALF_D: tl.constexpr = HEAD_DIM // 2
    NEG_INF: tl.constexpr = float("-inf")

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h_q = off_hz % H_Q
    off_h_kv = off_h_q // (H_Q // H_KV)

    # Base pointers
    q_base = Q_rot + off_z * stride_qz + off_h_q * stride_qh
    kp_base = K_packed + off_z * stride_kpz + off_h_kv * stride_kph
    kn_base = K_norms + off_z * stride_knz + off_h_kv * stride_knh
    vp_base = V_packed + off_z * stride_vpz + off_h_kv * stride_vph
    vn_base = V_norms + off_z * stride_vnz + off_h_kv * stride_vnh
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh
    mask_b_base = Mask + off_z * stride_mb

    # Block offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    d_mask = offs_d < HEAD_DIM
    offs_d_half = tl.arange(0, HALF_D_PAD)
    d_half_mask = offs_d_half < HALF_D

    # Load Q_rot tile
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    row_valid = offs_m < N_CTX_Q
    q = tl.load(q_ptrs, mask=row_valid[:, None] & d_mask[None, :], other=0.0)

    # Online softmax state (fp32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PAD], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX_KV)
    else:
        hi = N_CTX_KV

    for start_n in range(0, hi, BLOCK_N):
        kv_valid = (start_n + offs_n) < N_CTX_KV

        # -- K decompression --
        kp_ptrs = (
            kp_base
            + (start_n + offs_n[:, None]) * stride_kpn
            + offs_d_half[None, :] * stride_kpd
        )
        k_packed = tl.load(
            kp_ptrs, mask=kv_valid[:, None] & d_half_mask[None, :], other=0
        )
        k_hi = (k_packed >> 4).to(tl.int32)
        k_lo = (k_packed & 0x0F).to(tl.int32)
        k = tl.join(tl.load(Centroids + k_hi), tl.load(Centroids + k_lo)).reshape(
            BLOCK_N, HEAD_DIM_PAD
        )
        k = tl.where(d_mask[None, :], k, 0.0)
        kn_ptrs = kn_base + (start_n + offs_n) * stride_knn
        k_norms_tile = tl.load(kn_ptrs, mask=kv_valid, other=0.0)
        k = (k * k_norms_tile[:, None]).to(Q_rot.dtype.element_ty)

        # Q_rot @ K^T
        qk = tl.dot(q, tl.trans(k))
        qk = (qk * qk_scale).to(tl.float32)

        if IS_CAUSAL:
            causal = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal, qk, NEG_INF)

        if HAS_MASK:
            m_ptrs = (
                mask_b_base
                + offs_m[:, None] * stride_mq
                + (start_n + offs_n[None, :]) * stride_mk
            )
            tile_valid = row_valid[:, None] & kv_valid[None, :]
            keep = tl.load(m_ptrs, mask=tile_valid, other=False)
            qk = tl.where(keep, qk, NEG_INF)

        qk = tl.where(kv_valid[None, :], qk, NEG_INF)

        # Online softmax
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        acc = (acc * alpha[:, None]).to(tl.float32)

        # -- V decompression --
        vp_ptrs = (
            vp_base
            + (start_n + offs_n[:, None]) * stride_vpn
            + offs_d_half[None, :] * stride_vpd
        )
        v_packed = tl.load(
            vp_ptrs, mask=kv_valid[:, None] & d_half_mask[None, :], other=0
        )
        v_hi = (v_packed >> 4).to(tl.int32)
        v_lo = (v_packed & 0x0F).to(tl.int32)
        v = tl.join(tl.load(Centroids + v_hi), tl.load(Centroids + v_lo)).reshape(
            BLOCK_N, HEAD_DIM_PAD
        )
        v = tl.where(d_mask[None, :], v, 0.0)
        vn_ptrs = vn_base + (start_n + offs_n) * stride_vnn
        v_norms_tile = tl.load(vn_ptrs, mask=kv_valid, other=0.0)
        v = (v * v_norms_tile[:, None]).to(Q_rot.dtype.element_ty)

        # P @ V (accumulate in fp32 — .to(tl.float32) for Triton 3.6 compat)
        l_ij = tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, acc).to(tl.float32)

        l_i = (l_i * alpha + l_ij).to(tl.float32)
        m_i = m_new

    # Epilogue
    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(
        o_ptrs,
        acc.to(Q_rot.dtype.element_ty),
        mask=row_valid[:, None] & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# @triton_op wrapper
# ---------------------------------------------------------------------------


@triton_op("triton::tq4_sdpa", mutates_args={})
def tq4_sdpa(
    query: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused TQ4 SDPA over nibble-packed compressed K/V cache.

    Decompresses K/V per tile in the attention inner loop. The full
    decompressed cache is never materialized (3.8x memory savings).

    Args:
        query: [B, H_Q, L_Q, D] bf16
        k_packed: [B, H_KV, L_KV, D//2] uint8 nibble-packed key indices
        k_norms: [B, H_KV, L_KV, 1] fp32 key vector norms
        v_packed: [B, H_KV, L_KV, D//2] uint8 nibble-packed value indices
        v_norms: [B, H_KV, L_KV, 1] fp32 value vector norms
        centroids: [16] fp32 Lloyd-Max codebook
        rotation: [D, D] fp32 orthogonal rotation matrix
        attn_mask: Optional [B, 1, L_Q, L_KV] bool mask

    Returns:
        [B, H_Q, L_Q, D] bf16 attention output
    """
    B, H_Q, N_Q, D = query.shape
    _, H_KV, N_KV, HALF_D = k_packed.shape

    sm_scale = 1.0 / math.sqrt(D)

    # Reshape norms: [B, H, S, 1] -> [B, H, S]
    k_n = k_norms.reshape(B, H_KV, N_KV).contiguous()
    v_n = v_norms.reshape(B, H_KV, N_KV).contiguous()

    # Pre-rotate Q: Q_rot = Q @ Pi^T
    q_rot = torch.matmul(query.float(), rotation.T).to(query.dtype)

    out_rot = torch.empty_like(query)

    HEAD_DIM_PAD = _next_pow2(D)
    HALF_D_PAD = _next_pow2(HALF_D)

    # Masking: use explicit mask when provided, never auto-causal.
    # The caller (FullAttention) always provides an explicit bool mask
    # that handles both prefill (lower-triangular) and decode (row mask).
    HAS_MASK = attn_mask is not None
    is_causal = False
    if HAS_MASK:
        Mask_ptr = attn_mask
        stride_mb = attn_mask.stride(0)
        stride_mq = attn_mask.stride(2)
        stride_mk = attn_mask.stride(3)
    else:
        Mask_ptr = k_packed  # dummy, won't be accessed
        stride_mb = 0
        stride_mq = 0
        stride_mk = 0

    def grid(META):
        return (triton.cdiv(N_Q, META["BLOCK_M"]), B * H_Q)

    wrap_triton(_fwd_tq4_kv_kernel)[grid](
        q_rot,
        k_packed,
        k_n,
        v_packed,
        v_n,
        centroids,
        Mask_ptr,
        out_rot,
        sm_scale,
        *q_rot.stride(),
        *k_packed.stride(),
        *k_n.stride(),
        *v_packed.stride(),
        *v_n.stride(),
        stride_mb,
        stride_mq,
        stride_mk,
        *out_rot.stride(),
        H_Q,
        H_KV,
        N_Q,
        N_KV,
        HEAD_DIM=D,
        HEAD_DIM_PAD=HEAD_DIM_PAD,
        HALF_D_PAD=HALF_D_PAD,
        IS_CAUSAL=is_causal,
        HAS_MASK=HAS_MASK,
    )

    # Post-rotate: convert from rotated space back to original space
    return torch.matmul(out_rot.float(), rotation).to(query.dtype)


@tq4_sdpa.register_fake
def _tq4_sdpa_fake(
    query: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(query)
