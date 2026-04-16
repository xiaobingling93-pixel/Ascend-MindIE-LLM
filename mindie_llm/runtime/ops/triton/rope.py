# SPDX-FileCopyrightText: Copyright contributors to the vllm-project
# SPDX-License-Identifier: Apache-2.0
# Part of this file implemented based on vllm-project.
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    num_tokens,
    n_qh,
    n_kh,
    hd,
    rope_dim,
    pad_rope_dim: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,  # 新增：每次处理的 Head 数量
    IS_NEOX_STYLE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)

    half_rope_dim = rope_dim // 2
    rope_offs = tl.arange(0, pad_rope_dim // 2)
    rope_mask = rope_offs < half_rope_dim

    for row_idx in tl.range(pid, num_tokens, row_block_size):
        q_start_ptr = q_ptr + row_idx * q_row_stride
        k_start_ptr = k_ptr + row_idx * k_row_stride
        cos_start_ptr = cos + row_idx * cos_row_stride
        sin_start_ptr = sin + row_idx * sin_row_stride
        cos_row = tl.load(cos_start_ptr + rope_offs, mask=rope_mask, other=0).to(tl.float32)
        sin_row = tl.load(sin_start_ptr + rope_offs, mask=rope_mask, other=0).to(tl.float32)
        for h_start in range(0, n_qh, BLOCK_HEADS):
            h_offs = h_start + tl.arange(0, BLOCK_HEADS)
            h_mask = h_offs < n_qh
            head_base_offsets = h_offs[:, None] * hd
            if IS_NEOX_STYLE:
                offs_1 = head_base_offsets + rope_offs[None, :]
                offs_2 = offs_1 + half_rope_dim
            else:
                offs_1 = head_base_offsets + (2 * rope_offs[None, :])
                offs_2 = offs_1 + 1

            mask = h_mask[:, None] & rope_mask[None, :]

            q1 = tl.load(q_start_ptr + offs_1, mask=mask, other=0).to(tl.float32)
            q2 = tl.load(q_start_ptr + offs_2, mask=mask, other=0).to(tl.float32)

            out_q1 = q1 * cos_row - q2 * sin_row
            out_q2 = q2 * cos_row + q1 * sin_row

            tl.store(q_start_ptr + offs_1, out_q1, mask=mask)
            tl.store(q_start_ptr + offs_2, out_q2, mask=mask)

        for h_start in range(0, n_kh, BLOCK_HEADS):
            h_offs = h_start + tl.arange(0, BLOCK_HEADS)
            h_mask = h_offs < n_kh

            head_base_offsets = h_offs[:, None] * hd

            if IS_NEOX_STYLE:
                offs_1 = head_base_offsets + rope_offs[None, :]
                offs_2 = offs_1 + half_rope_dim
            else:
                offs_1 = head_base_offsets + (2 * rope_offs[None, :])
                offs_2 = offs_1 + 1

            mask = h_mask[:, None] & rope_mask[None, :]

            k1 = tl.load(k_start_ptr + offs_1, mask=mask, other=0).to(tl.float32)
            k2 = tl.load(k_start_ptr + offs_2, mask=mask, other=0).to(tl.float32)

            out_k1 = k1 * cos_row - k2 * sin_row
            out_k2 = k2 * cos_row + k1 * sin_row

            tl.store(k_start_ptr + offs_1, out_k1, mask=mask)
            tl.store(k_start_ptr + offs_2, out_k2, mask=mask)


def rope_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int = -1,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    num_tokens, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[1]

    # Flatten cos/sin to [num_tokens, dim]
    cos = cos.view(num_tokens, -1)
    sin = sin.view(num_tokens, -1)

    if rope_dim == -1:
        rope_dim = cos.shape[-1] * 2

    pad_rope_dim = triton.next_power_of_2(rope_dim)

    BLOCK_HEADS = 16

    grid = (num_tokens,)

    _triton_rope[grid](
        q,
        q.stride(0),
        k,
        k.stride(0),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        num_tokens,
        n_q_head,
        n_kv_head,
        head_dim,
        rope_dim,
        pad_rope_dim=pad_rope_dim,
        BLOCK_HEADS=BLOCK_HEADS,
        IS_NEOX_STYLE=is_neox_style,
        num_stages=1,
    )
    return q, k


@triton.jit
def triton_rope_kernel_in_place(
    x_ptr,
    sin_ptr,
    cos_ptr,
    x_stride,
    cos_stride,
    hidden_size: tl.constexpr,
    rope_dim: tl.constexpr,
    head_num: tl.constexpr,
):
    cur_b = tl.program_id(0)
    dim_start = hidden_size - rope_dim
    # load x
    offset_x = cur_b * x_stride + dim_start + tl.arange(0, rope_dim)
    x = tl.load(x_ptr + offset_x).to(tl.float32)
    # load sin cos
    offset_sin_cos = cur_b // head_num * cos_stride + tl.arange(0, rope_dim)
    sin = tl.load(sin_ptr + offset_sin_cos).to(tl.float32)
    cos = tl.load(cos_ptr + offset_sin_cos).to(tl.float32)

    even = tl.extract_slice(x, [0], [rope_dim // 2], [2])
    odd = tl.extract_slice(x, [1], [rope_dim // 2], [2])
    odd = -odd

    x_rotate = tl.zeros([rope_dim], dtype=tl.float32)
    x_rotate = tl.insert_slice(x_rotate, odd, [0], [rope_dim // 2], [2])
    x_rotate = tl.insert_slice(x_rotate, even, [1], [rope_dim // 2], [2])

    out = x * cos + x_rotate * sin
    tl.store(x_ptr + offset_x, out.to(tl.bfloat16))


def triton_apply_rope_partial_in_place(x, sin, cos):
    rope_dim = sin.shape[-1]
    org_shape = x.shape
    if x.dim() == 2:
        bsz, hidden_size = x.shape
        head_num = 1
    elif x.dim() == 3:
        bsz, head_num, hidden_size = x.shape
        x = x.view(-1, hidden_size)
    else:
        raise NotImplementedError(f"x_shape={x.shape} not supported")
    cores = bsz * head_num
    if cores >= 65535:
        raise ValueError(f"cores ({cores}) must be less than 65535 (uint16 limit for triton grid)")
    triton_rope_kernel_in_place[(cores,)](
        x,
        sin,
        cos,
        x.stride(0),
        sin.stride(0),
        hidden_size,
        rope_dim,
        head_num,
    )
    return x.view(org_shape)
