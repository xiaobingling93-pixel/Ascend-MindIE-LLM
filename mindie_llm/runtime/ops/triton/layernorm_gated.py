# SPDX-FileCopyrightText: Copyright contributors to the sgl-project
# SPDX-License-Identifier: Apache-2.0
# Part of this file implemented based on sgl-project.
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


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel_npu(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    # Compute row indices for this program
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    # Mask for valid rows and cols
    row_mask = rows < M
    col_mask = cols < N

    # Load weight once (broadcasted over rows)
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    # Load X: shape [BLOCK_M, BLOCK_N]
    x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)

    # Load Z if needed
    if HAS_Z:
        z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N
        z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
        if not NORM_BEFORE_GATE:
            x *= z * tl.sigmoid(z)

    # Compute statistics per row
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # [BLOCK_M]
        xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        tl.store(Mean + rows, mean, mask=row_mask)
    else:
        xbar = tl.where(col_mask[None, :], x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N

    rstd = 1.0 / tl.sqrt(var + eps)  # [BLOCK_M]
    tl.store(Rstd + rows, rstd, mask=row_mask)

    # Normalize
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :]
    if HAS_BIAS:
        y += b[None, :]

    # Post-gate
    if HAS_Z and NORM_BEFORE_GATE:
        y *= z * tl.sigmoid(z)

    # Store output
    y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :] + group * N
    tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


def layer_norm_fwd_npu(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    if N % group_size != 0:
        raise ValueError(f"N ({N}) must be divisible by group_size ({group_size})")
    ngroups = N // group_size

    if x.stride(-1) != 1:
        raise ValueError("x must be contiguous in the last dimension (stride(-1) == 1)")
    if z is not None:
        if z.stride(-1) != 1:
            raise ValueError("z must be contiguous in the last dimension (stride(-1) == 1)")
        if z.shape != (M, N):
            raise ValueError(f"z.shape must be (M, N) = ({M}, {N}), got {z.shape}")
    if weight.shape != (N,):
        raise ValueError(f"weight.shape must be (N,) = ({N},), got {weight.shape}")
    if weight.stride(-1) != 1:
        raise ValueError("weight must be contiguous in the last dimension (stride(-1) == 1)")
    if bias is not None:
        if bias.stride(-1) != 1:
            raise ValueError("bias must be contiguous in the last dimension (stride(-1) == 1)")
        if bias.shape != (N,):
            raise ValueError(f"bias.shape must be (N,) = ({N},), got {bias.shape}")
    # allocate output
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(f"out.shape must match x.shape {x.shape}, got {out.shape}")
    else:
        out = torch.empty_like(x)
    if out.stride(-1) != 1:
        raise ValueError("out must be contiguous in the last dimension (stride(-1) == 1)")
    mean = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("Feature dim too large.")

    # Choose BLOCK_M: e.g., 16, 32, 64 — depends on NPU vector core capacity
    BLOCK_M = 64  # Tune this based on your NPU's register/shared memory

    # Now grid is (num blocks over M, num groups)
    grid = (triton.cdiv(M, BLOCK_M), ngroups)
    _layer_norm_fwd_1pass_kernel_npu[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        # Remove multibuffer if not needed
    )
    return out, mean, rstd
