# Licensed under the Apache License, Version 2.0 (the "License");
#
# Implement part of this file based on vllm-project/vllm-ascend
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional

import torch
import torch_npu

SOFTMAX = "softmax"


def select_experts(hidden_states: torch.Tensor,
                   router_logits: torch.Tensor,
                   top_k: int,
                   use_grouped_topk: bool,
                   renormalize: bool,
                   topk_group: Optional[int] = None,
                   num_expert_group: Optional[int] = None,
                   scoring_func: str = SOFTMAX,
                   routed_scaling_factor: float = 1.0,
                   e_score_correction_bias: Optional[torch.Tensor] = None,
                   global_num_experts: int = -1):
    if scoring_func == SOFTMAX:
        # group topk is not used when softmax activation is applied
        norm_type = 0
        topk_group = 1
        num_expert_group = 1
    else:
        norm_type = 1

    if e_score_correction_bias is not None and \
        e_score_correction_bias.dtype != router_logits.dtype:
        e_score_correction_bias = e_score_correction_bias.to(
            router_logits.dtype)

    topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
        router_logits,
        k=top_k,
        bias=e_score_correction_bias,
        k_group=topk_group,             # number of selected experts group
        group_count=num_expert_group,   # number of experts groups
        group_select_mode=1,
        renorm=0,
        norm_type=norm_type,            # 0 Softmax, 1 Sigmoid
        routed_scaling_factor=routed_scaling_factor,
        eps=float(1e-20)
    )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids
