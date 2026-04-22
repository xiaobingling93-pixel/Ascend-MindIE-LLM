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
import torch.distributed as torch_dist


def all_gather(input_, process_group):
    group_size = process_group.size()
    target = torch.zeros_like(input_).repeat(group_size, *(1,) * (input_.dim() - 1))
    torch_dist.all_gather_into_tensor(target, input_, group=process_group)
    return target


def gather_tensor(input_, index):
    if index is None:
        return input_
    index = index.reshape(index.shape[0], *(1,) * (input_.dim() - 1)).expand(-1, *input_.shape[1:])
    target = torch.gather(input_, dim=0, index=index)
    return target


def allgather_and_reorder(input_, process_group, gather_index):
    # all-gather
    input_ = all_gather(input_, process_group=process_group)
    # gather
    input_ = gather_tensor(input_, gather_index)
    return input_
