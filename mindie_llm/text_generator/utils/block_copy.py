# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import numpy as np

from ...utils.log.logging import logger


class BlockCopy:
    def __init__(self, block_copy_type: str, kv_cache: list, to_tensor):
        # The shape of kv_cache here is [num_layer, 2, num_block, block_size, num_head, head_size].
        self.kv_cache = kv_cache
        self.to_tensor = to_tensor
        if block_copy_type == "atb":
            import torch_npu

            soc_version = torch_npu._C._npu_get_soc_version()
            if soc_version in (100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 205):
                self.block_copy = self.golden_copy_block
            else:
                self.block_copy_op = None
                self.kv_cache_map = {}
                self.init_atb_block_copy_op(len(kv_cache))
                self.block_copy = self.atb_copy_block
            self.block_copy_init = True
        else:
            logger.warning("Block copy backend type has not been supported yet.")

    @staticmethod
    def process_mapping(mapping):
        src_block_indices = []
        dst_block_indices = []
        for pair in mapping:
            src, dst = pair.tolist()
            src_block_indices.append(src)
            dst_block_indices.append(dst)
        return src_block_indices, dst_block_indices

    def init_atb_block_copy_op(self, num_layers: int):
        import atb_llm.nn as nn
        from atb_llm.nn.network_manager import get_default_net
        from atb_llm.nn.tensor import Tensor

        for layer_index in range(num_layers):
            nn.functional.copy_blocks(
                Tensor(f"in0_layer_{layer_index}"),
                Tensor(f"in1_layer_{layer_index}"),
                Tensor("in2"),
                Tensor("in3"),
                Tensor("in4"),
            )
        self.block_copy_op = get_default_net().build_engine()
        for layer_index, (key_cache, value_cache) in enumerate(self.kv_cache):
            self.kv_cache_map[f"in0_layer_{layer_index}"] = key_cache
            self.kv_cache_map[f"in1_layer_{layer_index}"] = value_cache

    def atb_copy_block(self, src_indices, dst_indices):
        device = "npu"
        src_block_indices = self.to_tensor(np.array(src_indices, dtype=np.int32)).to(device=device, non_blocking=False)
        dst_block_indices = self.to_tensor(np.array(dst_indices, dtype=np.int32)).to(device=device, non_blocking=False)
        cum_sum = self.to_tensor(np.arange(1, src_block_indices.size(0) + 1, dtype=np.int32)).to(
            device=device, non_blocking=False
        )
        inputs = {"in2": src_block_indices, "in3": dst_block_indices, "in4": cum_sum}
        inputs.update(self.kv_cache_map)
        outputs = {}
        self.block_copy_op.forward(inputs, outputs)

    def copy_blocks(self, src_dst_map):
        if not self.block_copy_init:
            raise RuntimeError("Block copy operator has not been successfully initialized.")
        src_idx, dst_idx = self.process_mapping(src_dst_map)
        self.block_copy(src_idx, dst_idx)

    def golden_copy_block(self, src_indices, dst_indices):
        for pair in zip(src_indices, dst_indices):
            src, dst = pair
            for key_cache, value_cache in self.kv_cache:
                key_cache.data[dst, :] = key_cache.data[src, :]
                value_cache.data[dst, :] = value_cache.data[src, :]
