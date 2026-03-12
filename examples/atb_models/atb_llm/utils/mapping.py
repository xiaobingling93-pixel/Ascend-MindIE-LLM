# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from copy import deepcopy
from dataclasses import dataclass
from typing import List
import json
import torch
import atb_llm.nn.distributed as atb_llm_dist
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.moe_utils import ExpertParallelDegree
from atb_llm.utils.env import ENV
from atb_llm.utils import file_utils
from atb_llm.utils.initial import NPUSocInfo

DP = "dp"
TP = "tp"
MOE_TP = "moe_tp"
PP = "pp"
MICROBATCH_SIZE = "microbatch_size"
MOE_EP = "moe_ep"
SP = "sp"
CP = "cp"
MAX_LCCL_COMM_DOMAIN_ID = 65535


@dataclass
class ParallelInfo:
    buffer_size: int
    group_size: int = 1
    num_group: int | None = None
    rank_per_group: List[List[int]] | None = None
    current_group_id: int | None = None
    rank: int | None = None
    domain: str = ""
    process_group: str = "-1"

    def to_dict(self):
        return {"groupId": self.current_group_id, "rankIds": self.rank_per_group[self.current_group_id],
                "rank": self.rank, "bufferSize": self.buffer_size}


class PipelineParallelInfo(ParallelInfo):
    def __init__(self, hccl_buffer):
        super().__init__(hccl_buffer)
        self.microbatch_size = 1
        self.pp_bcast_group = None
        self.tp = ParallelInfo(hccl_buffer)


class Mapping:
    def __init__(self, world_size, rank, llm_config=None, **kwargs):
        if llm_config is not None:
            self.parallel_config = llm_config.llm.parallel_options
        else:
            self.parallel_config = None

        self.lm_head_local_tp = self.parallel_config.lm_head_local_tp \
            if self.parallel_config is not None and isinstance(self.parallel_config.lm_head_local_tp, int) \
            else -1
        self.enable_lm_head_local_tp = (self.lm_head_local_tp > 1)
        self.o_proj_local_tp = self.parallel_config.o_proj_local_tp \
            if self.parallel_config is not None and isinstance(self.parallel_config.o_proj_local_tp, int) \
            else -1
        self.enable_o_proj_local_tp = (self.o_proj_local_tp > 1)
        self.hccl_buffer = 128 if self.parallel_config is None else self.parallel_config.hccl_buffer
        if self.parallel_config is not None and isinstance(self.parallel_config.dense_mlp_local_tp, int) \
           and self.parallel_config.dense_mlp_local_tp > 1:
            self.enable_dense_tp = True
        else:
            self.enable_dense_tp = False

        self.world_size = world_size
        self.rank = rank
        self.rank_table_file = ENV.rank_table_file
        self.num_nodes = self.get_num_nodes()
        self.local_world_size = self.world_size // self.num_nodes
        self.enable_node_based_a2a = (not self.is_super_node()) and self.num_nodes > 1 and \
            llm_config is not None and llm_config.llm.ep_level == ExpertParallelDegree.DYNAMIC_EP
        self.word_embed_tp = ParallelInfo(self.hccl_buffer)
        self.word_embed_dp = ParallelInfo(self.hccl_buffer)
        self.attn_tp = ParallelInfo(self.hccl_buffer)
        self.attn_o_proj_tp = ParallelInfo(self.hccl_buffer)
        self.attn_dp = ParallelInfo(self.hccl_buffer)
        self.attn_o_proj_dp = ParallelInfo(self.hccl_buffer)
        self.mlp_tp = ParallelInfo(self.hccl_buffer)
        self.mlp_dp = ParallelInfo(self.hccl_buffer)
        self.pp = PipelineParallelInfo(self.hccl_buffer)
        self.moe_tp = ParallelInfo(self.hccl_buffer)
        self.moe_ep = ParallelInfo(self.hccl_buffer)
        self.moe_ep_intra_node = ParallelInfo(self.hccl_buffer)
        self.moe_ep_inter_node = ParallelInfo(self.hccl_buffer)
        self.lm_head_tp = ParallelInfo(self.hccl_buffer)
        self.lm_head_dp = ParallelInfo(self.hccl_buffer)
        self.attn_inner_sp = ParallelInfo(self.hccl_buffer)
        self.attn_cp = ParallelInfo(self.hccl_buffer)
        self.attn_prefix_cache_cp = ParallelInfo(self.hccl_buffer)
        self.dense_tp = ParallelInfo(self.hccl_buffer)
        self.dense_dp = ParallelInfo(self.hccl_buffer)
        self.parse_parallel_info(**kwargs)
        self.validate()
        self.get_tp_group(self.word_embed_tp)
        self.get_dp_group(self.word_embed_dp)
        self.get_tp_group(self.attn_tp)
        self.get_tp_group(self.attn_o_proj_tp)
        self.get_dp_group(self.attn_dp)
        self.get_dp_group(self.attn_o_proj_dp)
        self.get_tp_group(self.mlp_tp)
        self.get_dp_group(self.mlp_dp)
        self.get_pp_group(self.pp)
        self.get_tp_group(self.moe_tp)
        self.get_dp_group(self.moe_ep)
        self.get_dp_group(self.moe_ep_inter_node)
        self.get_tp_group(self.moe_ep_intra_node)
        self.get_tp_group(self.lm_head_tp)
        self.get_dp_group(self.lm_head_dp)
        self.get_tp_group(self.attn_inner_sp)
        self.get_dp_group(self.attn_cp)
        self.get_prefix_cache_cp_group(self.attn_prefix_cache_cp)
        self.get_tp_group(self.dense_tp)
        self.get_dp_group(self.dense_dp)

        # 开启o_proj tpdp混合并行
        if self.enable_o_proj_local_tp:
            self.get_domain(self.attn_o_proj_tp, self.attn_o_proj_dp, 0)
            self.get_domain(self.attn_tp, self.attn_dp, self.attn_o_proj_tp.group_size)
            self.get_domain(self.attn_dp, self.attn_tp, self.attn_o_proj_tp.group_size + self.attn_dp.group_size)
            # 同时开启lmhead tpdp混合并行，目前仅支持A2 机内tp=8并行，机间dp并行。复用o_proj的通信域id
            if self.enable_lm_head_local_tp:
                self.lm_head_dp.domain = self.attn_o_proj_dp.domain
                self.lm_head_tp.domain = self.attn_o_proj_tp.domain
        # 开启lmhead tpdp混合并行
        elif self.enable_lm_head_local_tp:
            self.get_domain(self.lm_head_tp, self.lm_head_dp, 0)
            self.get_domain(self.attn_tp, self.attn_dp, self.attn_o_proj_tp.group_size)
            self.get_domain(self.attn_dp, self.attn_tp, self.attn_o_proj_tp.group_size + self.attn_dp.group_size)
        else:
            self.get_domain(self.attn_tp, self.attn_dp, 0)
            self.get_domain(self.attn_dp, self.attn_tp, self.attn_dp.group_size)
            self.get_domain(self.attn_cp, self.attn_tp, self.attn_cp.group_size)
        self.get_domain(self.moe_tp, self.moe_ep, 2 * world_size)
        self.get_domain(self.moe_ep, self.moe_tp, 2 * world_size + self.moe_ep.group_size)
        self.lccl_comm_domain_lower_bound, self.lccl_comm_domain_upper_bound = self.get_lccl_domain_range(
            kwargs.get("num_lccl_comm_shards", 1),
            kwargs.get("lccl_comm_shard_id", 0),
        )
        if self.enable_dense_tp:
            self.get_domain(self.dense_tp, self.dense_dp, 3 * world_size)
        # 设置默认通信域为63
        self.default_domain = str(MAX_LCCL_COMM_DOMAIN_ID)
        self.mlp_tp.domain = self.default_domain
        self.moe_ep.buffer_size = 512 if self.parallel_config is None else self.parallel_config.hccl_moe_ep_buffer
        self.moe_tp.buffer_size = 64 if self.parallel_config is None else self.parallel_config.hccl_moe_tp_buffer
        if self.enable_node_based_a2a:
            self.moe_ep_inter_node.buffer_size = 512 \
                if self.parallel_config is None else self.parallel_config.hccl_moe_ep_buffer
            self.moe_ep_intra_node.buffer_size = 512 \
                if self.parallel_config is None else self.parallel_config.hccl_moe_ep_buffer
        self.attn_inner_sp.domain = self.attn_tp.domain
        self.dynamic_eplb = deepcopy(self.moe_ep)
        self.dynamic_eplb.buffer_size = 128 # 动态eplb的通信域128MB够用

        self.update_pp()
        self.lwd_global_comm = None

    def __repr__(self):
        return (
            "Mapping("
            + f"world_size={self.world_size}, "
            + f"rank={self.rank}, "
            + f"num_nodes={self.num_nodes},"
            + f"pp_rank={self.pp.rank}, "
            + f"pp_groups={self.pp.rank_per_group}, "
            + f"micro_batch_size={self.pp.microbatch_size}, "
            + f"attn_dp_groups={self.attn_dp.rank_per_group}, "
            + f"attn_tp_groups={self.attn_tp.rank_per_group}, "
            + f"attn_inner_sp_groups={self.attn_inner_sp.rank_per_group}, "
            + f"attn_cp_groups={self.attn_cp.rank_per_group}, "
            + f"attn_o_proj_tp_groups={self.attn_o_proj_tp.rank_per_group}, "
            + f"mlp_tp_groups={self.mlp_tp.rank_per_group}, "
            + f"dense_dp_groups={self.dense_dp.rank_per_group}, "
            + f"dense_tp_groups={self.dense_tp.rank_per_group}, "
            + f"moe_ep_groups={self.moe_ep.rank_per_group}, "
            + f"moe_tp_groups={self.moe_tp.rank_per_group})"
        )

    @staticmethod
    def get_current_group_id(rank_per_group, target_rank_id):
        for idx, group in enumerate(rank_per_group):
            if target_rank_id in group:
                return idx
        return None

    @staticmethod
    def get_domain(src_module, dst_module, start_idx):
        current_idx = dst_module.rank
        src_module.domain = str(start_idx + current_idx)

    @staticmethod
    def get_lccl_domain_range(num_lccl_comm_shards, lccl_comm_shard_id):
        lower_bound = (MAX_LCCL_COMM_DOMAIN_ID + 1) // num_lccl_comm_shards * lccl_comm_shard_id
        upper_bound = (MAX_LCCL_COMM_DOMAIN_ID + 1) // num_lccl_comm_shards * (lccl_comm_shard_id + 1)
        return lower_bound, upper_bound

    @staticmethod
    def get_num_nodes():
        if ENV.rank_table_file == "":
            return 1
        with file_utils.safe_open(ENV.rank_table_file, 'r', encoding='utf-8') as f:
            ranktable = json.load(f)
        return int(ranktable["server_count"])
    
    @staticmethod
    def is_super_node():
        if ENV.rank_table_file == "":
            return False
        with file_utils.safe_open(ENV.rank_table_file, 'r', encoding='utf-8') as f:
            ranktable = json.load(f)
            for server in ranktable.get("server_list", []):
                for device in server.get("device", []):
                    if "super_device_id" in device:
                        return True
        return False

    def init_python_comm_process_group(self):
        atb_llm_dist.init_process_group(
            init_method=self.rank_table_file,
            world_size=self.world_size,
            rank=self.rank,
            buffer_size=self.hccl_buffer,
            backend=NPUSocInfo().communication_backend.value
        )
        python_comm_list = [self.word_embed_tp, self.word_embed_dp, self.attn_tp, self.attn_o_proj_tp,
            self.attn_dp, self.attn_o_proj_dp, self.mlp_tp, self.mlp_dp, self.moe_tp, self.moe_ep,
            self.lm_head_tp, self.lm_head_dp, self.attn_inner_sp, self.attn_cp]
        for comm in python_comm_list:
            comm.process_group = atb_llm_dist.new_group(comm.rank_per_group[comm.current_group_id],
                buffer_size=self.hccl_buffer, backend=NPUSocInfo().communication_backend.value)

    def update_pp(self):
        if self.has_pp():
            import torch.distributed as dist
            master_ip = ENV.master_ip
            if not master_ip:
                raise ValueError("Master ip cannot be None when pipeline parallel is used. Please export MASTER_IP.")
            master_port = ENV.master_port
            if not master_port:
                raise ValueError("MASTER_PORT or Network adapter cannot be None when pipeline parallel is used. \
                    Please export environment.")
            init_method = f"tcp://{master_ip}:{master_port}"
            logger.debug(f"rank: {self.rank}, init_method: {init_method}, start to init distributed")
            dist.init_process_group(backend='gloo', init_method=init_method, world_size=self.world_size, rank=self.rank)
            self.pp.pp_bcast_group = torch.distributed.group.WORLD
            logger.debug(f"rank: {self.rank}, init_method: {init_method}, init distributed successfully")     

    def parse_parallel_info(self, **kwargs):
        if kwargs.get(DP, -1) != -1:
            self.attn_dp.group_size = kwargs.get(DP, -1)
        # tp默认值为world_size
        self.attn_tp.group_size = self.world_size
        self.mlp_tp.group_size = self.world_size
        self.pp.tp.group_size = self.world_size
        self.attn_prefix_cache_cp.group_size = self.world_size
        if self.enable_lm_head_local_tp:
            # 保证lmhead tp切分 不跨机
            self.lm_head_tp.group_size = min(self.lm_head_local_tp, self.world_size // self.num_nodes)
            self.lm_head_dp.group_size = self.world_size // self.lm_head_tp.group_size
        # pp默认值为1
        self.pp.group_size = 1
        # microbatch_size
        self.pp.microbatch_size = kwargs.get(MICROBATCH_SIZE)
        self.moe_tp.group_size = self.world_size
        if kwargs.get(TP, -1) != -1:
            self.attn_tp.group_size = kwargs.get(TP, self.world_size)
            self.moe_tp.group_size = kwargs.get(TP, self.world_size)
            self.pp.tp.group_size = kwargs.get(TP, self.world_size)
        # moe_tp
        if kwargs.get(MOE_TP, -1) != -1:
            self.moe_tp.group_size = kwargs.get(MOE_TP, self.moe_tp.group_size)
        # moe_ep
        if kwargs.get(MOE_EP, -1) != -1:
            self.moe_ep.group_size = kwargs.get(MOE_EP, self.moe_ep.group_size)
            if self.enable_node_based_a2a:
                self.moe_ep_inter_node.group_size = self.num_nodes
                self.moe_ep_intra_node.group_size = self.moe_ep.group_size // self.num_nodes
        # pp
        if kwargs.get(PP, -1) != -1:
            self.pp.group_size = kwargs.get(PP, self.pp.group_size)
        # sp
        if kwargs.get(SP, -1) != -1:
            self.attn_inner_sp.group_size = kwargs.get(SP, self.attn_inner_sp.group_size)
        # cp
        if kwargs.get(CP, -1) != -1:
            self.attn_cp.group_size = kwargs.get(CP, self.attn_cp.group_size)
        # word embed
        self.word_embed_tp = self.attn_tp
        self.word_embed_dp = self.attn_dp
        # lm_head
        if ENV.enable_dp_partition_up:
            if self.enable_lm_head_local_tp:
                self.lm_head_tp = self.lm_head_tp
                self.lm_head_dp = self.lm_head_dp
            else:
                self.lm_head_tp = self.attn_tp
                self.lm_head_dp = self.attn_dp
        else:
            self.lm_head_tp = self.mlp_tp
            self.lm_head_dp = self.mlp_dp
        self.convert_attn_tp_type()
        self.parse_dense_tp_info()

    def convert_attn_tp_type(self):
        if self.enable_o_proj_local_tp:
            self.attn_o_proj_tp.group_size = self.o_proj_local_tp

    def parse_dense_tp_info(self):
        if self.enable_dense_tp:
            self.dense_tp.group_size = self.parallel_config.dense_mlp_local_tp
            self.dense_dp.group_size = self.world_size // self.parallel_config.dense_mlp_local_tp
        else:
            self.dense_tp.group_size = self.attn_tp.group_size
            self.dense_dp.group_size = self.attn_dp.group_size

    def validate(self):
        if self.world_size % self.num_nodes != 0:
            error_msg = "World size should be multiple of the number of nodes. " \
                        "Please check `world_size` and `ranktablefile`."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(error_msg)

        if self.attn_cp.group_size != 1 and self.attn_dp.group_size != 1:
            error_msg = "The attention module cannot support context parallel and data parallel simultaneously. " \
                        "Please check `cp` and `dp`."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(error_msg)

        if self.pp.group_size != 1 and self.attn_dp.group_size != 1:
            error_msg = "The attention module cannot support data parallel and pipeline parallel simultaneously. " \
                        "Please check `dp` and `pp`."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(error_msg)

        self.validate_pp()
        self.validate_dense_tp()

    def validate_pp(self):
        if self.has_pp():
            if self.pp.tp.group_size * self.pp.group_size != self.world_size:
                error_msg = f"World size must equal to attention's tp_size * pp_size. " \
                            f"pp_size is {self.pp.group_size}. " \
                            f"pp's tp_size is {self.pp.tp.group_size}. " \
                            f"World size is {self.world_size}. " \
                            f"Please check `tp`, `pp` and `world_size`."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)
        else:
            if self.attn_tp.group_size * self.attn_cp.group_size * self.attn_dp.group_size != self.world_size:
                error_msg = f"World size must equal to" \
                            f" attention's dp_size * attention's cp_size * attention's tp_size. " \
                            f"Attention's tp_size is {self.attn_tp.group_size}. " \
                            f"Attention's dp_size is {self.attn_dp.group_size}. " \
                            f"Attention's cp_size is {self.attn_cp.group_size}. " \
                            f"World size is {self.world_size}. " \
                            f"Please check `dp`, `cp`, `tp` and `world_size`."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)

            if self.attn_tp.group_size > self.local_world_size and self.attn_tp.group_size != self.world_size:
                error_msg = f"Attention's tp_size should be no greater than local world size, " \
                            f"or equal to world size. " \
                            f"Attention's tp_size is {self.attn_tp.group_size}. " \
                            f"World size is {self.world_size}. " \
                            f"Local world size is {self.local_world_size}. " \
                            f"Please check `tp`, `world_size` and `ranktablefile`."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)

            if self.has_moe_ep():
                if self.moe_ep.group_size * self.moe_tp.group_size != self.world_size:
                    error_msg = f"World size must equal to MoE's ep_size * MoE's tp_size. " \
                            f"MoE's tp_size is {self.moe_tp.group_size}. " \
                            f"MoE's dp_size is {self.moe_ep.group_size}. World size is {self.world_size}. " \
                            f"Please check `moe_tp`, `moe_ep` and `world_size`."
                    logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                    raise ValueError(error_msg)
                if self.moe_tp.group_size > self.local_world_size and self.moe_tp.group_size != self.world_size:
                    error_msg = f"MoE's tp_size should be no greater than local world size, or equal to world size. " \
                                f"MoE's tp_size is {self.moe_tp.group_size}. " \
                                f"World size is {self.world_size}. " \
                                f"Local world size is {self.local_world_size}. " \
                                f"Please check `moe_tp`, `world_size` and `ranktablefile`."
                    logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                    raise ValueError(error_msg)
            else:
                if self.moe_tp.group_size != self.world_size:
                    error_msg = f"World size must equal to MoE's tp_size. " \
                        f"MoE's tp_size is {self.moe_tp.group_size}. " \
                        f"World size is {self.world_size}. " \
                        f"Please check `tp`, `moe_tp` and `world_size`."
                    logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                    raise ValueError(error_msg)
            
            if self.has_attn_inner_sp() and self.attn_inner_sp.group_size != self.attn_tp.group_size:
                error_msg = f"Attention's sp_size must equal to attention's tp_size. " \
                    f"Attention's sp_size is {self.attn_inner_sp.group_size}. " \
                    f"Attention's tp_size is {self.attn_tp.group_size}. " \
                    f"Please check `attn_sp` and `attn_tp`."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)

            if self.enable_o_proj_local_tp and self.attn_o_proj_tp.group_size > self.local_world_size:
                error_msg = f"O_proj's tp_size should be no greater than local world size. " \
                    f"O_proj's tp_size is {self.attn_o_proj_tp.group_size}. " \
                    f"Local world size is {self.local_world_size}. " \
                    f"Please check the value of `o_proj_local_tp` in the config.json."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)
            
            if self.enable_lm_head_local_tp and self.lm_head_tp.group_size > self.local_world_size:
                error_msg = f"Lm_head's tp_size should be no greater than local world size. " \
                    f"Lm_head's tp_size is {self.lm_head_tp.group_size}. " \
                    f"Local world size is {self.local_world_size}. " \
                    f"Please check the value of `lm_head_local_tp` in the config.json."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)

            if self.enable_lm_head_local_tp and self.attn_tp.group_size > 1:
                error_msg = f"Lm_head's local tp is not supported by Attention tp > 1." \
                    f"Lm_head's tp_size is {self.lm_head_tp.group_size}. " \
                    f"Attention tp size is {self.attn_tp.group_size}. " \
                    f"Please unset the value of `lm_head_local_tp` in the config.json."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)

    def validate_dense_tp(self):
        if self.enable_dense_tp:
            if self.dense_tp.group_size > self.local_world_size and self.dense_tp.group_size != self.world_size:
                error_msg = f"Dense tp_size should be no greater than local world size, or equal to world size. " \
                    f"Dense tp_size is {self.dense_tp.group_size}. " \
                    f"World size is {self.world_size}. " \
                    f"Local world size is {self.local_world_size}. " \
                    f"Please check `dense_tp` in the config.json."
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)
            if self.world_size % self.dense_tp.group_size != 0:
                error_msg = f"World size should be multiple of dense tp_size. " \
                    f"World size is {self.world_size}. " \
                    f"Dense tp_size is {self.dense_tp.group_size}. Please check!"
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)
            if self.attn_tp.group_size > 1 and self.attn_tp.group_size >= self.dense_tp.group_size:
                error_msg = f"Dense tp is not supported by Attention tp >= Dense tp. " \
                    f"Attention tp_size is {self.attn_tp.group_size}. " \
                    f"Dense tp_size is {self.dense_tp.group_size}. Please check!"
                logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                raise ValueError(error_msg)

    def get_tp_group(self, module):
        module.num_group = self.world_size // module.group_size
        module.rank_per_group = []
        for i in range(module.num_group):
            ranks = range(i * module.group_size, (i + 1) * module.group_size)
            module.rank_per_group.append(list(ranks))
        module.current_group_id = self.get_current_group_id(module.rank_per_group, self.rank)
        module.rank = module.rank_per_group[module.current_group_id].index(self.rank)

    def get_dp_group(self, module):
        module.num_group = self.world_size // module.group_size
        module.rank_per_group = []
        for j in range(module.num_group):
            ranks = range(j, self.world_size, module.num_group)
            module.rank_per_group.append(list(ranks))
        module.current_group_id = self.get_current_group_id(module.rank_per_group, self.rank)
        module.rank = module.rank_per_group[module.current_group_id].index(self.rank)

    def get_prefix_cache_cp_group(self, module):
        module.num_group = self.world_size // module.group_size
        module.rank_per_group = []
        for j in range(module.num_group):
            ranks = range(j, self.world_size, module.num_group)
            module.rank_per_group.append(list(ranks))
        module.current_group_id = self.get_current_group_id(module.rank_per_group, self.rank)
        module.rank = self.rank

    def get_pp_group(self, module):
        self.get_tp_group(module.tp)
        module.num_group = self.world_size // module.group_size
        pp_groups = []
        for i in range(module.num_group):
            ranks = range(i, self.world_size, module.num_group)
            pp_groups.append(list(ranks))
        module.rank_per_group = pp_groups
        module.rank = self.rank // (module.tp.group_size * self.attn_dp.group_size)

    def has_attn_tp(self) -> bool:
        return self.attn_tp.group_size > 1

    def has_attn_o_proj_tp(self) -> bool:
        return self.attn_o_proj_tp.group_size > 1

    def has_dp(self) -> bool:
        return self.attn_dp.group_size > 1

    def has_mlp_tp(self) -> bool:
        return self.mlp_tp.group_size > 1
    
    def is_last_pp_rank(self):
        return self.pp.rank == self.pp.group_size - 1
 
    def is_first_pp_rank(self):
        return self.pp.rank == 0
    
    def has_pp(self):
        return self.pp.group_size > 1
    
    def prev_pp_rank(self):
        p = self.rank - self.pp.tp.group_size
        if p < 0:
            p = p + self.world_size
        return p
 
    def next_pp_rank(self):
        p = self.rank + self.pp.tp.group_size
        if p >= self.world_size:
            p = p - self.world_size
        return p
 
    def pp_layers(self, num_layers: int) -> List[int]:
        layers_per_pipeline_stage = num_layers // self.pp.group_size
        layers_range = range(self.pp.rank * layers_per_pipeline_stage,
                             (self.pp.rank + 1) * layers_per_pipeline_stage)
        return list(layers_range)
    
    def has_moe_tp(self) -> bool:
        return self.moe_tp.group_size > 1

    def has_moe_ep(self) -> bool:
        return self.moe_ep.group_size > 1
    
    def has_attn_inner_sp(self) -> bool:
        return self.attn_inner_sp.group_size > 1

    def has_attn_cp(self) -> bool:
        return self.attn_cp.group_size > 1

    def to_dict(self):
        # 开启lmhead后数据以dp策略形式输出
        if ENV.enable_dp_partition_up:
            # 仅支持A2场景，机器间dp形式输出
            if self.enable_lm_head_local_tp:
                self.lm_head_tp.domain = self.lm_head_tp.domain
            # 数据以attention部分的dp策略形式输出
            else:
                self.lm_head_tp.domain = self.attn_tp.domain
        # 若不开启lmhead后数据以cp策略形式输出，lmhead将走纯tp策略，输出完整数据
        else:
            self.lm_head_tp.domain = self.mlp_tp.domain

        parallel_dict = {
            "worldSize": self.world_size,
            "rank": self.rank,
            "localWorldSize": self.local_world_size,
            "hasAttnTp": self.has_attn_tp(),
            "attnTpRank": self.attn_tp.rank,
            "attnTpSize": self.attn_tp.group_size,
            "hasAttnOprojTp": self.has_attn_o_proj_tp(),
            "attnOprojTpRank": self.attn_o_proj_tp.rank,
            "attnOprojTpSize": self.attn_o_proj_tp.group_size,
            "hasAttnDp": self.has_dp(),
            "attnDpRank": self.attn_dp.rank,
            "attnDpSize": self.attn_dp.group_size,
            "hasMlpTp": self.has_mlp_tp(),
            "mlpTpRank": self.mlp_tp.rank,
            "mlpTpSize": self.mlp_tp.group_size,
            "hasMoeTp": self.has_moe_tp(),
            "moeTpRank": self.moe_tp.rank,
            "moeTpSize": self.moe_tp.group_size,
            "hasMoeEp": self.has_moe_ep(),
            "moeEpRank": self.moe_ep.rank,
            "moeEpSize": self.moe_ep.group_size,
            "lmHeadTpRank": self.lm_head_tp.rank if ENV.enable_dp_partition_up else self.mlp_tp.rank,
            "lmHeadTpSize": self.lm_head_tp.group_size if ENV.enable_dp_partition_up else self.mlp_tp.group_size,
            "attnTpDomain": self.attn_tp.domain,
            "attnOprojTpDomain": self.attn_o_proj_tp.domain,
            "attnDpDomain": self.attn_dp.domain,
            "mlpTpDomain": self.mlp_tp.domain,
            "moeTpDomain": self.moe_tp.domain,
            "moeEpDomain": self.moe_ep.domain,
            "lmHeadTpDomain": self.lm_head_tp.domain,
        }
        return parallel_dict

    def to_dict_v2(self):
        lmhead_tp = {}
        lmhead_dp = {}
        # 开启lmhead后数据以dp策略形式输出
        if ENV.enable_dp_partition_up:
            # 仅支持A2场景，机器间dp形式输出
            if self.enable_lm_head_local_tp:
                lmhead_tp = self.lm_head_tp.to_dict()
                lmhead_dp = self.lm_head_dp.to_dict()
            # 数据以attention部分的dp策略形式输出
            else:
                lmhead_tp = self.attn_tp.to_dict()
                lmhead_dp = self.attn_dp.to_dict()
        # 若不开启lmhead后数据以cp策略形式输出，lmhead将走纯tp策略，输出完整数据
        else:
            lmhead_tp = self.mlp_tp.to_dict()
            lmhead_dp = self.mlp_dp.to_dict()
        parallel_dict = {
            "worldSize": self.world_size,
            "rank": self.rank,
            "rankTableFile": self.rank_table_file,
            "localWorldSize": self.local_world_size,
            "lcclCommDomainLowerBound": self.lccl_comm_domain_lower_bound,
            "lcclCommDomainUpperBound": self.lccl_comm_domain_upper_bound,
            "wordEmbedTp": self.word_embed_tp.to_dict(),
            "wordEmbedDp": self.word_embed_dp.to_dict(),
            "attnTp": self.attn_tp.to_dict(),
            "attnDp": self.attn_dp.to_dict(),
            "attnInnerSp": self.attn_inner_sp.to_dict(),
            "attnCp": self.attn_cp.to_dict(),
            "attnPrefixcacheCp": self.attn_prefix_cache_cp.to_dict(),
            "attnOProjTp": self.attn_o_proj_tp.to_dict(),
            "attnOProjDp": self.attn_o_proj_dp.to_dict(),
            "mlpTp": self.mlp_tp.to_dict(),
            "mlpDp": self.mlp_dp.to_dict(),
            "moeTp": self.moe_tp.to_dict(),
            "moeEp": self.moe_ep.to_dict(),
            "moeEpIntraNode": self.moe_ep_intra_node.to_dict(),
            "moeEpInterNode": self.moe_ep_inter_node.to_dict(),
            "lmHeadTp": lmhead_tp,
            "lmHeadDp": lmhead_dp,
            "denseTp": self.dense_tp.to_dict(),
            "dynamicEplb": self.dynamic_eplb.to_dict()
        }
        if self.lwd_global_comm is not None:
            parallel_dict.update({"lwdGlobalComm": self.lwd_global_comm})
        return parallel_dict

    def set_lwd_global_comm(self, lwd_global_comm):
        self.lwd_global_comm = str(lwd_global_comm)