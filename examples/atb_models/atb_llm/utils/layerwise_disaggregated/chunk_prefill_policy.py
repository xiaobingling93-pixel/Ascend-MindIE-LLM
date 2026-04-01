#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import acl
from atb_llm.utils.layerwise_disaggregated.cloud_cut_policy import CloudCutModelType


class ChunkPrefilPolicy():
    _instance = None 

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ChunkPrefilPolicy, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name_or_path='qwen', batch_p_num=1, moe_quantize=None):
        self.soc_name = acl.get_soc_name()
        if not hasattr(self, 'initialized'):
            self.model_type = self.__get_model_name(model_name_or_path)
            self.batch_p_num = batch_p_num
            self.multi_nodes_enable = False
            self.moe_quantize = moe_quantize
            self.cp_size = 1
            # the map means to {prefill len(K) : ratio list for chunk ([chunk_ratio] * chunk num)}
            self.ratio_list_default_edge_map = {125: [1] * 33, 63: [1] * 20, 31: [1] * 10, 15: [1] * 6, 7: [1] * 2}
            self.__ajust_prefill_chunk_map_for_diff_npu_soc_qwen()
            if self.model_type == CloudCutModelType.DEEP_SEEK:
                self.ratio_list_default_edge_map = {31: [1] * 20, 15: [1] * 5, 7: [1] * 2}
            # 当前认为边云侧的chunk ratio list是一致的; 其实边云可以不一样长, 但是必须保持云的每一段是边段的整数倍长
            # 如云: [10, 6], 边: [6, 4, 2, 2, 2], 也就是云10 = 边6+4; 云6 = 边2+2+2; 边/云的总数也得相等
            # 也就是云的第一段等于边的前两段之和, 云的第二段等于边段的后三段之和
            self.ratio_list_default_cloud_map = self.ratio_list_default_edge_map

    @staticmethod
    def split_long_seq_by_chunk_num(total_len, chunk_num):
        average_len = int(total_len / chunk_num)
        mod = total_len % chunk_num
        prefill_chunk_len_policy: list = ([average_len] * chunk_num if mod == 0 else 
            [average_len + 1] * mod + [average_len] * (chunk_num - mod)
        )
        
        return prefill_chunk_len_policy

    @staticmethod
    def split_long_seq_by_ratio(total_len, ratio_list, min_unit=2048):
        """
        按比例切分数据，最小单位为指定值，剩余补到最后一份
        :param total_len: 总数据长度（如32468）
        :param ratio_list: 比例列表（如[22,16,12,8,4,2]）
        :param min_unit: 最小单位（如2048，每份必为该值的整数倍）
        :return: 最终切分结果列表
        """
        # 1. 计算比例总和
        ratio_sum = sum(ratio_list)
        if ratio_sum == 0 or (total_len / min_unit <= (len(ratio_list) - 1)):
            raise ValueError(f"比例总和不能为0, 至少要有{len(ratio_list) - 1}个单位")

        # 2. 按比例计算每份基础长度（非整数）
        base_lengths = [total_len * r / ratio_sum for r in ratio_list]

        # 3. 调整为最小单位的整数倍（向下取min_unit）
        split_lengths = []
        for bl in base_lengths:
            # 向下取到最近的min_unit倍数
            rounded = (int(bl) // min_unit) * min_unit
            # 确保至少为最小单位（避免0）
            split_lengths.append(max(rounded, min_unit))

        # 4. 计算剩余长度，全部补到最后一份
        current_sum = sum(split_lengths)
        remaining = total_len - current_sum
        split_lengths[-1] += remaining

        return split_lengths
    
    @staticmethod
    def __get_model_name(model_name_or_path):
        model_name_or_path_ = model_name_or_path.lower()
        if 'qwen' in model_name_or_path_:
            return CloudCutModelType.QWEN
        elif 'deepseek' in model_name_or_path_ or 'ds' in model_name_or_path_:
            return CloudCutModelType.DEEP_SEEK
        return CloudCutModelType.QWEN
        
    def initialize(self, multi_nodes_enable, cp_size=1):
        self.multi_nodes_enable = multi_nodes_enable
        self.cp_size = cp_size
        self.__ajust_prefill_chunk_map_for_multi_nodes()
        
    def initialize_standard_card(self):
        self.ratio_list_default_edge_map = {125: [1] * 33, 64: [1] * 20, 32: [1] * 10, 16: [1] * 4, 8: [1] * 2}
        if self.model_type == CloudCutModelType.DEEP_SEEK:
            self.ratio_list_default_edge_map = {31.5: [1] * 20, 15.5: [1] * 5, 7.5: [1] * 2}
        self.ratio_list_default_cloud_map = self.ratio_list_default_edge_map
        
    def get_chunk_len_policy(self, prefill_seq_len, is_edge=True):
        prefill_chunk_list_map = self.ratio_list_default_edge_map if is_edge else self.ratio_list_default_cloud_map
        tmp_k_len = math.ceil(prefill_seq_len / 1024)
        tmp_chunk_len_list = [1, 1] # 默认两段, 1:1
        for key, value in prefill_chunk_list_map.items():
            if tmp_k_len >= key:
                tmp_chunk_len_list = value
                break
        unit = 1024 if is_edge else 2048

        if self.cp_size > 1:    # 开cp之后, 按比例切分
            policy_split_lens = self.split_long_seq_by_ratio(prefill_seq_len, tmp_chunk_len_list, unit)
        else:
            policy_split_lens = self.split_long_seq_by_chunk_num(prefill_seq_len, len(tmp_chunk_len_list))
        return policy_split_lens

    def __ajust_prefill_chunk_map_for_multi_nodes(self):
        # DS双机INT4以及单/双机INT8
        if self.model_type == CloudCutModelType.DEEP_SEEK:
            if self.moe_quantize == 'w4a8_dynamic' and self.multi_nodes_enable:
                self.ratio_list_default_edge_map = {31: [1] * 20, 15: [1] * 2, 7: [1] * 2}
                self.ratio_list_default_cloud_map = self.ratio_list_default_edge_map
            elif self.moe_quantize != 'w4a8_dynamic':
                # 双机DS INT8: 注意, 开cp之后会传入切分cp之后的长度; 这里是指cp切分之后的chunk ratio list(如32K开cp之后是16K)
                self.ratio_list_default_edge_map = {63: [6] + [2] * 13 + [1] * 32,
                                                    31: [6] + [2] * 13, 15: [6, 4, 2, 2, 2]}
                self.ratio_list_default_cloud_map = {63: [8] * 3 + [4] * 7 + [2] * 6, 31: [8, 8, 8, 4, 4], 15: [10, 6]}
            
    def __ajust_prefill_chunk_map_for_diff_npu_soc_qwen(self):
        if self.soc_name.startswith('Ascend910B4') and self.batch_p_num == 2:
            self.ratio_list_default_edge_map = {125: [1] * 33, 63: [1] * 20, 31: [1] * 10, 15: [1] * 4, 7: [1] * 2}
            return
