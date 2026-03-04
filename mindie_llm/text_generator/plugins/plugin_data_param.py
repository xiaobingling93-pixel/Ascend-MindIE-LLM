# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# 构建plugin_data_param类
# 其中存储q_len, mask, random_values_list, la_all_input_ids(仅LA惩罚采样需要), parallel_decoding_enable_flag(并行解码使能)
# all_input_ids用于LA惩罚采样
class PluginDataParam:
    def __init__(self):
        self.q_len = None
        self.mask = None
        # LA惩罚采样需要
        self.la_all_input_ids = None
        # LA和MD, 采样使用随机数种子
        self.random_values_list = []
        # mtp新增输入
        self.num_speculative_tokens = None
        # mtp小模型的输入input
        self.mtp_model_inputs = None
        # forward输入的hidden_states
        self.hidden_states = None