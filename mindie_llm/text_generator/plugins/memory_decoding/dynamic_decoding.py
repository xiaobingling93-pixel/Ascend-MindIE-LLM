# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
VERY_SMALL_VALUE = 1e-6
DYNAMIC_LOWER_BS = 13
DYNAMIC_UPPER_BS = 19
INITIAL_BRANCH_LENGTH = 3
DELTA_BRANCH_LENGTH = 1
DYNAMIC_GAMMA_BS = 4


class DynamicBranch:
    def __init__(self, decode_length):
        self.hit_ratio = 0.0
        self.hit_ratio_filtered = 0.0
        self.alpha = 0.1
        self.hit_ratio_thd_lower = 0.01
        self.hit_ratio_thd_upper = 0.1
        self.hit_ratio_two = 0.0
        self.hit_ratio_filtered_two = 0.0
        self.alpha_two = 0.3
        self.hit_ratio_thd_lower_two = 0.001
        self.hit_ratio_thd_upper_two = 0.01
        self.max_decode_length = decode_length

    def dynamic_decoding(self, cache_ids, branch_length, last_accepted_token, last_decoding_len):
        decoding_lengths = []
        if len(cache_ids) > DYNAMIC_GAMMA_BS:
            sum_accept_len, sum_decoding_len = 0, 0
            for _, cid in enumerate(cache_ids):
                if cid in last_accepted_token:
                    sum_accept_len += last_accepted_token[cid]
                    sum_decoding_len += last_decoding_len[cid]
            self.hit_ratio = sum_accept_len / len(cache_ids) / (sum_decoding_len + VERY_SMALL_VALUE)

            if len(cache_ids) < DYNAMIC_LOWER_BS:
                hit_ratio_thd_lower = self.hit_ratio_thd_lower
                hit_ratio_thd_upper = self.hit_ratio_thd_upper
            elif len(cache_ids) > DYNAMIC_UPPER_BS:
                hit_ratio_thd_lower = 0.0
                hit_ratio_thd_upper = 1.0
                self.hit_ratio_filtered = 0.0
                branch_length = INITIAL_BRANCH_LENGTH
            else:
                self.alpha = 0.5
                hit_ratio_thd_lower = self.hit_ratio_thd_lower
                hit_ratio_thd_upper = self.hit_ratio_thd_upper

            self.hit_ratio_filtered = self.alpha * self.hit_ratio + (1 - self.alpha) * self.hit_ratio_filtered
            if self.hit_ratio_filtered < hit_ratio_thd_lower:
                branch_length = max(branch_length - DELTA_BRANCH_LENGTH, 0)
            elif self.hit_ratio_filtered > hit_ratio_thd_upper:
                branch_length = min(branch_length + DELTA_BRANCH_LENGTH, 16)
        else:
            sum_accept_len, sum_decoding_len = 0, 0
            for _, cid in enumerate(cache_ids):
                if cid in last_accepted_token:
                    sum_accept_len += last_accepted_token[cid]
                    sum_decoding_len += last_decoding_len[cid]
            self.hit_ratio_two = sum_accept_len / len(cache_ids) / (sum_decoding_len + VERY_SMALL_VALUE)

            self.hit_ratio_filtered_two = (
                self.alpha_two * self.hit_ratio_two + (1 - self.alpha_two) * self.hit_ratio_filtered_two
            )
            if self.hit_ratio_filtered_two < self.hit_ratio_thd_lower_two:
                branch_length = max(branch_length - DELTA_BRANCH_LENGTH, 0)
            elif self.hit_ratio_filtered_two > self.hit_ratio_thd_upper_two:
                branch_length = min(branch_length + DELTA_BRANCH_LENGTH, 15)
        branch_length = min(branch_length, self.max_decode_length)
        decoding_lengths = [branch_length] * len(cache_ids)
        return branch_length, decoding_lengths
