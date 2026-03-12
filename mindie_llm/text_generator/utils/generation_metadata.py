# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class GenerationParams:
    adapter_id: Optional[str] = None
    best_of: Optional[int] = None
    ignore_eos: Optional[bool] = None
    include_stop_str_in_output: Optional[bool] = None
    length_penalty: Optional[float] = None
    logprobs: Optional[List[float]] = None
    max_new_tokens: Optional[int] = None
    n: Optional[int] = None
    seed: Optional[int] = None
    skip_special_tokens: Optional[bool] = None
    stop_strings: Optional[List[str]] = None
    stop_token_ids: Optional[List[Union[int, List[int]]]] = None
    use_beam_search: Optional[bool] = None
    response_format: Optional[str] = None  # JSON 结构化输出约束 (response_format JSON 字符串)