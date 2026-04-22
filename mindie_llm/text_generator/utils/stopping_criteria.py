# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from functools import partial, reduce
from typing import List, Union
import numpy as np


def single_eos(output_ids_without_padding: np.ndarray, eos_token_id: int):
    if len(output_ids_without_padding) == 0:
        return False
    return output_ids_without_padding[-1] == eos_token_id


def continuous_eos(output_ids_without_padding: np.ndarray, eos_token_id: List[int]):
    eos_length = len(eos_token_id)
    if len(output_ids_without_padding) < eos_length:
        return False
    return np.all(output_ids_without_padding[-eos_length:] == eos_token_id)


def make_mixed_eos(eos_token_id: List[Union[int, List[int]]]):
    eos_funcs = []
    for eos in eos_token_id:
        if isinstance(eos, int):
            eos_funcs.append(partial(single_eos, eos_token_id=eos))
        else:
            eos_funcs.append(partial(continuous_eos, eos_token_id=eos))

    def mixed_eos(output_ids_without_padding):
        return reduce(lambda acc, f: acc or f(output_ids_without_padding), eos_funcs, False)

    return mixed_eos


def strings_eos(output_text, new_token_string, stop_strings: List[str], include_stop: bool = False):
    for stop_string in stop_strings:
        start_index = -len(new_token_string) - len(stop_string)
        stop_idx = output_text.find(stop_string, start_index)
        if stop_idx != -1:
            if include_stop:
                reversed_truncation_idx = stop_idx + len(stop_string) - len(output_text)
            else:
                reversed_truncation_idx = stop_idx - len(output_text)
            return reversed_truncation_idx
    return None
