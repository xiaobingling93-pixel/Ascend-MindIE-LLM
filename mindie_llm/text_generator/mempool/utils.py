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

import hashlib
import re
from typing import List
import torch


def flatten_tensors(nested_list: List) -> List[torch.Tensor]:
    flat_list = []
    if isinstance(nested_list, torch.Tensor):
        flat_list.append(nested_list)
    elif isinstance(nested_list, List):
        for item in nested_list:
            flat_list.extend(flatten_tensors(item))
    else:
        raise TypeError(f"Unsupported type in tensors list: {type(nested_list)}")

    return flat_list


def str_to_md5_hex(key: str) -> str:
    data = key.encode("utf-8")
    hash_bytes = hashlib.md5(data).digest()
    hash_hex = hash_bytes.hex()
    return hash_hex


def parse_global_segment_size(value) -> int:
    """
    Parse storage size strings with support for units: TB, GB, MB, KB, B
    """

    if isinstance(value, int):
        return value
    elif not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for global_segment_size: {type(value)}") from e

    cleaned_input = value.strip().lower()
    if not cleaned_input:
        raise ValueError("global segment size cannot be empty.")

    units = {
        "tb": 1024**4,  # 1 TB = 1024^4 bytes
        "gb": 1024**3,  # 1 GB = 1024^3 bytes
        "mb": 1024**2,  # 1 MB = 1024^2 bytes
        "kb": 1024,  # 1 KB = 1024 bytes
        "b": 1,  # 1 B = 1 byte
    }
    pattern = r"^([0-9]+(\.[0-9]+)?)\s*(tb|gb|mb|kb|b)?$"
    match = re.match(pattern, cleaned_input)

    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = float(match.group(1))
    unit = match.group(3) or "b"

    multiplier = units.get(unit, units["b"])
    return int(number_str * multiplier)
