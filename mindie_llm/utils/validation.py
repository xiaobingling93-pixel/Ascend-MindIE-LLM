# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json

from typing import Any, List, Union, Dict, Optional
from enum import Enum


SAFE_STRING_LENGTH = 10 ** 6
SAFE_LIST_LENGTH = 10 ** 3

UPPER_SAFE_BLOCK_SIZE = 2 ** 10
LOWER_SAFE_BLOCK_SIZE = 1
UPPER_SAFE_NPU_MEM = 64
LOWER_SAFE_NPU_MEM = 1

UPPER_SAFE_BATCH_SIZE = 819200
LOWER_SAFE_BATCH_SIZE = 1
UPPER_SAFE_SEQUENCE_LENGTH = 10 ** 6
LOWER_SAFE_SEQUENCE_LENGTH = 1

LOWER_SAFE_REPETITION_PENALTY = 0
LOWER_SAFE_TEMPERATURE = 0
SAFE_TOP_K = 0
UPPER_SAFE_TOP_P = 1
LOWER_SAFE_TOP_P = 0
UPPER_SAFE_LOGPROBS = 20
LOWER_SAFE_LOGPROBS = 0
UPPER_SAFE_SEED = 2 ** 63 - 1
LOWER_SAFE_SEED = 0
UPPER_SAFE_BEAM_WIDTH = 8192
LOWER_SAFE_BEAM_WIDTH = 1
UPPER_SAFE_BEST_OF = 8192
LOWER_SAFE_BEST_OF = 1


class ValidationError(Exception):
    def __init__(self, param_name: str, detail: str) -> None:
        message = f'The parameter `{param_name}` is invalid: ' + detail
        super().__init__(message)


class InconsistencyError(ValidationError):
    def __init__(self, param_name: str, param_attr: str, other_attr: str) -> None:
        detail = f'The {param_attr} is not equal to {other_attr}.'
        super().__init__(param_name, detail)


class OutOfBoundsError(ValidationError):
    def __init__(self, param_name: str, limit_key: str, limit_value: Union[int, float, str]) -> None:
        detail = (f'It exceeds the safety limit `{limit_value}` as `{limit_key}`. If you believe this boundary value is'
                  f' unreasonable, please modify the boundary value under path `mindie_llm.utils.validation`.')
        super().__init__(param_name, detail)


class UnsupportedTypeError(ValidationError):
    def __init__(self, param_name: str, valid_type: str) -> None:
        detail = f'Its type is not supported, which should be `{valid_type}`.'
        super().__init__(param_name, detail)


def validate_list(param_name: str, param_list: List[Any]) -> None:
    if not isinstance(param_list, list):
        raise UnsupportedTypeError(param_name, 'List[Any]')
    if len(param_list) > SAFE_LIST_LENGTH:
        raise OutOfBoundsError(param_name, 'SAFE_LIST_LENGTH', SAFE_LIST_LENGTH)


def validate_string(param_name: str, param_str: str) -> None:
    if len(param_str) > SAFE_STRING_LENGTH:
        raise OutOfBoundsError(param_name, 'SAFE_STRING_LENGTH', SAFE_STRING_LENGTH)


class ParseType(int, Enum):
    TO_STR = 0
    TO_INT = 1
    TO_FLOAT = 2
    TO_BOOL = 3
    TO_JSON = 4


MODEL_CONFIG_KEY_TYPE = {
    "load_tokenizer": ParseType.TO_BOOL,
    "max_position_embeddings": ParseType.TO_INT,
    "max_sequence_length": ParseType.TO_INT,
    "max_seq_len": ParseType.TO_INT,
    "bos_token_id": ParseType.TO_INT,
    "eos_token_id": ParseType.TO_JSON,
    "pad_token_id": ParseType.TO_INT,
    "cpu_mem": ParseType.TO_INT,
    "npu_mem": ParseType.TO_INT,
    "block_size": ParseType.TO_INT,
    "temperature": ParseType.TO_FLOAT,
    "top_k": ParseType.TO_INT,
    "top_p": ParseType.TO_FLOAT,
    "typical_p": ParseType.TO_FLOAT,
    "do_sample": ParseType.TO_BOOL,
    "seed": ParseType.TO_INT,
    "repetition_penalty": ParseType.TO_FLOAT,
    "frequency_penalty": ParseType.TO_FLOAT,
    "presence_penalty": ParseType.TO_FLOAT,
    "watermark": ParseType.TO_BOOL,
    "length_penalty": ParseType.TO_FLOAT,
    'num_speculative_tokens': ParseType.TO_INT,
}


def parse_config(
    model_config: Dict[str, str],
    item_name: str,
    required: bool = False,
    parse_type: ParseType = ParseType.TO_STR,
    default_value: Optional[Any] = None
):
    value = model_config.get(item_name)
    if value is None:
        if required:
            raise ValueError(f"model_config: `{item_name}` is required, but not set")
        if default_value is not None:
            value = default_value
    elif parse_type == ParseType.TO_INT:
        value = int(value)
    elif parse_type == ParseType.TO_FLOAT:
        value = float(value)
    elif parse_type == ParseType.TO_BOOL:
        value = value.lower() if isinstance(value, str) else value
        value = value is True or value == 'true' or value == '1'
    elif parse_type == ParseType.TO_JSON:
        value = json.loads(value)
    return value
