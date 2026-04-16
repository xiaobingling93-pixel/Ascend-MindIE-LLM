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

from enum import Enum

import ast
import ipaddress
import numpy as np
from mindie_llm.utils.log.logging import logger


# PADDING保证IPv4和IPv6兼容处理
IP_PADDING_VALUE = -1


class TaskType(Enum):
    PREFILL = "1"
    DECODE = "2"
    CLEAR = "3"
    ASSIGN_ROLE = "4"
    UNLINK_P_NODE = "5"
    D_NODE_TRANSFER = "10"
    P_NODE_TRANSFER = "11"
    MIX_INFER = "12"
    DELETE_HOST_KEYS = "13"


class TransferType(Enum):
    H2D = 0
    D2H = 1
    H2H_PULL = 2
    PREEMPT_SWAP = 3
    DELETE = 4
    PREEMPT_RECOMPUTE = 5


BLOCK_OP_SIZE = 3
transfer_op_map = {
    TransferType.H2D: "IBIS_H2D_BLOCK_OP_MEMPOOL",
    TransferType.D2H: "IBIS_D2H_BLOCK_OP_MEMPOOL",
    TransferType.H2H_PULL: "IBIS_H2H_BLOCK_OP_MEMPOOL",
    TransferType.PREEMPT_SWAP: "IBIS_PREEMPT_BLOCK_OP_MEMPOOL",
    TransferType.DELETE: "IBIS_CLEAR_BLOCK_OP_MEMPOOL",
    TransferType.PREEMPT_RECOMPUTE: "IBIS_RECOMPUTE_BLOCK_OP_MEMPOOL",
}


def get_ipv4_obj(ip_str, ip_field_name):
    try:
        ipv4 = ipaddress.IPv4Address(ip_str)
    except ipaddress.AddressValueError as e:
        raise ValueError(f"{ip_field_name}={ip_str} is invalid IPv4 address.") from e

    return ipv4


def get_ipv6_obj(ip_str, ip_field_name):
    try:
        ipv6 = ipaddress.IPv6Address(ip_str)
    except ipaddress.AddressValueError as e:
        raise ValueError(f"{ip_field_name}={ip_str} is invalid IPv6 address.") from e

    return ipv6


def get_ip_obj(ip_str, ip_field_name):
    """通过 IPv4 或 IPv6 字符串获取对应的IP对象"""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError as e:
        raise ValueError(f"{ip_field_name}={ip_str} is invalid IP address.") from e

    if ip_obj.version == 4:
        return get_ipv4_obj(ip_str, ip_field_name)
    elif ip_obj.version == 6:
        return get_ipv6_obj(ip_str, ip_field_name)
    else:
        raise ValueError(f"{ip_field_name}={ip_str} is unsupported IP version.")


def ipv4_to_list(ip_str):
    """将 IPv4 字符串转换为前 4 个为真实值，后 4 个为 IP_PADDING_VALUE 的列表"""
    parts = ip_str.split(".")
    if len(parts) != 4:
        raise ValueError(f"{ip_str} is invalid IPv4 format.")
    ip_list = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"IPv4 segment '{part}' is not a valid number.")
        num = int(part)
        if not (0 <= num <= 255):
            raise ValueError(f"IPv4 segment '{num}' out of range [0, 255].")
        ip_list.append(num)
    ip_list.extend([IP_PADDING_VALUE] * 4)

    return ip_list


def ipv6_to_list(ip_str):
    """将 IPv6 字符串解析为 8 个 16 位整数（不填充，长度必须为 8)"""
    try:
        ipv6 = ipaddress.IPv6Address(ip_str)
    except ipaddress.AddressValueError as e:
        raise ValueError(f"{ip_str} is invalid IPv6 address.") from e

    # 从 packed 字节中提取 8 个 16 位整数
    ip_list = []
    for i in range(8):
        value = (ipv6.packed[i * 2] << 8) + ipv6.packed[i * 2 + 1]
        ip_list.append(value)

    return ip_list


def ip_string_to_list(ip_str):
    """
    将 IPv4 或 IPv6 字符串转换为长度为 8 的整数列表。
    IPv4: 前 4 个为真实值，后 4 个为 IP_PADDING_VALUE (-1)
    IPv6: 8 个 16 位整数，无填充（本身就是 8 段）
    """
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError as e:
        raise ValueError(f"{ip_str} is invalid IP address.") from e

    if ip_obj.version == 4:
        return ipv4_to_list(ip_str)
    elif ip_obj.version == 6:
        return ipv6_to_list(ip_str)
    else:
        raise ValueError(f"{ip_str} is unsupported IP version.")


def ip_array_to_ipv4(ip_array):
    """将前 4 个为真实值, 后 4 个为 IP_PADDING_VALUE 的数组还原为 IPv4 字符串"""
    parts = []
    for i in range(4):
        num = ip_array[i]
        if not (0 <= num <= 255):
            raise ValueError(f"IPv4 segment '{num}' is out of range [0, 255].")
        parts.append(str(num))
    ipv4_str = ".".join(parts)

    return ipv4_str


def ip_array_to_ipv6(ip_array):
    """将 8 个整数的数组还原为 IPv6 字符串（自动压缩输出）"""
    byte_parts = []
    for num in ip_array:
        if not (0 <= num <= 0xFFFF):
            raise ValueError(f"IPv6 segment '{hex(num)}' is out of range [0, 0xFFFF].")
        byte_parts.append(num >> 8)
        byte_parts.append(num & 0xFF)
    ipv6_obj = ipaddress.IPv6Address(bytes(byte_parts))  # 自动压缩输出
    ipv6_str = str(ipv6_obj)

    return ipv6_str


def ip_array_to_string(ip_array):
    """
    将长度为 8 的整数数组还原为 IP 字符串
    规则：如果后 4 个元素全为 IP_PADDING_VALUE, 则视为 IPv4; 否则视为 IPv6
    """
    if len(ip_array) != 8:
        raise ValueError("ip_array must be an array of 8 integers.")

    # 检查是否符合 IPv4 模式 (后 4 个为 IP_PADDING_VALUE)
    if all(num == IP_PADDING_VALUE for num in ip_array[4:]):
        return ip_array_to_ipv4(ip_array)
    else:
        return ip_array_to_ipv6(ip_array)


def generate_lora_strings(request):
    lora_id: str = request.get_tensor_by_name("LORA_ID")
    return None if lora_id == "None" else lora_id


def generate_user_request_id_string(request):
    batch_input_id_strings_arr = np.array(request.get_tensor_by_name("USER_REQUEST_ID"), copy=False)
    try:
        batch_input_id_strings = list(map(lambda _: "".join(map(lambda __: chr(__), _)), batch_input_id_strings_arr))
    except Exception:
        batch_input_id_strings = [None]
    return batch_input_id_strings[0]


def generate_mem_pool_decisions(requests, transfer_operation):
    transfer_tensor_name = transfer_op_map.get(transfer_operation)
    if transfer_tensor_name is None:
        logger.error("transfer operation %s is not supported.", transfer_operation)
        return None
    if requests is None:
        raise ValueError("requests is not set.")
    tensor_data = requests[0].get_tensor_by_name(transfer_tensor_name)
    if not tensor_data:
        return None
    try:
        raw_bytes = bytes(tensor_data)
        decisions_str = raw_bytes.decode("utf-8").strip()
        decisions = np.array(ast.literal_eval(decisions_str))
        if transfer_operation == TransferType.H2H_PULL:
            decisions = decisions.reshape(-1, BLOCK_OP_SIZE + 1)
        else:
            decisions = decisions.reshape(-1, BLOCK_OP_SIZE)
        return decisions
    except Exception as e:
        logger.error("Generate %s decisions failed: %s", transfer_tensor_name, e)
        return None


def generate_dp_inst_id(inst_id, dp_size: int):
    dp_rank_to_id = list()
    for i in range(dp_size):
        dp_inst_id = int(inst_id) * 10000000 + i
        dp_rank_to_id.append(str(dp_inst_id))
    return dp_rank_to_id


def split_list_equally(lst, n):
    """
    将列表平均分割成n个子列表, 每个子列表包含相同数量的元素
    """
    if n <= 0:
        raise ValueError(f"Number of chunks {n} must be greater than 0")
    if len(lst) % n != 0:
        raise ValueError(f"Length {len(lst)} of the list cannot be divided evenly by {n}")

    chunk_size = len(lst) // n
    return [lst[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]
