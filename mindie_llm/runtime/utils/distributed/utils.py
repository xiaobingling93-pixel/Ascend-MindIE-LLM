# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import time
import torch

from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.utils.helpers.safety.file import safe_open
from mindie_llm.runtime.utils.helpers.env import ENV


def get_device_from_ranktable(rank: int, rank_table: str) -> torch.device:
    """Selects and returns the NPU device for a given rank from a rank table file.

    Parses the provided rank table JSON file to locate the device ID associated
    with the specified rank. The rank table is expected to contain a "server_list"
    key, where each server has a list of devices, each with "rank_id" and "device_id".

    Args:
        rank (int): The global rank for which to find the corresponding NPU device.
        rank_table (str): Path to the rank table JSON file.

    Returns:
        torch.device: The NPU device (e.g., `torch.device("npu:2")`) assigned to the given rank.

    Raises:
        ValueError: If no device entry matches the provided rank in the rank table.
        FileNotFoundError: If the rank table file does not exist (propagated from file open).
        json.JSONDecodeError: If the rank table file is not valid JSON (propagated from json.load).
    """
    device_found_flag = False
    logger.info(f"Selecting device from rank table for rank {rank}.")
    with safe_open(rank_table, "r", encoding="utf-8") as device_file:
        data = json.load(device_file)

        for server in data["server_list"]:
            for device in server["device"]:
                if int(device["rank_id"]) == rank:
                    device_id = int(device["device_id"])
                    device = torch.device(f"npu:{device_id}")
                    device_found_flag = True
                    break
            if device_found_flag:
                break
    if not device_found_flag:
        raise ValueError(f"ERROR: Rank id is not in the rankTableFile, the input rank is {rank}.")
    return device


def even_divide(numerator: int, denominator: int) -> int:
    """Divides two integers and ensures the division is exact.

    Raises a ValueError if the numerator is not evenly divisible by the denominator.

    Args:
        numerator (int): The dividend.
        denominator (int): The divisor. Must be non-zero.

    Returns:
        int: The result of integer division (numerator // denominator).

    Raises:
        ValueError: If `denominator` is zero or if `numerator % denominator != 0`.
    """
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not evenly divisible by {denominator}.")
    return numerator // denominator


def set_device(rank: int, npu_id: int = None) -> torch.device:
    """Sets the NPU device for the current process based on rank and environment.

    If a rank table file is provided via environment variable `rank_table_file`,
    the device is determined by parsing that file. Otherwise, it defaults to
    `npu:{npu_id}` where `npu_id` defaults to `rank` if not specified.

    Args:
        rank (int): The global rank of the current process.
        npu_id (int): The NPU device ID to use. If None, defaults to `rank`.

    Returns:
        torch.device: The NPU device that was set.

    Raises:
        RuntimeError: If NPU device setting fails (propagated from `torch.npu.set_device`).
    """
    if npu_id is None:
        npu_id = rank
    rank_table = ENV.rank_table_file
    if rank_table:
        device = get_device_from_ranktable(rank, rank_table)
    else:
        device = torch.device(f"npu:{npu_id}")

    # Try to set device, it will retry 12 times when it failed.
    retry_max = 12
    for i in range(retry_max):
        try:
            torch.npu.set_device(device)
        except Exception as e:
            if i == retry_max - 1:
                err_msg = f"Set device {device} for rank {rank} fails, {str(e)}"
                logger.error(err_msg)
                raise RuntimeError(err_msg) from e

            warning_msg = (
                f"Set device {device} for rank {rank} fails."
                f"Now wait 5 seconds to retry setting, retry times: {i + 1} / 12"
            )
            logger.warning(warning_msg)
            time.sleep(5)  # Wait 5s to retry setting
            continue
        break
    logger.info(f"Device {device} has been set to rank {rank}.")
    return device
