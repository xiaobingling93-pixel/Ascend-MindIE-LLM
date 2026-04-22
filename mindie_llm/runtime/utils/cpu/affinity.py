# Copyright (c) Huawei Technologies Co., Ltd. 2023-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import threading
from itertools import accumulate
import psutil

from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.utils.helpers.env import ENV
from mindie_llm.runtime.utils.helpers.command_executor import execute_command
from mindie_llm.runtime.utils.npu.device_utils import get_npu_node_info
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager


_LSCPU_STRING: str | None = None
_LSCPU_LOCK = threading.Lock()


def _get_lscpu() -> str:
    global _LSCPU_STRING
    if _LSCPU_STRING is None:
        with _LSCPU_LOCK:
            if _LSCPU_STRING is None:
                _LSCPU_STRING = execute_command(["lscpu"])
    return _LSCPU_STRING


def _get_numa_info_by_pci(
    device2pcie: dict[int, str], keyword: str = "NUMAnode"
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """
    Infer NUMA node affinity for NPUs based on their PCIe bus IDs.

    Uses `lspci -s <PCIe> -vvv` to query the NUMA node associated with each PCIe device.
    Builds two mappings:
      - device → NUMA node
      - NUMA node → list of devices

    Args:
        device2pcie (dict[int, str]): Mapping from logical device ID to PCIe bus ID.
        keyword (str): Keyword to search in `lspci -vvv` output. Default is "NUMAnode".

    Returns:
        tuple[dict[int, int], dict[int, list[int]]]:
            - First: device_id → numa_id
            - Second: numa_id → list of device_ids

    Note:
        If hardware does not expose NUMA affinity via lspci, this may return empty dicts,
        prompting fallback to `_get_balanced_numa_info`.
    """
    device2numa: dict[int, int] = dict()  # key is device id, value is numa id
    numa2devices: dict[int, list[int]] = dict()  # key is numa id, value is device id list

    for device, pcie_no in device2pcie.items():
        numa_info = execute_command(["lspci", "-s", f"{pcie_no}", "-vvv"]).split("\n")
        for _ in numa_info:
            line = "".join(_.split())
            if line.startswith(keyword):
                numa_id = int(line[len(keyword) + 1 :])
                device2numa[device] = numa_id

                devices = numa2devices.get(numa_id, None)
                if devices is None:
                    numa2devices[numa_id] = list()

                numa2devices[numa_id].append(device)
                break

    return device2numa, numa2devices


def _get_balanced_numa_info(
    devices: list[int], keyword: str = "NUMAnode(s)"
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """
    Fallback NUMA assignment when PCIe-based NUMA detection is unavailable.

    Distributes visible NPU devices evenly across detected NUMA nodes using round-robin.
    NUMA node count is inferred from `lscpu` output (via "NUMAnode(s)" field).

    Args:
        devices (list[int]): Sorted list of logical NPU device IDs.
        keyword (str): Keyword in `lscpu` output to detect NUMA node count.
            Default is "NUMAnode(s)".

    Returns:
        tuple[dict[int, int], dict[int, list[int]]]:
            - device_id → numa_id (assigned evenly)
            - numa_id → list of device_ids

    Example:
        4 devices, 2 NUMA nodes → NUMA 0: [0,1], NUMA 1: [2,3]
    """
    # Optimize NUMA node detection.
    numa_nodes = 1
    numa_info = _get_lscpu().split("\n")
    for _ in numa_info:
        line = "".join(_.split())
        if keyword not in line:
            continue
        numa_nodes = int(line[-1])
        break

    # Distribute devices evenly to each NUMA node.
    device_per_numa, tail_device = divmod(len(devices), numa_nodes)
    device_count_per_numa_list = [device_per_numa + (i < tail_device) for i in range(numa_nodes)]

    # Assign devices in ascending, continuous order to each NUMA node.
    ends = list(accumulate(device_count_per_numa_list))
    starts = [0] + ends[:-1]

    numa2devices: dict[int, list[int]] = {ind: devices[start:end] for ind, (start, end) in enumerate(zip(starts, ends))}

    device2numa: dict[int, int] = {device: numa for numa, _devices in numa2devices.items() for device in _devices}

    return device2numa, numa2devices


def _get_numa_cpu_affinity(
    numa_ids: list[int], keyword1: str = "NUMAnode", keyword2: str = "CPU(s)"
) -> dict[int, list[int]]:
    """
    Retrieve CPU core lists for given NUMA nodes using `lscpu`.

    Parses `lscpu` output to extract CPU core ranges (e.g., "0-31,64-95") for each NUMA node,
    then expands them into explicit integer lists.

    Args:
        numa_ids (list[int]): List of NUMA node IDs to query.
        keyword1 (str): Prefix for NUMA identifier (default: "NUMAnode").
        keyword2 (str): Suffix indicating CPU set (default: "CPU(s)").

    Returns:
        dict[int, list[int]]: Mapping from NUMA node ID to list of CPU core indices.

    Raises:
        RuntimeError: If CPU range parsing fails (e.g., malformed `lscpu` output).

    Example:
        Input: numa_ids=[0,1]
        Output: {0: [0,1,...,31], 1: [32,...,63]}
    """

    cpu_info = _get_lscpu().split("\n")
    numa2cpus: dict[int, list[int]] = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    for _ in cpu_info:
        line = "".join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    warn_msg = "Cannot obtain CPU range for NUMA while executing `lscpu`."
                    logger.warning(warn_msg)
                    raise RuntimeError(warn_msg)

                ranges.extend(range(int(endpoints[0]), int(endpoints[1]) + 1))

            numa_id = int(split_info[0].replace(keyword1, "").replace(keyword2, ""))
            numa2cpus[numa_id] = ranges
    return numa2cpus


def bind_cpus(ratio: float = 0.5) -> None:
    """
    Bind the current process to a subset of CPU cores based on NPU-NUMA topology.

    This function:
      1. Resolves visible NPU devices (from ENV or auto-detection).
      2. Maps each NPU to its PCIe bus and then to a NUMA node (fallback to even distribution if needed).
      3. Retrieves all CPU cores on the NUMA node of the current rank's NPU.
      4. Splits those cores among all NPUs sharing the same NUMA node.
      5. Sets CPU affinity for the current process using `psutil`.

    CPU core count per device is determined by:
      - `ENV.cpu_binding_num` if set, or
      - `total_cores_on_numa * ratio // num_sharing_np us` otherwise.

    Args:
        ratio (float): Fraction of NUMA-local CPU cores to use if `CPU_BINDING_NUM` is not set.
            Must be in (0, 1]. Default is 0.5.

    Raises:
        ValueError: If `CPU_BINDING_NUM` is set but exceeds available cores or is negative.
        RuntimeError: If device or CPU topology cannot be resolved.

    Side Effects:
        - Modifies the CPU affinity of the current process via `psutil.Process().cpu_affinity()`.
        - Logs detailed binding info (rank, device, NUMA, CPUs) at INFO level.

    Note:
        This function assumes homogeneous NUMA topology and that all ranks call it
        with consistent `visible_devices`. It is designed for multi-process,
        multi-NPU training/inference on Ascend-based servers.
        'CPU_BINDING_NUM' sets cores per process. If unset, it's calculated via NUMA utilization ratio.
        E.g., ratio 0.5 on 64 cores uses 32 cores, distributed evenly among NPUs on the NUMA node.
    """
    devices = get_npu_node_info().visible_device_ids
    # Retrieve the mapping between NPU and PCIe.
    device2pcie = get_npu_node_info().get_pcie_info(devices)
    # Retrieve the mapping between NPU and NUMA based on PCIe information.
    device2numa, numa2devices = _get_numa_info_by_pci(device2pcie)
    if not device2numa or not numa2devices:
        device2numa, numa2devices = _get_balanced_numa_info(devices)
    # Retrieve CPU core allocation information corresponding to the used NUMA node.
    numa2cpus = _get_numa_cpu_affinity(list(numa2devices.keys()))

    # NPU ID for the current rank.
    cur_device = devices[get_parallel_info_manager().rank]
    # Retrieve the NUMA ID corresponding to the NPU.
    numa_id = device2numa.get(cur_device)

    # Retrieve information about NPUs sharing this NUMA node.
    shard_devices = numa2devices.get(numa_id)
    # Sort by NPU ID.
    shard_devices.sort()

    # Retrieve all CPU ID information on this NUMA node.
    all_cpus = numa2cpus.get(numa_id)
    logger.info(
        f"rank_id: {get_parallel_info_manager().rank}, device_id: {cur_device}, "
        f"numa_id: {numa_id}, shard_devices: {shard_devices}, cpus: {all_cpus}"
    )

    cpu_nums = len(all_cpus)
    # Calculate the number of cores allocated to the NPU sharing this NUMA node.
    if ENV.cpu_binding_num is None:
        cpu_num_per_device = int(cpu_nums * ratio // len(shard_devices))
    else:
        cpu_num_per_device = int(ENV.cpu_binding_num)
        if len(shard_devices) * cpu_num_per_device > cpu_nums:
            err_msg = (
                f"CPU num in numa {numa_id} to assign {cpu_num_per_device} for every device is not enough, "
                f"please decrease the value of CPU_BINDING_NUM!"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        if cpu_num_per_device < 0:
            err_msg = "CPU_BINDING_NUM should not be less than 0."
            logger.error(err_msg)
            raise ValueError(err_msg)

    # Retrieve the index information for this NPU.
    idx = shard_devices.index(cur_device)
    # Allocate the CPU IDs to be bound to this NPU.
    binding_cpus = [all_cpus[_] for _ in range(idx * cpu_num_per_device, (idx + 1) * cpu_num_per_device)]

    # cpu bind
    p = psutil.Process()
    p.cpu_affinity(binding_cpus)
    new_affinity = p.cpu_affinity()
    logger.info(f"process {p.pid}, new_affinity is {new_affinity}, cpu count {cpu_num_per_device}")
