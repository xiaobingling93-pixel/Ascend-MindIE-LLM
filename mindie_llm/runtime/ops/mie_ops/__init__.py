# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import importlib

from mindie_llm.runtime.utils.npu.device_utils import DeviceType, get_npu_node_info


def import_mie_ops_by_device():
    mie_ops_version_map = {
        DeviceType.ASCEND_910B: "mie_ops_ascend910b",
        DeviceType.ASCEND_910_93: "mie_ops_ascend910_93"
    }
    device_type = get_npu_node_info().get_device_type()
    if device_type not in mie_ops_version_map:
        raise EnvironmentError(f"Unsupported device type: {device_type} for mie_ops. "
                               f"{mie_ops_version_map.values()} are supported currently.")
    importlib.import_module(mie_ops_version_map[device_type])


import_mie_ops_by_device()
