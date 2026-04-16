# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


_global_attn_dict: dict | None = None


def update_global_attn_dict(prefix, attn):
    global _global_attn_dict
    if _global_attn_dict is None:
        _global_attn_dict = {}
    _global_attn_dict[prefix] = attn


def get_global_attn_dict():
    return _global_attn_dict


def clear_global_attn_dict():
    global _global_attn_dict
    _global_attn_dict.clear()


def flush_global_attn_dict(attns):
    global _global_attn_dict
    _global_attn_dict.clear()
    _global_attn_dict.update(attns)
