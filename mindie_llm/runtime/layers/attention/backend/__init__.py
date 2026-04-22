# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from .abstract import AttentionBackend
from .fia_attention import FiaAttentionBackend
from .sparse_attention import SfaBackend


def get_attn_backend(
    use_mla=False,  # NOTE: Interfaces to be redesigned
    use_sfa=False,
) -> type[AttentionBackend]:
    if use_mla:
        raise NotImplementedError("MLA is not implemented.")
    elif use_sfa:
        return SfaBackend()
    else:
        return FiaAttentionBackend()
