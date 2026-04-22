# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any

import torch.nn as nn


class CustomLayer(nn.Module):
    """Base class for custom layers.

    This class extends PyTorch's nn.Module to serve as an abstract base class
    for custom layers. It provides no implementation itself, only a forward method
    signature that all subclasses must implement.

    Args:
        None. This is a base class with no initialization arguments.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("`forward` function is not defined in the `CustomLayer` class.")
