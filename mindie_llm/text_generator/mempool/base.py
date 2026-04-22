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

from abc import abstractmethod
from typing import List, Union
import torch
from .factory import CreatePool


class MemPool:
    @classmethod
    def create_pool(cls, backend: str, config_path: str, role: str = "scheduler", **kwargs):
        return CreatePool(backend, config_path, role, **kwargs)

    @abstractmethod
    def exists(self, key: str, **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def put(
        self,
        keys: Union[str, List[str]],
        tensors: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
        **kwargs,
    ) -> List[bool]:
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        keys: Union[str, List[str]],
        tensors: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]],
        **kwargs,
    ) -> List[bool]:
        raise NotImplementedError

    def close(self) -> None:
        pass
