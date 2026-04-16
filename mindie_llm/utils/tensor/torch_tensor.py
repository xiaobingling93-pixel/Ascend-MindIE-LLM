#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from mindie_llm.utils.tensor.llm_tensor import LLMBackend


class TorchBackend(LLMBackend):
    def __init__(self):
        import torch

        self._backend = torch

    def get_backend(self):
        return self._backend

    def get_op(self):
        return self._backend

    def get_npu(self):
        import importlib

        importlib.import_module("torch_npu")
        return self._backend.npu

    def get_hal(self):
        return self.get_npu()

    def ones(self, *args, **kwargs):
        return self.get_op().ones(*args, **kwargs)

    def equal(self, *args, **kwargs):
        return self.get_op().equal(*args, **kwargs)

    def repeat(self, value, size):
        return value.repeat(size)

    def softmax(self, *args, **kwargs):
        return self.get_op().softmax(*args, **kwargs)

    def shape(self, value, dim=None):
        return value.size(dim)

    def cumsum(self, *args, **kwargs):
        return self._backend.cumsum(*args, **kwargs)

    def gather(self, input_params, index, dim):
        return self._backend.gather(input_params, dim, index)

    def numpy(self, value):
        if value.dtype == self._backend.bfloat16:
            return value.cpu().to(self._backend.float16).numpy()
        return value.cpu().numpy()

    def where(self, condition, input_param=None, other=None):
        if input_param is None and other is None:
            return self._backend.where(condition)
        return self._backend.where(condition, input_param, other)

    def full(self, *args, **kwargs):
        return self.get_op().full(*args, **kwargs)

    def tensor(self, *args, **kwargs):
        return self._backend.tensor(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return self._backend.zeros(*args, **kwargs)

    def fill_diagonal(self, mask, fill_value):
        mask.fill_diagonal_(fill_value)
        return mask

    def to(self, value, device):
        return value.to(device)

    def get_device(self, value):
        return value.device

    def cpu(self, value):
        return value.cpu()

    def scatter(self, input_params, axis, index, src):
        input_params.scatter_(axis, index, src)
        return input_params

    def masked_fill(self, input_params, mask, value):
        input_params.masked_fill_(mask, value)
        return input_params

    def pad(self, input_tensor, pad, mode, value):
        return self._backend.nn.functional.pad(input_tensor, pad, mode, value)

    def cat(self, *args, **kwargs):
        return self.get_op().cat(*args, **kwargs)
