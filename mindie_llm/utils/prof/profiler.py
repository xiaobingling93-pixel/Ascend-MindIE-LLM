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

try:
    import functools
    import numpy as np
    from ms_service_profiler import Profiler, Level
    from ms_service_profiler.mstx import service_profiler
    from ..log.logging import logger
    from ...utils.tensor import npu

    def no_error(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as err:
                logger.debug(f"[prof error] {err}")
                return None

        return wrapper

    @no_error
    def span_start(name, sync=False, domain="ModelExecute", level=Level.INFO, *args, **kwargs):
        if sync and service_profiler.is_enable(level):
            npu.current_stream().synchronize()
        return Profiler(level).domain(domain).span_start(name)

    @no_error
    def span_end(prof, sync=False, *args, **kwargs):
        if sync and prof and getattr(prof, "_enable", True):
            npu.current_stream().synchronize()
        if prof:
            prof.span_end()

    @no_error
    def span_req(prof, req_list):
        if prof and getattr(prof, "_enable", True) and req_list:
            return prof.res([{"rid": str(rid)} for rid in req_list])

        return prof

    @no_error
    def span_attr(prof, name, value):
        if prof and getattr(prof, "_enable", True):
            return prof.attr(name, value() if callable(value) else value)

        return prof

    @no_error
    def tensor_attr(tensor_value, statistics=True):
        if isinstance(tensor_value, tuple):
            return tensor_value
        if statistics:
            if tensor_value.numel() == 0:
                return {}
            tensor_record = {
                "min": tensor_value.min().item(),
                "max": tensor_value.max().item(),
                "mean": tensor_value.mean().item(),
                "first_10": [t.item() if hasattr(t, "item") else float(t) for t in tensor_value.flatten()[:10]],
                "shape": list(tensor_value.shape),
            }
            return tensor_record
        else:
            return tensor_value.tolist()

    def count_block(block_tables):
        if block_tables.ndim > 2:
            return np.count_nonzero(block_tables, axis=(1, 2)).tolist()
        else:
            return [int(x) for x in np.count_nonzero(block_tables, axis=-1)]

except ImportError:
    from enum import Enum

    class Level(int, Enum):
        INFO = 20
        L0 = 10
        L1 = 20
        L2 = 30

    class Profiler:
        def __init__(self, *args, **kwargs):
            pass

        def __getattribute__(self, name):
            if name != "empty_func":
                return self.empty_func
            else:
                return super().__getattribute__(name)

        def empty_func(self, *args, **kwargs):
            return self

    def span_start(*args, **kwargs):
        return 0

    def span_end(*args, **kwargs):
        pass

    def span_req(*args, **kwargs):
        pass

    def span_attr(*args, **kwargs):
        pass

    def count_block(*args, **kwargs):
        return 0

    def tensor_attr(tensor_value, statistics=True):
        return tensor_value
