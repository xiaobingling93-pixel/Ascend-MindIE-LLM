# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from abc import ABC, abstractmethod
import queue

from mindie_llm.utils.log.logging import logger


class ObjectPool(ABC):
    def __init__(self, size=500):
        if size < 0:
            raise ValueError(f"Pool size cannot be negative (got {size})")
        self._pool_size = size
        self._pool = queue.Queue()
        self._expand_pool()  # initialize the pool

    @abstractmethod
    def _create_object(self):
        """
        an interface to create a new object, subclasses should override it
        """
        return None

    @abstractmethod
    def _reset_object(self, obj):
        """
        an interface to reset a given object, subclasses should override it
        """
        pass

    def acquire(self):
        try:
            return self._pool.get(block=False)
        except queue.Empty:
            logger.info("Object pool is empty")
            self._expand_pool()  # in case of an empty pool
            logger.info("Object pool is expanded")
            return self._pool.get(block=False)

    def release(self, obj):
        try:
            self._reset_object(obj)  # reset before release
            self._pool.put(obj, block=False)
        except queue.Full as e:
            raise Exception("Object pool is full, object release failed") from e

    def _expand_pool(self):
        for _ in range(self._pool_size):  # expand by size
            obj = self._create_object()
            self._reset_object(obj)  # reset before adding to the pool
            self._pool.put(obj)
