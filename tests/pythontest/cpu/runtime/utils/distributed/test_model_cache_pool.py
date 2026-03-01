# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import sys
from unittest.mock import MagicMock, patch, Mock
import unittest
import threading
from collections import namedtuple

import torch
# Isolate NPU dependencies for CPU execution

from mindie_llm.runtime.utils.distributed.model_cache_pool import (
    ModelCachePool, GROUP_KEY
)
from mindie_llm.runtime.utils.cache_spec import CacheGroupInfo, CacheType
sys.modules['torch_npu'] = MagicMock()


class TestModelCachePool(unittest.TestCase):
    def setUp(self):
        """Reset singleton state for test isolation"""
        ModelCachePool._instance = None
        ModelCachePool._lock = threading.Lock()

    def _create_mock_attn(self, shape=(16, 32, 128), ratio=1.0, cache_type=CacheType.TOKEN, 
                         dtype=torch.float16, acl_format=29):
        """Create mock attention object with required interface"""
        mock_attn = Mock()
        mock_attn.get_cache_spec.return_value = Mock(
            ratio=[ratio],
            type=[cache_type],
            dtype=[dtype],
            shape=[shape],
            format=acl_format
        )
        mock_attn.bind_model_cache = Mock()
        mock_attn.clear = Mock()
        return mock_attn

    def test_singleton_pattern(self):
        """Test singleton pattern"""
        instance1 = ModelCachePool()
        instance2 = ModelCachePool()
        self.assertIs(instance1, instance2)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    def test_initialize_and_update_caches_info(self, mock_get_attn_dict):
        """Test initialize and _update_caches_info"""
        mock_attn = self._create_mock_attn()
        mock_get_attn_dict.return_value = {'layer_0': mock_attn}
        
        pool = ModelCachePool()
        pool.initialize(Mock(), device='cpu', max_batch_size=8)
        
        self.assertTrue(pool.initialized)
        self.assertEqual(len(pool._groups), 1)
        self.assertEqual(len(pool._caches), 1)
        self.assertEqual(pool._max_batch_size, 8)

    def test_get_groups_info(self):
        """Test get_groups_info"""
        pool = ModelCachePool()
        pool._groups = {
            GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN): 
                CacheGroupInfo(ratio=1.0, block_size=16, type=CacheType.TOKEN, num_blocks=100)
        }
        groups = pool.get_groups_info()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].num_blocks, 100)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.torch_npu.empty_with_format')
    def test_warmup_device_cache(self, mock_empty, mock_get_attn_dict):
        """Test warmup_device_cache"""
        mock_attn = self._create_mock_attn()
        mock_get_attn_dict.return_value = {'layer_0': mock_attn}
        
        mock_tensor = MagicMock()
        mock_tensor.data_ptr.return_value = 12345
        mock_tensor.fill_ = MagicMock(return_value=mock_tensor)
        mock_empty.return_value = mock_tensor
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._max_batch_size = 8
        pool._device_caches_addrs = []
        pool._caches = [[(100, (16, 32, 128), 0, torch.float16, 29)]]
        pool._group_keys = [GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN)]
        pool._groups = {
            pool._group_keys[0]: CacheGroupInfo(
                ratio=1.0, block_size=16, type=CacheType.TOKEN,
                bytes_of_blocks=16*32*128*2,
                num_blocks=0
            )
        }
        pool._device = 'cpu'
        
        cache_size = pool.warmup_device_cache(device_mem=1024*1024*1024)
        
        self.assertGreater(cache_size, 0)
        self.assertEqual(len(pool._device_caches_addrs), 1)
        mock_attn.bind_model_cache.assert_called_once()

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    def test_cal_num_blocks_with_token(self, mock_get_attn_dict):
        """Test _cal_num_blocks with TOKEN cache type"""
        mock_get_attn_dict.return_value = {}
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._max_batch_size = 8
        pool._groups = {
            GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN): 
                CacheGroupInfo(
                    ratio=1.0, block_size=16, type=CacheType.TOKEN,
                    bytes_of_blocks=16*32*128*2,
                    num_blocks=0
                )
        }
        
        pool._cal_num_blocks(1024 * 1024 * 1024)
        
        group = pool._groups[GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN)]
        self.assertGreater(group.num_blocks, 0)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    def test_calculate_groups_info(self, mock_get_attn_dict):
        """Test calculate_groups_info"""
        mock_get_attn_dict.return_value = {}
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._max_batch_size = 8
        pool._groups = {
            GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN): 
                CacheGroupInfo(
                    ratio=1.0, block_size=16, type=CacheType.TOKEN,
                    bytes_of_blocks=16*32*128*2,
                    num_blocks=50
                )
        }
        
        groups_info = pool.calculate_groups_info(device_mem=1024*1024*1024)
        
        self.assertEqual(len(groups_info), 1)
        self.assertGreater(groups_info[0].num_blocks, 0)
        self.assertEqual(pool._groups[GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN)].num_blocks, 50)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.torch_npu.empty_with_format')
    def test_allocate_device_cache_flow(self, mock_empty, mock_get_attn_dict):
        """Test allocate_device_cache"""
        mock_attn = self._create_mock_attn()
        mock_get_attn_dict.return_value = {'layer_0': mock_attn}
        
        mock_tensor = MagicMock()
        mock_tensor.data_ptr.return_value = 54321
        mock_tensor.fill_ = MagicMock(return_value=mock_tensor)
        mock_empty.return_value = mock_tensor
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._max_batch_size = 8
        pool._device_caches_addrs = []
        pool._caches = [[(0, (16, 32, 128), 0, torch.float16, 2)]]
        pool._group_keys = [GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN)]
        pool._groups = {
            pool._group_keys[0]: CacheGroupInfo(
                ratio=1.0, block_size=16, type=CacheType.TOKEN,
                bytes_of_blocks=16*32*128*2,
                num_blocks=0
            )
        }
        pool._device = 'cpu'
        
        pool.allocate_device_cache(device_mem=1024*1024*1024, is_dmi=False)
        
        updated_num_blocks = pool._caches[0][0][0]
        self.assertGreater(updated_num_blocks, 0)
        mock_empty.assert_called_once()
        mock_attn.bind_model_cache.assert_called_once()

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    def test_clear_method(self, mock_get_attn_dict):
        """Test _clear method"""
        mock_attn1 = self._create_mock_attn()
        mock_attn2 = self._create_mock_attn()
        mock_get_attn_dict.return_value = {'layer_0': mock_attn1, 'layer_1': mock_attn2}
        
        pool = ModelCachePool()
        pool._clear()
        
        mock_attn1.clear.assert_called_once()
        mock_attn2.clear.assert_called_once()

    def test_get_caches_info(self):
        """Test get_caches_info"""
        pool = ModelCachePool()
        pool._caches = [[(100, (16, 32, 128), 0, torch.float16, 29)]]
        result = pool.get_caches_info()
        self.assertEqual(result, pool._caches)

    def test_get_caches_addrs(self):
        """Test get_caches_addrs"""
        pool = ModelCachePool()
        pool._device_caches_addrs = [[12345, 67890]]
        result = pool.get_caches_addrs()
        self.assertEqual(result, pool._device_caches_addrs)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    def test_cal_num_blocks_sliding_window_and_sequence(self, mock_get_attn_dict):
        """Test _cal_num_blocks with SLIDING_WINDOW and SEQUENCE types"""
        mock_get_attn_dict.return_value = {}
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._max_batch_size = 8
        pool._groups = {
            GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN): 
                CacheGroupInfo(
                    ratio=1.0, block_size=16, type=CacheType.TOKEN,
                    bytes_of_blocks=16*32*128*2,
                    num_blocks=0
                ),
            GROUP_KEY(ratio=1.0, block_size=32, type=CacheType.SLIDING_WINDOW): 
                CacheGroupInfo(
                    ratio=1.0, block_size=32, type=CacheType.SLIDING_WINDOW,
                    bytes_of_blocks=32*16*64*2,
                    num_blocks=0
                ),
            GROUP_KEY(ratio=1.0, block_size=64, type=CacheType.SEQUENCE): 
                CacheGroupInfo(
                    ratio=1.0, block_size=64, type=CacheType.SEQUENCE,
                    bytes_of_blocks=64*8*32*2,
                    num_blocks=0
                )
        }
        
        pool._cal_num_blocks(1024 * 1024 * 1024)
        
        token_group = pool._groups[GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN)]
        sliding_group = pool._groups[GROUP_KEY(ratio=1.0, block_size=32, type=CacheType.SLIDING_WINDOW)]
        sequence_group = pool._groups[GROUP_KEY(ratio=1.0, block_size=64, type=CacheType.SEQUENCE)]
        
        self.assertGreater(token_group.num_blocks, 0)
        self.assertEqual(sliding_group.num_blocks, 12 * 8 + 2)
        self.assertEqual(sequence_group.num_blocks, 8 + 2)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    def test_calculate_groups_info_with_sliding_window(self, mock_get_attn_dict):
        """Test calculate_groups_info with SLIDING_WINDOW"""
        mock_get_attn_dict.return_value = {}
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._max_batch_size = 8
        pool._groups = {
            GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN): 
                CacheGroupInfo(
                    ratio=1.0, block_size=16, type=CacheType.TOKEN,
                    bytes_of_blocks=16*32*128*2,
                    num_blocks=50
                ),
            GROUP_KEY(ratio=1.0, block_size=32, type=CacheType.SLIDING_WINDOW): 
                CacheGroupInfo(
                    ratio=1.0, block_size=32, type=CacheType.SLIDING_WINDOW,
                    bytes_of_blocks=32*16*64*2,
                    num_blocks=100
                )
        }
        
        groups_info = pool.calculate_groups_info(device_mem=1024*1024*1024)
        
        self.assertEqual(len(groups_info), 2)
        
        sliding_info = next(g for g in groups_info if g.type == CacheType.SLIDING_WINDOW)
        self.assertEqual(sliding_info.num_blocks, 12 * 8)
        
        self.assertEqual(pool._groups[GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN)].num_blocks, 50)
        self.assertEqual(pool._groups[GROUP_KEY(ratio=1.0, block_size=32, type=CacheType.SLIDING_WINDOW)].num_blocks, 100)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    def test_cal_num_blocks_oom_exception(self, mock_get_attn_dict):
        """Test OOM exception in _cal_num_blocks"""
        mock_get_attn_dict.return_value = {}
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._max_batch_size = 8
        pool._groups = {
            GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN): 
                CacheGroupInfo(
                    ratio=1.0, block_size=16, type=CacheType.TOKEN,
                    bytes_of_blocks=16*32*128*2,
                    num_blocks=0
                )
        }
        
        with self.assertRaises(RuntimeError) as cm:
            pool._cal_num_blocks(device_mem=100)
        
        self.assertIn("Npu out of memory", str(cm.exception))
        self.assertIn("negative number", str(cm.exception))

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.torch.empty')
    def test_create_aligned_tensor_nd_format(self, mock_torch_empty, mock_get_attn_dict):
        """Test _create_aligned_tensor with ND format (format=2)"""
        mock_tensor = MagicMock()
        mock_tensor.data_ptr.return_value = 1048576
        mock_tensor.__getitem__.return_value = mock_tensor
        mock_tensor.contiguous.return_value = mock_tensor
        mock_tensor.view.return_value = mock_tensor
        mock_torch_empty.return_value = mock_tensor
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._device = 'cpu'
        
        result = pool._create_aligned_tensor(
            target_shape=(100, 16, 32, 128),
            dtype=torch.float16,
            device='cpu',
            format=2
        )
        
        mock_torch_empty.assert_called_once()
        self.assertEqual(result, mock_tensor)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.torch_npu.empty_with_format')
    def test_create_aligned_tensor_nz_format(self, mock_empty_with_format, mock_get_attn_dict):
        """Test _create_aligned_tensor with NZ format (format != 2)"""
        mock_tensor = MagicMock()
        mock_empty_with_format.return_value = mock_tensor
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._device = 'cpu'
        
        result = pool._create_aligned_tensor(
            target_shape=(100, 16, 32, 128),
            dtype=torch.float16,
            device='cpu',
            format=29
        )
        
        mock_empty_with_format.assert_called_once()
        self.assertEqual(result, mock_tensor)

    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.get_global_attn_dict')
    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.torch.empty')
    @patch('mindie_llm.runtime.utils.distributed.model_cache_pool.torch_npu.empty_with_format')
    def test_allocate_device_cache_with_dmi(self, mock_npu_empty, mock_torch_empty, mock_get_attn_dict):
        """Test allocate_device_cache with is_dmi=True"""
        mock_attn = self._create_mock_attn()
        mock_get_attn_dict.return_value = {'layer_0': mock_attn}
        
        mock_tensor = MagicMock()
        mock_tensor.data_ptr.return_value = 54321
        mock_tensor.fill_ = MagicMock(return_value=mock_tensor)
        mock_torch_empty.return_value = mock_tensor
        
        pool = ModelCachePool()
        pool.initialized = True
        pool._max_batch_size = 8
        pool._device_caches_addrs = []
        pool._caches = [[(0, (16, 32, 128), 0, torch.float16, 2)]]
        pool._group_keys = [GROUP_KEY(ratio=1.0, block_size=16, type=CacheType.TOKEN)]
        pool._groups = {
            pool._group_keys[0]: CacheGroupInfo(
                ratio=1.0, block_size=16, type=CacheType.TOKEN,
                bytes_of_blocks=16*32*128*2,
                num_blocks=0
            )
        }
        pool._device = 'cpu'
        
        pool.allocate_device_cache(device_mem=1024*1024*1024, is_dmi=True)
        
        mock_torch_empty.assert_called_once()
        mock_npu_empty.assert_not_called()
        mock_attn.bind_model_cache.assert_called_once()


if __name__ == '__main__':
    unittest.main()
