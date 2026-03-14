# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from mindie_llm.runtime.layers.linear.linear import ReplicatedLinear
from mindie_llm.runtime.layers.parameter import ModelWeightParameter
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.layers.quantization.ms_model_slim.w8a8sc import (
    W8A8SCLinearMethod,
    sparse_compressed_weight_loader,
)
from mindie_llm.runtime.utils.distributed import set_parallel_info_manager


class TestSparseCompressedWeightLoader(unittest.TestCase):
    """Test cases for sparse_compressed_weight_loader function"""

    def test_same_shape(self):
        """Test direct copy when shapes are the same"""
        param = ModelWeightParameter(torch.randn(100))
        loaded_weight = torch.randn(100)

        sparse_compressed_weight_loader(param, loaded_weight)

        self.assertTrue(torch.equal(param.data, loaded_weight))

    def test_different_shape(self):
        """Test recreating tensor when shapes are different"""
        param = ModelWeightParameter(torch.randn(50))
        original_device = param.data.device
        loaded_weight = torch.randn(100)

        sparse_compressed_weight_loader(param, loaded_weight)

        self.assertEqual(param.data.shape, (100,))
        self.assertTrue(torch.equal(param.data, loaded_weight))
        self.assertEqual(param.data.device, original_device)


class TestW8A8SCLinearMethod(unittest.TestCase):
    """Test cases for W8A8SCLinearMethod class"""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1
        set_parallel_info_manager(self.mock_parallel_info_manager)

        self.quant_config = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8SC,
            "layer.weight": QuantType.W8A8SC,
        })

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_create_weights(self):
        """Test create_weights method creates all required parameters"""
        layer = ReplicatedLinear(input_size=512, output_size=1024)
        method = W8A8SCLinearMethod()

        method.create_weights(
            layer=layer,
            input_size_per_partition=512,
            output_partition_sizes=[1024],
            bias=False,
            weight_dtype=torch.float16,
            bias_dtype=torch.float16,
        )

        # Verify all parameters are created
        self.assertTrue(hasattr(layer, 'weight'))
        self.assertTrue(hasattr(layer, 'input_scale'))
        self.assertTrue(hasattr(layer, 'input_offset'))
        self.assertTrue(hasattr(layer, 'deq_scale'))
        self.assertTrue(hasattr(layer, 'quant_bias'))
        self.assertTrue(hasattr(layer, 'index'))

        # Verify weights have weight_loader
        self.assertTrue(callable(layer.weight.weight_loader))
        self.assertTrue(callable(layer.index.weight_loader))

    def test_weight_loader_same_shape(self):
        """Test weight_loader handles weights with same shape"""
        layer = ReplicatedLinear(input_size=512, output_size=1024)
        method = W8A8SCLinearMethod()

        method.create_weights(
            layer=layer,
            input_size_per_partition=512,
            output_partition_sizes=[1024],
            bias=False,
            weight_dtype=torch.float16,
            bias_dtype=torch.float16,
        )

        # Simulate loading weight with same shape
        loaded_weight = torch.randn(100)
        layer.weight.weight_loader(layer.weight, loaded_weight)

        self.assertEqual(layer.weight.data.shape, (100,))

    def test_weight_loader_different_shape(self):
        """Test weight_loader handles weights with different shape"""
        layer = ReplicatedLinear(input_size=512, output_size=1024)
        method = W8A8SCLinearMethod()

        method.create_weights(
            layer=layer,
            input_size_per_partition=512,
            output_partition_sizes=[1024],
            bias=False,
            weight_dtype=torch.float16,
            bias_dtype=torch.float16,
        )

        # Initial shape is (1,)
        self.assertEqual(layer.weight.data.shape, (1,))

        # Load weight with different shape
        loaded_weight = torch.randn(100)
        layer.weight.weight_loader(layer.weight, loaded_weight)

        # Shape should be adjusted
        self.assertEqual(layer.weight.data.shape, (100,))

    def test_index_loader(self):
        """Test weight_loader for index"""
        layer = ReplicatedLinear(input_size=512, output_size=1024)
        method = W8A8SCLinearMethod()

        method.create_weights(
            layer=layer,
            input_size_per_partition=512,
            output_partition_sizes=[1024],
            bias=False,
            weight_dtype=torch.float16,
            bias_dtype=torch.float16,
        )

        # Load index
        loaded_index = torch.randn(50)
        layer.index.weight_loader(layer.index, loaded_index)

        self.assertEqual(layer.index.data.shape, (50,))


if __name__ == '__main__':
    unittest.main()
