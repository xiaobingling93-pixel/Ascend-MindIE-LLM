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
from unittest.mock import MagicMock, patch, Mock
import torch
import torch.distributed as dist

from mindie_llm.runtime.layers.linear.linear import (
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from mindie_llm.runtime.layers.quantization.unquantized import UnquantizedLinearMethod
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.quantization.ms_model_slim.w8a8 import (
    W8A8PerTensorLinearMethod,
    W8A8PerTokenLinearMethod,
    W8A8MixLinearMethod,
)
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType, InferenceMode
from mindie_llm.runtime.layers.parameter import BaseParameter, RowParameter, ColumnParameter, BiasParameter
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager, set_parallel_info_manager


class TestLinearBase(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.output_size = 20
        self.bias = True
        self.quant_config = MagicMock()
        self.prefix = "test_prefix"
        self.weight_dtype = torch.bfloat16

        self.mock_set_parallel_info_manager()

    def mock_set_parallel_info_manager(self):
        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 1
        self.mock_parallel_info.world_size = 1

        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def test_init_bias_false_skip_bias_add_true_error(self):
        """test bias=False conflict with skip_bias_add=True"""
        with self.assertRaises(ValueError) as context:
            LinearBase(
                input_size=self.input_size,
                output_size=self.output_size,
                bias=False,
                skip_bias_add=True
            )
        self.assertIn("Cannot set `bias` to False and `skip_bias_add` to True", str(context.exception))

    def test_create_weights_without_quant_config(self):
        with patch("mindie_llm.runtime.layers.linear.linear.UnquantizedLinearMethod") as mock_unquant_method_cls, \
                patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager",
                      return_value=self.mock_parallel_info):
            mock_quant_method = MagicMock(spec=QuantizationMethodBase)
            mock_unquant_method_cls.return_value = mock_quant_method

            linear = LinearBase(
                input_size=self.input_size,
                output_size=self.output_size,
                bias=self.bias,
                weight_dtype=self.weight_dtype,
                bias_dtype=self.weight_dtype,
                quant_config=None,
                prefix=self.prefix
            )

            mock_quant_method.create_weights.assert_called_once_with(
                layer=linear,
                input_size_per_partition=self.input_size,
                output_partition_sizes=[self.output_size],
                bias=self.bias,
                weight_dtype=self.weight_dtype,
                bias_dtype=self.weight_dtype,
                weight_loader=linear.weight_loader
            )

    def test_create_weights_with_quant_config(self):
        with patch("mindie_llm.runtime.layers.linear.linear.get_parallel_info_manager",
                   return_value=self.mock_parallel_info):
            mock_quant_method = MagicMock()
            self.quant_config.get_quant_method.return_value = mock_quant_method

            linear = LinearBase(
                input_size=self.input_size,
                output_size=self.output_size,
                bias=self.bias,
                weight_dtype=self.weight_dtype,
                bias_dtype=self.weight_dtype,
                quant_config=self.quant_config,
                prefix=self.prefix
            )

            mock_quant_method.create_weights.assert_called_once_with(
                layer=linear,
                input_size_per_partition=self.input_size,
                output_partition_sizes=[self.output_size],
                bias=self.bias,
                weight_dtype=self.weight_dtype,
                bias_dtype=self.weight_dtype,
                weight_loader=linear.weight_loader
            )


class TestReplicatedLinear(unittest.TestCase):
    """Test cases for ReplicatedLinear with UnquantizedLinearMethod."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_set_parallel_info_manager()

    def mock_set_parallel_info_manager(self):
        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 1
        self.mock_parallel_info.world_size = 1

        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_init_without_bias(self):
        """Test initialization without bias."""
        # Create mock parallel info manager
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            bias=False,
        )

        self.assertEqual(layer.input_size, 512)
        self.assertEqual(layer.output_size, 1024)
        self.assertFalse(layer.has_bias)
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
        self.assertIsNone(layer.bias)

    def test_init_with_bias(self):
        """Test initialization with bias."""
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            bias=True,
        )

        self.assertTrue(layer.has_bias)
        self.assertIsNotNone(layer.bias)
        self.assertEqual(layer.bias.data.shape, (1024,))

    def test_init_with_custom_dtypes(self):
        """Test initialization with custom weight and bias dtypes."""
        # Create mock parallel info manager
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            bias=True,
            weight_dtype=torch.float16,
            bias_dtype=torch.float32,
        )

        self.assertEqual(layer.weight_dtype, torch.float16)
        self.assertEqual(layer.bias_dtype, torch.float32)
        self.assertEqual(layer.weight.data.dtype, torch.float16)
        self.assertEqual(layer.bias.data.dtype, torch.float32)

    def test_weight_shape(self):
        """Test that weight shape is correct."""
        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
        )

        # For linear layer, weight shape should be [output_size, input_size]
        expected_shape = (1024, 512)
        self.assertEqual(layer.weight.data.shape, expected_shape)

    def test_weight_loader(self):
        """Test weight loading."""
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
        )

        loaded_weight = torch.randn(1024, 512)
        param = layer.weight

        # Mock the load_weight method
        with patch.object(param, 'load_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight=loaded_weight)

    @patch('torch.nn.functional.linear')
    def test_forward(self, mock_linear):
        """Test forward pass."""
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            bias=True,
        )

        # Initialize weight and bias
        layer.weight.data = torch.randn(1024, 512)
        layer.bias.data = torch.randn(1024)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 1024)

        output = layer.forward(x)

        # Verify linear was called
        mock_linear.assert_called_once()
        call_args = mock_linear.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.allclose(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # Verify output shape
        self.assertEqual(output.shape, (2, 3, 1024))

    def test_forward_with_return_bias(self):
        """Test forward pass with return_bias=True."""
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            bias=True,
            return_bias=True,
        )

        # Initialize weight and bias
        layer.weight.data = torch.randn(1024, 512)
        layer.bias.data = torch.randn(1024)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock the quant_method.apply
        with patch.object(layer.quant_method, 'apply') as mock_apply:
            mock_apply.return_value = torch.randn(2, 3, 1024)
            output, bias = layer.forward(x)

            mock_apply.assert_called_once_with(layer, x)
            self.assertEqual(output.shape, (2, 3, 1024))
            self.assertIsNone(bias)  # skip_bias_add is False by default

    def test_forward_with_skip_bias_add(self):
        """Test forward pass with skip_bias_add=True."""
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(
            input_size=512,
            output_size=1024,
            bias=True,
            skip_bias_add=True,
            return_bias=True,
        )

        # Initialize weight and bias
        layer.weight.data = torch.randn(1024, 512)
        layer.bias.data = torch.randn(1024)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock the quant_method.apply
        with patch.object(layer.quant_method, 'apply') as mock_apply:
            mock_apply.return_value = torch.randn(2, 3, 1024)
            output, bias = layer.forward(x)

            mock_apply.assert_called_once_with(layer, x)
            self.assertEqual(output.shape, (2, 3, 1024))
            torch.testing.assert_close(bias, layer.bias)  # Should return bias when skip_bias_add=True

    def test_init_error_bias_false_skip_bias_add_true(self):
        """Test initialization error when bias=False and skip_bias_add=True."""
        self.mock_set_parallel_info_manager()

        with self.assertRaises(ValueError) as context:
            ReplicatedLinear(
                input_size=512,
                output_size=1024,
                bias=False,
                skip_bias_add=True,
            )
        self.assertIn("bias", str(context.exception))
        self.assertIn("skip_bias_add", str(context.exception))

    def test_init_error_bias_false_return_bias_true(self):
        """Test initialization error when bias=False and return_bias=True."""
        # Create mock parallel info manager
        self.mock_set_parallel_info_manager()

        with self.assertRaises(ValueError) as context:
            ReplicatedLinear(
                input_size=512,
                output_size=1024,
                bias=False,
                return_bias=True,
            )
        self.assertIn("bias", str(context.exception))
        self.assertIn("return_bias", str(context.exception))


class TestRowParallelLinear(unittest.TestCase):
    """Test cases for RowParallelLinear."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_set_parallel_info_manager()

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def mock_set_parallel_info_manager(self):
        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.process_group = MagicMock()

        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 2

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_init(self):
        """Test initialization."""
        self.mock_set_parallel_info_manager()

        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            parallel_info=self.mock_parallel_info,
        )

        self.assertEqual(layer.input_size, 1024)
        self.assertEqual(layer.output_size, 512)
        self.assertEqual(layer.tp_rank, 0)
        self.assertEqual(layer.tp_size, 2)
        # input_size_per_partition should be divided
        self.assertEqual(layer.input_size_per_partition, 512)  # 1024 / 2

    def test_init_error_input_is_parallel_false(self):
        """Test initialization error when input_is_parallel=False."""
        self.mock_set_parallel_info_manager()

        with self.assertRaises(NotImplementedError):
            RowParallelLinear(
                input_size=1024,
                output_size=512,
                parallel_info=self.mock_parallel_info,
                input_is_parallel=False,
            )

    def test_init_error_reduce_false_with_bias(self):
        """Test initialization error when reduce_results=False and bias=True."""
        self.mock_set_parallel_info_manager()

        with self.assertRaises(ValueError) as context:
            RowParallelLinear(
                input_size=1024,
                output_size=512,
                parallel_info=self.mock_parallel_info,
                reduce_results=False,
                bias=True,
            )
        self.assertIn("reduce", str(context.exception))

    def test_weight_shape(self):
        """Test that weight shape is correct."""
        self.mock_set_parallel_info_manager()

        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            parallel_info=self.mock_parallel_info,
        )

        # Weight shape should be [output_size, input_size_per_partition]
        expected_shape = (512, 512)  # output_size=512, input_size_per_partition=512
        self.assertEqual(layer.weight.data.shape, expected_shape)

    def test_weight_loader_row_parameter(self):
        """Test weight loading with RowParameter."""
        self.mock_set_parallel_info_manager()

        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            parallel_info=self.mock_parallel_info,
        )

        param = RowParameter(torch.randn(512, 512))
        loaded_weight = torch.randn(512, 512)

        # Mock the load_row_parallel_weight method
        with patch.object(param, 'load_row_parallel_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight=loaded_weight, tp_rank=0)

    def test_weight_loader_base_parameter(self):
        """Test weight loading with BaseParameter."""
        self.mock_set_parallel_info_manager()

        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            parallel_info=self.mock_parallel_info,
        )

        param = BaseParameter(torch.randn(512))
        loaded_weight = torch.randn(512)

        # Mock the load_weight method
        with patch.object(param, 'load_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight=loaded_weight)

    @patch('torch.distributed.all_reduce')
    @patch('torch.nn.functional.linear')
    def test_forward_with_reduce(self, mock_linear, mock_all_reduce):
        """Test forward pass with reduce_results=True."""
        self.mock_set_parallel_info_manager()

        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            parallel_info=self.mock_parallel_info,
            reduce_results=True,
        )

        # Initialize weight
        layer.weight.data = torch.randn(512, 512)

        # Create input (already partitioned)
        x = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 512)

        output = layer.forward(x)

        # Verify linear was called
        mock_linear.assert_called_once()
        # Verify all_reduce was called
        mock_all_reduce.assert_called_once()
        self.assertEqual(output.shape, (2, 3, 512))

    @patch('torch.nn.functional.linear')
    def test_forward_without_reduce(self, mock_linear):
        """Test forward pass with reduce_results=False."""
        self.mock_set_parallel_info_manager()

        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            parallel_info=self.mock_parallel_info,
            reduce_results=False,
            bias=False,
        )

        # Initialize weight
        layer.weight.data = torch.randn(512, 512)

        # Create input (already partitioned)
        x = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 512)

        output = layer.forward(x)

        # Verify linear was called
        mock_linear.assert_called_once()
        # Should not call all_reduce
        self.assertEqual(output.shape, (2, 3, 512))

    def test_extra_repr(self):
        """Test extra_repr method."""
        self.mock_set_parallel_info_manager()
        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            parallel_info=self.mock_parallel_info,
        )

        repr_str = layer.extra_repr()
        self.assertIn("in_features=512", repr_str)
        self.assertIn("output_features=512", repr_str)
        self.assertIn("tp_rank=0", repr_str)
        self.assertIn("tp_size=2", repr_str)


class TestColumnParallelLinear(unittest.TestCase):
    """Test cases for ColumnParallelLinear."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock parallel info
        self.mock_set_parallel_info_manager()

    def mock_set_parallel_info_manager(self):
        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.process_group = MagicMock()

        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 2

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_init(self):
        """Test initialization."""
        # Create mock parallel info manager
        self.mock_set_parallel_info_manager()

        layer = ColumnParallelLinear(
            input_size=512,
            output_size=1024,
            parallel_info=self.mock_parallel_info,
        )

        self.assertEqual(layer.input_size, 512)
        self.assertEqual(layer.output_size, 1024)
        self.assertEqual(layer.tp_rank, 0)
        self.assertEqual(layer.tp_size, 2)
        # output_partition_sizes should be divided
        self.assertEqual(layer.output_partition_sizes, [512])  # 1024 / 2

    def test_init_error_gather_output_true(self):
        """Test initialization error when gather_output=True."""
        self.mock_set_parallel_info_manager()

        with self.assertRaises(NotImplementedError):
            ColumnParallelLinear(
                input_size=512,
                output_size=1024,
                parallel_info=self.mock_parallel_info,
                gather_output=True,
            )

    def test_weight_shape(self):
        """Test that weight shape is correct."""
        self.mock_set_parallel_info_manager()

        layer = ColumnParallelLinear(
            input_size=512,
            output_size=1024,
            parallel_info=self.mock_parallel_info,
        )

        # Weight shape should be [output_partition_size, input_size]
        expected_shape = (512, 512)  # output_partition_size=512, input_size=512
        self.assertEqual(layer.weight.data.shape, expected_shape)

    def test_weight_loader_column_parameter(self):
        """Test weight loading with ColumnParameter."""
        self.mock_set_parallel_info_manager()

        layer = ColumnParallelLinear(
            input_size=512,
            output_size=1024,
            parallel_info=self.mock_parallel_info,
        )

        param = ColumnParameter(torch.randn(512, 512))
        loaded_weight = torch.randn(512, 512)

        # Mock the load_column_parallel_weight method
        with patch.object(param, 'load_column_parallel_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight=loaded_weight, tp_rank=0)

    def test_weight_loader_base_parameter(self):
        """Test weight loading with BaseParameter."""
        self.mock_set_parallel_info_manager()

        layer = ColumnParallelLinear(
            input_size=512,
            output_size=1024,
            parallel_info=self.mock_parallel_info,
        )

        param = BaseParameter(torch.randn(512))
        loaded_weight = torch.randn(512)

        # Mock the load_weight method
        with patch.object(param, 'load_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight=loaded_weight)

    @patch('torch.nn.functional.linear')
    def test_forward(self, mock_linear):
        """Test forward pass."""
        self.mock_set_parallel_info_manager()

        layer = ColumnParallelLinear(
            input_size=512,
            output_size=1024,
            parallel_info=self.mock_parallel_info,
        )

        # Initialize weight
        layer.weight.data = torch.randn(512, 512)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 512)

        output = layer.forward(x)

        # Verify linear was called
        mock_linear.assert_called_once()
        # Output should be partitioned
        self.assertEqual(output.shape, (2, 3, 512))


class TestMergedColumnParallelLinear(unittest.TestCase):
    """Test cases for MergedColumnParallelLinear."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock parallel info
        self.mock_set_parallel_info_manager()

    def mock_set_parallel_info_manager(self):
        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.process_group = MagicMock()

        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 2

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_init(self):
        """Test initialization."""
        self.mock_set_parallel_info_manager()

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            parallel_info=self.mock_parallel_info,
        )

        self.assertEqual(layer.input_size, 512)
        self.assertEqual(layer.output_sizes, [256, 256])
        self.assertEqual(layer.output_size, 512)  # sum of output_sizes
        self.assertEqual(layer.output_partition_sizes, [128, 128])  # Each divided by tp_size

    def test_init_error_gather_output_true(self):
        """Test initialization error when gather_output=True."""
        self.mock_set_parallel_info_manager()

        with self.assertRaises(NotImplementedError):
            MergedColumnParallelLinear(
                input_size=512,
                output_sizes=[256, 256],
                parallel_info=self.mock_parallel_info,
                gather_output=True,
            )

    def test_init_error_output_sizes_not_divisible(self):
        """Test initialization error when output_sizes are not divisible by tp_size."""
        self.mock_set_parallel_info_manager()

        with self.assertRaises(ValueError) as context:
            MergedColumnParallelLinear(
                input_size=512,
                output_sizes=[255, 256],  # 255 is not divisible by 2
                parallel_info=self.mock_parallel_info,
            )
        self.assertIn("must be multiples", str(context.exception))

    def test_weight_shape(self):
        """Test that weight shape is correct."""
        self.mock_set_parallel_info_manager()

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            parallel_info=self.mock_parallel_info,
        )

        # Weight shape should be [sum(output_partition_sizes), input_size]
        expected_shape = (256, 512)  # sum([128, 128])=256, input_size=512
        self.assertEqual(layer.weight.data.shape, expected_shape)

    def test_weight_loader_column_parameter(self):
        """Test weight loading with ColumnParameter."""
        self.mock_set_parallel_info_manager()

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            parallel_info=self.mock_parallel_info,
        )

        param = ColumnParameter(torch.randn(256, 512))
        loaded_weight = torch.randn(256, 512)
        loaded_shard_id = 0

        # Mock the load_merged_column_weight method
        with patch.object(param, 'load_merged_column_weight') as mock_load:
            layer.weight_loader(param, loaded_weight, loaded_shard_id=loaded_shard_id)
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            torch.testing.assert_close(call_kwargs['loaded_weight'], loaded_weight)
            self.assertEqual(call_kwargs['tp_rank'], 0)
            self.assertEqual(call_kwargs['shard_offset'], 0)  # sum([0]) = 0
            self.assertEqual(call_kwargs['shard_size'], 128)  # 256 / 2

    def test_weight_loader_error_invalid_shard_id(self):
        """Test weight loading error with invalid shard_id."""
        self.mock_set_parallel_info_manager()

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            parallel_info=self.mock_parallel_info,
        )

        param = ColumnParameter(torch.randn(256, 512))
        loaded_weight = torch.randn(256, 512)

        # Invalid shard_id (exceeds output_sizes length)
        with self.assertRaises(ValueError) as context:
            layer.weight_loader(param, loaded_weight, loaded_shard_id=2)
        self.assertIn("exceeds the valid range", str(context.exception))

    @patch('torch.nn.functional.linear')
    def test_forward(self, mock_linear):
        """Test forward pass."""
        self.mock_set_parallel_info_manager()

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            parallel_info=self.mock_parallel_info,
        )

        # Initialize weight
        layer.weight.data = torch.randn(256, 512)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 256)

        output = layer.forward(x)

        # Verify linear was called
        mock_linear.assert_called_once()
        # Output should be partitioned
        self.assertEqual(output.shape, (2, 3, 256))

    def test_init_with_different_quant_types(self):
        """Test initialization when MergedColumnParallelLinear has multiple linear sub-layers with different quant types."""
        self.mock_set_parallel_info_manager()

        # Create mock quant config where two prefixes map to different quant types
        mock_quant_config = MagicMock()
        mock_quant_config.get_quant_type_by_weight_name = MagicMock(side_effect=[
            QuantType.W8A8,  # Quant type for first prefix
            QuantType.W8A8_DYNAMIC,  # Quant type for second prefix
        ])

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            prefix=["gate", "up"],
            quant_config=mock_quant_config,
            parallel_info=self.mock_parallel_info,
        )

        # Verify multiple independent linear_modules were created
        self.assertEqual(len(layer.linear_modules), 2)
        self.assertEqual(layer.prefix, ["gate", "up"])
        self.assertFalse(layer.return_bias)
        # Verify each linear_module has the correct output size
        self.assertEqual(layer.linear_modules[0].output_size, 256)
        self.assertEqual(layer.linear_modules[1].output_size, 256)

    def test_init_with_same_quant_types(self):
        """Test initialization when MergedColumnParallelLinear has multiple linear sub-layers with the same quant type."""
        self.mock_set_parallel_info_manager()

        # Create mock quant config where both prefixes map to the same quant type
        mock_quant_config = MagicMock()
        mock_quant_config.get_quant_type_by_weight_name = MagicMock(return_value=QuantType.W8A8)

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            prefix=["gate", "up"],
            quant_config=mock_quant_config,
            parallel_info=self.mock_parallel_info,
        )

        # Verify no independent linear_modules; traditional merged init is used
        self.assertEqual(len(layer.linear_modules), 0)
        self.assertEqual(layer.output_size, 512)  # sum of output_sizes

    @patch('torch.nn.functional.linear')
    def test_forward_with_multiple_linear_modules(self, mock_linear):
        """Test forward pass when MergedColumnParallelLinear has multiple linear_modules."""
        self.mock_set_parallel_info_manager()

        # Create mock quant config with different quant types per prefix
        mock_quant_config = MagicMock()
        mock_quant_config.get_quant_type_by_weight_name = MagicMock(side_effect=[
            QuantType.W8A8,
            QuantType.W8A8_DYNAMIC,
        ])

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            prefix=["gate", "up"],
            quant_config=mock_quant_config,
            parallel_info=self.mock_parallel_info,
        )

        # Verify multiple independent linear_modules were created
        self.assertEqual(len(layer.linear_modules), 2)

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock forward of each linear_module
        mock_output1 = torch.randn(2, 3, 256)
        mock_output2 = torch.randn(2, 3, 256)
        layer.linear_modules[0].forward = MagicMock(return_value=mock_output1)
        layer.linear_modules[1].forward = MagicMock(return_value=mock_output2)

        # Run forward
        output = layer.forward(x)

        # Verify output is the concatenation of sub-module outputs
        expected_output = torch.cat([mock_output1, mock_output2], dim=-1)
        torch.testing.assert_close(output, expected_output)
        # Verify forward was called on each linear_module
        layer.linear_modules[0].forward.assert_called_once_with(x)
        layer.linear_modules[1].forward.assert_called_once_with(x)

    @patch('torch.nn.functional.linear')
    def test_forward_with_multiple_linear_modules_return_bias(self, mock_linear):
        """Test forward when MergedColumnParallelLinear has multiple linear_modules and return_bias=True."""
        self.mock_set_parallel_info_manager()

        # Create mock quant config with different quant types per prefix
        mock_quant_config = MagicMock()
        mock_quant_config.get_quant_type_by_weight_name = MagicMock(side_effect=[
            QuantType.W8A8,
            QuantType.W8A8_DYNAMIC,
        ])

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            prefix=["gate", "up"],
            quant_config=mock_quant_config,
            parallel_info=self.mock_parallel_info,
            return_bias=True,
        )

        # Create input
        x = torch.randn(2, 3, 512)

        # Mock forward of each linear_module to return (output, bias)
        mock_output1 = torch.randn(2, 3, 256)
        mock_output2 = torch.randn(2, 3, 256)
        mock_bias = MagicMock()
        layer.linear_modules[0].forward = MagicMock(return_value=(mock_output1, mock_bias))
        layer.linear_modules[1].forward = MagicMock(return_value=(mock_output2, mock_bias))

        # Run forward
        output, bias = layer.forward(x)

        # Verify output is concatenation and bias is returned
        expected_output = torch.cat([mock_output1, mock_output2], dim=-1)
        torch.testing.assert_close(output, expected_output)
        self.assertEqual(bias, mock_bias)

    def test_extra_repr_with_multiple_linear_modules(self):
        """Test extra_repr when MergedColumnParallelLinear has multiple linear_modules."""
        self.mock_set_parallel_info_manager()

        # Create mock quant config with different quant types per prefix
        mock_quant_config = MagicMock()
        mock_quant_config.get_quant_type_by_weight_name = MagicMock(side_effect=[
            QuantType.W8A8,
            QuantType.W8A8_DYNAMIC,
        ])

        layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            prefix=["gate", "up"],
            quant_config=mock_quant_config,
            parallel_info=self.mock_parallel_info,
        )

        # Mock extra_repr of each linear_module
        layer.linear_modules[0].extra_repr = MagicMock(return_value="module1_repr")
        layer.linear_modules[1].extra_repr = MagicMock(return_value="module2_repr")

        # Get string representation
        repr_str = layer.extra_repr()

        # Verify representation includes both modules
        self.assertIn("module1_repr", repr_str)
        self.assertIn("module2_repr", repr_str)


class TestQKVParallelLinear(unittest.TestCase):
    """Test cases for QKVParallelLinear."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock parallel info
        self.mock_set_parallel_info_manager()

    def mock_set_parallel_info_manager(self):
        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.process_group = MagicMock()

        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 2

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_init(self):
        """Test initialization."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            total_num_kv_heads=4,
            parallel_info=self.mock_parallel_info,
        )

        self.assertEqual(layer.hidden_size, 512)
        self.assertEqual(layer.head_size, 64)
        self.assertEqual(layer.total_num_heads, 8)
        self.assertEqual(layer.total_num_kv_heads, 4)
        self.assertEqual(layer.num_heads, 4)  # 8 / 2
        self.assertEqual(layer.num_kv_heads, 2)  # 4 / 2
        self.assertEqual(layer.num_kv_head_replicas, 1)

    def test_init_with_default_kv_heads(self):
        """Test initialization with default kv_heads (None)."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            parallel_info=self.mock_parallel_info,
        )

        self.assertEqual(layer.total_num_kv_heads, 8)  # Should default to total_num_heads

    def test_weight_loader_q_proj(self):
        """Test weight loading for Q projection (shard_id=0)."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            total_num_kv_heads=4,
            parallel_info=self.mock_parallel_info,
        )

        param = ColumnParameter(torch.randn(256, 512))
        loaded_weight = torch.randn(256, 512)
        loaded_shard_id = 0  # Q projection

        # Mock the load_qkv_weight method
        with patch.object(param, 'load_qkv_weight') as mock_load:
            layer.weight_loader(param, loaded_weight, loaded_shard_id=loaded_shard_id)
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            torch.testing.assert_close(call_kwargs['loaded_weight'], loaded_weight)
            self.assertEqual(call_kwargs['tp_rank'], 0)
            self.assertEqual(call_kwargs['shard_id'], 0)
            self.assertEqual(call_kwargs['shard_offset'], 0)  # Q offset
            self.assertEqual(call_kwargs['shard_size'], 256)  # 4 * 64

    def test_weight_loader_k_proj(self):
        """Test weight loading for K projection (shard_id=1)."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            total_num_kv_heads=4,
            parallel_info=self.mock_parallel_info,
        )

        param = ColumnParameter(torch.randn(128, 512))
        loaded_weight = torch.randn(128, 512)
        loaded_shard_id = 1  # K projection

        # Mock the load_qkv_weight method
        with patch.object(param, 'load_qkv_weight') as mock_load:
            layer.weight_loader(param, loaded_weight, loaded_shard_id=loaded_shard_id)
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            self.assertEqual(call_kwargs['shard_id'], 1)
            self.assertEqual(call_kwargs['shard_offset'], 256)  # K offset after Q

    def test_weight_loader_v_proj(self):
        """Test weight loading for V projection (shard_id=2)."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            total_num_kv_heads=4,
            parallel_info=self.mock_parallel_info,
        )

        param = ColumnParameter(torch.randn(128, 512))
        loaded_weight = torch.randn(128, 512)
        loaded_shard_id = 2  # V projection

        # Mock the load_qkv_weight method
        with patch.object(param, 'load_qkv_weight') as mock_load:
            layer.weight_loader(param, loaded_weight, loaded_shard_id=loaded_shard_id)
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            self.assertEqual(call_kwargs['shard_id'], 2)
            self.assertEqual(call_kwargs['shard_offset'], 384)  # V offset after Q+K

    def test_weight_loader_error_invalid_shard_id(self):
        """Test weight loading error with invalid shard_id."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            parallel_info=self.mock_parallel_info,
        )

        param = ColumnParameter(torch.randn(256, 512))
        loaded_weight = torch.randn(256, 512)

        # Invalid shard_id
        with self.assertRaises(ValueError) as context:
            layer.weight_loader(param, loaded_weight, loaded_shard_id=3)
        self.assertIn("Invalid", str(context.exception))
        self.assertIn("loaded_shard_id", str(context.exception))

    def test_get_shard_offset_mapping(self):
        """Test _get_shard_offset_mapping method."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            total_num_kv_heads=4,
            parallel_info=self.mock_parallel_info,
        )

        self.assertEqual(layer._get_shard_offset_mapping(0), 0)  # Q
        self.assertEqual(layer._get_shard_offset_mapping(1), 256)  # K: 4 * 64
        self.assertEqual(layer._get_shard_offset_mapping(2), 384)  # V: (4 + 2) * 64
        self.assertEqual(layer._get_shard_offset_mapping("total"), 512)  # (4 + 2 + 2) * 64

    def test_get_shard_size_mapping(self):
        """Test _get_shard_size_mapping method."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            total_num_kv_heads=4,
            parallel_info=self.mock_parallel_info,
        )

        self.assertEqual(layer._get_shard_size_mapping(0), 256)  # Q: 4 * 64
        self.assertEqual(layer._get_shard_size_mapping(1), 128)  # K: 2 * 64
        self.assertEqual(layer._get_shard_size_mapping(2), 128)  # V: 2 * 64

    def test_output_partition_sizes(self):
        """Test output_partition_sizes."""
        self.mock_set_parallel_info_manager()

        layer = QKVParallelLinear(
            hidden_size=512,
            head_size=64,
            total_num_heads=8,
            total_num_kv_heads=4,
            parallel_info=self.mock_parallel_info,
        )

        # Should have 3 partitions: Q, K, V
        self.assertEqual(layer.output_partition_sizes, [256, 128, 128])


if __name__ == '__main__':
    unittest.main()