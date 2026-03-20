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
from torch import nn

from mindie_llm.runtime.utils.loader.default_model_loader import DefaultModelLoader
from mindie_llm.runtime.layers.fused_moe.fused_moe import FusedMoE
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.linear.linear import MergedColumnParallelLinear
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType
from mindie_llm.runtime.utils.distributed import set_parallel_info_manager


class TestDefaultModelLoader(unittest.TestCase):
    """Test cases for DefaultModelLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = DefaultModelLoader()

    def test_init(self):
        """Test __init__ method."""
        self.assertEqual(self.loader._counter_before_loading_weights, 0.0)
        self.assertEqual(self.loader._counter_after_loading_weights, 0.0)
        self.assertEqual(self.loader._loaded_weight_names, [])
        self.assertIsNone(self.loader._weight_file_handler)

    def test_get_total_leaf_modules_single_module(self):
        """Test _get_total_leaf_modules with a single leaf module."""
        module = nn.Linear(10, 20)
        result = self.loader._get_total_leaf_modules(module)

        self.assertEqual(len(result), 1)
        self.assertIn("", result)
        self.assertEqual(result[""], module)

    def test_get_total_leaf_modules_nested_modules(self):
        """Test _get_total_leaf_modules with nested modules."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 30)

        model = NestedModel()
        result = self.loader._get_total_leaf_modules(model)

        self.assertEqual(len(result), 2)
        self.assertIn("layer1", result)
        self.assertIn("layer2", result)
        self.assertEqual(result["layer1"], model.layer1)
        self.assertEqual(result["layer2"], model.layer2)

    def test_get_total_leaf_modules_deeply_nested(self):
        """Test _get_total_leaf_modules with deeply nested modules."""

        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = nn.ModuleDict({
                    "linear1": nn.Linear(10, 20),
                    "linear2": nn.Linear(20, 30)
                })

        model = DeepModel()
        result = self.loader._get_total_leaf_modules(model)

        self.assertEqual(len(result), 2)
        self.assertIn("submodule.linear1", result)
        self.assertIn("submodule.linear2", result)

    def test_get_total_leaf_modules_with_prefix(self):
        """Test _get_total_leaf_modules with custom prefix."""
        module = nn.Linear(10, 20)
        result = self.loader._get_total_leaf_modules(module, prefix="test")

        self.assertEqual(len(result), 1)
        self.assertIn("test", result)
        self.assertEqual(result["test"], module)

    @patch('mindie_llm.runtime.utils.loader.default_model_loader.logger')
    @patch('mindie_llm.runtime.utils.loader.default_model_loader.WeightsFileHandler')
    @patch('mindie_llm.runtime.utils.loader.default_model_loader.get_parallel_info_manager')
    def test_load_weights(self, mock_get_parallel_info_manager, mock_weights_file_handler_class, mock_logger):
        """Test load_weights method."""
        mock_parallel_info = MagicMock()
        mock_parallel_info.rank = 0
        mock_get_parallel_info_manager.return_value = mock_parallel_info

        mock_weight_file_handler = MagicMock()
        mock_weights_file_handler_class.return_value = mock_weight_file_handler

        model = nn.Linear(10, 20)
        # Add config attribute since load_weights now accesses model.config.quantize
        model.config = MagicMock()
        model.config.quantize = None

        with patch.object(self.loader, '_load_modules') as mock_load_modules:
            self.loader.load_weights(model, "/fake/path")

            mock_weights_file_handler_class.assert_called_once_with("/fake/path", ".safetensors", None)
            mock_load_modules.assert_called_once_with(model)
            mock_weight_file_handler.release_file_handler.assert_called_once()
            mock_logger.info.assert_called_once()
            self.assertIsNotNone(self.loader._counter_before_loading_weights)
            self.assertIsNotNone(self.loader._counter_after_loading_weights)

    def test_load_modules_with_progress_simple_module(self):
        """Test _load_modules_with_progress with simple module."""
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_module = MagicMock()
        mock_module.named_parameters.return_value = [("weight", mock_param)]
        mock_module.prefix = None

        modules_dict = {"test": mock_module}
        mock_pbar = MagicMock()

        mock_weight_file_handler.get_tensor.return_value = torch.tensor([1.0])

        self.loader._load_modules_with_progress(modules_dict, mock_pbar)

        mock_weight_file_handler.get_tensor.assert_called_once_with("test.weight")
        mock_param.weight_loader.assert_called_once_with(mock_param, torch.tensor([1.0]))
        mock_pbar.update.assert_called_once_with(1)

    def test_load_modules_with_progress_module_with_prefix_list(self):
        """Test _load_modules_with_progress with module having prefix list."""
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_module = MagicMock()
        mock_module.prefix = ["prefix1", "prefix2"]
        mock_module.named_parameters.return_value = [("weight", mock_param)]

        modules_dict = {"test": mock_module}
        mock_pbar = MagicMock()

        mock_weight_file_handler.get_tensor.return_value = torch.tensor([1.0])

        self.loader._load_modules_with_progress(modules_dict, mock_pbar)

        # Should be called twice (once for each prefix)
        self.assertEqual(mock_weight_file_handler.get_tensor.call_count, 2)
        self.assertEqual(mock_param.weight_loader.call_count, 2)
        # First call with shard_id=0, second with shard_id=1
        mock_param.weight_loader.assert_any_call(mock_param, torch.tensor([1.0]), 0)
        mock_param.weight_loader.assert_any_call(mock_param, torch.tensor([1.0]), 1)
        mock_pbar.update.assert_called_once_with(1)

    def test_load_modules_with_progress_none_param(self):
        """Test _load_modules_with_progress with None parameter."""
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        mock_module = MagicMock()
        mock_module.named_parameters.return_value = []
        mock_module.prefix = None

        modules_dict = {"test": mock_module}
        mock_pbar = MagicMock()

        self.loader._load_modules_with_progress(modules_dict, mock_pbar)

        # Should not call get_tensor for None param
        mock_weight_file_handler.get_tensor.assert_not_called()
        mock_pbar.update.assert_called_once_with(1)

    def test_load_modules_with_progress_value_error_without_prefix(self):
        """Test _load_modules_with_progress with ValueError but no module prefix."""
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_module = MagicMock()
        mock_module.named_parameters.return_value = [("weight", mock_param)]
        mock_module.prefix = None

        modules_dict = {"test": mock_module}
        mock_pbar = MagicMock()

        # Raises ValueError
        mock_weight_file_handler.get_tensor.side_effect = ValueError("Weight file was not found")

        # Should raise the exception
        with self.assertRaises(ValueError):
            self.loader._load_modules_with_progress(modules_dict, mock_pbar)

    def test_load_single_prefix_module_raises_on_non_weight_file_error(self):
        """Test _load_single_prefix_module raises clear ValueError when get_tensor fails with non-weight-file error."""
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_module = MagicMock()
        mock_module.named_parameters.return_value = [("weight", mock_param)]
        mock_module.prefix = "layer"

        mock_weight_file_handler.get_tensor.side_effect = ValueError("Invalid tensor format")

        with self.assertRaises(ValueError) as ctx:
            self.loader._load_single_prefix_module(mock_module, "prefix")
        self.assertIn("Cannot load weights of prefix.weight", str(ctx.exception))
        self.assertIn("Invalid tensor format", str(ctx.exception.__cause__))

    @patch('mindie_llm.runtime.utils.loader.default_model_loader.get_parallel_info_manager')
    def test_load_modules_with_progress_merged_column_linear_multiple_modules(
        self, mock_get_parallel_info_manager
    ):
        """Test _load_modules_with_progress loads weights and processes quant for MergedColumnParallelLinear with multiple linear_modules."""
        mock_get_parallel_info_manager.return_value = MagicMock(rank=0, world_size=2)
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler
        # Return correctly shaped tensors: weight (128, 512) and bias (128) per partition

        def get_tensor_side_effect(name):
            if "weight" in name:
                return torch.randn(128, 512)
            return torch.randn(128)
        mock_weight_file_handler.get_tensor.side_effect = get_tensor_side_effect

        mock_parallel_info = MagicMock()
        mock_parallel_info.rank = 0
        mock_parallel_info.group_size = 2
        mock_parallel_info.process_group = MagicMock()

        mock_quant_config = MagicMock()
        mock_quant_config.get_quant_type_by_weight_name = MagicMock(side_effect=[
            QuantType.W8A8,
            QuantType.W8A8_DYNAMIC,
        ])
        mock_quant_method = MagicMock(spec=QuantizationMethodBase)
        mock_quant_method.process_weights_after_loading = MagicMock()
        mock_quant_config.get_quant_method = MagicMock(return_value=mock_quant_method)

        merged_layer = MergedColumnParallelLinear(
            input_size=512,
            output_sizes=[256, 256],
            prefix=["gate", "up"],
            quant_config=mock_quant_config,
            parallel_info=mock_parallel_info,
        )
        self.assertEqual(len(merged_layer.linear_modules), 2)

        modules_dict = {"mlp": merged_layer}
        mock_pbar = MagicMock()
        self.loader._load_modules_with_progress(modules_dict, mock_pbar)

        # _load_single_prefix_module is invoked for each (prefix, linear_module) pair
        self.assertGreaterEqual(mock_weight_file_handler.get_tensor.call_count, 4)  # 2 modules * (weight + bias)
        # process_weights_after_loading should be called on each linear_module
        self.assertEqual(mock_quant_method.process_weights_after_loading.call_count, 2)
        mock_quant_method.process_weights_after_loading.assert_any_call(merged_layer.linear_modules[0])
        mock_quant_method.process_weights_after_loading.assert_any_call(merged_layer.linear_modules[1])
        mock_pbar.update.assert_called_once_with(1)

    def test_load_modules_with_progress_fused_moe(self):
        """Test _load_modules_with_progress with FusedMoE module."""
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_module = MagicMock(spec=FusedMoE)
        mock_module.named_parameters.return_value = [("weight", mock_param)]
        mock_module.prefix = None
        mock_module.weight_loader = MagicMock()
        mock_module.expert_list = ["expert0"]
        mock_module.suffix = ["linear"]
        mock_module.weight_list = ["weight"]  # Add missing attribute
        mock_module.get_weight_components_suffix = MagicMock(return_value=["weight"])

        modules_dict = {"test": mock_module}
        mock_pbar = MagicMock()

        mock_weight_file_handler.get_tensor.return_value = torch.tensor([1.0])

        self.loader._load_modules_with_progress(modules_dict, mock_pbar)

        mock_module.weight_loader.assert_called_once_with(torch.tensor([1.0]), "expert0", "linear", "weight")
        mock_pbar.update.assert_called_once_with(1)

    def test_load_modules_with_progress_with_quant_method(self):
        """Test _load_modules_with_progress with quantization method."""
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_module = MagicMock()
        mock_module.named_parameters.return_value = [("weight", mock_param)]
        mock_module.prefix = None

        mock_quant_method = MagicMock(spec=QuantizationMethodBase)
        mock_quant_method.process_weights_after_loading = MagicMock()
        mock_module.quant_method = mock_quant_method

        modules_dict = {"test": mock_module}
        mock_pbar = MagicMock()

        mock_weight_file_handler.get_tensor.return_value = torch.tensor([1.0])

        self.loader._load_modules_with_progress(modules_dict, mock_pbar)

        mock_quant_method.process_weights_after_loading.assert_called_once_with(mock_module)
        mock_pbar.update.assert_called_once_with(1)

    def test_load_modules_with_progress_without_quant_method(self):
        """Test _load_modules_with_progress without quantization method."""
        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_module = MagicMock()
        mock_module.named_parameters.return_value = [("weight", mock_param)]
        mock_module.prefix = None
        # No quant_method attribute

        modules_dict = {"test": mock_module}
        mock_pbar = MagicMock()

        mock_weight_file_handler.get_tensor.return_value = torch.tensor([1.0])

        # Should not raise error
        self.loader._load_modules_with_progress(modules_dict, mock_pbar)

        mock_pbar.update.assert_called_once_with(1)

    @patch('mindie_llm.runtime.utils.loader.default_model_loader.tqdm')
    @patch('mindie_llm.runtime.utils.loader.default_model_loader.get_parallel_info_manager')
    def test_load_modules(self, mock_get_parallel_info_manager, mock_tqdm_class):
        """Test _load_modules method."""
        mock_parallel_info = MagicMock()
        mock_parallel_info.rank = 0
        mock_get_parallel_info_manager.return_value = mock_parallel_info

        mock_pbar = MagicMock()
        mock_tqdm_class.return_value = mock_pbar

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 30)
        )

        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        with patch.object(self.loader, '_load_modules_with_progress') as mock_load_modules_with_progress:
            self.loader._load_modules(model)

            mock_tqdm_class.assert_called_once()
            mock_load_modules_with_progress.assert_called_once()
            mock_pbar.close.assert_called_once()

    @patch('mindie_llm.runtime.utils.loader.default_model_loader.tqdm')
    @patch('mindie_llm.runtime.utils.loader.default_model_loader.get_parallel_info_manager')
    def test_load_modules_disables_progress_bar_for_non_rank0(self, mock_get_parallel_info_manager, mock_tqdm_class):
        """Test _load_modules disables progress bar for non-rank 0."""
        mock_parallel_info = MagicMock()
        mock_parallel_info.rank = 1  # Non-zero rank
        mock_get_parallel_info_manager.return_value = mock_parallel_info

        mock_pbar = MagicMock()
        mock_tqdm_class.return_value = mock_pbar

        model = nn.Linear(10, 20)

        mock_weight_file_handler = MagicMock()
        self.loader._weight_file_handler = mock_weight_file_handler

        with patch.object(self.loader, '_load_modules_with_progress'):
            self.loader._load_modules(model)

            # Check that disable parameter was set to True (rank != 0)
            call_kwargs = mock_tqdm_class.call_args[1]
            self.assertTrue(call_kwargs.get('disable', False))

    @patch('mindie_llm.runtime.utils.loader.default_model_loader.get_parallel_info_manager')
    @patch('mindie_llm.runtime.utils.loader.default_model_loader.WeightsFileHandler')
    def test_load_weights_passes_quantize_to_handler(self, mock_handler_class, mock_get_parallel_info):
        """Test that load_weights passes quantize from config to WeightsFileHandler."""
        mock_parallel_info = MagicMock()
        mock_parallel_info.rank = 0
        mock_get_parallel_info.return_value = mock_parallel_info

        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler

        # Create a mock model with quantize in config
        model = nn.Linear(10, 20)
        model.config = MagicMock()
        model.config.quantize = 'w8a8sc'

        with patch.object(self.loader, '_load_modules'):
            self.loader.load_weights(model, "/fake/model/path")

        # Verify WeightsFileHandler was called with quantize parameter
        mock_handler_class.assert_called_once_with("/fake/model/path", ".safetensors", 'w8a8sc')

    @patch('mindie_llm.runtime.utils.loader.default_model_loader.check_and_reuse_global_param_dict')
    @patch('mindie_llm.runtime.utils.loader.default_model_loader.get_weight_mapper_cls')
    def test_weight_name_mapping_for_w8a8sc_quantize(self, mock_get_weight_mapper_cls, mock_check_reuse):
        """Test that W8A8SC quantize config uses weight name mapping."""
        mock_check_reuse.return_value = False  # Don't skip loading
        mock_handler = MagicMock()
        mock_handler.get_tensor.return_value = torch.randn(20, 10)
        self.loader._weight_file_handler = mock_handler

        # Mock the mapper class
        mock_mapper_cls = MagicMock()
        mock_mapper_cls.map_model_to_weight.return_value = "transformer.h.0.attn.c_attn"
        mock_get_weight_mapper_cls.return_value = mock_mapper_cls

        # Create a mock layer
        mock_layer = MagicMock()
        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_layer.named_parameters.return_value = [("weight", mock_param)]
        mock_layer.prefix = None

        # Create a mock model with w8a8sc quantize
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantize = 'w8a8sc'

        modules_dict = {"model.layers.0.self_attn.qkv_proj": mock_layer}

        from tqdm.auto import tqdm
        pbar = tqdm(total=1, disable=True)

        self.loader._load_modules_with_progress(modules_dict, pbar, mock_model)

        # Verify the mapper was called to convert the name
        mock_mapper_cls.map_model_to_weight.assert_called_once_with("model.layers.0.self_attn.qkv_proj")
        # Verify get_tensor was called with the mapped name
        mock_handler.get_tensor.assert_called_once_with("transformer.h.0.attn.c_attn.weight")

    @patch('mindie_llm.runtime.utils.loader.default_model_loader.check_and_reuse_global_param_dict')
    @patch('mindie_llm.runtime.utils.loader.default_model_loader.get_weight_mapper_cls')
    def test_weight_name_mapping_skipped_for_non_w8a8sc_quantize(self, mock_get_weight_mapper_cls, mock_check_reuse):
        """Test that non-W8A8SC quantize config does not use weight name mapping."""
        mock_check_reuse.return_value = False  # Don't skip loading
        mock_handler = MagicMock()
        mock_handler.get_tensor.return_value = torch.randn(20, 10)
        self.loader._weight_file_handler = mock_handler

        # Mock the mapper class to return None (non-W8A8SC config)
        mock_get_weight_mapper_cls.return_value = None

        # Create a mock layer
        mock_layer = MagicMock()
        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_layer.named_parameters.return_value = [("weight", mock_param)]
        mock_layer.prefix = None

        # Create a mock model with float quantize (non-w8a8sc)
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantize = 'float'

        modules_dict = {"model.layers.0.self_attn.qkv_proj": mock_layer}

        from tqdm.auto import tqdm
        pbar = tqdm(total=1, disable=True)

        self.loader._load_modules_with_progress(modules_dict, pbar, mock_model)

        # Verify get_tensor was called with the original name (no mapping applied)
        mock_handler.get_tensor.assert_called_once_with("model.layers.0.self_attn.qkv_proj.weight")

    @patch('mindie_llm.runtime.utils.loader.default_model_loader.check_and_reuse_global_param_dict')
    @patch('mindie_llm.runtime.utils.loader.default_model_loader.get_weight_mapper_cls')
    def test_w8a8sc_skips_multi_prefix_handling(self, mock_get_weight_mapper_cls, mock_check_reuse):
        """Test that W8A8SC quantize config skips multi-prefix handling."""
        mock_check_reuse.return_value = False  # Don't skip loading
        mock_handler = MagicMock()
        mock_handler.get_tensor.return_value = torch.randn(20, 10)
        self.loader._weight_file_handler = mock_handler

        # Mock the mapper class
        mock_mapper_cls = MagicMock()
        mock_mapper_cls.map_model_to_weight.return_value = "transformer.h.0.attn.c_attn"
        mock_get_weight_mapper_cls.return_value = mock_mapper_cls

        # Create a mock layer with multi-prefix
        mock_layer = MagicMock()
        mock_param = MagicMock()
        mock_param.weight_loader = MagicMock()
        mock_layer.named_parameters.return_value = [("weight", mock_param)]
        mock_layer.prefix = ["prefix1", "prefix2"]  # Multi-prefix

        # Create a mock model with w8a8sc quantize
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.quantize = 'w8a8sc'

        modules_dict = {"model.layers.0.self_attn.qkv_proj": mock_layer}

        from tqdm.auto import tqdm
        pbar = tqdm(total=1, disable=True)

        self.loader._load_modules_with_progress(modules_dict, pbar, mock_model)

        # Verify the layer was NOT handled as multi-prefix (no _load_multi_prefix_module call)
        # Instead, it should go through the single prefix path with name mapping
        mock_mapper_cls.map_model_to_weight.assert_called_once()
        mock_handler.get_tensor.assert_called_once()


if __name__ == '__main__':
    unittest.main()
