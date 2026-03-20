#!/usr/bin/env python
# coding=utf-8
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

from atb_llm.utils.data.layer_adapter import (
    RowParallelLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    VocabParallelEmbedding,
    ParallelLMHead,
    RMSNorm,
)
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.utils.quantize.pack_type import TransposeType


def make_fake_layer(has_bias=True, weight_shape=(64, 64), dtype=torch.float16):
    layer = MagicMock()
    layer.weight.data = torch.randn(weight_shape, dtype=dtype, device='cpu')
    if has_bias:
        layer.bias.data = torch.randn(weight_shape[0], dtype=dtype, device='cpu')
    else:
        layer.bias = None
    layer.quant_method = MagicMock()
    layer.quant_method.get_weights_for_atb_graph.return_value = [torch.tensor([1.0], device='cpu')]
    desc = LinearTypeV2.BFLOAT16 if dtype == torch.bfloat16 else LinearTypeV2.FLOAT16
    layer.quant_method.get_linear_descs.return_value = desc
    layer.quant_method.get_weight_transpose_type.return_value = TransposeType.NOT_TRANSPOSE
    layer._PLACEHOLDER = torch.tensor([999], dtype=torch.float32, device='cpu')
    return layer


class TestLayerAdapter(unittest.TestCase):

    def test_row_parallel_linear(self):
        layer = make_fake_layer()
        weights = RowParallelLinear.get_weights_for_atb_graph(layer, padding=True, is_swiglu_quant_enabled=True)
        self.assertEqual(len(weights), 1)
        descs = RowParallelLinear.get_linear_descs(layer)
        trans = RowParallelLinear.get_weight_transpose_type(layer)
        self.assertEqual(descs, [LinearTypeV2.FLOAT16])
        self.assertEqual(trans, [TransposeType.NOT_TRANSPOSE])

    def test_row_parallel_linear_no_bias(self):
        layer = make_fake_layer(has_bias=False)
        weights = RowParallelLinear.get_weights_for_atb_graph(layer, padding=True)
        # When bias is None, it should append _PLACEHOLDER after weight
        self.assertEqual(len(weights), 1)

    def test_column_parallel_linear(self):
        layer = make_fake_layer()
        weights = ColumnParallelLinear.get_weights_for_atb_graph(layer, padding=True)
        self.assertEqual(len(weights), 1)

        with self.assertRaises(ValueError) as cm:
            ColumnParallelLinear.get_weights_for_atb_graph(layer, is_swiglu_quant_enabled=True)
        self.assertIn("Cannot set `is_swiglu_quant_enabled` to True", str(cm.exception))

        descs = ColumnParallelLinear.get_linear_descs(layer)
        trans = ColumnParallelLinear.get_weight_transpose_type(layer)
        self.assertEqual(descs, [LinearTypeV2.FLOAT16])
        self.assertEqual(trans, [TransposeType.NOT_TRANSPOSE])

    def test_merged_column_parallel_linear(self):
        with patch('mindie_llm.runtime.layers.linear.linear.MergedColumnParallelLinear.__init__', return_value=None):
            layer = MergedColumnParallelLinear(input_size=64, output_sizes=[32, 32])

        layer.weight = MagicMock()
        layer.weight.data = torch.randn(64, 64, dtype=torch.float16, device='cpu')
        layer.bias = None
        layer.quant_method = MagicMock()
        layer.quant_method.get_weights_for_atb_graph.return_value = [torch.tensor([1.0], device='cpu')]
        layer.quant_method.get_linear_descs.return_value = LinearTypeV2.FLOAT16
        layer.quant_method.get_weight_transpose_type.return_value = TransposeType.NOT_TRANSPOSE
        layer._PLACEHOLDER = torch.tensor([888], dtype=torch.float32, device='cpu')
        layer.linear_modules = []

        weights = layer.get_weights_for_atb_graph(padding=True)
        self.assertEqual(len(weights), 7)
        for w in weights[-6:]:
            self.assertEqual(w.item(), 888)

        descs = MergedColumnParallelLinear.get_linear_descs(layer)
        trans = MergedColumnParallelLinear.get_weight_transpose_type(layer)
        self.assertListEqual(descs, [LinearTypeV2.FLOAT16, LinearTypeV2.INVALID])
        self.assertListEqual(trans, [TransposeType.NOT_TRANSPOSE, TransposeType.INVALID])

    def test_qkv_parallel_linear(self):
        with patch('mindie_llm.runtime.layers.linear.linear.QKVParallelLinear.__init__', return_value=None):
            layer = QKVParallelLinear(num_heads=2, num_kv_heads=1, head_size=32)

        layer.weight = MagicMock()
        layer.weight.data = torch.randn(96, 64, dtype=torch.float16, device='cpu')  # q+k+v
        layer.bias = None
        layer.quant_method = MagicMock()
        layer.quant_method.get_weights_for_atb_graph.return_value = [torch.tensor([1.0], device='cpu')]
        layer.quant_method.get_linear_descs.return_value = LinearTypeV2.FLOAT16
        layer.quant_method.get_weight_transpose_type.return_value = TransposeType.NOT_TRANSPOSE
        layer._PLACEHOLDER = torch.tensor([777], dtype=torch.float32, device='cpu')

        weights = layer.get_weights_for_atb_graph(padding=True)
        self.assertEqual(len(weights), 13)
        for w in weights[-12:]:
            self.assertEqual(w.item(), 777)

        with self.assertRaises(ValueError):
            layer.get_weights_for_atb_graph(is_swiglu_quant_enabled=True)

        descs = layer.get_linear_descs()
        trans = layer.get_weight_transpose_type()
        self.assertListEqual(descs, [LinearTypeV2.FLOAT16, LinearTypeV2.INVALID, LinearTypeV2.INVALID])
        self.assertListEqual(trans, [TransposeType.NOT_TRANSPOSE, TransposeType.INVALID, TransposeType.INVALID])

    def test_vocab_parallel_embedding(self):
        layer = MagicMock()
        layer.quant_method.get_weights_for_atb_graph.return_value = [torch.tensor([[1.0]])]
        layer._PLACEHOLDER = torch.tensor([0])
        weights = VocabParallelEmbedding.get_weights_for_atb_graph(layer, padding=True)
        expected = [torch.tensor([[1.0]])]
        self.assertEqual(len(weights), len(expected))
        self.assertTrue(torch.allclose(weights[0], expected[0]))

    def test_parallel_lm_head(self):
        layer = make_fake_layer()
        weights = ParallelLMHead.get_weights_for_atb_graph(layer, padding=True)
        self.assertEqual(len(weights), 1)
        descs = ParallelLMHead.get_linear_descs(layer)
        trans = ParallelLMHead.get_weight_transpose_type(layer)
        self.assertEqual(descs, [LinearTypeV2.FLOAT16])
        self.assertEqual(trans, [TransposeType.NOT_TRANSPOSE])

    def test_rms_norm(self):
        layer = MagicMock()
        layer.quant_method.get_weights_for_atb_graph.return_value = [torch.tensor([0.5])]
        weights = RMSNorm.get_weights_for_atb_graph(layer, padding=True)
        expected = [torch.tensor([0.5])]
        self.assertEqual(len(weights), len(expected))
        self.assertTrue(torch.allclose(weights[0], expected[0]))


if __name__ == '__main__':
    unittest.main()
