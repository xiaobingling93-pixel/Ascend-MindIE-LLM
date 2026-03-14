# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
Test cases for Qwen2Config weight name mapping.

This module tests the bidirectional weight name mapping between:
- Transformer format (W8A8SC): transformer.h.N.attn.c_attn
- Model format (HuggingFace): model.layers.N.self_attn.qkv_proj

Note: Qwen2 now uses the same mapping logic as Qwen3,
as the mapping methods have been moved from Qwen3Config to Qwen2Config.
"""

import unittest

from mindie_llm.runtime.models.qwen2.config_qwen2 import Qwen2Config


class TestQwen2ConfigMapWeightToModel(unittest.TestCase):
    """Test cases for Qwen2Config.map_weight_to_model method: transformer format to model format"""

    def test_basic_prefix_replacement(self):
        """Test basic prefix replacement: transformer -> model"""
        result = Qwen2Config.map_weight_to_model("transformer.wte.weight")
        self.assertEqual(result[0], "model.embed_tokens.weight")

    def test_layer_numbering(self):
        """Test layer numbering: h -> layers"""
        result = Qwen2Config.map_weight_to_model("transformer.h.0.attn.c_attn.weight")
        self.assertEqual(result[0], "model.layers.0.self_attn.qkv_proj.weight")

    def test_qkv_packed_weights(self):
        """Test QKV packed weights, should return primary key and split keys"""
        result = Qwen2Config.map_weight_to_model("transformer.h.0.attn.c_attn.weight")
        # Primary key
        self.assertIn("model.layers.0.self_attn.qkv_proj.weight", result)
        # Split keys
        self.assertIn("model.layers.0.self_attn.q_proj.weight", result)
        self.assertIn("model.layers.0.self_attn.k_proj.weight", result)
        self.assertIn("model.layers.0.self_attn.v_proj.weight", result)

    def test_gate_up_packed_weights(self):
        """Test gate_up packed weights, should return primary key and split keys"""
        result = Qwen2Config.map_weight_to_model("transformer.h.0.mlp.w2_w1.weight")
        # Primary key
        self.assertIn("model.layers.0.mlp.gate_up_proj.weight", result)
        # Split keys
        self.assertIn("model.layers.0.mlp.gate_proj.weight", result)
        self.assertIn("model.layers.0.mlp.up_proj.weight", result)

    def test_mlp_c_proj_to_down_proj(self):
        """Test MLP c_proj -> down_proj"""
        result = Qwen2Config.map_weight_to_model("transformer.h.0.mlp.c_proj.weight")
        self.assertEqual(result[0], "model.layers.0.mlp.down_proj.weight")

    def test_attn_c_proj_to_o_proj(self):
        """Test Attention c_proj -> o_proj"""
        result = Qwen2Config.map_weight_to_model("transformer.h.0.attn.c_proj.weight")
        self.assertEqual(result[0], "model.layers.0.self_attn.o_proj.weight")

    def test_all_suffixes_for_qkv(self):
        """Test split key generation for all QKV suffixes"""
        result = Qwen2Config.map_weight_to_model("transformer.h.0.attn.c_attn.input_scale")
        self.assertIn("model.layers.0.self_attn.qkv_proj.input_scale", result)
        self.assertIn("model.layers.0.self_attn.q_proj.input_scale", result)
        self.assertIn("model.layers.0.self_attn.k_proj.input_scale", result)
        self.assertIn("model.layers.0.self_attn.v_proj.input_scale", result)


class TestQwen2ConfigMapModelToWeight(unittest.TestCase):
    """Test cases for Qwen2Config.map_model_to_weight method: model format to transformer format"""

    def test_basic_prefix_replacement(self):
        """Test basic prefix replacement: model.layers -> transformer.h"""
        result = Qwen2Config.map_model_to_weight("model.layers.0.self_attn.qkv_proj")
        self.assertEqual(result, "transformer.h.0.attn.c_attn")

    def test_qkv_proj_mapping(self):
        """Test QKV mapping: qkv_proj -> c_attn"""
        result = Qwen2Config.map_model_to_weight("model.layers.0.self_attn.qkv_proj")
        self.assertEqual(result, "transformer.h.0.attn.c_attn")

    def test_o_proj_mapping(self):
        """Test o_proj mapping: o_proj -> c_proj"""
        result = Qwen2Config.map_model_to_weight("model.layers.0.self_attn.o_proj")
        self.assertEqual(result, "transformer.h.0.attn.c_proj")

    def test_gate_up_proj_mapping(self):
        """Test gate_up_proj mapping: gate_up_proj -> w2_w1"""
        result = Qwen2Config.map_model_to_weight("model.layers.0.mlp.gate_up_proj")
        self.assertEqual(result, "transformer.h.0.mlp.w2_w1")

    def test_down_proj_mapping(self):
        """Test down_proj mapping: down_proj -> c_proj"""
        result = Qwen2Config.map_model_to_weight("model.layers.0.mlp.down_proj")
        self.assertEqual(result, "transformer.h.0.mlp.c_proj")

    def test_input_layernorm_mapping(self):
        """Test input_layernorm mapping: input_layernorm -> ln_1"""
        result = Qwen2Config.map_model_to_weight("model.layers.0.input_layernorm")
        self.assertEqual(result, "transformer.h.0.ln_1")

    def test_post_attention_layernorm_mapping(self):
        """Test post_attention_layernorm mapping: post_attention_layernorm -> ln_2"""
        result = Qwen2Config.map_model_to_weight("model.layers.0.post_attention_layernorm")
        self.assertEqual(result, "transformer.h.0.ln_2")

    def test_norm_mapping(self):
        """Test norm mapping: model.norm -> transformer.ln_f"""
        result = Qwen2Config.map_model_to_weight("model.norm")
        self.assertEqual(result, "transformer.ln_f")

    def test_embed_tokens_mapping(self):
        """Test embed_tokens mapping: model.embed_tokens -> transformer.wte"""
        result = Qwen2Config.map_model_to_weight("model.embed_tokens")
        self.assertEqual(result, "transformer.wte")

    def test_lm_head_mapping(self):
        """Test lm_head mapping: model.lm_head -> transformer.wte (tied weights)"""
        result = Qwen2Config.map_model_to_weight("model.lm_head")
        self.assertEqual(result, "transformer.wte")


class TestQwen2ConfigNameMappingRoundTrip(unittest.TestCase):
    """Test name mapping round-trip consistency: model -> transformer -> model"""

    def test_qkv_weight_round_trip(self):
        """Test QKV weight round-trip mapping consistency"""
        original = "model.layers.0.self_attn.qkv_proj.weight"
        to_transformer = Qwen2Config.map_model_to_weight(original)
        back_to_model = Qwen2Config.map_weight_to_model(to_transformer)
        self.assertIn(original, back_to_model)

    def test_gate_up_weight_round_trip(self):
        """Test gate_up weight round-trip mapping consistency"""
        original = "model.layers.0.mlp.gate_up_proj.weight"
        to_transformer = Qwen2Config.map_model_to_weight(original)
        back_to_model = Qwen2Config.map_weight_to_model(to_transformer)
        self.assertIn(original, back_to_model)


if __name__ == '__main__':
    unittest.main()
