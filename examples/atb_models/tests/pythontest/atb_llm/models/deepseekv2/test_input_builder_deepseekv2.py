# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from ddt import ddt
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.models.deepseekv2.input_builder_deepseekv2 import Deepseekv2InputBuilder

ROLE = "role"
CONTENT = "content"
INPUT_STR = "who are you"
FAKE_CONFIG_DICT = {
    'model_type': 'deepseekv2',
    'num_hidden_layers': 61,
    'max_position_embeddings': 4096,
    'vocab_size': 163840,
    'rope_scaling': 1.0,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
}


class MockTokenizer:
    def __init__(self, use_fast=False):
        self.use_fast = use_fast
        self.init_kwargs = {}
        self.chat_template = "test"

    def apply_chat_template(self, conversation, thinking=False, **kwargs):
        return f"test string. Thinking: {thinking}"

    @classmethod
    def encode(cls, prompt, add_special_tokens):
        return prompt


class MockTokenizerNoChatTemplate:
    def __init__(self, use_fast=False):
        self.use_fast = use_fast
        self.init_kwargs = {}
        self.chat_template = "test"

    @classmethod
    def encode(cls, prompt, add_special_tokens):
        return prompt


@ddt
class TestDeepseekInputBuilder(unittest.TestCase):
    def setUp(self):
        self.deepseek_config = DeepseekV2Config(**FAKE_CONFIG_DICT)
        self.tokenizer = MockTokenizer()
        self.input_builder = Deepseekv2InputBuilder(self.tokenizer)

    def test_apply_chat_template_default(self):

        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        chat_template_kwargs_think = {"enable_thinking": "true"}
        chat_template_kwargs_nothink = {"enable_thinking": "false"}
        user_prompt_think = self.input_builder._apply_chat_template(
            user_conversation, chat_template_kwargs=chat_template_kwargs_think
        )
        user_prompt_nothink = self.input_builder._apply_chat_template(
            user_conversation, chat_template_kwargs=chat_template_kwargs_nothink
        )
        self.assertIsNotNone(user_prompt_think)
        self.assertIsInstance(user_prompt_think, str)
        self.assertEqual(user_prompt_think, "test string. Thinking: true")
        self.assertIsNotNone(user_prompt_nothink)
        self.assertIsInstance(user_prompt_nothink, str)
        self.assertEqual(user_prompt_nothink, "test string. Thinking: false")

    def test_apply_chat_template_raise(self):
        # no apply_chat_template
        tokenizer = MockTokenizerNoChatTemplate()
        input_builder = Deepseekv2InputBuilder(tokenizer)

        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        with self.assertRaises(RuntimeError) as cm:
            input_builder._apply_chat_template(user_conversation)
        self.assertIn("transformers version is detected to be <4.34", str(cm.exception))

        # no chat_template
        tokenizer = MockTokenizer()
        tokenizer.chat_template = ""
        input_builder = Deepseekv2InputBuilder(tokenizer)
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        with self.assertRaises(RuntimeError) as cm:
            input_builder._apply_chat_template(user_conversation)
        self.assertIn("it is not configured with a `chat_template`", str(cm.exception))


if __name__ == '__main__':
    unittest.main()