#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import random
import string
import unittest

from mindie_llm.text_generator.plugins import PluginParameterValidator


class TestPluginInit(unittest.TestCase):
    def test_validator_length_error(self):
        speculation_gamma = 0
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=1025))
        plugin_params = random_string
        validator = PluginParameterValidator(speculation_gamma)
        with self.assertRaises(ValueError) as cm:
            _, _, _ = validator.validate(plugin_params)
        self.assertIn("The length of plugin_params is too long, it should be within (0, 1024]", str(cm.exception))

    def test_validator_json_format_error(self):
        speculation_gamma = 0
        plugin_params = "{\"plugin_type\":\"mtp\",\"num_speculative_tokens\": 1"
        validator = PluginParameterValidator(speculation_gamma)
        with self.assertRaises(json.JSONDecodeError) as cm:
            _, _, _ = validator.validate(plugin_params)
        expect_msg = ("The 'plugin_params' field does not conform to JSON format."
                      " Please check.: line 1 column 49 (char 48)")
        self.assertIn(expect_msg, str(cm.exception))

    def test_validator_plugin_type_none(self):
        speculation_gamma = 0
        plugin_params = "{\"plugin_type\":\"\",\"num_speculative_tokens\": 1}"
        validator = PluginParameterValidator(speculation_gamma)

        _, _, enabled_plugins_list = validator.validate(plugin_params)
        expected = []
        self.assertEqual(enabled_plugins_list, expected)

    def test_validator_plugin_type_error(self):
        speculation_gamma = 0
        plugin_params = "{\"plugin_type\":\"ttt\"}"
        validator = PluginParameterValidator(speculation_gamma)
        expect_msg = ("Unsupported plugin type: ttt, Only 'la', 'memory_decoding',"
                      " 'prefix_cache', 'mtp' and 'splitfuse' are supported.")
        with self.assertRaises(NotImplementedError) as cm:
            _, _, _ = validator.validate(plugin_params)
        self.assertIn(expect_msg, str(cm.exception))

    def test_validator_missing_fields(self):
        speculation_gamma = 0
        plugin_params = "{\"plugin_type\":\"la\",\"level\": 4,\"window\": 5}"
        validator = PluginParameterValidator(speculation_gamma)
        with self.assertRaises(NotImplementedError) as cm:
            _, _, _ = validator.validate(plugin_params)
        self.assertIn("Missing fields for plugin_type 'la': {'guess_set_size'}", str(cm.exception))

    def test_validator_parameter_error(self):
        speculation_gamma = 0
        plugin_params = "{\"plugin_type\":\"la\",\"level\": 4,\"window\": 5,\"guess_set_size\": 5}"
        validator = PluginParameterValidator(speculation_gamma)
        expect_msg = ("Validation failed for plugin_type 'la'."
                      " Please check the parameter configuration against the user manual.")
        with self.assertRaises(ValueError) as cm:
            _, _, _ = validator.validate(plugin_params)
        self.assertIn(expect_msg, str(cm.exception))

if __name__ == "__main__":
    unittest.main()

