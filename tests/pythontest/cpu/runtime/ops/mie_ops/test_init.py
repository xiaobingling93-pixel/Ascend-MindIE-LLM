# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import patch, MagicMock
from mindie_llm.runtime.utils.npu.device_utils import DeviceType
from mindie_llm.runtime.ops.mie_ops import import_mie_ops_by_device


class TestMieOpsImport(unittest.TestCase):

    @patch("mindie_llm.runtime.ops.mie_ops.get_npu_node_info")
    @patch("mindie_llm.runtime.ops.mie_ops.importlib")
    def test_import_success_ascend910b(self, mock_importlib, mock_get_info):
        # Mock get_npu_node_info().get_device_type() 返回 ASCEND_910B
        mock_node_info = MagicMock()
        mock_node_info.get_device_type.return_value = DeviceType.ASCEND_910B
        mock_get_info.return_value = mock_node_info
        mock_importlib.import_module = MagicMock()

        # 调用被测函数
        import_mie_ops_by_device()

        # 验证 import_module 被正确调用
        mock_importlib.import_module.assert_called_with("mie_ops_ascend910b")

    @patch("mindie_llm.runtime.ops.mie_ops.get_npu_node_info")
    @patch("mindie_llm.runtime.ops.mie_ops.importlib")
    def test_import_success_ascend910_93(self, mock_importlib, mock_get_info):
        mock_node_info = MagicMock()
        mock_node_info.get_device_type.return_value = DeviceType.ASCEND_910_93
        mock_get_info.return_value = mock_node_info
        mock_importlib.import_module = MagicMock()

        import_mie_ops_by_device()

        mock_importlib.import_module.assert_called_with("mie_ops_ascend910_93")

    @patch("mindie_llm.runtime.ops.mie_ops.get_npu_node_info")
    def test_unsupported_device_type_raises_error(self, mock_get_info):
        # 模拟一个不支持的设备类型
        unsupported_device = DeviceType.ASCEND_310P
        mock_node_info = MagicMock()
        mock_node_info.get_device_type.return_value = unsupported_device
        mock_get_info.return_value = mock_node_info

        # 验证抛出 EnvironmentError
        with self.assertRaises(EnvironmentError) as cm:
            import_mie_ops_by_device()

        error_msg = str(cm.exception)
        self.assertIn("Unsupported device type", error_msg)
        self.assertIn("mie_ops_ascend910b", error_msg)
        self.assertIn("mie_ops_ascend910_93", error_msg)


if __name__ == "__main__":
    unittest.main()
