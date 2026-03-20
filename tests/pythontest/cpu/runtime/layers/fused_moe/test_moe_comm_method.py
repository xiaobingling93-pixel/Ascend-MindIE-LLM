# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Unit tests for moe_comm_method module.

This module contains test cases for MoE communication method selection
and dispatcher caching functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from enum import Enum


class MockMoECommType(Enum):
    """Mock enumeration for MoE communication types."""
    ALLGATHER = "allgather"
    MC2 = "mc2"
    ALLTOALL = "alltoall"


class TestMoECommMethod(unittest.TestCase):
    """Test cases for MoE communication method selection and dispatcher."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock strategy classes
        self.mock_strategy_allgather = Mock()
        self.mock_strategy_allgather.is_applicable = Mock(return_value=True)
        self.mock_strategy_allgather.get_comm_type = Mock(
            return_value=MockMoECommType.ALLGATHER
        )

        self.mock_strategy_mc2 = Mock()
        self.mock_strategy_mc2.is_applicable = Mock(return_value=False)
        self.mock_strategy_mc2.get_comm_type = Mock(
            return_value=MockMoECommType.MC2
        )

        self.mock_strategy_alltoall = Mock()
        self.mock_strategy_alltoall.is_applicable = Mock(return_value=False)
        self.mock_strategy_alltoall.get_comm_type = Mock(
            return_value=MockMoECommType.ALLTOALL
        )

        # Mock dispatcher classes
        self.mock_dispatcher_allgather = Mock()
        self.mock_dispatcher_mc2 = Mock()
        self.mock_dispatcher_alltoall = Mock()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        pass

    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MOE_COMM_STRATEGIES'
    )
    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType',
        MockMoECommType
    )
    def test_select_moe_comm_method_returns_first_applicable(
        self,
        mock_strategies
    ):
        """Test selection returns first applicable strategy.
        
        Verifies that the function traverses strategies and returns
        the communication type of the first applicable one.
        """
        mock_strategies.__iter__ = Mock(
            return_value=iter([
                self.mock_strategy_allgather,
                self.mock_strategy_mc2,
                self.mock_strategy_alltoall
            ])
        )

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            select_moe_comm_method
        )

        result = select_moe_comm_method(
            quant_type="W4A8_DYNAMIC",
            max_num_tokens_per_device=1024
        )

        self.assertEqual(
            result,
            MockMoECommType.ALLGATHER
        )
        self.assertTrue(
            self.mock_strategy_allgather.is_applicable.called
        )

    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MOE_COMM_STRATEGIES'
    )
    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType',
        MockMoECommType
    )
    def test_select_moe_comm_method_no_applicable_returns_none(
        self,
        mock_strategies
    ):
        """Test selection returns None when no strategy is applicable.
        
        Verifies that the function returns None if no strategy matches
        the given conditions.
        """
        self.mock_strategy_allgather.is_applicable = Mock(return_value=False)
        self.mock_strategy_mc2.is_applicable = Mock(return_value=False)
        self.mock_strategy_alltoall.is_applicable = Mock(return_value=False)

        mock_strategies.__iter__ = Mock(
            return_value=iter([
                self.mock_strategy_allgather,
                self.mock_strategy_mc2,
                self.mock_strategy_alltoall
            ])
        )

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            select_moe_comm_method
        )
        with self.assertRaises(RuntimeError) as ctx:
            select_moe_comm_method(
                quant_type="UNKNOWN",
                max_num_tokens_per_device=999999
            )

    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method._COMM_TYPE_TO_DISPATCHER'
    )
    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType',
        MockMoECommType
    )
    def test_get_cached_dispatcher_returns_instance(
        self,
        mock_dispatcher_map
    ):
        """Test dispatcher retrieval returns correct instance.
        
        Verifies that the function returns a dispatcher instance
        for a valid communication type.
        """
        mock_dispatcher_map.__getitem__ = Mock(
            return_value=self.mock_dispatcher_allgather
        )
        mock_dispatcher_map.__contains__ = Mock(return_value=True)

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            get_cached_dispatcher
        )

        result = get_cached_dispatcher(MockMoECommType.ALLGATHER)

        self.assertIsNotNone(
            result
        )
        self.assertTrue(
            mock_dispatcher_map.__getitem__.called
        )

    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method._COMM_TYPE_TO_DISPATCHER'
    )
    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType',
        MockMoECommType
    )
    def test_get_cached_dispatcher_none_type_returns_none(
        self,
        mock_dispatcher_map
    ):
        """Test dispatcher retrieval with None type returns None.
        
        Verifies that the function handles None input gracefully.
        """
        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            get_cached_dispatcher
        )

        result = get_cached_dispatcher(None)

        self.assertIsNone(
            result
        )
        self.assertFalse(
            mock_dispatcher_map.__getitem__.called,
            "Should not lookup dispatcher for None input"
        )

    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method._COMM_TYPE_TO_DISPATCHER'
    )
    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType',
        MockMoECommType
    )
    def test_get_cached_dispatcher_unsupported_type_returns_none(
        self,
        mock_dispatcher_map
    ):
        """Test dispatcher retrieval with unsupported type returns None.
        
        Verifies that the function returns None for unsupported
        communication types.
        """
        mock_dispatcher_map.__contains__ = Mock(return_value=False)

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            get_cached_dispatcher
        )

        result = get_cached_dispatcher(MockMoECommType.MC2)

        self.assertIsNone(
            result
        )

    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MOE_COMM_STRATEGIES'
    )
    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType',
        MockMoECommType
    )
    def test_select_moe_comm_method_with_quant_type_only(
        self,
        mock_strategies
    ):
        """Test selection with quantization type parameter only.
        
        Verifies that the function works with partial parameters.
        """
        mock_strategies.__iter__ = Mock(
            return_value=iter([self.mock_strategy_allgather])
        )

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            select_moe_comm_method
        )

        result = select_moe_comm_method(quant_type="W8A8")

        self.assertEqual(
            result,
            MockMoECommType.ALLGATHER
        )
        self.assertEqual(
            self.mock_strategy_allgather.is_applicable.call_count,
            1
        )

    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MOE_COMM_STRATEGIES'
    )
    @patch(
        'mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType',
        MockMoECommType
    )
    def test_select_moe_comm_method_with_no_parameters(
        self,
        mock_strategies
    ):
        """Test selection with no parameters provided.
        
        Verifies that the function works with default parameters.
        """
        mock_strategies.__iter__ = Mock(
            return_value=iter([self.mock_strategy_allgather])
        )

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            select_moe_comm_method
        )

        result = select_moe_comm_method()

        self.assertEqual(
            result,
            MockMoECommType.ALLGATHER
        )

    def test_comm_type_to_dispatcher_mapping_exists(self):
        """Test communication type to dispatcher mapping is defined.
        
        Verifies that the mapping dictionary is properly initialized.
        """
        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            _COMM_TYPE_TO_DISPATCHER
        )

        self.assertIsInstance(
            _COMM_TYPE_TO_DISPATCHER,
            dict
        )
        self.assertGreater(
            len(_COMM_TYPE_TO_DISPATCHER),
            0
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)