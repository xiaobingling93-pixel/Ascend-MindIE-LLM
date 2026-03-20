"""
Unit tests for MoE communication strategy selection logic.

Mocks all NPU/distributed dependencies to ensure fast, isolated testing.
Uses unittest assertions (assertTrue/assertFalse/assertEqual) per coding standards.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from typing import Optional

# Import module under test
from mindie_llm.runtime.layers.fused_moe.moe_comm_strategy import (
    MoECommType,
    MoECommStrategyBase,
    FusedMC2Strategy,
    MC2Strategy,
    All2AllStrategy,
    AllGatherStrategy,
    MOE_COMM_STRATEGIES,
)


class TestMoECommStrategies(unittest.TestCase):
    """Test suite for MoE communication strategy selection."""

    def setUp(self) -> None:
        """Set up common mocks for NPU/distributed dependencies."""
        # Mock device info
        self.mock_device_info = MagicMock()
        # Mock parallel manager
        self.mock_parallel_mgr = MagicMock()
        # Mock forward context
        self.mock_forward_ctx = MagicMock()
        # Mock batch descriptor
        self.mock_batch_desc = MagicMock()
        self.mock_forward_ctx.batch_descriptor = self.mock_batch_desc

        # Patchers for external dependencies
        self.patchers = [
            patch(
                "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.get_npu_node_info",
                return_value=self.mock_device_info,
            ),
            patch(
                "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.get_parallel_info_manager",
                return_value=self.mock_parallel_mgr,
            ),
            patch(
                "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.get_forward_context",
                return_value=self.mock_forward_ctx,
            ),
            patch(
                "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.get_mc2_token_capacity",
                return_value=8192,  # Default capacity for testing
            ),
            patch(
                "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.logger"
            ),  # Mock logger to suppress output
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self) -> None:
        """Stop all patchers after each test."""
        for p in self.patchers:
            p.stop()

    def _setup_parallel_info(self, **kwargs) -> None:
        """
        Helper: configure mock parallel manager with common settings.
        
        Args:
            **kwargs: Optional override parameters for parallel configuration.
                      Available keys:
                      - ep_enabled (bool): Whether MoE EP is enabled. Default: True
                      - ep_mc2_group_size (int): EP MC2 group size. Default: 16
                      - moe_tp_enabled (bool): Whether MoE TP is enabled. Default: False
                      - attn_tp_group_size (int): Attention TP group size. Default: 1
                      - attn_dp_enabled (bool): Whether Attention DP is enabled. Default: False
                      - world_size (int): Distributed world size. Default: 8
        
        Example:
            >>> self._setup_parallel_info(ep_enabled=False)  # Only override EP enabled
            >>> self._setup_parallel_info(world_size=32, ep_mc2_group_size=64)  # Multiple overrides
        """
        # Extract parameters with defaults
        ep_enabled = kwargs.get("ep_enabled", True)
        ep_mc2_group_size = kwargs.get("ep_mc2_group_size", 16)
        moe_tp_enabled = kwargs.get("moe_tp_enabled", False)
        attn_tp_group_size = kwargs.get("attn_tp_group_size", 1)
        attn_dp_enabled = kwargs.get("attn_dp_enabled", False)
        world_size = kwargs.get("world_size", 8)
        
        # Warn if unknown kwargs passed (helps catch typos in tests)
        valid_keys = {
            "ep_enabled", "ep_mc2_group_size", "moe_tp_enabled",
            "attn_tp_group_size", "attn_dp_enabled", "world_size"
        }
        unknown_keys = set(kwargs.keys()) - valid_keys
        if unknown_keys:
            logger.warning(
                f"Unknown kwargs in _setup_parallel_info: {unknown_keys}. "
                f"Valid keys: {valid_keys}"
            )
        
        def get_side_effect(pt):
            mock_pt = MagicMock()
            if pt.name == "MOE_EP":
                mock_pt.is_enabled.return_value = ep_enabled
            elif pt.name == "MOE_EP_MC2":
                mock_pt.group_size = ep_mc2_group_size
            elif pt.name == "MOE_TP":
                mock_pt.is_enabled.return_value = moe_tp_enabled
            elif pt.name == "ATTN_TP":
                mock_pt.group_size = attn_tp_group_size
            elif pt.name == "ATTN_DP":
                mock_pt.is_enabled.return_value = attn_dp_enabled
            return mock_pt

        self.mock_parallel_mgr.get.side_effect = get_side_effect
        self.mock_parallel_mgr.world_size = world_size

    def _setup_device(self, device_type: str) -> None:
        """Helper: set mock device type."""
        from mindie_llm.runtime.utils.npu.device_utils import DeviceType
        self.mock_device_info.get_device_type.return_value = getattr(
            DeviceType, device_type
        )

    def _setup_forward_context(
        self, is_prefill: bool, num_tokens: int
    ) -> None:
        """Helper: configure mock forward context."""
        self.mock_forward_ctx.is_prefill = is_prefill
        self.mock_batch_desc.num_tokens = num_tokens

    # ==================== Base Class Tests ====================

    def test_base_class_not_implemented(self) -> None:
        """Base strategy methods should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            MoECommStrategyBase.is_applicable(None, None)
        with self.assertRaises(NotImplementedError):
            MoECommStrategyBase.get_comm_type()

    # ==================== FusedMC2Strategy Tests ====================

    def test_fused_mc2_happy_path(self) -> None:
        """FusedMC2: applicable on 910C with valid EP size and token count."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(
            ep_enabled=True,
            ep_mc2_group_size=16,
            moe_tp_enabled=False,
            attn_tp_group_size=1,
        )
        self._setup_forward_context(is_prefill=False, num_tokens=1024)

        result = FusedMC2Strategy.is_applicable(None, max_num_tokens_per_device=2048)
        self.assertTrue(result)
        self.assertEqual(
            FusedMC2Strategy.get_comm_type(),
            MoECommType.FUSED_MC2,
            "FusedMC2 should return FUSED_MC2 comm type",
        )

    def test_fused_mc2_reject_wrong_device(self) -> None:
        """FusedMC2: reject non-910C devices."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info()
        self._setup_forward_context(is_prefill=False, num_tokens=1024)

        result = FusedMC2Strategy.is_applicable(None, max_num_tokens_per_device=2048)
        self.assertFalse(result, "FusedMC2 should reject non-910C devices")

    def test_fused_mc2_reject_large_ep_group(self) -> None:
        """FusedMC2: reject EP group size > 32."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(ep_mc2_group_size=64)
        self._setup_forward_context(is_prefill=False, num_tokens=1024)

        result = FusedMC2Strategy.is_applicable(None, max_num_tokens_per_device=2048)
        self.assertFalse(result, "FusedMC2 should reject EP group size > 32")

    def test_fused_mc2_reject_token_overflow(self) -> None:
        """FusedMC2: reject when tokens exceed per-device capacity."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(attn_tp_group_size=1)
        self._setup_forward_context(is_prefill=False, num_tokens=4096)

        result = FusedMC2Strategy.is_applicable(None, max_num_tokens_per_device=2048)
        self.assertFalse(result, "FusedMC2 should reject excessive token count")

    def test_fused_mc2_reject_moe_tp(self) -> None:
        """FusedMC2: incompatible with MoE TP parallelism."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(moe_tp_enabled=True)
        self._setup_forward_context(is_prefill=False, num_tokens=1024)

        result = FusedMC2Strategy.is_applicable(None, max_num_tokens_per_device=2048)
        self.assertFalse(result, "FusedMC2 should reject MoE TP enabled")

    # ==================== MC2Strategy Tests ====================

    def test_mc2_910b_decode_large_cluster(self) -> None:
        """MC2 on 910B: applicable in decode phase with world_size >= 16."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(world_size=16)
        self._setup_forward_context(is_prefill=False, num_tokens=512)

        result = MC2Strategy.is_applicable(None, max_num_tokens_per_device=2048)
        self.assertTrue(result)
        self.assertEqual(
            MC2Strategy.get_comm_type(),
            MoECommType.MC2,
            "MC2 should return MC2 comm type",
        )

    def test_mc2_910b_reject_prefill(self) -> None:
        """MC2 on 910B: reject prefill phase."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(world_size=16)
        self._setup_forward_context(is_prefill=True, num_tokens=512)

        result = MC2Strategy.is_applicable(None, max_num_tokens_per_device=2048)
        self.assertFalse(result, "MC2 should reject prefill phase on 910B")

    def test_mc2_910b_reject_small_cluster(self) -> None:
        """MC2 on 910B: reject world_size < 16."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(world_size=8)
        self._setup_forward_context(is_prefill=False, num_tokens=512)

        result = MC2Strategy.is_applicable(None, max_num_tokens_per_device=2048)
        self.assertFalse(result, "MC2 should reject small cluster on 910B")

    def test_mc2_910c_decode_valid(self) -> None:
        """MC2 on 910C: applicable in decode phase within token capacity."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info()
        self._setup_forward_context(is_prefill=False, num_tokens=4096)

        result = MC2Strategy.is_applicable(None, max_num_tokens_per_device=8192)
        self.assertTrue(result)

    def test_mc2_910c_reject_token_exceed(self) -> None:
        """MC2 on 910C: raise ValueError when tokens exceed capacity."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info()
        self._setup_forward_context(is_prefill=False, num_tokens=10000)

        with self.assertRaises(ValueError) as ctx:
            MC2Strategy.is_applicable(None, max_num_tokens_per_device=8192)
        self.assertIn("MC2 operator limitation", str(ctx.exception))
        self.assertIn("exceed the limit", str(ctx.exception))

    def test_mc2_reject_moe_tp(self) -> None:
        """MC2: incompatible with MoE TP parallelism."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(moe_tp_enabled=True)
        self._setup_forward_context(is_prefill=False, num_tokens=1024)

        result = MC2Strategy.is_applicable(None, max_num_tokens_per_device=8192)
        self.assertFalse(result, "MC2 should reject MoE TP enabled")

    def test_mc2_unsupported_device(self) -> None:
        """MC2: raise RuntimeError on unsupported device type."""
        # Simulate unknown device type
        self.mock_device_info.get_device_type.return_value = "UNKNOWN_DEVICE"
        self._setup_parallel_info()
        self._setup_forward_context(is_prefill=False, num_tokens=1024)

        with self.assertRaises(RuntimeError) as ctx:
            MC2Strategy.is_applicable(None, max_num_tokens_per_device=8192)
        self.assertIn("Unsupported device type", str(ctx.exception))

    # ==================== All2AllStrategy Tests ====================

    def test_all2all_910b_w4a8_dynamic(self) -> None:
        """All2All on 910B: applicable with W4A8_DYNAMIC quantization."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info()

        from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType
        result = All2AllStrategy.is_applicable(QuantType.W4A8_DYNAMIC, None)
        self.assertTrue(result)
        self.assertEqual(
            All2AllStrategy.get_comm_type(),
            MoECommType.ALLTOALL,
            "All2All should return ALLTOALL comm type",
        )

    def test_all2all_910b_fallback_with_attn_dp(self) -> None:
        """All2All on 910B: fallback applicable if Attn DP enabled."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(attn_dp_enabled=True)

        result = All2AllStrategy.is_applicable("OTHER_QUANT", None)
        self.assertTrue(result)

    def test_all2all_910b_reject_non_w4a8_no_dp(self) -> None:
        """All2All on 910B: reject if not W4A8_DYNAMIC and Attn DP disabled."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(attn_dp_enabled=False)

        result = All2AllStrategy.is_applicable("OTHER_QUANT", None)
        self.assertFalse(result, "All2All should reject non-W4A8 without Attn DP on 910B")

    def test_all2all_reject_moe_tp_with_attn_dp(self) -> None:
        """All2All: raise error when MoE TP + Attn DP both enabled."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(moe_tp_enabled=True, attn_dp_enabled=True)
        from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType
        with self.assertRaises(RuntimeError) as ctx:
            All2AllStrategy.is_applicable(QuantType.W4A8_DYNAMIC, None)
        self.assertIn("Do not support moe_tp > 1", str(ctx.exception))

    def test_all2all_reject_moe_tp_alone(self) -> None:
        """All2All: return False when MoE TP enabled but Attn DP disabled."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(moe_tp_enabled=True, attn_dp_enabled=False)

        result = All2AllStrategy.is_applicable(None, None)
        self.assertFalse(result, "All2All should reject MoE TP without Attn DP")

    # ==================== AllGatherStrategy Tests ====================

    def test_allgather_always_applicable(self) -> None:
        """AllGather: universal fallback, always applicable."""
        # Test with various inputs
        test_cases = [
            (None, None),
            ("W4A8_DYNAMIC", 1024),
            ("OTHER", 99999),
        ]
        for quant, tokens in test_cases:
            with self.subTest(quant=quant, tokens=tokens):
                result = AllGatherStrategy.is_applicable(quant, tokens)
                self.assertTrue(result)
        self.assertEqual(
            AllGatherStrategy.get_comm_type(),
            MoECommType.ALLGATHER,
            "AllGather should return ALLGATHER comm type",
        )

    # ==================== Strategy Selection Order Tests ====================

    def test_strategy_priority_order(self) -> None:
        """Verify strategies are ordered by priority: FusedMC2 > MC2 > All2All > AllGather."""
        expected_order = [
            FusedMC2Strategy,
            MC2Strategy,
            All2AllStrategy,
            AllGatherStrategy,
        ]
        self.assertEqual(
            MOE_COMM_STRATEGIES,
            expected_order,
            "Strategy list should follow priority order",
        )

    def test_first_applicable_strategy_selection(self) -> None:
        """Simulate strategy selection: first applicable strategy wins."""
        # Setup: 910C with valid fused MC2 conditions
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(
            ep_enabled=True,
            ep_mc2_group_size=16,
            moe_tp_enabled=False,
            attn_tp_group_size=1,
        )
        self._setup_forward_context(is_prefill=False, num_tokens=1024)

        selected = None
        for strategy in MOE_COMM_STRATEGIES:
            if strategy.is_applicable(None, max_num_tokens_per_device=2048):
                selected = strategy.get_comm_type()
                break

        self.assertEqual(
            selected,
            MoECommType.FUSED_MC2,
            "FusedMC2 should be selected first when applicable",
        )

    def test_fallback_to_allgather(self) -> None:
        """When no strategy matches, AllGather should be selected as fallback."""
        # Setup: EP disabled -> all strategies except AllGather reject
        self._setup_parallel_info(ep_enabled=False)

        selected = None
        for strategy in MOE_COMM_STRATEGIES:
            if strategy.is_applicable(None, max_num_tokens_per_device=2048):
                selected = strategy.get_comm_type()
                break

        self.assertEqual(
            selected,
            MoECommType.ALLGATHER,
            "AllGather should be selected as universal fallback",
        )


if __name__ == "__main__":
    # Run with: python -m unittest test_moe_comm_strategy.py -v
    unittest.main(verbosity=2)