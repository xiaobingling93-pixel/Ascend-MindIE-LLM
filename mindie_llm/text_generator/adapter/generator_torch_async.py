# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Any, Dict

import torch
import torch_npu

from .generator_torch import GeneratorTorch, reorder_tensor
from ..utils.model_input import ModelInput
from ..utils.model_output import ModelOutput
from ...utils.log.error_code import ErrorCode
from ...utils.log.logging import logger


class GeneratorTorchAsync(GeneratorTorch):
    def __init__(self, model_config: Dict[str, Any]) -> None:
        self.layerwise_disaggregated = model_config.get("layerwise_disaggregated", False)
        super().__init__(model_config)
        self.new_stream = torch_npu.npu.Stream(self.device)

    @staticmethod
    def synchronize():
        torch_npu.npu.current_stream().synchronize()

    def get_new_stream(self):
        return torch.npu.stream(self.new_stream)

    def to_tensor_async(self, array):
        host_tensor = torch.from_numpy(array).pin_memory()
        device_tensor = host_tensor.to(self.device, non_blocking=True)
        return device_tensor

    def prepare_model_inputs(self, model_input: ModelInput, **kwargs):
        if self.mapping.has_dp() and model_input.dp_rank_ids is None:
            error_msg = "The `dp_rank_ids` is not given when data parallel size > 1."
            logger.error(error_msg, ErrorCode.TEXT_GENERATOR_INTERNAL_ERROR)
            raise AssertionError(error_msg)

        do_reorder_requests, revert_adapter_idx = self._prepare_model_inputs(model_input, kwargs)

        model_input, kwargs = self.model_wrapper.prepare_model_inputs(model_input, **kwargs)

        kwargs["do_reorder_requests"] = do_reorder_requests
        kwargs["revert_adapter_idx"] = revert_adapter_idx
        return model_input, kwargs

    def forward_from_model_inputs(self, model_input: ModelInput, **kwargs):
        result = self.model_wrapper.forward_from_model_inputs(model_input, self.cache_pool.npu_cache, **kwargs)

        if isinstance(result, tuple):
            if len(result) == 2:
                logits, hidden_states = result
                model_output = ModelOutput(logits=logits, hidden_states=hidden_states)
            else:
                logits, hidden_states, draft_tokens = result
                model_output = ModelOutput(logits=logits, hidden_states=hidden_states, draft_tokens=draft_tokens)
        else:
            model_output = ModelOutput(logits=result)
        model_output.original_result = result

        # sort logits back to the original order (related to lm_head_indices)
        if kwargs.get("do_reorder_requests", False):
            model_output.logits = reorder_tensor(model_output.logits, kwargs.get("revert_adapter_idx", []))

        return model_output
