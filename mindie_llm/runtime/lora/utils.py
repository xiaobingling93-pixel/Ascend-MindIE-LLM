# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.runtime.config.mindie_llm_config import LoraModelConfig
from .lora_layers import RowParallelLinearWithLoRA, ColumnParallelLinearWithLoRA


def replace_submodule(model, module_name: str, new_module: CustomLayer) -> CustomLayer:
    """Replace a submodule in a model with a new module."""
    target_name = module_name.split(".")[-1]
    parent_module = model.get_submodule(".".join(module_name.split(".")[:-1]))
    setattr(parent_module, target_name, new_module)
    return new_module


def from_layer(layer: CustomLayer, lora_model_config: LoraModelConfig, dtype, device) -> CustomLayer:
    supported_lora_classes = {RowParallelLinearWithLoRA, ColumnParallelLinearWithLoRA}
    for lora_class in supported_lora_classes:
        if lora_class.can_replace_layer(layer):
            instance_layer = lora_class(layer)
            instance_layer.create_lora_weights(lora_model_config, dtype, device)
            return instance_layer
    return layer
