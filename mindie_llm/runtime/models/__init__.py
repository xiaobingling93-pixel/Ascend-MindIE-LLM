# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import importlib
import os

from mindie_llm.runtime.config.load_config import LoadConfig
from mindie_llm.runtime.utils.helpers.safety.file import check_file_permission, safe_listdir
from mindie_llm.runtime.utils.helpers.safety.hf import safe_get_config_dict
from mindie_llm.runtime.models.base import tool_calls_processor_registry


def get_router_ins(load_config: LoadConfig):
    """
    Loads the model router instance based on the configuration.

    Args:
        load_config: Configuration object containing model path and other parameters

    Returns:
        router_ins: Instance of the model router class for the specified model type
    """
    model_name_or_path = load_config.model_name_or_path
    check_file_permission(model_name_or_path)
    model_type_key = "model_type"
    config_dict = safe_get_config_dict(model_name_or_path)
    config_dict[model_type_key] = config_dict[model_type_key].lower()
    model_type = config_dict[model_type_key]

    # safe check
    current_path = os.path.dirname(os.path.abspath(__file__))
    supported_models = []
    for foldername in safe_listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, foldername))
        skip_base_folder = foldername != "base"
        skip_invalid_folder = not foldername.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(foldername)
    if model_type not in supported_models:
        raise NotImplementedError(
            f"unsupported model type: {model_type};"
            f"Please check if a folder named {model_type} exists in the"
            f" mindie_llm.runtime.models directory."
        )

    router_path = f"mindie_llm.runtime.models.{model_type}.router_{model_type}"
    router = importlib.import_module(router_path)
    router_cls = getattr(router, "".join(part.capitalize() for part in model_type.split("_")) + "Router")
    router_ins = router_cls(
        config_dict,
        load_config,
    )
    return router_ins
