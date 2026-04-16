# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# Implement part of this file based on vllm-project/vllm
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
ToolCallProcessor Registry

Import all ToolCallProcessors to ensure that all ToolCallProcessors are registered to ToolCallsProcessorManager.
"""

from mindie_llm.utils.log.logging import logger


def register_all_tool_calls_processors():
    """
    Import to register all ToolCallProcessors
    """
    # Qwen3
    logger.debug("Successfully imported Qwen3 ToolCallProcessors")

    # DeepSeekV3
    logger.debug("Successfully imported DeepSeekV3.2 ToolCallProcessors")


register_all_tool_calls_processors()
