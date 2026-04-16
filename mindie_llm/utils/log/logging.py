# Copyright (c) Huawei Technologies Co., Ltd. 2023-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import re
from mindie_llm.utils.log.logging_base import get_logger, Component


_MAX_MSG_LEN = 4096
_SPECIAL_CHARS = [
    "\n",
    "\r",
    "\f",
    "\t",
    "\v",
    "\b",
    "//",
    "\\",
    "&",
    "\u000a",
    "\u000d",
    "\u000c",
    "\u000b",
    "\u0008",
    "\u007f",
]


def print_log(rank_id, logger_fn, msg, need_filter=False):
    if rank_id != 0:
        return
    if need_filter:
        msg = message_filter(str(msg))
    logger_fn(msg, stacklevel=2)


def message_filter(msg: str):
    """
    Truncate message exceeding the limit and filter special characters.
    """
    if len(msg) > _MAX_MSG_LEN:
        msg = msg[:_MAX_MSG_LEN] + "..."
    for item in _SPECIAL_CHARS:
        msg = msg.replace(item, " ")
    msg = re.sub(r" {5,}", "    ", msg)
    return msg


logger = get_logger(Component.LLM)
