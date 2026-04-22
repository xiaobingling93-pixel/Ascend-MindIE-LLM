# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import re


def filter_urls_from_error(error_message: Exception) -> Exception:
    """
    Filter out URLs from an exception's arguments by replacing them with '***'.

    This function processes all string arguments in the exception's `args` tuple,
    replacing any domain names, IPv4 addresses, or IPv6 addresses
    with '***' to prevent sensitive information leakage in error messages.

    Args:
        error_message (Exception): The exception object containing the error message.

    Returns:
        Exception: The modified exception object with URLs filtered out.
    """
    # Define regex patterns for different URL types
    domain_pattern = r"://(?:[a-zA-Z0-9.-]{1,253}(?:\.[a-zA-Z]{2,63})(?::[0-9]{1,5})?)"
    ipv4_pattern = (
        r"://(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?::[0-9]{1,5})?"
    )
    ipv6_pattern = r"://(?:\[[0-9a-fA-F:]{3,39}\])(?::[0-9]{1,5})?"
    url_pattern = rf"{domain_pattern}|{ipv4_pattern}|{ipv6_pattern}"

    # Process each argument in the exception's args tuple
    args = list(error_message.args)
    for i, arg in enumerate(args):
        if isinstance(arg, str):
            args[i] = re.sub(url_pattern, "***", arg)

    # Update the exception's args with filtered values
    error_message.args = tuple(args)
    return error_message
