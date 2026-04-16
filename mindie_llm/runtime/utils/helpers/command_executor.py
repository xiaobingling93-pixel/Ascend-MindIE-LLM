# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import subprocess


def execute_command(cmd_list: list[str]) -> str:
    """
    Execute a shell command with the given arguments and return its stdout as a string.

    This function runs a command without invoking a shell (shell=False) for security reasons.
    It captures stdout and stderr, waits for the process to complete (with a timeout of 1000 seconds),
    and returns the decoded stdout as a UTF-8 string. If the command fails or times out,
    the exception is propagated to the caller.

    Args:
        cmd_list (List[str]): A list of command and its arguments, e.g., ["ls", "-l", "/tmp"].

    Returns:
        str: The decoded stdout output from the command.
    """

    with subprocess.Popen(cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        out, _ = p.communicate(timeout=1000)
    res = out.decode()
    return res
