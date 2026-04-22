# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import os
import threading

from .. import file_utils

MAX_KEY_LENGTH = 4096 * 10

MB = 1024 * 1024
MAX_LOG_FILE_SIZE_MB = 500  # Maximum file size: 500MB
DEFAULT_MAX_FILE_SIZE_MB = 20  # Default rotation file size: 20MB
DEFAULT_MAX_FILES = 10  # Default number of rotation files: 10

_log_lock = threading.Lock()


def get_component_config(env_config: str, component: str) -> str:
    """
    Get the configuration value for a specific component from environment configuration string

    Args:
        env_config: Environment configuration string, format like "llm:DEBUG;llmmodels:INFO"
        component: Name of the component to get configuration for

    Returns:
        str: Configuration value for the component, returns empty string if not found
    """
    if len(env_config) >= MAX_KEY_LENGTH:
        env_config = env_config[:MAX_KEY_LENGTH]

    for comp_config in env_config.split(";")[::-1]:
        parts = comp_config.split(":")
        if len(parts) == 1:
            return parts[0].strip()
        elif len(parts) == 2 and parts[0].strip().lower() == component:
            return parts[1].strip()
    return ""


def create_log_dir_and_check_permission(file_path: str):
    file_path = file_utils.standardize_path(file_path)
    if os.path.isdir(file_path):
        raise argparse.ArgumentTypeError("`MINDIE_LOG_PATH` only supports paths that end with a file.")
    dirs = os.path.dirname(file_path)
    try:
        _log_lock.acquire()

        if os.path.exists(file_path):
            # Owner and OTH-permission check
            file_utils.check_path_permission(file_path)
        elif os.path.exists(dirs):
            if file_utils.has_owner_write_permission(dirs):
                # Owner and OTH-permission check
                file_utils.check_path_permission(dirs)
            else:
                raise PermissionError("{dirs} should have write permission.")
        else:
            try:
                os.makedirs(dirs, mode=0o750, exist_ok=True)
            except PermissionError as e:
                err_msg = (
                    f"Failed to create the log directory: {dirs}.Please add write permissions to the parent directory."
                )
                raise PermissionError(err_msg) from e
    finally:
        _log_lock.release()


def update_log_file_param(
    rotate_config: str, max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB, max_files: int = DEFAULT_MAX_FILES
) -> tuple[int, int]:
    """
    Update log file parameters

    Args:
        rotate_config: Configuration string, format like "-fs 100 -r 5"
        max_file_size_mb: Maximum file size (MB)
        max_files: Maximum number of files

    Returns:
        tuple[int, int]: (max_file_size_bytes, max_files)
    """
    max_file_size_bytes = max_file_size_mb * MB

    if not rotate_config:
        return max_file_size_bytes, max_files

    def validate_numeric_value(s: str, param_name: str) -> int:
        """Validate and convert numeric string

        Args:
            s: String to be validated
            param_name: Parameter name (for error messages)

        Returns:
            int: Converted number

        Raises:
            ValueError: When input is not a valid number, including original error message
        """
        try:
            return int(s)
        except ValueError as e:
            raise ValueError(f"{param_name} should be an integer, but got '{s}'. Original error: {str(e)}") from e

    # Split configuration string into list
    config_list = rotate_config.split()

    # Iterate through configuration, taking two elements at a time (option and value)
    for i in range(0, len(config_list), 2):
        if i + 1 >= len(config_list):
            continue

        option = config_list[i]
        value = config_list[i + 1]

        if option == "-fs":
            file_size_mb = validate_numeric_value(value, "Log file size (-fs)")
            if not (1 <= file_size_mb <= MAX_LOG_FILE_SIZE_MB):
                raise ValueError(
                    f"Log file size (-fs) should be between 1 and {MAX_LOG_FILE_SIZE_MB} MB, but got {file_size_mb} MB."
                )
            max_file_size_mb = file_size_mb
            max_file_size_bytes = max_file_size_mb * MB
        elif option == "-r":
            files = validate_numeric_value(value, "Log rotation count (-r)")
            if not (1 <= files <= 64):
                raise ValueError(f"Log rotation count (-r) should be between 1 and 64, but got {files}.")
            max_files = files

    return max_file_size_bytes, max_files
