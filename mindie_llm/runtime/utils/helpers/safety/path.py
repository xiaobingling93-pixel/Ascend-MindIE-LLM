# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import re

MAX_PATH_LENGTH = 4096


def standardize_path(path: str, max_path_length: int = MAX_PATH_LENGTH, check_link: bool = True) -> str:
    """
    Standardize and validate the path according to security requirements.

    Args:
        path (str): Input path string.
        max_path_length (int): Maximum allowed path length. Default is MAX_PATH_LENGTH.
        check_link (bool): Whether to check for symbolic links. Default is True.

    Returns:
        str: Standardized path after validation.

    Raises:
        TypeError: If path is None or not a string.
        ValueError: If path contains special characters, exceeds length limits, or is a symbolic link.
    """
    check_path_is_none(path)
    check_path_is_str(path)
    check_path_length_lt(path, max_path_length)
    if check_link:
        check_path_is_link(path)
    path = os.path.realpath(path)
    check_path_has_special_characters(path)
    return path


def check_path_is_none(path: str) -> None:
    """
    Validate that the path is not None.

    Args:
        path (str): Path to check.

    Raises:
        TypeError: If path is None.
    """
    if path is None:
        raise TypeError("The file path should not be None.")


def check_path_is_link(path: str) -> None:
    """
    Validate that the path is not a symbolic link.

    Args:
        path (str): Path to check.

    Raises:
        ValueError: If path is a symbolic link.
    """
    if os.path.islink(os.path.normpath(path)):
        raise ValueError("The path should not be a symbolic link file. Please check the input path.")


def check_path_is_str(path: str) -> None:
    """
    Validate that the path is a string.

    Args:
        path (str): Path to check.

    Raises:
        TypeError: If path is not a string.
    """
    if not isinstance(path, str):
        raise TypeError(f"The file path's type should be str, but get {type(path)}. Please check the input path.")


def check_path_has_special_characters(path: str) -> None:
    """
    Validate that the path contains no special characters.

    Args:
        path (str): Path to check.

    Raises:
        ValueError: If path contains special characters.
    """
    pattern = re.compile(r"[^0-9a-zA-Z_./-]")
    match_name = pattern.findall(path)
    if match_name:
        raise ValueError("The file path should not contain special characters.")


def check_path_length_lt(path: str, max_path_length: int = MAX_PATH_LENGTH) -> None:
    """
    Validate that the path length is within the allowed limit.

    Args:
        path (str): Path to check.
        max_path_length (int): Maximum allowed path length.

    Raises:
        ValueError: If path length exceeds the limit.
    """
    path_length = len(path)
    if path_length > max_path_length:
        raise ValueError(
            f"The length of path should not be greater than {max_path_length}, but got {path_length}. "
            f"Please check the input path within the valid length range."
        )
