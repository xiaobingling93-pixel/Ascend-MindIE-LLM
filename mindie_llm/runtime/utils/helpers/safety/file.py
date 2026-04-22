# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from functools import reduce
import os
import stat
from typing import IO, Optional, Any
from mindie_llm.runtime.utils.helpers.safety.path import standardize_path, MAX_PATH_LENGTH

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_FILENUM_PER_DIR = 1024
MAX_LINENUM_PER_FILE = 10 * 1024 * 1024

FLAG_OS_MAP = {
    "r": os.O_RDONLY,
    "r+": os.O_RDWR,
    "w": os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
    "w+": os.O_CREAT | os.O_TRUNC | os.O_RDWR,
    "a": os.O_CREAT | os.O_APPEND | os.O_WRONLY,
    "a+": os.O_CREAT | os.O_APPEND | os.O_RDWR,
    "x": os.O_CREAT | os.O_EXCL,
    "b": getattr(os, "O_BINARY", 0),
}


def safe_open(
    file_path: str,
    mode: str = "r",
    encoding: Optional[str] = None,
    permission_mode: int = 0o600,
    is_exist_ok: bool = True,
    max_path_length: int = MAX_PATH_LENGTH,
    max_file_size: int = MAX_FILE_SIZE,
    check_link: bool = True,
) -> IO[Any]:
    """
    Safely open a file with comprehensive security checks.

    This function performs path standardization, safety validation, and file opening with appropriate permissions.
    It includes checks for path length, symbolic links, special characters, file size, and directory permissions.

    Args:
        file_path (str): Path to the target file.
        mode (str): File open mode (e.g., 'r', 'w', 'a', 'rb', 'wb'). Default is 'r'.
        encoding (Optional[str]): Text encoding for text mode (e.g., 'utf-8'). Default is None.
        permission_mode (int): File permission mode (e.g., 0o600 for read/write by owner only). Default is 0o600.
        is_exist_ok (bool): Whether to allow existing files (for write operations). Default is True.
        **kwargs: Additional parameters for path standardization and safety checks:
            - max_path_length (int): Maximum allowed path length. Default is MAX_PATH_LENGTH.
            - max_file_size (int): Maximum allowed file size in bytes. Default is MAX_FILE_SIZE.
            - check_link (bool): Whether to check for symbolic links. Default is True.

    Returns:
        IO: A file object corresponding to the opened file.

    Raises:
        TypeError: If file_path is None or not a string.
        ValueError: If path contains special characters, exceeds length limits, or is a symbolic link.
        PermissionError: If current user lacks permission to access the file or directory.
        FileExistsError: If file exists but is not allowed (when is_exist_ok is False).
        FileNotFoundError: If file does not exist but is required (for read mode).
        OSError: If system-level errors occur during file operations.
    """
    file_path = standardize_path(file_path, max_path_length, check_link)
    check_file_safety(file_path, mode, is_exist_ok, max_file_size)

    flags = []
    for item in list(mode):
        if item == "+" and flags:
            flags[-1] = f"{flags[-1]}+"
            continue
        flags.append(item)
    flags = [FLAG_OS_MAP.get(mode, os.O_RDONLY) for mode in flags]
    total_flag = reduce(lambda a, b: a | b, flags)

    return os.fdopen(os.open(file_path, total_flag, permission_mode), mode, encoding=encoding)


def check_user_write_permission(file_path: str) -> bool:
    """
    Check if the current user has write permission for the specified file.

    Args:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the current user has write permission, False otherwise.
    """
    st = os.stat(file_path)
    return st.st_mode & stat.S_IWUSR


def check_uid_permission(path: str) -> None:
    """
    Check if the current user has permission to access the path based on ownership and group.

    Args:
        path (str): Path to check.

    Raises:
        PermissionError: If the current user is not root, not the owner, and not in the same group as the owner.
    """
    path_stat = os.stat(path)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    cur_uid = os.geteuid()
    cur_gid = os.getgid()
    if not (cur_uid == 0 or cur_uid == path_owner or path_gid == cur_gid):
        raise PermissionError(
            "The current user does not have permission to access the input path. "
            "Because he is neither the root user nor the path owner, "
            "nor a member of the same user group as the path owner. "
            "Please check and make sure to satisfy at least one of the conditions above."
        )


def check_world_write_permission(file_path: str) -> None:
    """
    Check if the specified file is writable by others who are neither the owner nor in the group.

    Args:
        file_path (str): Path to the file to check.

    Raises:
        PermissionError: If the file is writable by others.
    """
    file_stat = os.stat(file_path)
    mode = file_stat.st_mode  # Extract the file's mode (permissions and type) from metadata.
    perm = mode & 0o777  # Mask the mode to retain only permission bits (0o777).
    if perm & stat.S_IWOTH:  # Check if others have write permission (S_IWOTH).
        required_perm = perm & ~stat.S_IWOTH  # Clear others' write permission bit to get required permissions.
        raise PermissionError(
            "The file should not be writable by others who are neither the owner nor in the group. "
            f"Current permission: {oct(perm)}. "
            f"Required permission: {oct(required_perm)}."
        )


def check_file_permission(file_path: str, is_internal_file: bool = False) -> None:
    """
    Check file permissions based on environment variable `MINDIE_CHECK_INPUTFILES_PERMISSION`
        and internal file flag.


    Args:
        file_path (str): Path to the file.
        is_internal_file (bool): Whether the file is internal (default is False).

    Raises:
        PermissionError: If permission checks fail.
    """
    check_inputfiles_permission = os.getenv("MINDIE_CHECK_INPUTFILES_PERMISSION", "1") != "0"
    check_permission_flag = is_internal_file or check_inputfiles_permission
    if check_permission_flag:
        check_uid_permission(file_path)
        check_world_write_permission(file_path)


def check_file_size_lt(path: str, max_file_size: int = MAX_FILE_SIZE) -> None:
    """
    Validate that the file size is within the allowed limit.

    Args:
        path (str): Path to the file.
        max_file_size (int): Maximum allowed file size in bytes.

    Raises:
        ValueError: If file size exceeds the limit.
    """
    file_size = os.path.getsize(path)
    if file_size > max_file_size:
        raise ValueError(f"The size of file should not be greater than {max_file_size}, but got {file_size}.")


def check_file_safety(
    file_path: str,
    mode: str = "r",
    is_exist_ok: bool = True,
    max_file_size: int = MAX_FILE_SIZE,
    is_check_file_size: bool = True,
) -> None:
    """
    Perform comprehensive safety checks on the file before opening.

    Args:
        file_path (str): Path to the file.
        mode (str): File open mode (e.g., 'r', 'w').
        is_exist_ok (bool): Whether to allow the file to exist (for write operations).
        max_file_size (int): Maximum allowed file size in bytes.
        is_check_file_size (bool): Whether to check file size.

    Raises:
        FileExistsError: If file exists but is not allowed (when is_exist_ok is False).
        FileNotFoundError: If file does not exist but is required (for read mode).
        PermissionError: If directory permissions are invalid.
        ValueError: If file size exceeds limit.
    """
    if os.path.exists(file_path):
        if not is_exist_ok:
            raise FileExistsError(
                "The file is expected not to exist, but it already does. Please check the input path."
            )
        if is_check_file_size:
            check_file_size_lt(file_path, max_file_size)
        file_dir = file_path
    else:
        if mode == "r" or mode == "r+":
            raise FileNotFoundError("The file is expected to exist, but it does not. Please check the input path.")
        file_dir = os.path.dirname(file_path)

    check_file_permission(file_dir)


def safe_readlines(file_obj: IO, max_line_num: int = MAX_LINENUM_PER_FILE) -> list[str]:
    """
    Read lines from a file object with line count validation.

    Args:
        file_obj (IO): Opened file object.
        max_line_num (int): Maximum allowed lines in the file.

    Returns:
        list[str]: list of lines read from the file.

    Raises:
        ValueError: If line count exceeds the limit.
    """
    lines = file_obj.readlines()
    line_num = len(lines)
    if line_num > max_line_num:
        raise ValueError(
            f"The file line num is {line_num}, which exceeds the limit {max_line_num}. Please check the input file."
        )
    return lines


def safe_listdir(file_path: str, max_file_num: int = MAX_FILENUM_PER_DIR) -> list[str]:
    """
    list directory contents with file count validation.

    Args:
        file_path (str): Directory path.
        max_file_num (int): Maximum allowed files in the directory.

    Returns:
        list[str]: list of filenames in the directory.

    Raises:
        ValueError: If file count exceeds the limit.
    """
    filenames = os.listdir(file_path)
    file_num = len(filenames)
    if file_num > max_file_num:
        raise ValueError(
            f"The file num in dir is {file_num}, which exceeds the limit {max_file_num}. Please check the input path."
        )
    return filenames


def safe_chmod(file_path: str, permission_mode: int) -> None:
    """
    Change file permissions safely after validating path and permissions.

    Args:
        file_path (str): Path to the file.
        permission_mode (int): New permission mode (e.g., 0o600).

    Raises:
        PermissionError: If current user lacks permission to change permissions.
    """
    standard_path = standardize_path(file_path)
    check_file_permission(standard_path)
    os.chmod(file_path, permission_mode)
