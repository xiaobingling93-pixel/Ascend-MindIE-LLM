# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
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

MAX_PATH_LENGTH = 4096
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_LINENUM_PER_FILE = 10 * 1024 * 1024
MAX_LOG_DIR_PERM = 0o750

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


def makedir_and_change_permissions(path, mode=MAX_LOG_DIR_PERM):
    parts = path.strip(os.sep).split(os.sep)
    current_path = os.sep

    for part in parts:
        current_path = os.path.join(current_path, part)
        if not os.path.exists(current_path):
            os.makedirs(current_path, mode, exist_ok=True)


def safe_open(file_path: str, mode="r", encoding=None, permission_mode=0o600, is_exist_ok=True, **kwargs):
    """
    :param file_path: File path
    :param mode: File opening mode
    :param encoding: File encoding
    :param permission_mode: File permission mode
    :param is_exist_ok: Whether to allow file existence
    :param max_path_length: Maximum path length
    :param max_file_size: Maximum file size in bytes, default 10MB
    :param check_link: Whether to check symbolic links
    :param kwargs:
    :return:
    """
    max_path_length = kwargs.get("max_path_length", MAX_PATH_LENGTH)
    max_file_size = kwargs.get("max_file_size", MAX_FILE_SIZE)
    check_link = kwargs.get("check_link", True)

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


def standardize_path(path: str, max_path_length=MAX_PATH_LENGTH, check_link=True):
    """
    Check path
    param: path
    return: data real path after check
    """
    check_path_is_none(path)
    check_path_length_lt(path, max_path_length)
    if check_link:
        check_path_is_link(path)
    path = os.path.realpath(path)
    return path


def is_path_exists(path: str):
    return os.path.exists(path)


def check_path_is_none(path: str):
    if path is None:
        raise TypeError("The file path should not be None.")


def check_path_is_link(path: str):
    if os.path.islink(os.path.normpath(path)):
        raise ValueError(f"The path should not be a symbolic link file. Please check the input path:{path}.")


def check_path_length_lt(path: str, max_path_length=MAX_PATH_LENGTH):
    path_length = path.__len__()
    if path_length > max_path_length:
        raise ValueError(
            f"The length of path should not be greater than {max_path_length}, but got {path_length}. "
            f"Please check the input path within the valid length range:{path[:max_path_length]}."
        )


def check_file_size_lt(path: str, max_file_size=MAX_FILE_SIZE):
    file_size = os.path.getsize(path)
    if file_size > max_file_size:
        raise ValueError(
            f"The size of file should not be greater than {max_file_size}, but got {file_size}. "
            f"Please check the input path:{path}."
        )


def check_owner(path: str):
    """
    Check the path owner
    param: the input path
    """
    path_stat = os.stat(path)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    cur_uid = os.geteuid()
    cur_gid = os.getgid()
    if not (cur_uid == 0 or cur_uid == path_owner or path_gid == cur_gid):
        raise PermissionError(
            f"The current user does not have permission to access the path:{path}. "
            "Because he is not root or the path owner, "
            "and not in the same user group with the path owner. "
            "Please check and make sure to satisfy at least one of the conditions above."
        )


def check_other_write_permission(file_path: str):
    """
    Check if the specified file is writable by group users or others who are neither the owner nor in the group
    param: the path to the file to be checked
    """
    # Get the status of the file
    file_stat = os.stat(file_path)
    # Get the mode (permission) of the file
    mode = file_stat.st_mode
    # check the write permission for group and others
    if mode & stat.S_IWGRP:
        raise PermissionError(
            "The file should not be writable by group users. "
            f"Please check the input path:{file_path}, and change mode to {mode & ~stat.S_IWGRP}."
        )
    if mode & stat.S_IWOTH:
        raise PermissionError(
            "The file should not be writable by others who are neither the owner nor in the group. "
            f"Please check the input path:{file_path}, and change mode to {mode & ~stat.S_IWOTH}."
        )


def check_path_permission(file_path: str):
    check_owner(file_path)
    check_other_write_permission(file_path)


def check_file_safety(file_path: str, mode="r", is_exist_ok=True, max_file_size=MAX_FILE_SIZE, is_check_file_size=True):
    if is_path_exists(file_path):
        if not is_exist_ok:
            raise FileExistsError(
                f"The file is expected not to exist, but it already does. Please check the input path:{file_path}."
            )
        if is_check_file_size:
            check_file_size_lt(file_path, max_file_size)
        file_dir = file_path
    else:
        if mode == "r" or mode == "r+":
            raise FileNotFoundError(
                f"The file is expected to exist, but it does not. Please check the input path:{file_path}."
            )
        file_dir = os.path.dirname(file_path)

    check_path_permission(file_dir)


def safe_chmod(file_path: str, permission_mode):
    standard_path = standardize_path(file_path)
    check_path_permission(standard_path)
    os.chmod(file_path, permission_mode)


def has_owner_write_permission(file_path: str):
    st = os.stat(file_path)
    return st.st_mode & stat.S_IWUSR


def safe_readlines(file_obj, max_line_num=MAX_LINENUM_PER_FILE):
    lines = file_obj.readlines()
    line_num = len(lines)
    if line_num > max_line_num:
        raise ValueError(
            f"The file line num is {line_num}, which exceeds the limit {max_line_num}. "
            f"Please check the input file:{file_obj.name}."
        )
    return lines
