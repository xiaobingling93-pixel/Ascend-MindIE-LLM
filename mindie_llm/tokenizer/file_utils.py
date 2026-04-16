#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import argparse
import stat
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

MAX_PATH_LENGTH = 4096
MAX_FILE_SIZE = 10 * 1024 * 1024
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


def safe_open(file_path: str, mode="r", encoding=None, permission_mode=0o600, is_exist_ok=True, **kwargs):
    """
    :param file_path: 文件路径
    :param mode: 文件打开模式
    :param encoding: 文件编码方式
    :param permission_mode: 文件权限模式
    :param is_exist_ok: 是否允许文件存在
    :param max_path_length: 文件路径最大长度
    :param max_file_size: 文件最大大小，单位: 字节, 默认值10MB
    :param check_link: 是否校验软链接
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
    check path
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
        raise argparse.ArgumentTypeError("The file path should not be None.")


def check_path_is_link(path: str):
    if os.path.islink(os.path.normpath(path)):
        raise argparse.ArgumentTypeError("The path should not be a symbolic link file.")


def check_path_length_lt(path: str, max_path_length=MAX_PATH_LENGTH):
    if path.__len__() > max_path_length:
        raise argparse.ArgumentTypeError(f"The length of path should not be greater than {max_path_length}.")


def check_file_size_lt(path: str, max_file_size=MAX_FILE_SIZE):
    if os.path.getsize(path) > max_file_size:
        raise argparse.ArgumentTypeError(f"The size of path should not be greater than {max_file_size}.")


def check_owner(path: str):
    """
    check the path owner
    param: the input path
    """
    try:
        path_stat = os.stat(path)
    except FileNotFoundError as e:
        raise argparse.ArgumentTypeError(f"The path {path} does not exist.") from e
    except PermissionError as e:
        raise argparse.ArgumentTypeError(f"No permission to access the path {path}.") from e
    except OSError as e:
        raise argparse.ArgumentTypeError("Failed to get the path status.") from e
    except Exception as e:
        raise argparse.ArgumentTypeError("Unknown error occurred when getting the path status.") from e
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    user_check = path_owner == os.getuid() and path_owner == os.geteuid()
    if not (path_owner == 0 or path_gid in os.getgroups() or user_check):
        raise argparse.ArgumentTypeError("The path is not owned by current user or root")


def check_access_rights(file_path: str, mode=0o750):
    try:
        file_stat = os.stat(file_path)
    except FileNotFoundError as e:
        raise argparse.ArgumentTypeError(f"The path {file_path} does not exist.") from e
    except PermissionError as e:
        raise argparse.ArgumentTypeError(f"No permission to access the path {file_path}.") from e
    except OSError as e:
        raise argparse.ArgumentTypeError("Failed to get the path status.") from e
    except Exception as e:
        raise argparse.ArgumentTypeError("Unknown error occurred when getting the path status.") from e

    current_permissions = file_stat.st_mode & 0o777
    required_permissions = mode & 0o777

    for i in range(3):
        cur_perm = (current_permissions >> (i * 3)) & 0o7
        max_perm = (required_permissions >> (i * 3)) & 0o7
        if (cur_perm | max_perm) != max_perm:
            err_msg = (
                f"File: {file_path} Check {['Other group', 'Owner group', 'Owner'][i]} failed: "
                f"Current permission is {cur_perm}, but required permission is {max_perm}. "
            )
            raise argparse.ArgumentTypeError(err_msg)


def check_path_permission(file_path: str, mode=0o750):
    check_owner(file_path)
    check_access_rights(file_path, mode)


def check_file_safety(file_path: str, mode="r", is_exist_ok=True, max_file_size=MAX_FILE_SIZE):
    if is_path_exists(file_path):
        if not is_exist_ok:
            raise argparse.ArgumentTypeError("The file already exists.")
        check_file_size_lt(file_path, max_file_size)
        file_dir = file_path
    else:
        if mode == "r" or mode == "r+":
            raise argparse.ArgumentTypeError("The file doesn't exist.")
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
    if len(lines) > max_line_num:
        raise argparse.ArgumentTypeError(f"The file line num is {len(lines)}, which exceeds the limit {max_line_num}.")
    return lines


LOG_LVL_ENV = "MINDIE_LOG_LEVEL"
LOG_PATH_ENV = "MINDIE_LOG_PATH"

LOG_MOD_ALL = "all"
LOG_MOD_SVR = "server"

LOG_LVL_INFO = "info"

LOG_PATH_BASE = "~/mindie/log"
LOG_PATH_MID = "log"
LOG_PATH_LAST = "debug"

LOG_FILE_MODE = 0o640

# 默认限制日志大小20M，缓存10个
LOG_FILE_SIZE = 20 * 1024 * 1024
LOG_FILE_NUM = 10


def parse_log_env(key: str, default_val: str):
    env_str = os.getenv(key, "")
    if len(env_str) == 0:
        return default_val

    env_list = env_str.split(";")
    modules = {}
    for s in env_list:
        s = s.strip()
        if ":" not in s and len(s) > 0:
            modules[LOG_MOD_ALL] = s
            continue
        pair = s.split(":")
        if len(pair) != 2:
            continue
        k = pair[0].strip()
        v = pair[1].strip()
        if len(k) > 0 and len(v) > 0:
            modules[k] = v

    if LOG_MOD_SVR in modules:
        return modules[LOG_MOD_SVR]
    elif LOG_MOD_ALL in modules:
        return modules[LOG_MOD_ALL]
    else:
        return default_val


def parse_log_path():
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y%m%d%H%S%f")[:-3]
    file_name = f"mindie-llm-tokenizer_{os.getpid()}_{now_str}.log"

    log_base_path = LOG_PATH_BASE
    path_str = parse_log_env(LOG_PATH_ENV, "")
    if len(path_str) == 0:
        dir_path = os.path.join(log_base_path, LOG_PATH_LAST)
    elif path_str.startswith("/") or path_str.startswith("~"):
        dir_path = os.path.join(path_str, LOG_PATH_MID, LOG_PATH_LAST)
    else:
        dir_path = os.path.join(log_base_path, path_str, LOG_PATH_LAST)

    dir_path = standardize_path(os.path.expanduser(dir_path))
    check_file_safety(dir_path, "w")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, mode=LOG_FILE_MODE, exist_ok=True)
    elif not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(f"The path {dir_path} is not a dir.")

    full_path = standardize_path(os.path.join(dir_path, file_name))
    return full_path


def parse_log_level():
    lvl_str = parse_log_env(LOG_LVL_ENV, LOG_LVL_INFO)
    lvl_map = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warn": logging.WARN,
        LOG_LVL_INFO: logging.INFO,
        "debug": logging.DEBUG,
    }
    if lvl_str in lvl_map:
        return lvl_map[lvl_str]
    else:
        return logging.INFO


def get_tokenizer_logger():
    logger = logging.getLogger(f"tokenizer-{os.getpid()}")
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    try:
        log_lvl = parse_log_level()
        logger.setLevel(log_lvl)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(process)d] [%(thread)d] [tokenizer] [%(levelname)s] "
            "[%(filename)s-%(lineno)d] : %(message)s"
        )

        console_hd = logging.StreamHandler()
        console_hd.setLevel(log_lvl)
        console_hd.setFormatter(formatter)
        logger.addHandler(console_hd)

        log_file = parse_log_path()
        file_hd = RotatingFileHandler(log_file, maxBytes=LOG_FILE_SIZE, backupCount=LOG_FILE_NUM, delay=True)
        file_hd.setLevel(log_lvl)
        file_hd.setFormatter(formatter)
        logger.addHandler(file_hd)
    except Exception as e:
        print(f"[WARN]Failed to initialize logger: {e}")

    return logger
