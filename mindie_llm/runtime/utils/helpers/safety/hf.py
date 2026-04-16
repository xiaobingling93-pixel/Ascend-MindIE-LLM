# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from functools import wraps

from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer

from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.utils.helpers.safety.file import check_file_permission
from mindie_llm.runtime.utils.helpers.safety.path import standardize_path
from mindie_llm.runtime.utils.helpers.safety.url import filter_urls_from_error


EXTRA_EXP_INFO = (
    "Please check the input parameters model_path, kwargs and the version of transformers. "
    + "If the input parameters are valid, the required files exist in model_path and "
    + "the version of transformers is correct, make sure the folder's owner has execute permission. "
    + "Otherwise, please check the function stack for detailed exception information "
    + "and the logs of the llmmodels."
)


def check_file_and_catch_exception_decorator(func):
    """
    Decorator for Huggingface model loading functions to enforce path validation and exception handling.

    Args:
        func: Huggingface model loading function to wrap

    Returns:
        Wrapped function with path validation and exception handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        model_path = standardize_path(args[0], check_link=False)
        check_file_permission(model_path)
        try:
            return func(model_path, **kwargs)
        except EnvironmentError as error:
            filtered_error = filter_urls_from_error(error)
            logger.error(filtered_error)
            raise EnvironmentError(f"{func.__name__} failed. " + EXTRA_EXP_INFO) from filtered_error
        except Exception as error:
            filtered_error = filter_urls_from_error(error)
            logger.error(filtered_error)
            raise ValueError(f"{func.__name__} failed. " + EXTRA_EXP_INFO) from filtered_error

    return wrapper


@check_file_and_catch_exception_decorator
def safe_get_tokenizer_from_pretrained(model_path: str, **kwargs) -> AutoTokenizer:
    """A wrapper function of `AutoTokenizer.from_pretrained` which validates the path."""
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)


@check_file_and_catch_exception_decorator
def safe_get_config_dict(model_path: str, **kwargs) -> PretrainedConfig:
    """A wrapper of `PretrainedConfig.get_config_dict` which will validate the path."""
    config, _ = PretrainedConfig.get_config_dict(model_path, local_files_only=True, **kwargs)
    return config
