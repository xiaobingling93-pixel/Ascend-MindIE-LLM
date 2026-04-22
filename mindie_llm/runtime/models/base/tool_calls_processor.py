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

import json
import random
import string
import re
from typing import Any, Optional, Literal, Pattern, Union, Callable

from pydantic import BaseModel

from mindie_llm.runtime.utils.helpers.json_completor import FillMode, complete_json_for_tool_calls
from mindie_llm.utils.log.logging import logger, message_filter

JSON_END_WITH_QUOTATION = '"}'
JSON_END_CURLY_BRACKET = "}"
SPECIAL_DELTA_PREFIXES = ['{"', "{'"]

INIT_RETURN_NONE = {}

CONTENT = "content"
TOOL_CALLS = "tool_calls"
NAME = "name"
ARGUMENTS = "arguments"

DELTA_TEXT = "delta_text"
DECODE_STREAM_RETURNVALUE = "decode_stream_returnvalue"
FILL_MODE = "fill_mode"
TOOL_CALL_PORTION = "tool_call_portion"

TOOL_CALL_ID_LEN = 8


class ToolCallsProcessor:
    """Base class for tool calls processor."""

    def __init__(self, model_version: str) -> None:
        """Initializes the tool call processor with a model version.

        Args:
            model_version: Version identifier of the model being used.
        """
        self.model_version = model_version

    @staticmethod
    def decode(content) -> dict:
        """Parse model output to extract tools call output.

        Args:
            content: Raw model output string.

        Returns:
            Dict: A dictionary containing the content under the 'content' key.
        """
        return {CONTENT: content}


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class DeltaToolCall(BaseModel):
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    index: int
    function: Optional[DeltaFunctionCall] = None


class ToolCallsProcessorWithXml(ToolCallsProcessor):
    def __init__(self, tokenizer) -> None:
        """Initializes the XML-based tool call processor.

        Args:
            tokenizer: Tokenizer instance used for decoding token IDs.
        """
        super().__init__(model_version="")
        self.tokenizer = tokenizer
        self.current_tool_name_sent: bool = False
        self.current_tool_arguments_sent: bool = False
        self.current_tool_id: int = -1

    @property
    def stream_tool_call_portion_regex(self) -> Pattern:
        return None

    @property
    def stream_tool_call_name_regex(self) -> Pattern:
        return None

    @property
    def tool_call_start_token(self) -> str:
        raise NotImplementedError("Subclasses must implement the 'tool_call_start_token' property.")

    @property
    def tool_call_end_token(self) -> str:
        raise NotImplementedError("Subclasses must implement the 'tool_call_end_token' property.")

    @property
    def tool_call_start_token_id(self) -> int:
        raise NotImplementedError("Subclasses must implement the 'tool_call_start_token_id' property.")

    @property
    def tool_call_end_token_id(self) -> int:
        raise NotImplementedError("Subclasses must implement the 'tool_call_end_token_id' property.")

    @property
    def tool_call_regex(self) -> Pattern:
        raise NotImplementedError("Subclasses must implement the 'tool_call_regex' property.")

    @property
    def decode_spilt_token(self) -> int:
        return self.tool_call_start_token

    @staticmethod
    def _random_tool_calls_id() -> str:
        """Generates a random tool call ID.

        Returns:
            str: A randomly generated tool call ID prefixed with 'call_'.
        """
        return "call_" + "".join(random.choice(string.ascii_letters + string.digits) for _ in range(TOOL_CALL_ID_LEN))

    @staticmethod
    def _get_tool_calls_json(matches: list[str]) -> list[dict[str, Any]]:
        """Parses matched strings into JSON objects and validates required fields.

        Args:
            matches: List of string matches expected to be JSON representations of tool calls.

        Returns:
            list[Dict[str, Any]]: List of valid tool call dictionaries containing 'name' and 'arguments',
                or an empty list if any parsing/validation fails.
        """
        try:
            tool_calls = [json.loads(match) for match in matches]
            for item in tool_calls:
                _ = item[NAME]
                _ = item[ARGUMENTS]
        except Exception:
            tool_calls = []
        return tool_calls

    def decode(self, content: str) -> dict[str, Any]:
        """
        example content:
        <tool_call>
            {"name": "get_rectangle_property", "arguments": {"perimeter": 14, "area": 15, "property": "length"}}
        </tool_call>
        """
        lines = content.strip()
        matches = self.tool_call_regex.findall(lines)
        tool_calls = self._get_tool_calls_json(matches) if matches else None
        if not tool_calls:
            return {CONTENT: lines}
        call_res = []
        for item in tool_calls:
            tool_call = {
                NAME: item[NAME],
                ARGUMENTS: json.dumps(item[ARGUMENTS], ensure_ascii=False)
                if isinstance(item[ARGUMENTS], dict)
                else item[ARGUMENTS],
            }
            res = {"type": "function", "id": self._random_tool_calls_id(), "function": tool_call}
            call_res.append(res)
        spilt_token = self.decode_spilt_token
        return {CONTENT: content.split(spilt_token)[0], TOOL_CALLS: call_res}

    def decode_stream(
        self,
        all_token_ids: list[int],
        prev_decode_index: int,
        curr_decode_index: int,
        skip_special_tokens: bool,
        delta_text: str,
    ) -> dict[str, Any]:
        """Processes streaming output to extract incremental tool call information.

        Args:
            all_token_ids: Full list of token IDs generated so far.
            prev_decode_index: Index up to which tokens were processed in the previous step.
            curr_decode_index: Current decoding index.
            skip_special_tokens: Whether to skip special tokens during decoding.
            delta_text: Newly generated text since the last step.

        Returns:
            Dict[str, Any]: A dictionary containing either content or tool calls, or empty if no update.
        """
        try:
            full_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=skip_special_tokens)
            history_token_ids = all_token_ids[:curr_decode_index]
            delta_text = self._preprocess_delta_text(delta_text)
            tool_call_portion_dict = self._decode_stream_tool_calls_portion(
                history_token_ids, all_token_ids, full_text, delta_text
            )
            if DECODE_STREAM_RETURNVALUE in tool_call_portion_dict.keys():
                return tool_call_portion_dict[DECODE_STREAM_RETURNVALUE]
            return self._decode_stream_tool_calls(tool_call_portion_dict)
        except Exception:
            logger.error("An exception occurred when parsing the function call. The large model response is invalid.")
            return INIT_RETURN_NONE

    def _preprocess_delta_text(self, delta_text: str) -> str:
        return delta_text

    def _decode_stream_tool_calls_portion(
        self, history_token_ids: list[int], all_token_ids: list[int], full_text: str, delta_text: str
    ) -> dict[str, Any]:
        """Determines the current state of tool call parsing in streaming mode.

        Args:
            history_token_ids: Token IDs processed before the current step.
            all_token_ids: All token IDs generated so far.
            full_text: Full decoded text from all_token_ids.
            delta_text: Newly generated text segment.

        Returns:
            Dict[str, Any]: A dictionary containing intermediate parsing state and results.
        """
        output_dict = {}
        output_dict.update({DELTA_TEXT: delta_text})
        # tool call has not yet started, return {CONTENT: delta_text}
        if self.tool_call_start_token_id not in all_token_ids:
            output_dict.update({DECODE_STREAM_RETURNVALUE: {CONTENT: delta_text}})
            return output_dict
        # count tool call symbol numbers
        prev_tool_start_count, prev_tool_end_count, cur_tool_start_count, cur_tool_end_count = self._count_tool_tokens(
            history_token_ids, all_token_ids
        )
        # set json fill mode
        fill_mode = FillMode.Full if self.current_tool_name_sent else FillMode.BraceOnly
        output_dict.update({FILL_MODE: fill_mode})
        # case1：common content
        if (
            cur_tool_start_count == cur_tool_end_count
            and prev_tool_end_count == cur_tool_end_count
            and self.tool_call_end_token not in delta_text
        ):
            output_dict.update({DECODE_STREAM_RETURNVALUE: {CONTENT: delta_text}})
            return output_dict
        # case2：start new tool_call
        elif cur_tool_start_count > cur_tool_end_count and cur_tool_start_count > prev_tool_start_count:
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.current_tool_arguments_sent = False
            text_portion, _ = delta_text.split(self.tool_call_start_token)
            if text_portion:
                output_dict.update({DECODE_STREAM_RETURNVALUE: {CONTENT: text_portion}})
            else:
                output_dict.update({DECODE_STREAM_RETURNVALUE: INIT_RETURN_NONE})
            return output_dict
        # case3：update tool_call
        elif cur_tool_start_count > cur_tool_end_count and cur_tool_start_count == prev_tool_start_count:
            tool_call_portion = full_text.split(self.tool_call_start_token)[-1]
            output_dict.update({TOOL_CALL_PORTION: tool_call_portion})
        # case4：end tool_call
        elif cur_tool_start_count == cur_tool_end_count and cur_tool_end_count >= prev_tool_end_count:
            # The arguments have been sent. The prevention part follows the end tag.
            if not self.current_tool_arguments_sent:
                matches = self.tool_call_regex.findall(full_text)
                current_tool_call = self._get_tool_calls_json(matches)
                cur_arguments = current_tool_call[-1].get(ARGUMENTS)
                if cur_arguments == {}:
                    output_dict.update(
                        {
                            DECODE_STREAM_RETURNVALUE: {
                                TOOL_CALLS: [
                                    DeltaToolCall(
                                        index=self.current_tool_id, function=DeltaFunctionCall(arguments="{}")
                                    ).model_dump(exclude_none=True)
                                ]
                            }
                        }
                    )
                else:
                    output_dict.update({DECODE_STREAM_RETURNVALUE: INIT_RETURN_NONE})
                return output_dict
            if JSON_END_WITH_QUOTATION not in delta_text:
                output_dict.update({DECODE_STREAM_RETURNVALUE: INIT_RETURN_NONE})
                return output_dict
            diff = delta_text[: delta_text.rindex(JSON_END_WITH_QUOTATION)] + JSON_END_WITH_QUOTATION

            output_dict.update(
                {
                    DECODE_STREAM_RETURNVALUE: {
                        TOOL_CALLS: [
                            DeltaToolCall(
                                index=self.current_tool_id, function=DeltaFunctionCall(arguments=diff)
                            ).model_dump(exclude_none=True)
                        ]
                    }
                }
            )
            return output_dict
        else:
            delta_text = delta_text.replace(self.tool_call_start_token, "").replace(self.tool_call_end_token, "")
            output_dict.update({DELTA_TEXT: delta_text, DECODE_STREAM_RETURNVALUE: {CONTENT: delta_text}})
        return output_dict

    def _count_tool_tokens(self, history_token_ids: list[int], all_token_ids: list[int]) -> tuple[int, int, int, int]:
        """Counts occurrences of tool call start and end tokens in token sequences.

        Args:
            history_token_ids: Token IDs from previous decoding steps.
            all_token_ids: All token IDs generated so far.

        Returns:
            tuple[int, int, int, int]: Previous start count, previous end count,
                current start count, current end count.
        """
        prev_tool_start_count = history_token_ids.count(self.tool_call_start_token_id)
        prev_tool_end_count = history_token_ids.count(self.tool_call_end_token_id)
        cur_tool_start_count = all_token_ids.count(self.tool_call_start_token_id)
        cur_tool_end_count = all_token_ids.count(self.tool_call_end_token_id)
        return prev_tool_start_count, prev_tool_end_count, cur_tool_start_count, cur_tool_end_count

    def _decode_stream_tool_calls(self, tool_call_portion_dict: dict[str, Any]) -> dict[str, Any]:
        """Processes the tool call portion of streaming output to generate delta updates.

        Args:
            tool_call_portion_dict: Dictionary containing tool call parsing state.

        Returns:
            Dict[str, Any]: Delta update for tool calls or empty dict if no valid update.

        _decode_stream_tool_calls()
        │
        ├── Phase 1: Function name not yet sent
        │   ├── JSON completion → extract "name"
        │   ├── Emit {function: {name: "xxx"}}
        │   └── Set current_tool_name_sent = True
        │
        └── Phase 2: Function name already sent
            ├── First argument emission (current_tool_arguments_sent = False)
            │   ├── Special handling for nested JSON
            │   ├── Compute the complete argument segment
            │   ├── Emit the argument structure
            │   └── Set current_tool_arguments_sent = True
            │
            └── Subsequent argument deltas (current_tool_arguments_sent = True)
            ├── Handle extraneous trailing closing braces
            └── Directly emit delta_text as argument delta
        """
        try:
            tool_call_portion = tool_call_portion_dict[TOOL_CALL_PORTION]
            delta_text = tool_call_portion_dict[DELTA_TEXT]
            fill_mode = tool_call_portion_dict[FILL_MODE]
            current_tool_call = (
                complete_json_for_tool_calls(tool_call_portion or "{}", fill_mode) if tool_call_portion else None
            )
        except Exception:
            # Invalid JSON fragment newline characters.
            return INIT_RETURN_NONE

        # case1：send function name
        if not self.current_tool_name_sent:
            if current_tool_call is None or not current_tool_call.get(NAME):
                return INIT_RETURN_NONE
            self.current_tool_name_sent = True
            # OpenAI format
            return {
                TOOL_CALLS: [
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=self._random_tool_calls_id(),
                        function=DeltaFunctionCall(name=current_tool_call.get(NAME)),
                    ).model_dump(exclude_none=True)
                ]
            }

        delta = {}
        # case2：send param
        cur_arguments = current_tool_call.get(ARGUMENTS)
        if cur_arguments and not self.current_tool_arguments_sent:
            # case2-1:send arguments contains structure.example {"arguments":"{\"order_id\": \""}
            # get the location where previous args differ from current
            cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
            last_brace_index = cur_arguments_json.rfind("}")
            if last_brace_index == -1:
                return INIT_RETURN_NONE
            # Detection of new JSON object begins
            if any(prefix in delta_text for prefix in SPECIAL_DELTA_PREFIXES):
                if delta_text not in tool_call_portion:
                    return INIT_RETURN_NONE
                pattern = r'["\']arguments["\']\s*:\s*(\{.*\}|\[.*\]|[^,;}\n]*)'
                matched = re.search(pattern, tool_call_portion, re.DOTALL)
                if matched:
                    argument_portion = matched.group(1).strip()
                else:
                    argument_portion = None
            else:
                argument_portion = cur_arguments_json[: last_brace_index + 1]
            # Calculate the real delta
            start_pos = argument_portion.rfind(delta_text)
            if start_pos == -1:
                return INIT_RETURN_NONE
            arguments_delta = argument_portion[: start_pos + len(delta_text)]
            # Send initial parameter delta and mark
            delta = {
                TOOL_CALLS: [
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=arguments_delta).model_dump(exclude_none=True),
                    ).model_dump(exclude_none=True)
                ]
            }
            self.current_tool_arguments_sent = True
        # Subsequent parameter incremental update
        elif cur_arguments and self.current_tool_arguments_sent:
            # case2-2:arguments delta content
            if (
                isinstance(delta_text, str)
                and len(delta_text.rstrip()) >= 1
                and delta_text.rstrip()[-1] == JSON_END_CURLY_BRACKET
            ):
                # delete the end brace of the whole tool call json
                if _count_closing_braces_at_end(tool_call_portion) > _count_closing_braces_at_end(
                    json.dumps(cur_arguments, ensure_ascii=False)
                ):
                    delta_text = delta_text.rstrip()[:-1]
            delta = {
                TOOL_CALLS: [
                    DeltaToolCall(
                        index=self.current_tool_id, function=DeltaFunctionCall(arguments=delta_text)
                    ).model_dump(exclude_none=True)
                ]
            }
        return delta


def _count_closing_braces_at_end(text: str) -> int:
    """Counts consecutive closing braces '}' at the end of a string.

    Args:
        text: Input string to inspect.

    Returns:
        int: Number of trailing '}' characters.
    """
    text = text.rstrip()
    count = 0
    for char in reversed(text):
        if char == "}":
            count += 1
        else:
            break
    return count


class ToolCallsProcessorManager:
    """
    A class for manager and register Tools Call Processors.
    Adapted from vllm/vllm/entrypoints/openai/tool_parsers/abstract_tool_parser.py

    Example for register a new module:
    >>> # import ToolCallsProcessorManager
    >>> from atb_llm.models.base.tool_call_parser import ToolCallsProcessorManager
    >>>
    >>> # using as decorator:
    >>> @ToolCallsProcessorManager.register_module(module_names=["module_name"], force=True)
    >>> class ModuleName(ToolCallsProcessor):
    >>>     # define your module
    >>>
    >>> # using as function:
    >>> class ModuleName(ToolCallsProcessor):
    >>>     # define your module
    >>> ToolCallsProcessorManager.register_module(module_names=["module_name"], force=True, module=ModuleName)
    >>> ..
    """

    _tool_calls_processors: dict[str, type] = {}

    @classmethod
    def get_tool_calls_processors(cls) -> dict[str, type]:
        """Retrieves all registered tool call processors.

        Returns:
            Dict[str, type]: A dictionary mapping processor names to their classes.
        """
        return cls._tool_calls_processors

    @classmethod
    def get_tool_calls_processor(cls, name: str) -> type:
        """Retrieves a specific tool call processor by name.

        Args:
            name: Name of the registered processor.

        Returns:
            type: The processor class.

        Raises:
            KeyError: If the specified name is not registered.
        """
        if name not in cls._tool_calls_processors:
            error_msg = f"'{name}' not found in registered tool_call_processors"
            raise KeyError(message_filter(error_msg))
        return cls._tool_calls_processors[name]

    @classmethod
    def remove_tool_calls_processor(cls, name: str) -> None:
        """Removes a registered tool call processor by name.

        Args:
            name: Name of the processor to remove.
        """
        if name not in cls._tool_calls_processors:
            warn_msg = f"'{name}' not found in tool_call_processors, cannot be removed"
            logger.warning(message_filter(warn_msg))
            return
        cls._tool_calls_processors.pop(name)
        debug_msg = f"'{name}' is successfully removed in tool_call_processors"
        logger.debug(message_filter(debug_msg))

    @classmethod
    def register_module(
        cls,
        module_names: Optional[Union[str, list[str], None]] = None,
        force: bool = True,
        module: Union[type, None] = None,
    ) -> Union[type, Callable]:
        """
        Register a ToolCallsProcessor module with given name(s) to the ToolCallParserManager.
        Can be used as a decorator or a function.

        Args:
            module_names (Optional[Union[str, list[str], None]]):
                The names of the module to be registered. Can be string, list of string or None.
                When set to None, the name of the given module will be used.
            force (bool, optional): Whether to force register the module even if the given name is already existed.
            module (Union[type, None], optional): The module to be registered.

        Returns:
            Union[type, Callable]: The registered module.

        """

        def _register_module(module: type, module_names: Optional[Union[str, list[str]]] = None, force: bool = True):
            if not issubclass(module, ToolCallsProcessor):
                raise TypeError("module to be registered must be subclass of ToolCallsProcessor")

            # ensure module_names is a list
            if not isinstance(module_names, list):
                module_names = [module_names]

            for name in module_names:
                # check name format
                if name is None:
                    name = module.__name__
                if not isinstance(name, str):
                    logger.warning(
                        f"'module_names' must be None, str or a list of str, but got '{type(name)}', "
                        f"will skip this module register"
                    )
                    continue
                # check if name is existed
                if not force and name in cls._tool_calls_processors:
                    warn_msg = (
                        f"'{name}' is already registered with module '{cls._tool_calls_processors[name].__name__}', "
                        f"will skip this module register"
                    )
                    logger.warning(message_filter(warn_msg))
                    continue
                cls._tool_calls_processors[name] = module
                debug_msg = f"The module '{module.__name__}' is successfully registered with name '{name}'"
                logger.debug(message_filter(debug_msg))

        if not isinstance(force, bool):
            raise TypeError(f"'force' must be a boolean, but got '{type(force)}'")

        def _register(module: type) -> type:
            _register_module(module=module, module_names=module_names, force=force)
            return module

        # When module is given, used as a function
        if module is not None:
            return _register(module)
        # otherwise, used as a decorator
        return _register
