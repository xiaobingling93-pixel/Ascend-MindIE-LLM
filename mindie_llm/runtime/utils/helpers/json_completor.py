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
import re
from enum import Enum

from mindie_llm.utils.log.logging import logger


class FillMode(Enum):
    """Enumeration defining strategies for completing partial JSON strings."""

    BraceOnly = 0
    Full = 1


def _skip_whitespace(text, index: int) -> int:
    """Advances the index past any whitespace characters.

    Args:
        text: The input string being parsed.
        index: Current position in the string.

    Returns:
        int: New index after skipping all consecutive whitespace characters.
    """
    n = len(text)
    while index < n and text[index].isspace():
        index += 1
    return index


def _parse_string(text: str, index: int) -> tuple:
    """Parses a JSON string value starting at the given index.

    Assumes the current character is a double quote (`"`).

    Args:
        text: The input string being parsed.
        index: Index of the opening double quote.

    Returns:
        tuple: A tuple containing the parsed string content and the new index after the closing quote.
    """
    # first char is '"', skip it
    index += 1
    result = ""
    n = len(text)
    while index < n:
        char = text[index]

        # Handle escape characters '\'
        if char == "\\":
            if index + 1 < n:
                next_char = text[index + 1]
                # Standard JSON escape sequences: \" \\ \/ \b \f \n \r \t
                if next_char in '"\\/bfnrt':
                    esc_map = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
                    result += esc_map.get(next_char, next_char)
                    index += 2  # 2: length of escape sequence
                    continue
                # Unicode escape sequence \uXXXX
                elif next_char == "u" and index + 5 < n:  # 5: last index of \uXXXX
                    hexcode = text[index + 2 : index + 6]
                    if re.fullmatch(r"[0-9a-fA-F]{4}", hexcode):  # '[0-9a-fA-F]{4}': 4 hexadecimal digits
                        result += chr(int(hexcode, 16))  # 16: hex to dec
                    index += 6  # 6: length of \uXXXX
                    continue
                # Unknown or invalid escape, just pass through the next char
                else:
                    if index + 1 < n:
                        result += text[index + 1]
                    index += 2  # 2: length of escape sequence
                    continue
            # Lone backslash at end of input
            index += 1
            break

        # End of string detected
        elif char == '"':
            index += 1
            return result, index

        # Normal character, append to result
        else:
            result += char
            index += 1
    return result, index


def _parse_number(text: str, index: int) -> tuple:
    """Parses a JSON number (integer or floating-point) starting at the given index.

    Args:
        text: The input string being parsed.
        index: Starting index of the number.

    Returns:
        tuple: A tuple containing the parsed number (or None if invalid)
        and the new index after the number.
    """
    n = len(text)
    start = index

    # Handle optional negative sign
    if index < n and text[index] == "-":
        index += 1

    # Parse integer part
    while index < n and text[index].isdigit():
        index += 1

    # Check for decimal point
    if index < n and text[index] == ".":
        index += 1
        # Parse fractional part
        while index < n and text[index].isdigit():
            index += 1

    # Check for exponent part
    if index < n and text[index] in "eE":
        index += 1
        # Optional exponent sign
        if index < n and text[index] in "+-":
            index += 1
        # Parse exponent digits
        while index < n and text[index].isdigit():
            index += 1

    num_str = text[start:index]

    try:
        # Empty or just '-' means invalid number
        if num_str == "" or num_str == "-":
            return None, index
        # Determine if float or int based on content
        return (float(num_str) if "." in num_str or "e" in num_str or "E" in num_str else int(num_str)), index
    except Exception as e:
        # Log parsing failure but continue gracefully
        logger.debug(f"Try parse number from {num_str} failed, exception is {str(e)}, skip it.")
        return None, index


def _parse_literal(text: str, index: int) -> tuple:
    """Parses JSON literals: true, false, or null.

    Args:
        text: The input string being parsed.
        index: Starting index to check for a literal.

    Returns:
        tuple: A tuple containing the parsed literal value (True, False, or None)
        or None if no literal matches, and the new index.
    """
    if text.startswith("true", index):
        return True, index + 4
    if text.startswith("false", index):
        return False, index + 5
    if text.startswith("null", index):
        return None, index + 4
    return None, index


def _skip_field(text: str, index: int) -> int:
    """Skips over an invalid or unparseable JSON field by balancing brackets.

    Used as a fallback when parsing fails.

    Args:
        text: The input string being parsed.
        index: Starting index of the invalid field.

    Returns:
        int: Index after the skipped field (after a comma or closing brace/bracket).
    """
    n = len(text)
    # Track nested { } and [ ] levels
    depth = 0
    while index < n:
        char = text[index]
        if char in "{[":
            depth += 1
        elif char in "}]":
            # unmatch closing bracket
            if depth == 0:
                return index
            depth -= 1
        elif char == "," and depth == 0:
            return index
        index += 1
    return index


def _parse_array(text: str, index: int) -> tuple:
    """Parses a JSON array starting at the given index.

    Assumes the current character is an opening bracket (`[`).

    Args:
        text: The input string being parsed.
        index: Index of the opening bracket.

    Returns:
        tuple: A tuple containing the parsed list and the new index after the closing bracket.
    """
    # skip opening '['
    index += 1
    arr = []
    index = _skip_whitespace(text, index)
    n = len(text)
    while index < n:
        if text[index] == "]":
            # end of array
            index += 1
            return arr, index

        if text[index] == '"':
            val, index = _parse_string(text, index)
        elif text[index] == "{":
            val, index = _parse_object(text, index)
        elif text[index] == "[":
            val, index = _parse_array(text, index)
        elif text[index].isdigit() or text[index] == "-":
            val, index = _parse_number(text, index)
        elif text.startswith("true", index) or text.startswith("false", index) or text.startswith("null", index):
            val, index = _parse_literal(text, index)

        else:
            # skip invalid field
            index = _skip_field(text, index)
            if index < n and text[index] == ",":
                index += 1
            continue

        arr.append(val)
        index = _skip_whitespace(text, index)

        if index < n and text[index] == ",":
            index += 1
            index = _skip_whitespace(text, index)
            continue
        elif index < n and text[index] == "]":
            index += 1
            return arr, index
        else:
            break

    return arr, index


def _parse_object(text: str, index: int) -> tuple:
    """Parses a JSON object starting at the given index.

    Assumes the current character is an opening brace (`{`).

    Args:
        text: The input string being parsed.
        index: Index of the opening brace.

    Returns:
        Tuple[Dict[str, Any], int]: A tuple containing the parsed dictionary and the new index after the closing brace.
    """
    # skip opening '{'
    index += 1
    obj = {}
    index = _skip_whitespace(text, index)
    n = len(text)

    while index < n:
        if text[index] == "}":
            # end of object
            index += 1
            return obj, index

        if text[index] == ",":
            # skip to next field
            index += 1
            index = _skip_whitespace(text, index)
            continue

        if text[index] == '"':
            # handle key
            key, index = _parse_string(text, index)
        else:
            index = _skip_field(text, index)
            if index < n and text[index] == ",":
                index += 1
            continue

        # skip whitespace between key and ":"
        index = _skip_whitespace(text, index)

        # make sure ":" exists
        if index >= n or text[index] != ":":
            index = _skip_field(text, index)
            if index < n and text[index] == ",":
                index += 1
            continue

        # skip ":" and whitespace between ":" and value
        index += 1
        index = _skip_whitespace(text, index)

        # handle value
        if index >= n:
            continue
        if text[index] == '"':
            val, index = _parse_string(text, index)
        elif text[index] == "{":
            val, index = _parse_object(text, index)
        elif text[index] == "[":
            val, index = _parse_array(text, index)
        elif text[index].isdigit() or text[index] == "-":
            val, index = _parse_number(text, index)
        elif text.startswith("true", index) or text.startswith("false", index) or text.startswith("null", index):
            val, index = _parse_literal(text, index)
        else:
            index = _skip_field(text, index)
            continue
        obj[key] = val
        index = _skip_whitespace(text, index)

        # handle comma
        if index < n and text[index] == ",":
            index += 1
            index = _skip_whitespace(text, index)
            continue

        # handle end of object
        elif index < n and text[index] == "}":
            index += 1
            return obj, index
        else:
            break
    return obj, index


def complete_json_for_tool_calls(json_str: str, mode: FillMode) -> dict:
    """Attempts to complete a potentially incomplete JSON string for tool call parsing.

    Args:
        json_str: Input string that should represent a partial or complete JSON object.
        mode: Strategy for completion:
            - FillMode.BraceOnly: Only appends missing closing braces (`}`).
            - FillMode.Full: Uses custom recursive descent parser to extract partial structure.

    Returns:
        dict: A dictionary representing the best-effort parsed JSON object.
                        Returns an empty dict if parsing fails.

    Raises:
        ValueError: If input is not a string or does not start with '{'.
    """
    if not isinstance(json_str, str):
        raise ValueError(f"Invalid input type to complete, expected str, got {type(json_str)}.")
    text = json_str.strip()
    if not text[0] == "{":
        raise ValueError("Invalid input to complete, not start with a `{`.")

    if mode == FillMode.BraceOnly:
        # BraceOnly
        text = text.rstrip("\n\r\t ")
        while text.endswith(","):
            text = text[:-1].rstrip()
        if text == "{" or text == '{"':
            return {}
        if text.endswith(":"):
            return {}

        try:
            return json.loads(text)
        except Exception:
            logger.debug(f"Text {text} is not a complete json, try to complete it.")
            open_count = text.count("{")
            close_count = text.count("}")
            if open_count > close_count:
                s2 = text + "}" * (open_count - close_count)
                try:
                    return json.loads(s2)
                except Exception:
                    logger.debug(f"Text {s2} cannot be fixed, return an empty json.")
                    return {}
            return {}

    elif mode == FillMode.Full:
        obj, _ = _parse_object(text, 0)
        return obj
    else:
        raise ValueError(
            f"Invalid input FillMode, which shall be either FillNode.BraceOnly or FillNode.Full, while got {mode}."
        )
