# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Tuple, List

from mindie_llm.utils.log.logging import logger


class ReasoningParser:
    def __init__(self, start_reasoning_token_id: int, end_reasoning_token_id: int) -> None:
        """Initializes the reasoning parser with start and end token IDs.

        Args:
            start_reasoning_token_id: Token ID representing the start of reasoning content (e.g., <think>).
            end_reasoning_token_id: Token ID representing the end of reasoning content (e.g., </think>).
        """
        self.start_reasoning_token_id = start_reasoning_token_id
        self.end_reasoning_token_id = end_reasoning_token_id

    def is_reasoning_end(self, all_token_ids: List[int]) -> bool:
        """Determines whether the end reasoning token appears in the token sequence.

        Args:
            all_token_ids: List of generated token IDs.

        Returns:
            bool: True if the end reasoning token is present, False otherwise.
        """
        return self.end_reasoning_token_id in all_token_ids


class CommonReasoningParser(ReasoningParser):
    """
    Common implementation.
    compatibility without start  <think>
    compatibility limited by length with unfinished
    Implementation logic:  </think>: reasoning content on the left and content on the right
    """

    def __init__(self, start_reasoning_token_id: int, end_reasoning_token_id: int) -> None:
        """Initializes the common reasoning parser by invoking the parent constructor.

        Args:
            start_reasoning_token_id: Token ID for the start of reasoning content.
            end_reasoning_token_id: Token ID for the end of reasoning content.
        """
        super().__init__(start_reasoning_token_id, end_reasoning_token_id)

    def stream_process_reasoning(self, all_token_ids: List[int], current_index: int) -> Tuple[List[int], List[int]]:
        """Processes token stream incrementally to separate reasoning and final output content.

        Args:
            all_token_ids: Full list of token IDs generated up to the current step.
            current_index: Index from which newly generated tokens are considered.

        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - Delta reasoning content tokens (from current_index up to the end reasoning token)
                - Delta final content tokens (after the end reasoning token)
        """
        # compatibility without start  <think>
        valid_text_start_index = 1 if self.start_reasoning_token_id == all_token_ids[0] else 0
        # only generated <think>, no content has been generated yet
        if len(all_token_ids) == 1 and valid_text_start_index == 1:
            return [], []
        delta_reasoning_content_token_ids = []
        delta_content_token_ids = []

        reasoning_end_token_index = (
            len(all_token_ids)
            if self.end_reasoning_token_id not in all_token_ids
            else all_token_ids.index(self.end_reasoning_token_id)
        )
        # get reasoning delta
        if current_index < reasoning_end_token_index:
            delta_reasoning_content_token_ids = all_token_ids[current_index:reasoning_end_token_index]
        # get content delta
        if reasoning_end_token_index < len(all_token_ids) - 1:
            delta_content_token_ids = all_token_ids[max(1 + reasoning_end_token_index, current_index) :]
        return delta_reasoning_content_token_ids, delta_content_token_ids

    def single_process_reasoning(self, all_token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Processes the complete token sequence to extract reasoning and final output content.

        Args:
            all_token_ids: Complete list of generated token IDs.

        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - Reasoning content tokens (between start and end tokens, if applicable)
                - Final output content tokens (after the end reasoning token)
        """
        reasoning_content_token_ids = []
        content_token_ids = []
        if not all_token_ids:
            return reasoning_content_token_ids, content_token_ids
        if self.end_reasoning_token_id is None:
            logger.error("ERROR: now in reasoning parser without given end_reasoning_token id.")
            return reasoning_content_token_ids, all_token_ids
        # compatibility without start  <think>
        reasoning_content_start_index = 1 if self.start_reasoning_token_id == all_token_ids[0] else 0
        if self.end_reasoning_token_id not in all_token_ids:
            # compatibility limited by length with unfinished
            return all_token_ids[reasoning_content_start_index:], []
        # common scene
        reasoning_content_end_index = all_token_ids.index(self.end_reasoning_token_id)
        reasoning_content_token_ids = all_token_ids[reasoning_content_start_index:reasoning_content_end_index]
        # final answer
        content_token_ids = (
            []
            if reasoning_content_end_index == len(all_token_ids) - 1
            else all_token_ids[reasoning_content_end_index + 1 :]
        )
        return reasoning_content_token_ids, content_token_ids

    def count_reasoning_tokens(self, all_token_ids: List[int]) -> int:
        """Counts the number of tokens up to the end reasoning token.

        Args:
            all_token_ids: List of token IDs to inspect.

        Returns:
            int: The index of the end reasoning token if found; otherwise, 0.
        """
        try:
            return all_token_ids.index(self.end_reasoning_token_id)
        except ValueError:
            return 0
