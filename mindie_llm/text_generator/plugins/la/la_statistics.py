# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import numpy as np
from ....utils.log.logging import logger, print_log
from ....utils.log.error_code import ErrorCode
from ....utils.env import ENV


class RequestStatistic:
    def __init__(self, level, window, guess_set_size, input_len, guess_size, req_id):
        self.level = level
        self.window = window
        self.guess_set_size = guess_set_size
        self.input_len = input_len
        self.guess_size = guess_size
        self.req_id = req_id

        self.pool_from_prompt = True
        self.generate_tokens = 0
        self.total_steps = 0
        self.guess_steps = 0
        self.all_accept_tokens = 0
        self.all_guess_tokens = 0

        self.preprocess_time = 0
        self.forward_time = 0
        self.postprocess_time = 0

        self.prefill_preprocess_time = 0
        self.prefill_forward_time = 0
        self.prefill_postprocess_time = 0

    def print_statistics(self):
        total_time = self.preprocess_time + self.forward_time + self.postprocess_time
        if self.total_steps == 0:
            message = "Total decoding steps of lookahead is 0, Exception occurred during the inference!"
            logger.error(message, ErrorCode.TEXT_GENERATOR_ZERO_ITER_ERR)
            raise ZeroDivisionError(message)
        if total_time == 0:
            message = "Total running time of lookahead is 0, Exception occurred during the inference!"
            logger.error(message, ErrorCode.TEXT_GENERATOR_ZERO_TIME_ERR)
            raise ZeroDivisionError(message)

        rank = ENV.rank

        accept_per_step = 0
        guess_group_per_step = 0
        acceptance_for_guess_size = 0
        if self.guess_steps != 0:
            accept_per_step = round(self.all_accept_tokens / self.guess_steps, 2)
            if self.guess_size != 0:
                guess_group_per_step = round(self.all_guess_tokens / self.guess_size / self.guess_steps, 2)
                acceptance_for_guess_size = round(
                    (self.all_accept_tokens / (self.guess_size * self.guess_steps)) * 100, 2
                )

        acceptance_for_guess_tokens = 0
        if self.all_guess_tokens != 0:
            acceptance_for_guess_tokens = round((self.all_accept_tokens / self.all_guess_tokens) * 100, 2)

        print_log(
            rank, logger.debug, "========================ACCELERATION===SUMMARY======================================="
        )
        print_log(rank, logger.debug, f"Statistics of request id {self.req_id} is as below:")
        print_log(
            rank,
            logger.debug,
            f"Lookahead decoding config, N: {self.level}, W: {self.window}, "
            f"G: {self.guess_set_size}, Pool from prompt: {self.pool_from_prompt}",
        )
        print_log(
            rank,
            logger.debug,
            f"Input tokens: {self.input_len}, Generated tokens: {self.generate_tokens}, "
            f"Total steps: {self.total_steps}, Compression ratio: "
            f"{round(self.generate_tokens / self.total_steps, 2)}",
        )
        print_log(
            rank,
            logger.debug,
            f"Guess steps: {self.guess_steps}, Guess tokens: {self.all_guess_tokens}, "
            f"Accept_tokens: {self.all_accept_tokens}, Guess_size: {self.guess_size}",
        )
        print_log(
            rank, logger.debug, f"Accept per step: {accept_per_step}, Guess group per step: {guess_group_per_step}"
        )
        print_log(
            rank,
            logger.debug,
            f"Acceptance for guess size: {acceptance_for_guess_size}%, "
            f"Acceptance for guess tokens: {acceptance_for_guess_tokens}%",
        )
        print_log(
            rank,
            logger.debug,
            f"Total Preprocess time: {round(self.preprocess_time, 2)}ms, Total Forward time: "
            f"{round(self.forward_time, 2)}ms, Total Postprocess time: {round(self.postprocess_time, 2)}ms",
        )
        print_log(
            rank,
            logger.debug,
            f"Total Time spent: {round(total_time, 2)}ms, Average Time per step: "
            f"{round((total_time / self.total_steps), 2)}ms, Average token num per sec: "
            f"{round((self.generate_tokens * 1000 / total_time), 2)}tokens/s",
        )
        print_log(
            rank,
            logger.debug,
            f"Prefill Preprocess time: {round(self.prefill_preprocess_time, 2)}ms, "
            f"Prefill Forward time: {round(self.prefill_forward_time, 2)}ms, Prefill Postprocess time: "
            f"{round(self.prefill_postprocess_time, 2)}ms",
        )
        print_log(
            rank,
            logger.debug,
            "=======================================================================================",
        )

    def update_statistics(self, accept_tokens, generate_tokens, guess_tokens):
        self.generate_tokens += generate_tokens
        self.total_steps += 1
        self.guess_steps += 1 if guess_tokens != 0 else 0
        self.all_accept_tokens += accept_tokens
        self.all_guess_tokens += guess_tokens

    def record_time(self, preprocess_time, forward_time, postprocess_time, is_prefill):
        self.preprocess_time += preprocess_time
        self.forward_time += forward_time
        self.postprocess_time += postprocess_time
        if is_prefill:
            self.prefill_preprocess_time += preprocess_time
            self.prefill_forward_time += forward_time
            self.prefill_postprocess_time += postprocess_time


class RequestStatsList:
    def __init__(self, level, window, guess_set_size):
        self.requests_idx_mapping = {}
        self.requests_statistics = []
        self.used_idx = []
        self.level = level
        self.window = window
        self.guess_set_size = guess_set_size
        self.guess_size = self.level - 1

    def prepare_statistics(self, req_id, req_stats):
        if req_id not in self.requests_idx_mapping:
            available_idx = np.flatnonzero(np.array(self.used_idx) == 0)
            if available_idx.shape[0] == 0:
                self.used_idx.append(1)
                self.requests_statistics.append(None)
                append_idx = len(self.used_idx) - 1
                self.requests_idx_mapping[req_id] = append_idx
            else:
                unused_idx = available_idx[0]
                self.used_idx[unused_idx] = 1
                self.requests_idx_mapping[req_id] = unused_idx
        map_id = self.requests_idx_mapping[req_id]
        self.requests_statistics[map_id] = req_stats
        return

    def del_request_stats(self, req_id):
        if req_id not in self.requests_idx_mapping:
            return
        map_id = self.requests_idx_mapping[req_id]
        self.used_idx[map_id] = 0
        self.requests_statistics[map_id] = None
        del self.requests_idx_mapping[req_id]
        return

    def new_request_in(self, input_len, req_id):
        new_stats = RequestStatistic(self.level, self.window, self.guess_set_size, input_len, self.guess_size, req_id)
        self.prepare_statistics(req_id, new_stats)

    def update_request_stats(self, accept_tokens, generate_tokens, guess_tokens, req_id):
        if req_id not in self.requests_idx_mapping:
            message = f"Request {req_id} has never been executed, so data statistics cannot be recorded"
            logger.error(message, ErrorCode.TEXT_GENERATOR_REQ_ID_UNUSED)
            return
        map_id = self.requests_idx_mapping[req_id]
        req_stats = self.requests_statistics[map_id]
        req_stats.update_statistics(accept_tokens, generate_tokens, guess_tokens)

    def final_print_req_stats(self, req_id):
        if req_id not in self.requests_idx_mapping:
            message = f"Request {req_id} has never been executed, so data statistics cannot be recorded"
            logger.error(message, ErrorCode.TEXT_GENERATOR_REQ_ID_UNUSED)
            return

        map_id = self.requests_idx_mapping[req_id]
        req_stats = self.requests_statistics[map_id]
        req_stats.print_statistics()

        self.del_request_stats(req_id)

    def record_req_time(self, req_id, preprocess_time, forward_time, postprocess_time, is_prefill):
        if req_id not in self.requests_idx_mapping:
            message = f"Request {req_id} has never been executed, so data statistics cannot be recorded"
            logger.error(message, ErrorCode.TEXT_GENERATOR_REQ_ID_UNUSED)
            return

        map_id = self.requests_idx_mapping[req_id]
        req_stats = self.requests_statistics[map_id]
        req_stats.record_time(preprocess_time, forward_time, postprocess_time, is_prefill)
