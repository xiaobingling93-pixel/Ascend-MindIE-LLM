/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef SAMPLING_H
#define SAMPLING_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <memory>

#include "basic_types.h"

namespace mindie_llm {
struct SamplingParams {
    std::optional<float> temperature;

    std::optional<uint32_t> topK;

    std::optional<float> topP;

    std::optional<float> typicalP;

    std::optional<bool> doSample;

    std::optional<uint64_t> seed;

    std::optional<float> repetitionPenalty;

    std::optional<bool> watermark;

    std::optional<float> frequencyPenalty;

    std::optional<float> presencePenalty;

    uint64_t maxOutputLen;

    std::vector<TokenId> stopTokenIds{};

    std::optional<bool> includeStopStrInOutput;

    std::string stopStrings;

    std::optional<bool> logprobs;

    std::optional<uint32_t> topLogprobs;

    std::optional<std::string> responseFormat; // JSON structured output format

    uint32_t n = 1;

    uint32_t bestOf = 1;

    bool useBeamsearch = false;

    bool enableParallelSampling{false};
};
using SamplingParamsSPtr = std::shared_ptr<SamplingParams>;
} // namespace mindie_llm

#endif