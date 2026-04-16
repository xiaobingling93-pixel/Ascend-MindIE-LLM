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
#include "endpoint_def.h"

#include <codecvt>
#include <locale>
#include <string>
#include <vector>

#include "common_util.h"
#include "config_manager.h"
#include "config_manager_impl.h"
#include "spdlog/common.h"

namespace mindie_llm {
std::atomic<bool> g_health(false);

std::atomic<bool> &HealthManager::GetHealth() { return g_health; }

void HealthManager::UpdateHealth(bool healthStatus) { g_health.store(healthStatus); }

std::string GetUriParameters(const httplib::Request &request, uint32_t index) {
    if (request.matches.size() > index && request.matches[index].matched) {
        return request.matches[index];
    }
    return "";
}

std::u16string GetU16Str(const std::string &inputStr, std::string *error) {
    try {
        std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16cvt;
        return utf16cvt.from_bytes(inputStr);
    } catch (...) {
        if (error != nullptr) {
            *error += "Can't convert string to UTF-16.";
        }
        return std::u16string{};
    }
}

std::wstring String2Wstring(const std::string &str, std::string *error) {
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        return conv.from_bytes(str);
    } catch (const std::range_error &e) {
        if (error != nullptr) {
            *error += "Can't convert string to wstring: ";
            *error += e.what();
        }
        return std::wstring{};
    }
}

std::string TransformTruncation(std::u16string inputStr, int64_t truncationStart, int64_t truncationEnd,
                                std::string *error) {
    inputStr = inputStr.substr(truncationStart, truncationEnd);
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> cvt;
    try {
        return cvt.to_bytes(inputStr);
    } catch (...) {
        if (error != nullptr) {
            *error += "Failed to convert UTF-16 string to UTF-8";
        }
        return "";
    }
}

std::string GetFinishReasonStr(InferStatusType finishReason) {
    if (finishReason == InferStatusType::END_OF_SENTENCE) {
        return "eos_token";
    } else if (finishReason == InferStatusType::ABORT || finishReason == InferStatusType::EXECUTE_ERROR ||
               finishReason == InferStatusType::ILLEGAL_INPUT) {
        return "stop_sequence";
    } else if (finishReason == InferStatusType::REACH_MAX_SEQ_LEN ||
               finishReason == InferStatusType::REACH_MAX_OUTPUT_LEN) {
        return "length";
    } else {
        return "invalid finishReason";
    }
}

uint32_t GetMaxInputLen() {
    auto serverConfig = GetServerConfig();
    uint32_t maxInputLen = serverConfig.maxRequestLength * 1024 * 1024;
    return maxInputLen;
}
}  // namespace mindie_llm
