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
#include "safe_result.h"

namespace mindie_llm {

Result Result::OK() { return Result(ResultCode::OK, ""); }

Result Result::Error(ResultCode code, std::string msg) { return Result(code, std::move(msg)); }

bool Result::IsOk() const noexcept { return code_ == ResultCode::OK; }

std::string Result::Result2Str() const { return IsOk() ? "" : CodeToString(code_); }

const std::string& Result::message() const noexcept {
    static thread_local std::string fullMsg;
    const std::string codeStr = Result2Str();
    if (!codeStr.empty()) {
        fullMsg = codeStr + " " + message_;
        return fullMsg;
    }
    return message_;
}

Result::Result(ResultCode code, std::string msg) : code_(code), message_(std::move(msg)) {}

std::string Result::CodeToString(ResultCode code) {
    auto it = resultCodeMap.find(code);
    if (it != resultCodeMap.end()) {
        return it->second;
    }
    return "UNDEFINE_ERROR";
}

}  // namespace mindie_llm
