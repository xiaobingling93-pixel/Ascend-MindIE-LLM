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

#ifndef SAFE_RESULT_H
#define SAFE_RESULT_H

#include <string>
#include <unordered_map>

namespace mindie_llm {

enum class ResultCode {
    OK,
    NONE_ARGUMENT,
    TYPE_MISMATCH,
    INVALID_ARGUMENT,
    NO_PERMISSION,
    RISK_ALERT,
    INIT_FAILURE,
    IO_FAILURE,
    PARSE_FAILURE
};

inline const std::unordered_map<ResultCode, std::string> resultCodeMap = {
    {ResultCode::OK, "[OK]"},
    {ResultCode::NONE_ARGUMENT, "[NONE_ARGUMENT]"},
    {ResultCode::TYPE_MISMATCH, "[TYPE_MISMATCH]"},
    {ResultCode::INVALID_ARGUMENT, "[INVALID_ARGUMENT]"},
    {ResultCode::NO_PERMISSION, "[NO_PERMISSION]"},
    {ResultCode::RISK_ALERT, "[RISK_ALERT]"},
    {ResultCode::INIT_FAILURE, "[INIT_FAILURE]"},
    {ResultCode::IO_FAILURE, "[IO_FAILURE]"},
    {ResultCode::PARSE_FAILURE, "[PARSE_FAILURE]"}};

class Result {
   public:
    static Result OK();
    static Result Error(ResultCode code, std::string msg);

    bool IsOk() const noexcept;
    const std::string& message() const noexcept;

   private:
    Result(ResultCode code, std::string msg);
    std::string Result2Str() const;

    static std::string CodeToString(ResultCode code);

   private:
    ResultCode code_;
    std::string message_;
};

}  // namespace mindie_llm

#endif  // SAFE_RESULT_H
