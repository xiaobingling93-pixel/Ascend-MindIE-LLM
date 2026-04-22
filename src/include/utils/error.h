/**
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

#ifndef MINDIE_LLM_ERROR_H
#define MINDIE_LLM_ERROR_H

#include <string>
#include <utility>

namespace mindie_llm {
/// Error class, which is used to return error code and message.
/// This class can contain a code in Code format and a message.
class Error {
   public:
    /// The enum class for error code.
    enum class Code {
        OK,
        ERROR,
        INVALID_ARG,
        NOT_FOUND,
    };

    /// The constructor of Error class, which initializes the code_ to code, and msg_ to "".
    ///
    /// \param code The error code.
    explicit Error(Code code = Code::OK) : code_(code) {}

    /// The constructor of Error class, which initializes the code_ to code, and msg_ to msg.
    ///
    /// \param code The error code.
    /// \param msg The error message.
    explicit Error(Code code, std::string msg) : code_(code), msg_(std::move(msg)) {}

    /// This function returns the error code.
    Code ErrorCode() const { return code_; }
    /// This function returns the error message.
    const std::string &Message() const { return msg_; }
    /// This function returns whether the error code is OK.
    bool IsOk() const { return code_ == Code::OK; }
    /// This function returns the error code as a string.
    std::string ToString() const;

    /// This function provides a static function to convert the error code to a string.
    static const char *CodeToString(const Code code);

   protected:
    Code code_;
    std::string msg_;
};
}  // namespace mindie_llm

#endif  // MIES_INFRA_ERROR_H
