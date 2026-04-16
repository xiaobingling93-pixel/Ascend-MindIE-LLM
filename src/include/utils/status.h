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

#ifndef MINDIE_LLM_STATUS_H
#define MINDIE_LLM_STATUS_H

#include <string>

#include "error.h"

namespace mindie_llm {
// Slave node Health Status
enum class NodeHealthStatus { READY, ABNORMAL };
/// Status class
/// This class is used to represent the error states and status information.
class Status {
   public:
    /// Construct a status from a code with no message.
    explicit Status(Error::Code code = Error::Code::OK) noexcept { error_ = Error(code); }

    /// Construct a status from a code and message.
    explicit Status(Error::Code code, const std::string &msg) { error_ = Error(code, msg); }

    /// Construct a status from an error.
    explicit Status(const Error &error) { error_ = error; }

    /// The function returns whether the status is ok.
    bool IsOk() const { return error_.IsOk(); }

    /// The function returns the error code.
    Error::Code StatusCode() const { return error_.ErrorCode(); }

    /// The function returns the error message.
    std::string StatusMsg() const { return error_.Message(); }

    /// The equal operator is to compare two status objects and return true if they are equal.
    bool operator==(const Status &other) const { return error_.ErrorCode() == other.error_.ErrorCode(); }

   private:
    Error error_;  /// the error object.
};

}  // namespace mindie_llm
#endif  // MINDIE_LLM_STATUS_H
