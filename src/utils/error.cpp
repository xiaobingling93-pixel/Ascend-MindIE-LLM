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

#include "error.h"

namespace mindie_llm {
std::string Error::ToString() const {
    std::string str(CodeToString(code_));
    str += ":" + msg_;
    return str;
}

const char *Error::CodeToString(const Code code) {
    switch (code) {
        case Error::Code::OK:
            return "OK";
        case Error::Code::ERROR:
            return "Error";
        case Error::Code::INVALID_ARG:
            return "Invalid argument";
        case Error::Code::NOT_FOUND:
            return "Not found";
        default:
            break;
    }
    return "invalid code";
}
}  // namespace mindie_llm
