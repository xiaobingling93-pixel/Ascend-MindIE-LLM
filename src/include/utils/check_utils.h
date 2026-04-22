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

#ifndef MINDIE_LLM_UTILS_CHECK_H
#define MINDIE_LLM_UTILS_CHECK_H

#include <sstream>

#include "nlohmann/json.hpp"

namespace mindie_llm {
constexpr uint32_t MAX_STRING_LENGTH = 256;
using Json = nlohmann::json;

/// int multiply with check overflow.
///
/// \param a         input num 1
/// \param b        input num 1
/// \param overflowFlag     True, if multiply detected overflow. False otherwise
/// \return multiply result.
template <typename T, typename U>
typename std::common_type<T, U>::type IntMulWithCheckOverFlow(const T a, const U b, bool& overflowFlag) {
    if (std::is_signed<T>::value != std::is_signed<U>::value) {
        overflowFlag = true;
        return 0;
    }
    using PromotedType = typename std::common_type<T, U>::type;

    PromotedType result;
    // 返回 true 表示溢出
    if (__builtin_mul_overflow(a, b, &result)) {
        overflowFlag = true;
        return 0;
    }
    return result;
}

template <typename T, typename U>
typename std::common_type<T, U>::type CheckIntMulOverFlow(const T a, const U b) {
    if (std::is_signed<T>::value != std::is_signed<U>::value) {
        throw std::runtime_error("Multiplication between signed and unsigned integer not supported, it's not safe");
    }
    using PromotedType = typename std::common_type<T, U>::type;
    if (a == 0 || b == 0) {
        return 0;
    }

    PromotedType pa = static_cast<PromotedType>(a);
    PromotedType pb = static_cast<PromotedType>(b);

    if constexpr (std::is_signed<PromotedType>::value) {
        const PromotedType maxVal = std::numeric_limits<PromotedType>::max();
        const PromotedType minVal = std::numeric_limits<PromotedType>::min();
        if (pa > 0 && pb > 0) {
            if (pa > maxVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa < 0 && pb < 0) {
            if (pa < maxVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa > 0 && pb < 0) {
            if (pb < minVal / pa) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa < minVal / pb) {
            throw std::overflow_error("Integer overflow detected.");
        }
    } else {
        const PromotedType maxVal = std::numeric_limits<PromotedType>::max();
        if (pa > maxVal / pb) {
            throw std::overflow_error("Integer overflow detected.");
        }
    }
    return pa * pb;
}
int CheckParamRange(const int& intParam, int min, int max);
bool IsWithinRange(const std::string& integerType, const Json& jsonValue);
uint32_t GetIntegerParamValue(const Json& jsonData, const std::string& configName, uint32_t defaultVal);
std::string GetStringParamValue(const Json& jsonData, const std::string& configName);
bool CheckStringInputLength(const std::string& stringValue, uint32_t maxValLength);
}  // namespace mindie_llm
#endif
