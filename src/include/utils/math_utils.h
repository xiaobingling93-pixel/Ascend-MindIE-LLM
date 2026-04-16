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

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <functional>
#include <stdexcept>

#include "basic_types.h"

namespace mindie_llm {

constexpr int HASH_SHIFT_LEFT = 6;
constexpr int HASH_SHIFT_RIGHT = 2;

template <class T>
inline void HashCombine(HashValue &seed, const T &v) {
    std::hash<T> hasher;
    HashValue hv = static_cast<HashValue>(hasher(v));
    constexpr HashValue kMul = 0x9e3779b97f4a7c15ULL;  // 64 位黄金分割常数，比 32 位版本更稳定
    seed ^= hv + kMul + (seed << HASH_SHIFT_LEFT) + (seed >> HASH_SHIFT_RIGHT);
    if (seed == INVALID_HASH_VALUE) {
        seed = 1;
    }
}

template <typename T>
T CeilDiv(T dividend, T divisor) {
    static_assert(std::is_integral<T>::value, "CeilDiv only supports integral types");
    if (divisor == 0) {
        throw std::invalid_argument("Divisor cannot be zero");
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
bool IsClose(T a, T b, T relTol = 1e-6f, T absTol = 1e-6f) {
    static_assert(std::is_floating_point_v<T>, "IsClose requires floating-point types");
    if (relTol < T(0) || absTol < T(0)) {
        throw std::invalid_argument("Tolerances must be non-negative");
    }
    return std::fabs(a - b) <= std::max(absTol, relTol * std::max(std::fabs(a), std::fabs(b)));
}

}  // namespace mindie_llm

#endif  // MATH_UTILS_H
