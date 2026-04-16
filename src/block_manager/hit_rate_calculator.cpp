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

#include "hit_rate_calculator.h"

#include <iostream>

#include "log.h"

namespace mindie_llm {
constexpr uint64_t MAX_HIT_NUM = 62;
/*
    Record the cache hit/miss when a cacheable block is allocated
    This should be called every time the allocator allocates an immutable block
    Only used for PrefixCachingAllocator
*/
void HitRateCalculator::Record(bool hit) {
    if (hit) {
        hitNum_++;
    } else {
        missNum_++;
    }
    if (hitNum_ >= (1ULL << MAX_HIT_NUM) || missNum_ >= (1ULL << MAX_HIT_NUM)) {
        MINDIE_LLM_LOG_WARN("HitRateCalculator may overflow!");
    }
}

// Calculate cache hit rate
double HitRateCalculator::GetHitRate() const {
    uint64_t totalNum = hitNum_ + missNum_;
    if (totalNum == 0) {
        return 0;
    }
    return static_cast<double>(hitNum_) / static_cast<double>(totalNum);
}
}  // namespace mindie_llm
