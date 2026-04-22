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

#ifndef HIT_RATE_CALCULATOR_H
#define HIT_RATE_CALCULATOR_H

#include <cstdint>

namespace mindie_llm {
class HitRateCalculator {
   public:
    HitRateCalculator() = default;

    ~HitRateCalculator() = default;

    virtual void Record(bool hit);

    virtual double GetHitRate() const;

   protected:
    uint64_t hitNum_ = 0;
    uint64_t missNum_ = 0;
};
}  // namespace mindie_llm

#endif
