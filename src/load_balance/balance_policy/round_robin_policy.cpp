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

#include "round_robin_policy.h"

#include <stdexcept>

namespace mindie_llm {

RoundRobinPolicy::RoundRobinPolicy() {}

void RoundRobinPolicy::SetConstraint(const BalancerConstraintParam &param) {
    maxReqNumPerWavePerDP_ = param.maxReqNumPerWavePerDP_;
    dispatchNum_ = param.dispatchNumAtThisWave_;
}

std::vector<std::vector<SequenceGroupSPtr>> RoundRobinPolicy::Apply(std::vector<SequenceGroupSPtr> &candidates) {
    if (dispatchNum_ == 0) {
        throw std::runtime_error("Invalid dispatch number: must be positive (got " + std::to_string(dispatchNum_) +
                                 ")");
    }
    static int lastOffset = -1;
    std::vector<std::vector<SequenceGroupSPtr>> res(dispatchNum_);
    for (size_t offset = 0; offset < candidates.size(); offset++) {
        lastOffset++;
        lastOffset %= static_cast<int>(dispatchNum_);
        res[lastOffset].push_back(candidates[offset]);  // blocknum is unused at round robin policy
    }
    candidates.clear();
    return res;
}

}  // namespace mindie_llm
