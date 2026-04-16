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

#ifndef ROUND_ROBIN_BALANCE_POLICY_H
#define ROUND_ROBIN_BALANCE_POLICY_H

#include <vector>

#include "ibalance_policy.h"

namespace mindie_llm {

class RoundRobinPolicy : public IBalancePolicy {
   public:
    explicit RoundRobinPolicy();
    void SetConstraint(const BalancerConstraintParam &param) override;
    std::vector<std::vector<SequenceGroupSPtr>> Apply(std::vector<SequenceGroupSPtr> &candidates) override;

   private:
    size_t maxReqNumPerWavePerDP_{0};
    size_t dispatchNum_{0};
};

}  // namespace mindie_llm

#endif
