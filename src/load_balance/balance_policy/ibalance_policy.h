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

#ifndef BALANCE_POLICY_H
#define BALANCE_POLICY_H

#include <vector>

#include "block_manager/block_manager_interface.h"

namespace mindie_llm {

enum class BalancePolicyType { ROUND_ROBIN, BALANCE_BLOCK, BALANCE_BLOCK_BATCH };

struct BalancerConstraintParam {
    // output limit
    size_t maxReqNumPerWavePerDP_;
    size_t dispatchNumAtThisWave_;
    uint64_t minFreeBlockNumForDP_;

    BalancerConstraintParam(size_t maxReqNum, size_t dispatchNum, size_t minFreeBlockForDp)
        : maxReqNumPerWavePerDP_(maxReqNum),
          dispatchNumAtThisWave_(dispatchNum),
          minFreeBlockNumForDP_(minFreeBlockForDp) {}
};

class IBalancePolicy {
   public:
    virtual ~IBalancePolicy() = default;

    virtual void SetConstraint(const BalancerConstraintParam &param) = 0;

    virtual std::vector<std::vector<SequenceGroupSPtr>> Apply(std::vector<SequenceGroupSPtr> &candidates) = 0;
};

using BalancePolicyPtr = std::unique_ptr<IBalancePolicy>;
BalancePolicyPtr MakeBalancePolicy(BalancePolicyType type);

}  // namespace mindie_llm

#endif
