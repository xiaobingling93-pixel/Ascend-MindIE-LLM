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

#ifndef __I_LOAD_BALANCER_H__
#define __I_LOAD_BALANCER_H__

#include <cstddef>
#include <memory>
#include <vector>

#include "sequence_group.h"

namespace mindie_llm {
class ILoadBalancer {
   public:
    virtual ~ILoadBalancer() = default;

    virtual void AddSeqGroup(SequenceGroupSPtr &seqGroup) = 0;

    virtual void Stop() = 0;
};

using LoadBalancerPtr = std::unique_ptr<ILoadBalancer>;

struct EnginePerDP;
LoadBalancerPtr MakeLoadBalancer(const std::vector<std::shared_ptr<EnginePerDP>> &enginePerDPs,
                                 size_t waveNumPerDP = 256, size_t thresholdPerDP = 512, size_t intervalMs = 1);
}  // namespace mindie_llm
#endif
