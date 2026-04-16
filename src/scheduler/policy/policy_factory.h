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

#ifndef POLICY_FACTORY_H
#define POLICY_FACTORY_H

#include "latency_predictor/latency_predictor.h"
#include "policy.h"
#include "stage_policy/stage_policy.h"

namespace mindie_llm {
class PolicyFactory {
   public:
    static std::shared_ptr<Policy> CreatePolicy(Role role, std::shared_ptr<SchedulerConfig> &schedulerConfig,
                                                BlockSpaceManagerSPtr &blockManager, PDPriorityType pdPriorityType);

    static std::shared_ptr<KVTransferPolicy> CreateTransferPolicy(Role role,
                                                                  std::shared_ptr<SchedulerConfig> &schedulerConfig,
                                                                  BlockSpaceManagerSPtr &blockManager);

    static std::shared_ptr<StagePolicy> CreateStagePolicy(const SchedulerConfigSPtr schedulerConfig,
                                                          std::shared_ptr<LatencyPredictor> predictor,
                                                          std::shared_ptr<BlockSpaceManager> blockManager, Role role);
};
}  // namespace mindie_llm

#endif
