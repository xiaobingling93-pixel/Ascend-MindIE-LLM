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

#ifndef SCHEDULER_PRE_SCHEDULER_H
#define SCHEDULER_PRE_SCHEDULER_H

#include "dataclass/metric.h"
#include "policy/seq_group_collection.h"

namespace mindie_llm {

struct SchedInfo {
    PDPriorityType pdPriority_;

    int64_t waitingSeqGroupNum_;

    int64_t runningSeqGroupNum_;

    int64_t swapSeqGroupNum_;

    SchedInfo() = default;

    SchedInfo(PDPriorityType pdPriority, const SchedulerMetric &metric)
        : pdPriority_(pdPriority),
          waitingSeqGroupNum_(static_cast<int64_t>(metric.reqsInfo.waitingRequestNum_)),
          runningSeqGroupNum_(static_cast<int64_t>(metric.reqsInfo.runningRequestNum_)),
          swapSeqGroupNum_(static_cast<int64_t>(metric.reqsInfo.swappedRequestNum_)) {}
};

class PreScheduler {
   public:
    static std::vector<SchedInfo> ShareSchedInfo(const SchedInfo &schedInfo, size_t dpRank, bool enableDistributed);

    static PDPriorityType DecidePDPriority(const std::vector<SchedInfo> &schedInfos);

   private:
    static std::vector<SchedInfo> ShareSchedInfoCrossNode(const SchedInfo &schedInfo);

    static std::vector<SchedInfo> ShareSchedInfoCrossDP(const SchedInfo &schedInfo, size_t dpRank);
};  // class PreScheduler

}  // namespace mindie_llm

#endif
