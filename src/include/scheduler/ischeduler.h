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

#ifndef __SCHEDULER_H__
#define __SCHEDULER_H__

#include <pthread.h>

#include <chrono>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_types.h"
#include "concurrent_deque.h"
#include "config_info.h"
#include "latency_predictor/latency_predictor.h"
#include "metric.h"
#include "policy/stage_policy/stage_policy.h"
#include "sequence_group.h"
#include "sequence_group_meta_data.h"

namespace mindie_llm {
/* 异步调度最多调度batch数 */
constexpr int MAX_ASYNC_SCHEDULE_TIMES = 1;

class IScheduler {
   public:
    virtual ~IScheduler() = default;

    virtual void AddSeqGroup(SequenceGroupSPtr &seqGroup) = 0;

    virtual std::pair<SequenceGroupMetaDatas, SchedulerOutputs> Schedule(bool needSync = false) = 0;

    virtual std::pair<SequenceGroupMetaDatas, SchedulerKVTransferOutput> ScheduleTransfer() = 0;

    virtual size_t GetUnFinishedSeqGroups() = 0;

    virtual std::unordered_set<SequenceId> &FetchFinishedSeqIds(ConcurrentDeque<SequenceId> &finishedSeqIds) = 0;
    virtual std::unordered_set<SequenceId> &FetchExceptionSeqIds(ConcurrentDeque<SequenceId> &exceptionSeqIds) = 0;
    virtual std::unordered_set<RequestId> &FetchAbortedReqIds(ConcurrentDeque<RequestId> &abortedReqIds) = 0;
    virtual void KVPulledReqEnterRunningQueue(ConcurrentDeque<RequestId> &pulledReqIds) = 0;
    virtual void NotifyMeKvPulledSeqIds(SequenceId seqId) = 0;
    virtual std::unordered_set<SequenceId> ClearAndReturnTerminatedSeqIds() = 0;
    virtual void FetchSeqGeneratedTokens(ConcurrentDeque<std::pair<SequenceId, TokenId>> &seqIdToOutputTokenQueue) = 0;
    virtual void MarkLastScheduleEmpty() = 0;
    virtual void ClearLastScheduleEmpty() = 0;
    virtual void PrepareNextSchedule(std::vector<ScheduledSequenceGroupSPtr> &scheduledSeqGroups) = 0;
    virtual SchedulerMetric CollectSchedulerMetric() = 0;
    virtual void ClearSeqGrp(SequenceGroupSPtr seqGroup, SequenceStatus finalStatus) = 0;
    virtual std::vector<SequenceGroupSPtr> &GetAbortedParallelSeqGroups() = 0;
    virtual void CollectAndClearAbortedParallelSeqGroups() = 0;
    virtual void SetPrefillPercentage(uint32_t prefillPercentage) = 0;
    virtual Role SwitchRole() = 0;
    // 在flex场景中，需要获取当前的调度策略去记录推理请求的开始和结束时间戳
    virtual std::shared_ptr<StagePolicy> GetStagePolicy() = 0;
    virtual void StopRunningRequest() = 0;
};

using SchedulerPtr = std::unique_ptr<IScheduler>;
SchedulerPtr MakeScheduler(SchedulerConfigSPtr schedulerConfig, std::shared_ptr<LatencyPredictor> latencypredictor,
                           Role pdRole, size_t localDPRank = 0);

}  // namespace mindie_llm

#endif
