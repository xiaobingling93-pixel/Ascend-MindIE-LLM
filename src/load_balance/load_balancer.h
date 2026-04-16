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

#ifndef MINDIE_LLM_LOAD_BALANCE_H
#define MINDIE_LLM_LOAD_BALANCE_H

#include <boost/asio.hpp>
#include <boost/asio/post.hpp>
#include <boost/bind.hpp>
#include <deque>
#include <thread>
#include <vector>

#include "balance_policy/ibalance_policy.h"
#include "basic_types.h"
#include "concurrent_deque.h"
#include "iload_balancer.h"
#include "llm_engine.h"
#include "sequence_group.h"

namespace mindie_llm {
class LoadBalancer : public ILoadBalancer {
   public:
    explicit LoadBalancer(const std::vector<EnginePerDPSPtr> &enginePerDPs, size_t waveNumPerDP = 256,
                          size_t thresholdPerDP = 1, size_t intervalMs = 1);  // every 1ms to dispatch requests

    ~LoadBalancer() override;

    // 添加sequence group
    void AddSeqGroup(SequenceGroupSPtr &seqGroup) override;

    void Stop() override;

   protected:
    void ExecutePeriodicTask(const boost::system::error_code &ec);

    void TriggerImmediatelyTask();

    void Distribute();

    std::vector<size_t> GetDistributedDpRankIds();

    void PrepCandidates(size_t numDp, size_t maxNumPerDp);

    void DistributeSeqGroups(const std::vector<size_t> &dpRankIds);

   private:
    std::vector<EnginePerDPSPtr> enginePerDPs_;

    BalancePolicyPtr balancePolicyPtr_;

    size_t dpSize_;

    size_t waveNumPerDP_;  // 每个wave下发的最大seqgroup数量

    size_t thresholdPerDP_;  // 每个DP可下发SeqGroup的个数超过该阈值，就会触发distribute

    ConcurrentDeque<SequenceGroupSPtr> seqGroups_;  // 存储待分配的SequenceGroup

    std::vector<SequenceGroupSPtr> candidates_;  // TBC_待优化PrepCandidates要么全取管理，要么取多少用多少

    WaveId waveId_{0};  // 当前wave的ID，用于负载均衡

    size_t intervalMs_;

    std::thread worker_;

    boost::asio::io_context io_;

    boost::asio::steady_timer timer_;

    std::atomic<bool> stop_{false};  // 停止标志

    std::atomic<size_t> lastIntervalTime_{0};
};
}  // namespace mindie_llm

#endif
