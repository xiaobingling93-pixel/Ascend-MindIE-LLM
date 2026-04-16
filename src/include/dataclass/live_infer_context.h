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

#ifndef SEQUENCE_GROUP_MAPPER_H
#define SEQUENCE_GROUP_MAPPER_H

#include <pthread.h>

#include <mutex>
#include <unordered_map>

#include "concurrent_map.h"
#include "sequence_group.h"

namespace mindie_llm {
constexpr size_t MAX_DP_COUNT = 32;
// 推理上下文，当前保存了reqId和seqId到SeqGroup的映射关系
// 其中SeqGroup的生命周期的开始和结束均在Sceduler类中管理
// 开始: Scheduler::AddSeqGroup函数
// 结束：Scheduler::LifeEndWrapUp_和Scheduler::ReleaseKVCache函数
// 增加了seqId到RootSeqGroup的映射关系，用于并行采样
// 增加了reqId到SeqGroup的映射关系，用于配比微调特性中，保存请求使用的实例角色，在处理响应时使用
class LiveInferContext {
   public:
    static std::shared_ptr<LiveInferContext> GetInstance(size_t localDPRank);

    LiveInferContext();
    ~LiveInferContext();

    void Add(SequenceGroupSPtr &seqGroup);
    void AddIntoSeqRootMap(SequenceId seqId, SequenceGroupSPtr &rootSeqGroup);  // used by parallel sampling

    void Remove(RequestId reqId);
    void Remove(SequenceId seqId);
    void RemoveFromSeqRootMap(SequenceId seqId);  // used by parallel sampling
    void RemoveAll();

    SequenceSPtr GetSeq(SequenceId seqId);
    SequenceGroupSPtr GetSeqGroup(RequestId reqId);
    SequenceGroupSPtr GetSeqGroup(SequenceId seqId);
    SequenceGroupSPtr GetSeqGroupFromSeqRootMap(SequenceId seqId);
    // 配比微调场景每次调度后，从队列中获取请求时，设置当前请求使用的实例角色
    void SetInferInstanceFlexRole4Req(RequestId reqId, Role role);
    // 配比微调场景每次处理响应时，判断是否是需要传递kv准备完成信息的prefill请求，
    Role GetInferInstanceFlexRole4Req(RequestId reqId);

    /**
     * 根据id从本机的所有的dp rank中查询seqgrp；只有主节点有可能有多个rank
     */
    template <typename T>
    static std::pair<size_t, SequenceGroupSPtr> FindSeqGroupInAllRank(T id);

   private:
    std::unordered_map<RequestId, SequenceGroupSPtr> reqId2SeqGroupMap_;

    std::unordered_map<SequenceId, SequenceGroupSPtr> seqId2SeqGroupMap_;

    std::unordered_map<SequenceId, SequenceGroupSPtr> seqId2RootSeqGroupMap_;

    std::unordered_map<RequestId, Role> reqId2UsedInstanceRoleMap_;

    mutable pthread_spinlock_t spinlock_;
};

using LiveInferContextSPtr = std::shared_ptr<LiveInferContext>;

template <typename T>
std::pair<size_t, SequenceGroupSPtr> LiveInferContext::FindSeqGroupInAllRank(T id) {
    for (size_t i = 0; i < MAX_DP_COUNT; i++) {
        LiveInferContextSPtr lifeContext = GetInstance(i);
        if (lifeContext == nullptr || lifeContext->GetSeqGroup(id) == nullptr) {
            continue;
        }
        return {i, lifeContext->GetSeqGroup(id)};
    }
    return {0, nullptr};
}

}  // namespace mindie_llm
#endif
