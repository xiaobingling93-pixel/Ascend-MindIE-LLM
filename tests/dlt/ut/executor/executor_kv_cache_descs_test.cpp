/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 */

#include <gtest/gtest.h>

#define private public
#include "executor.h"
#undef private

namespace mindie_llm {

static KVCacheOverview::KVCacheDesc MakeDesc(uint32_t npu, uint32_t bs, uint32_t cr, int32_t type)
{
    KVCacheOverview::KVCacheDesc d;
    d.npuBlockNum = npu;
    d.blockSize = bs;
    d.compressionRatio = cr;
    d.cacheType = type;
    return d;
}

TEST(ExecutorKVCacheDescsTest, MasterHandleSlaveInitResponse_UpdatesWhenEmptyAndKeepsWhenMismatch)
{
    auto executor = std::make_shared<Executor>();

    // Reset baseline to avoid cross-test contamination.
    {
        std::lock_guard<std::mutex> lock(IExecutor::kvCacheOverview_.updateValueMutex);
        IExecutor::kvCacheOverview_.cpuBlockNum = 0xFFFFFFFF;
        IExecutor::kvCacheOverview_.npuBlockNum = 0xFFFFFFFF;
        IExecutor::kvCacheOverview_.maxPositionEmbeddings = 0xFFFFFFFF;
        IExecutor::kvCacheOverview_.kvCacheDescs.clear();
    }

    ExecuteResponse resp;
    resp.set_msg_type(REMOTE_MODEL_INIT);
    auto *init = resp.mutable_remote_model_init_results();
    init->set_cpu_block_num(100);
    init->set_max_position_embeddings(300);

    {
        auto *p0 = init->add_kv_cache_descs();
        p0->set_npu_block_num(16);
        p0->set_block_size(128);
        p0->set_compression_ratio(1);
        p0->set_cache_type(0);
        auto *p1 = init->add_kv_cache_descs();
        p1->set_npu_block_num(32);
        p1->set_block_size(256);
        p1->set_compression_ratio(2);
        p1->set_cache_type(1);
    }

    EXPECT_TRUE(executor->MasterHandleSlaveInitResponse(resp));
    ASSERT_EQ(IExecutor::kvCacheOverview_.kvCacheDescs.size(), 2u);
    EXPECT_EQ(IExecutor::kvCacheOverview_.kvCacheDescs[0], MakeDesc(16, 128, 1, 0));
    EXPECT_EQ(IExecutor::kvCacheOverview_.kvCacheDescs[1], MakeDesc(32, 256, 2, 1));

    // Second response with mismatch should be ignored (keep existing descs).
    ExecuteResponse resp2;
    resp2.set_msg_type(REMOTE_MODEL_INIT);
    auto *init2 = resp2.mutable_remote_model_init_results();
    init2->set_cpu_block_num(100);
    init2->set_max_position_embeddings(300);
    auto *q0 = init2->add_kv_cache_descs();
    q0->set_npu_block_num(16);
    q0->set_block_size(999); // mismatch
    q0->set_compression_ratio(1);
    q0->set_cache_type(0);
    EXPECT_TRUE(executor->MasterHandleSlaveInitResponse(resp2));
    ASSERT_EQ(IExecutor::kvCacheOverview_.kvCacheDescs.size(), 2u);
    EXPECT_EQ(IExecutor::kvCacheOverview_.kvCacheDescs[0], MakeDesc(16, 128, 1, 0));
}

TEST(ExecutorKVCacheDescsTest, MasterHandleSlaveInitResponse_InvalidResponseReturnsFalse)
{
    auto executor = std::make_shared<Executor>();
    ExecuteResponse resp;
    resp.set_msg_type(MODEL_INIT); // wrong type
    EXPECT_FALSE(executor->MasterHandleSlaveInitResponse(resp));
}

} // namespace mindie_llm

