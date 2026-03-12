/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 */

#include <gtest/gtest.h>

#include "executor_interface.h"

namespace mindie_llm {

TEST(KVCacheOverviewTest, UpdateKvCacheDescsIfEmptyOrEqual)
{
    KVCacheOverview ov;

    std::vector<KVCacheOverview::KVCacheDesc> empty;
    EXPECT_TRUE(ov.UpdateKvCacheDescsIfEmptyOrEqual(empty));
    EXPECT_TRUE(ov.kvCacheDescs.empty());

    std::vector<KVCacheOverview::KVCacheDesc> d1;
    KVCacheOverview::KVCacheDesc a;
    a.npuBlockNum = 16;
    a.blockSize = 128;
    a.compressionRatio = 1;
    a.cacheType = 0;
    d1.push_back(a);

    KVCacheOverview::KVCacheDesc b;
    b.npuBlockNum = 32;
    b.blockSize = 256;
    b.compressionRatio = 2;
    b.cacheType = 1;
    d1.push_back(b);

    EXPECT_TRUE(ov.UpdateKvCacheDescsIfEmptyOrEqual(d1));
    EXPECT_EQ(ov.kvCacheDescs, d1);

    EXPECT_TRUE(ov.UpdateKvCacheDescsIfEmptyOrEqual(d1));
    EXPECT_EQ(ov.kvCacheDescs, d1);

    auto d2 = d1;
    d2[0].blockSize += 1;
    EXPECT_FALSE(ov.UpdateKvCacheDescsIfEmptyOrEqual(d2));
    EXPECT_EQ(ov.kvCacheDescs, d1);
}

} // namespace mindie_llm

