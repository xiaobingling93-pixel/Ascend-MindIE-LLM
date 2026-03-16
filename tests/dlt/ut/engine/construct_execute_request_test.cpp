/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "construct_execute_request.h"
#include "sampling.h"

namespace mindie_llm {

static std::vector<BlockId> DecodeBlockIds(const std::string &bytes)
{
    EXPECT_EQ(bytes.size() % sizeof(BlockId), 0u);
    std::vector<BlockId> out(bytes.size() / sizeof(BlockId));
    if (!out.empty()) {
        std::memcpy(out.data(), bytes.data(), bytes.size());
    }
    return out;
}

TEST(ConstructExecuteRequestTest, ConstructExecuteModelRequest_SerializesRepeatedBlockTables)
{
    SequenceGroupMetaData meta{};
    meta.requestId_ = "req_0";
    meta.serverid_ = "srv";
    meta.seqIds_ = {123};
    meta.samplingParams_ = std::make_shared<SamplingParams>();
    meta.samplingParams_->n = 1;
    meta.samplingParams_->bestOf = 1;

    meta.blockIds_.push_back(BlockIds{1, 2, 3});
    meta.blockIds_.push_back(BlockIds{}); // empty manager table should serialize as empty bytes
    meta.blockIds_.push_back(BlockIds{9});

    SequenceGroupMetaDatas metas{};
    metas.metaList.push_back(meta);

    SchedulerOutputs out{};
    out.forwardMode_ = ForwardMode::DECODE; // avoid prefill-only fields for this test

    ExecuteModelRequestPtr req = std::make_unique<model_execute_data::ExecuteModelRequest>();
    ConstructExecuteRequest::ConstructExecuteModelRequest(req, metas, out, /*dpRankId=*/0);

    ASSERT_NE(req, nullptr);
    ASSERT_EQ(req->seq_group_metadata_list_size(), 1);
    const auto &protoMeta = req->seq_group_metadata_list(0);

    ASSERT_EQ(static_cast<size_t>(protoMeta.block_tables_size()), meta.blockIds_.size());
    EXPECT_EQ(DecodeBlockIds(protoMeta.block_tables(0)), (std::vector<BlockId>{1, 2, 3}));
    EXPECT_TRUE(DecodeBlockIds(protoMeta.block_tables(1)).empty());
    EXPECT_EQ(DecodeBlockIds(protoMeta.block_tables(2)), (std::vector<BlockId>{9}));
}

TEST(ConstructExecuteRequestTest, ConstructPullKVRequest_SerializesSrcDstRepeatedBlockTables)
{
    SequenceGroupMetaData meta{};
    meta.requestId_ = "req_kv";
    meta.serverid_ = "srv";
    meta.dpInstanceId_ = 42;
    meta.seqIds_ = {7};
    meta.samplingParams_ = std::make_shared<SamplingParams>();
    meta.samplingParams_->n = 1;
    meta.samplingParams_->bestOf = 1;

    // Make prefill serialization paths safe (non-empty arrays).
    meta.promptLens_ = {4};
    meta.tokenIds_ = {11, 12, 13, 14};
    meta.computedLens_ = {0};
    meta.remoteComputedLens_ = {0};

    meta.blockIds_.push_back(BlockIds{100, 101});
    meta.blockIds_.push_back(BlockIds{}); // keep manager dimension aligned
    meta.srcBlockIds_.push_back(BlockIds{200});
    meta.srcBlockIds_.push_back(BlockIds{}); // empty allowed

    SequenceGroupMetaDatas metas{};
    metas.metaList.push_back(meta);

    PullKVRequestPtr req = ConstructExecuteRequest::ConstructPullKVRequest(metas);
    ASSERT_NE(req, nullptr);
    ASSERT_EQ(req->pull_kv_infos_size(), 1);

    const auto &info = req->pull_kv_infos(0);
    ASSERT_EQ(info.dst_block_tables_size(), 2);
    ASSERT_EQ(info.src_block_tables_size(), 2);

    EXPECT_EQ(DecodeBlockIds(info.dst_block_tables(0)), (std::vector<BlockId>{100, 101}));
    EXPECT_TRUE(DecodeBlockIds(info.dst_block_tables(1)).empty());
    EXPECT_EQ(DecodeBlockIds(info.src_block_tables(0)), (std::vector<BlockId>{200}));
    EXPECT_TRUE(DecodeBlockIds(info.src_block_tables(1)).empty());
}

} // namespace mindie_llm

