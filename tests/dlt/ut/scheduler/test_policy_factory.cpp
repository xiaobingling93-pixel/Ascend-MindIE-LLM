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
 
#include <gtest/gtest.h>
#define private public
#define protected public
#include "policy/policy_factory.h"
#include "stage_policy/latency_stage_policy.h"
#include "stage_policy/tpt_stage_policy.h"
#include "stage_policy/prefill_first_policy.h"
#include "stage_policy/time_division_policy.h"
#include "stage_policy/edge_cloud_policy.h"

using namespace mindie_llm;
class PolicyFactoryTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        role_ = Role::PnD;
        schedulerConfig_ = std::make_shared<SchedulerConfig>();
        predictor_ = std::make_shared<LatencyPredictor>();
        blockConf_ = {schedulerConfig_->cacheBlockSize,
                     schedulerConfig_->cpuBlockNum,
                     schedulerConfig_->npuBlockNum,
                     0,
                     0,
                     schedulerConfig_->enablePrefixCache,
                     schedulerConfig_->spSize * schedulerConfig_->cpSize};
        blockManager_ = BlockManagerFactory::CreateBlockSpaceManager(BlockManagerType::SELFATTNBLOCKMANAGER,
                                                                    std::move(blockConf_), 0);
    }

    std::shared_ptr<SchedulerConfig> schedulerConfig_;
    std::shared_ptr<LatencyPredictor> predictor_;
    BlockManagerConfig blockConf_;
    BlockSpaceManagerSPtr blockManager_;
    Role role_;
};

TEST_F(PolicyFactoryTest, CreateStagePolicy_PrefillFirstPolicy)
{
    schedulerConfig_->stageSelectPolicy = 0;
    auto stagePolicy = PolicyFactory::CreateStagePolicy(schedulerConfig_, predictor_, blockManager_, role_);
    EXPECT_NE(stagePolicy, nullptr);
    EXPECT_NE(dynamic_cast<PrefillFirstPolicy*>(stagePolicy.get()), nullptr);
}

TEST_F(PolicyFactoryTest, CreateStagePolicy_TptStagePolicy)
{
    schedulerConfig_->stageSelectPolicy = 1;
    auto stagePolicy = PolicyFactory::CreateStagePolicy(schedulerConfig_, predictor_, blockManager_, role_);
    EXPECT_NE(stagePolicy, nullptr);
    EXPECT_NE(dynamic_cast<TptStagePolicy*>(stagePolicy.get()), nullptr);
}

TEST_F(PolicyFactoryTest, CreateStagePolicy_LatencyStagePolicy)
{
    schedulerConfig_->stageSelectPolicy = 2;
    auto stagePolicy = PolicyFactory::CreateStagePolicy(schedulerConfig_, predictor_, blockManager_, role_);
    EXPECT_NE(stagePolicy, nullptr);
    EXPECT_NE(dynamic_cast<LatencyStagePolicy*>(stagePolicy.get()), nullptr);
}

TEST_F(PolicyFactoryTest, CreateStagePolicy_TimeDivisionPolicy)
{
    schedulerConfig_->stageSelectPolicy = 0;
    role_ = Role::FlexP;
    auto stagePolicy = PolicyFactory::CreateStagePolicy(schedulerConfig_, predictor_, blockManager_, role_);
    EXPECT_NE(stagePolicy, nullptr);
    EXPECT_NE(dynamic_cast<TimeDivisionPolicy*>(stagePolicy.get()), nullptr);
}

TEST_F(PolicyFactoryTest, CreateStagePolicy_EdgeCloudPolicy)
{
    schedulerConfig_->stageSelectPolicy = 3;
    role_ = Role::PnD;
    auto stagePolicy = PolicyFactory::CreateStagePolicy(schedulerConfig_, predictor_, blockManager_, role_);
    EXPECT_NE(stagePolicy, nullptr);
    EXPECT_NE(dynamic_cast<EdgeCloudPolicy*>(stagePolicy.get()), nullptr);
}

TEST_F(PolicyFactoryTest, CreateStagePolicy_InvalidPolicy)
{
    schedulerConfig_->stageSelectPolicy = 99;
    EXPECT_THROW(PolicyFactory::CreateStagePolicy(schedulerConfig_, predictor_, blockManager_, role_),
                std::runtime_error);
}