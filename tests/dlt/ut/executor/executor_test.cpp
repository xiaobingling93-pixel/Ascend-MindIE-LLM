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
#include <mockcpp/mockcpp.hpp>
#include <pthread.h>
#include <thread>
#include <sys/mman.h>
#define private public
#include "executor.h"
#include "communicator.h"

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

namespace mindie_llm {

void InitA2MasterConfig(std::map<std::string, std::string> &config, int dp, int tp)
{
    config["isMaster"] = "1";          // Set as master node
    config["dp"] = std::to_string(dp); // Set dp
    config["tp"] = std::to_string(tp); // Set tp
    config["globalWorldSize"] = "16";  // Set global world size
}

void InitA2SlaveConfig(std::map<std::string, std::string> &config, int dp, int tp)
{
    config["isMaster"] = "0";                          // Set as slave node
    config["dp"] = std::to_string(dp);                 // Set dp
    config["tp"] = std::to_string(tp);                 // Set tp
    config["globalRankIds"] = "8,9,10,11,12,13,14,15"; // Set global rank ids for slaves
    config["globalWorldSize"] = "16";                  // Set global world size
    config["localIP"] = "2.2.2.2";                     // Set local IP for slave
}

class ExecutorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        executor_ = std::make_shared<Executor>();
        // Reset global KV cache overview to avoid cross-test contamination.
        {
            std::lock_guard<std::mutex> lock(IExecutor::kvCacheOverview_.updateValueMutex);
            IExecutor::kvCacheOverview_.cpuBlockNum = 0xFFFFFFFF;
            IExecutor::kvCacheOverview_.npuBlockNum = 0xFFFFFFFF;
            IExecutor::kvCacheOverview_.maxPositionEmbeddings = 0xFFFFFFFF;
            IExecutor::kvCacheOverview_.kvCacheDescs.clear();
        }
        executeModelResponseHandled_ = false;
        pullKVResponseHandled_ = false;
        configFromManager_ = {
            {"isMaster", "0"},
            {"localIP", "1.1.1.1"},
            {"slaveIPs", "2.2.2.2"},
            {"backend_modelInstance_id", "0"},
            {"globalWorldSize", "8"},
            {"globalRankIds", "0,1,2,3,4,5,6,7"},
            {"model_instance_type", "Standard"},
            {"world_size", "8"},
            {"npu_device_ids", "0,1,2,3,4,5,6,7"},
            {"deploy_type", "INTER_PROCESS"},
            {"executor_type", "LLM_EXECUTOR_PYTHON"},
            {"backend_log_file", "/usr/local/Ascend/mindie/2.0.RC1/mindie-service/logs/mindie-server.log"},
            {"asyncBatchscheduler", "21"}};
    }

    void TearDown() override { MOCKCPP_NS::GlobalMockObject::reset(); }

    std::shared_ptr<Executor> executor_;
    bool executeModelResponseHandled_;
    bool pullKVResponseHandled_;
    std::map<std::string, std::string> configFromManager_;
};

static void MockForExecutorInstanceInit(bool isIPC = true, bool isGRPC = true)
{
    MOCKER_CPP(&Communicator::InitIPCCommunicators, bool (*)(const std::string &, uint32_t))
        .stubs()
        .will(returnValue(isIPC));
    MOCKER_CPP(&Executor::ExecuteCommand, bool (*)(const std::string &)).stubs().will(returnValue(true));
    MOCKER_CPP(&Communicator::SendModelInitRequestAndReceive,
               bool (*)(ExecuteRequest &, std::vector<ExecuteResponse> &))
        .stubs()
        .will(returnValue(true));
    MOCKER_CPP(&Executor::HandleInitResult, bool (*)(std::vector<ExecuteResponse> &)).stubs().will(returnValue(true));
    MOCKER_CPP(&Communicator::LaunchIPCHandleResponseThreads, bool (*)(ResponseHandler))
        .stubs()
        .will(returnValue(true));
    MOCKER_CPP(&Communicator::InitGRPCCommunicator,
               bool (*)(std::unordered_map<std::string, std::string> &, ResponseHandler, uint32_t))
        .stubs()
        .will(returnValue(isGRPC));
    MOCKER_CPP(&Executor::MasterHandleSlaveInitResponse, bool (*)(ExecuteResponse &)).stubs().will(returnValue(true));
    MOCKER_CPP(&Executor::SlaveSendInitResponseToMaster, bool (*)()).stubs().will(returnValue(true));
}

TEST_F(ExecutorTest, ParseFromModelConfig_Valid_SingleNode)
{
    std::unordered_map<std::string, std::string> config(configFromManager_.begin(), configFromManager_.end());
    ModelLaunchConfig modelLaunchConfig;
    bool ok = executor_->ParseFromModelConfig(config, modelLaunchConfig, /*isMultiNodesInfer=*/false);
    EXPECT_TRUE(ok);

    // Check that each field got set appropriately:
    EXPECT_EQ(modelLaunchConfig.deployType, config["deploy_type"]);
    EXPECT_EQ(modelLaunchConfig.executorType, config["executor_type"]);
    EXPECT_EQ(modelLaunchConfig.npuNumPerNode, std::stoul(config.at("world_size")));
    EXPECT_EQ(modelLaunchConfig.globalWorldSize, std::stoul(config["globalWorldSize"]));
    ASSERT_EQ(modelLaunchConfig.npuDeviceIds.size(), 8u);
    EXPECT_EQ(modelLaunchConfig.npuDeviceIds[0], "0");
    EXPECT_EQ(modelLaunchConfig.npuDeviceIds[1], "1");
    EXPECT_EQ(modelLaunchConfig.modelInstanceType, config["model_instance_type"]);
    EXPECT_FALSE(modelLaunchConfig.isMultiNodesInfer);
    EXPECT_TRUE(modelLaunchConfig.globalRankIds.empty());
    EXPECT_FALSE(modelLaunchConfig.isMasterNode);
    EXPECT_EQ(modelLaunchConfig.npuNumPerDP, 8u);
    EXPECT_EQ(modelLaunchConfig.ipcCommunicatorNum, 1u);
}

TEST_F(ExecutorTest, ParseFromModelConfig_Valid_MultiNode)
{
    std::unordered_map<std::string, std::string> config(configFromManager_.begin(), configFromManager_.end());
    ModelLaunchConfig modelLaunchConfig;
    bool ok = executor_->ParseFromModelConfig(config, modelLaunchConfig, /*isMultiNodesInfer=*/true);
    EXPECT_TRUE(ok);

    // Check that each field got set appropriately:
    EXPECT_TRUE(modelLaunchConfig.isMultiNodesInfer);
    ASSERT_EQ(modelLaunchConfig.globalRankIds.size(), 8u);
    EXPECT_EQ(modelLaunchConfig.globalRankIds[0], "0");
    EXPECT_EQ(modelLaunchConfig.globalRankIds[1], "1");
}

TEST_F(ExecutorTest, ParseFromModelConfig_Invalid_Param)
{
    ModelLaunchConfig modelLaunchConfig;

    std::unordered_map<std::string, std::string> config1(configFromManager_.begin(), configFromManager_.end());
    config1.erase("executor_type"); // remove one required key
    bool res1 = executor_->ParseFromModelConfig(config1, modelLaunchConfig, /*isMultiNodesInfer=*/false);
    EXPECT_FALSE(res1);

    std::unordered_map<std::string, std::string> config2(configFromManager_.begin(), configFromManager_.end());
    config2["deploy_type"] = "REMOTE_PROCEDURE"; // not “INTER_PROCESS”
    bool res2 = executor_->ParseFromModelConfig(config2, modelLaunchConfig, /*isMultiNodesInfer=*/false);
    EXPECT_FALSE(res2);

    std::unordered_map<std::string, std::string> config4(configFromManager_.begin(), configFromManager_.end());
    config4["world_size"] = "0"; // invalid world size < 1
    bool res4 = executor_->ParseFromModelConfig(config4, modelLaunchConfig, /*isMultiNodesInfer=*/false);
    EXPECT_FALSE(res4);
}

TEST_F(ExecutorTest, RegisterExecuteModelResponseHandler)
{
    // Before registration, the internal pointer should be nullptr:
    EXPECT_EQ(executor_->executeModelResponseHandler_, nullptr);

    auto handler = [&](ModelBatchResultSPtr) { executeModelResponseHandled_ = true; };
    executor_->RegisterExecuteModelResponseHandler(handler);
    EXPECT_NE(executor_->executeModelResponseHandler_, nullptr);

    ExecuteResponse response;
    response.mutable_execute_model_response(); // mark as having execute_model_response
    executor_->HandleExecuteModelResponse(response);
    EXPECT_TRUE(executeModelResponseHandled_);
}

TEST_F(ExecutorTest, RegisterPullKVResponseHandler)
{
    // Before registration, the internal pointer should be nullptr:
    EXPECT_EQ(executor_->pullKVResponseHandler_, nullptr);

    auto handler = [&](PullKVResponseSPtr) { pullKVResponseHandled_ = true; };
    executor_->RegisterPullKVResponseHandler(handler);
    EXPECT_NE(executor_->pullKVResponseHandler_, nullptr);

    ExecuteResponse response;
    response.mutable_pull_kv_response(); // mark as having pull_kv_response
    executor_->HandleKVTransferResponse(response);
    EXPECT_TRUE(pullKVResponseHandled_);
}

TEST_F(ExecutorTest, ExecutorInstanceInit_MissingConfig)
{
    MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/true);
    EXPECT_TRUE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/false));

    configFromManager_.erase("world_size"); // Remove a required key from configFromManager_:
    EXPECT_FALSE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/false));
}

TEST_F(ExecutorTest, ExecutorInstanceInit_SingleNode) // Only IPC is required
{
    // Even if GRPC "fails", it should never be called in this case
    MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/false);
    EXPECT_TRUE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/false, /*rankIdx=*/0));
    MOCKCPP_NS::GlobalMockObject::reset();

    // If IPC "fails", it should return false:
    MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/false);
    EXPECT_FALSE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/false, /*rankIdx=*/0));
    MOCKCPP_NS::GlobalMockObject::reset();
}

TEST_F(ExecutorTest, ExecutorInstanceInit_SingleNode_DP8TP1)
{
    configFromManager_["tp"] = "1"; // Set tp = 1
    configFromManager_["dp"] = "8"; // Set dp = 8

    // For rankIdx = 0, 1, 2, 3, 4, 5, 6, 7, only IPC is required
    for (size_t rankIdx = 0; rankIdx < 8; ++rankIdx) {
        MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/false);
        EXPECT_TRUE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/false, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();

        MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/false);
        EXPECT_FALSE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/false, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();
    }
}

TEST_F(ExecutorTest, ExecutorInstanceInit_A2_DP2TP8_Master)
{
    InitA2MasterConfig(configFromManager_, /*dp=*/2, /*tp=*/8);

    // For rankIdx = 0, only IPC is required
    MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/false);
    EXPECT_TRUE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/0));
    MOCKCPP_NS::GlobalMockObject::reset();

    MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/false);
    EXPECT_FALSE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/0));
    MOCKCPP_NS::GlobalMockObject::reset();

    // For rankIdx = 1, only GRPC is required
    MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/true);
    EXPECT_TRUE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/1));
    MOCKCPP_NS::GlobalMockObject::reset();

    MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/false);
    EXPECT_FALSE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/1));
    MOCKCPP_NS::GlobalMockObject::reset();
}

TEST_F(ExecutorTest, ExecutorInstanceInit_A2_DP2TP8_Slave) // Both IPC and GRPC are required
{
    InitA2SlaveConfig(configFromManager_, /*dp=*/2, /*tp=*/8);

    // For rankIdx = 0, both IPC and GRPC are required
    MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/true);
    EXPECT_TRUE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/0));
    MOCKCPP_NS::GlobalMockObject::reset();

    MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/false);
    EXPECT_FALSE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/0));
    MOCKCPP_NS::GlobalMockObject::reset();

    MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/true);
    EXPECT_FALSE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/0));
    MOCKCPP_NS::GlobalMockObject::reset();
}

TEST_F(ExecutorTest, ExecutorInstanceInit_A2_DP8TP2_Master)
{
    InitA2MasterConfig(configFromManager_, /*dp=*/8, /*tp=*/2);

    // For rankIdx = 0, 1, 2, 3, only IPC is required
    for (size_t rankIdx = 0; rankIdx < 4; ++rankIdx) {
        MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/false);
        EXPECT_TRUE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();

        MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/false);
        EXPECT_FALSE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();
    }

    // For rankIdx = 4, 5, 6, 7, only GRPC is required
    for (size_t rankIdx = 4; rankIdx < 8; ++rankIdx) {
        MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/true);
        EXPECT_TRUE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();

        MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/false);
        EXPECT_FALSE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();
    }
}

TEST_F(ExecutorTest, ExecutorInstanceInit_A2_DP8TP2_Slave)
{
    InitA2SlaveConfig(configFromManager_, /*dp=*/8, /*tp=*/2);

    // For rankIdx = 0, 1, 2, 3, both IPC and GRPC are required
    for (size_t rankIdx = 0; rankIdx < 4; ++rankIdx) {
        MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/true);
        EXPECT_TRUE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();

        MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/false);
        EXPECT_FALSE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();

        MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/true);
        EXPECT_FALSE(
            executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/rankIdx));
        MOCKCPP_NS::GlobalMockObject::reset();
    }
}

TEST_F(ExecutorTest, ExecutorInstanceInit_A2_DP1TP16) // Both IPC and GRPC are required
{
    configFromManager_["tp"] = "16";              // Set tp = 16
    configFromManager_["dp"] = "1";               // Set dp = 1
    configFromManager_["globalWorldSize"] = "16"; // Set global world size

    for (bool isMaster : {true, false}) {
        configFromManager_["isMaster"] = isMaster ? "1" : "0"; // Set as master or slave node

        // For rankIdx = 0, both IPC and GRPC are required
        MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/true);
        EXPECT_TRUE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/0));
        MOCKCPP_NS::GlobalMockObject::reset();

        MockForExecutorInstanceInit(/*isIPC=*/true, /*isGRPC=*/false);
        EXPECT_FALSE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/0));
        MOCKCPP_NS::GlobalMockObject::reset();

        MockForExecutorInstanceInit(/*isIPC=*/false, /*isGRPC=*/true);
        EXPECT_FALSE(executor_->ExecutorInstanceInit(configFromManager_, /*isMultiNodesInfer=*/true, /*rankIdx=*/0));
        MOCKCPP_NS::GlobalMockObject::reset();
    }
}

TEST_F(ExecutorTest, AsyncExecuteModel)
{
    MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(true));

    // Returns false when modelRequest is null
    ExecuteModelRequestPtr nullRequest = nullptr;
    EXPECT_FALSE(executor_->AsyncExecuteModel(nullRequest, nullptr));

    ExecuteModelRequestPtr request = std::make_unique<model_execute_data::ExecuteModelRequest>();
    model_execute_data::SequenceGroupMetadata metadata;
    std::vector<BlockId> blockIds = {1, 2, 3};
    metadata.add_block_tables(
        std::string(reinterpret_cast<const char *>(blockIds.data()), blockIds.size() * sizeof(BlockId)));
    request->mutable_seq_group_metadata_list()->Add(std::move(metadata));

    bool ret = executor_->AsyncExecuteModel(request);
    EXPECT_EQ(ret, true);
}

TEST_F(ExecutorTest, AsyncTGCleanup)
{
    MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(true));

    TGCleanupRequestPtr request = std::make_unique<model_execute_data::TGCleanupRequest>();
    request->add_seq_ids(0);
    request->add_seq_ids(1);
    request->add_seq_ids(2);
    EXPECT_TRUE(executor_->AsyncTGCleanup(request));
}

TEST_F(ExecutorTest, AsyncEOSCleanupSucc)
{
    MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(true));
    TGCleanupRequestPtr request = std::make_unique<model_execute_data::TGCleanupRequest>();
    request->add_seq_ids(0);
    request->add_seq_ids(1);
    request->add_seq_ids(2);
    EXPECT_TRUE(executor_->AsyncEOSCleanup(request));
}

TEST_F(ExecutorTest, AsyncEOSCleanupFail)
{
    MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(false));
    TGCleanupRequestPtr request = std::make_unique<model_execute_data::TGCleanupRequest>();
    request->add_seq_ids(0);
    request->add_seq_ids(1);
    request->add_seq_ids(2);
    EXPECT_FALSE(executor_->AsyncEOSCleanup(request));
}

TEST_F(ExecutorTest, SetupPDLink)
{
    MOCKER_CPP(&Communicator::SendSharedSyncRequestAndReceive, bool (*)(ExecuteRequest &, ExecuteResponse &))
        .stubs()
        .will(returnValue(true));
    MOCKER_CPP(&Executor::HandlePDLinkResponse, bool (*)(ExecuteResponse &)).stubs().will(returnValue(true));
    MOCKER_CPP(&Communicator::LaunchIPCHandleResponseThreads, bool (*)(ResponseHandler))
        .stubs()
        .will(returnValue(true));

    PDLinkRequest pdLinkRequest;
    EXPECT_TRUE(executor_->SetupPDLink(pdLinkRequest));
}

TEST_F(ExecutorTest, ExecuteKVTransfer)
{
    MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(true));

    // Returns false when modelRequest is null
    PullKVRequestPtr nullRequest = nullptr;
    EXPECT_FALSE(executor_->ExecuteKVTransfer(nullRequest, nullptr));

    PullKVRequestPtr pullKVRequest = std::make_unique<model_execute_data::PullKVRequest>();
    PullKVResponseHandler pullKVResponseHandler = [](std::shared_ptr<model_execute_data::PullKVResponse>) {};
    EXPECT_TRUE(executor_->ExecuteKVTransfer(pullKVRequest, pullKVResponseHandler));
}

TEST_F(ExecutorTest, ExecutorInstanceFinalize)
{
    MOCKER_CPP(&Communicator::SendAsyncRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(true));
    MOCKER_CPP(&Communicator::CleanUp, void (*)()).stubs();

    EXPECT_TRUE(executor_->ExecutorInstanceFinalize());
}

TEST_F(ExecutorTest, HandleInitResult_Valid)
{
    ExecuteResponse response;
    response.set_msg_type(model_execute_data::MODEL_INIT);
    auto *initResults = response.mutable_init_results();
    auto &initResultMap = *initResults->mutable_init_result_map();
    initResultMap["cpuBlockNum"] = "1024";
    initResultMap["maxPositionEmbeddings"] = "2048";
    auto *desc0 = initResults->add_kv_cache_descs();
    desc0->set_npu_block_num(1024);
    desc0->set_block_size(128);
    desc0->set_compression_ratio(1);
    desc0->set_cache_type(0);
    std::vector<ExecuteResponse> responses;
    responses.push_back(response);

    EXPECT_TRUE(executor_->HandleInitResult(responses));
    EXPECT_EQ(executor_->GetCpuBlockNum(), 1024);
    EXPECT_EQ(executor_->GetNpuBlockNum(), 1024);
    EXPECT_EQ(executor_->GetMaxPositionEmbeddings(), 2048);
}

TEST_F(ExecutorTest, HandleInitResult_Invalid)
{
    ExecuteResponse response;
    response.set_msg_type(model_execute_data::MODEL_INIT);
    std::vector<ExecuteResponse> responses;
    responses.push_back(response);
    EXPECT_FALSE(executor_->HandleInitResult(responses)); // Missing keys

    auto *initResults = responses[0].mutable_init_results();
    auto &initResultMap = *initResults->mutable_init_result_map();
    initResultMap["cpuBlockNum"] = "not_an_int"; // Invalid format
    initResultMap["maxPositionEmbeddings"] = "2048";
    auto *desc0 = initResults->add_kv_cache_descs();
    desc0->set_npu_block_num(1024);
    desc0->set_block_size(128);
    desc0->set_compression_ratio(1);
    desc0->set_cache_type(0);

    EXPECT_FALSE(executor_->HandleInitResult(responses));
}

TEST_F(ExecutorTest, HandlePDLinkResponse)
{
    ExecuteResponse response;
    response.set_msg_type(model_execute_data::PD_LINK);
    EXPECT_FALSE(executor_->HandlePDLinkResponse(response)); // Missing keys

    PDLinkResponse pdLinkResponse;
    *response.mutable_pd_link_response() = pdLinkResponse;
    EXPECT_TRUE(executor_->HandlePDLinkResponse(response));

    PDLinkResponse out = executor_->GetPDLinkResponse();
    EXPECT_EQ(out.SerializeAsString(), pdLinkResponse.SerializeAsString());
}

TEST_F(ExecutorTest, AsyncResponseHandler_WrongType)
{
    ExecuteResponse response;
    response.set_msg_type(model_execute_data::PD_LINK);
    bool ret = executor_->AsyncResponseHandler(response);
    EXPECT_FALSE(ret);
}

TEST_F(ExecutorTest, ProposeWorkerCommands)
{
    ModelLaunchConfig cfg;
    cfg.npuNumPerNode = 2;
    cfg.npuNumPerDP = 2;
    cfg.globalWorldSize = 2;
    cfg.npuDeviceIds = {"7", "8"};
    cfg.isMultiNodesInfer = false;

    std::string prefix = "shared_mem_prefix";
    pid_t pid = getpid();
    const std::string pid_s = std::to_string(pid);

    // --- single‐node test ----
    for (uint32_t localRank = 0; localRank < 2; ++localRank) {
        std::vector<std::string> cmd = executor_->BuildConnectorCommand(cfg, prefix, localRank);
        std::ostringstream expected;
        expected << "mindie_llm_backend"
                 << " --local_rank " << localRank << " --local_world_size " << cfg.npuNumPerNode << " --npu_num_per_dp "
                 << cfg.npuNumPerDP << " --npu_device_id " << cfg.npuDeviceIds[localRank] << " --parent_pid " << pid_s
                 << " --shm_name_prefix " << prefix;
        std::ostringstream cmd_oss;
        for (size_t i = 0; i < cmd.size(); ++i) {
            if (i > 0) cmd_oss << " ";
            cmd_oss << cmd[i];
        }
        EXPECT_EQ(cmd_oss.str(), expected.str());
    }

    // --- now switch to multi‐node mode ----
    cfg.npuNumPerNode = 4;
    cfg.npuNumPerDP = 4;
    cfg.globalWorldSize = 4;
    cfg.globalRankIds = {"0", "1", "2", "3"};
    cfg.npuDeviceIds = {"0", "1", "2", "3"};
    cfg.isMultiNodesInfer = true;

    for (uint32_t globalRank = 0; globalRank < 4; ++globalRank) {
        // localRank wraps around within each node
        uint32_t localRank = globalRank % cfg.npuNumPerNode;
        std::vector<std::string> cmd = executor_->BuildConnectorCommand(cfg, prefix, localRank);

        std::ostringstream expected;
        expected << "mindie_llm_backend"
                 << " --local_rank " << localRank << " --local_world_size " << cfg.npuNumPerNode << " --npu_num_per_dp "
                 << cfg.npuNumPerDP << " --npu_device_id " << cfg.npuDeviceIds[localRank] << " --parent_pid " << pid_s
                 << " --shm_name_prefix " << prefix << " --global_rank " << localRank << " --global_world_size "
                 << cfg.globalWorldSize;
        std::ostringstream cmd_oss;
        for (size_t i = 0; i < cmd.size(); ++i) {
            if (i > 0) cmd_oss << " ";
            cmd_oss << cmd[i];
        }

        EXPECT_EQ(cmd_oss.str(), expected.str());
    }
}

TEST_F(ExecutorTest, GetRemoteDPRankIdx_A2_DP2TP8_Master)
{
    InitA2MasterConfig(configFromManager_, /*dp=*/2, /*tp=*/8);

    std::unordered_map<std::string, std::string> config(configFromManager_.begin(), configFromManager_.end());
    ModelLaunchConfig modelLaunchConfig;
    ASSERT_TRUE(executor_->ParseFromModelConfig(config, modelLaunchConfig, /*isMultiNodesInfer=*/true));

    // For rankIdx = 0, the remote DP rank index should be -1
    EXPECT_EQ(executor_->GetRemoteDPRankIdx(modelLaunchConfig, 0, false), -1);
    // For rankIdx = 1, the remote DP rank index should be 0
    EXPECT_EQ(executor_->GetRemoteDPRankIdx(modelLaunchConfig, 1, false), 0);
}

TEST_F(ExecutorTest, GetRemoteDPRankIdx_A2_DP2TP8_Slave)
{
    InitA2SlaveConfig(configFromManager_, /*dp=*/2, /*tp=*/8);

    std::unordered_map<std::string, std::string> config(configFromManager_.begin(), configFromManager_.end());
    ModelLaunchConfig modelLaunchConfig;
    ASSERT_TRUE(executor_->ParseFromModelConfig(config, modelLaunchConfig, /*isMultiNodesInfer=*/true));

    // For rankIdx = 0, the remote DP rank index should be 1
    EXPECT_EQ(executor_->GetRemoteDPRankIdx(modelLaunchConfig, 0, false), 1);
}

TEST_F(ExecutorTest, GetRemoteDPRankIdx_A2_DP8TP2_Master)
{
    InitA2MasterConfig(configFromManager_, /*dp=*/8, /*tp=*/2);

    std::unordered_map<std::string, std::string> config(configFromManager_.begin(), configFromManager_.end());
    ModelLaunchConfig modelLaunchConfig;
    ASSERT_TRUE(executor_->ParseFromModelConfig(config, modelLaunchConfig, /*isMultiNodesInfer=*/true));

    // For rankIdx = 0, 1, 2, 3, the remote DP rank index should be -1
    for (size_t rankIdx = 0; rankIdx < 4; ++rankIdx) {
        EXPECT_EQ(executor_->GetRemoteDPRankIdx(modelLaunchConfig, rankIdx, false), -1);
    }
    // For rankIdx = 4, 5, 6, 7, the remote DP rank index should be 0, 1, 2, 3 respectively
    for (size_t rankIdx = 4; rankIdx < 8; ++rankIdx) {
        EXPECT_EQ(executor_->GetRemoteDPRankIdx(modelLaunchConfig, rankIdx, false), rankIdx - 4);
    }
}

TEST_F(ExecutorTest, GetRemoteDPRankIdx_A2_DP8TP2_Slave)
{
    InitA2SlaveConfig(configFromManager_, /*dp=*/8, /*tp=*/2);

    std::unordered_map<std::string, std::string> config(configFromManager_.begin(), configFromManager_.end());
    ModelLaunchConfig modelLaunchConfig;
    ASSERT_TRUE(executor_->ParseFromModelConfig(config, modelLaunchConfig, /*isMultiNodesInfer=*/true));

    // For rankIdx = 0, 1, 2, 3, the remote DP rank index should be 4, 5, 6, 7 respectively
    for (size_t rankIdx = 0; rankIdx < 4; ++rankIdx) {
        EXPECT_EQ(executor_->GetRemoteDPRankIdx(modelLaunchConfig, rankIdx, false), rankIdx + 4);
    }
}

TEST_F(ExecutorTest, GetRemoteDPRankIdx_A2_DP1TP16)
{
    InitA2MasterConfig(configFromManager_, /*dp=*/1, /*tp=*/16);

    std::unordered_map<std::string, std::string> config(configFromManager_.begin(), configFromManager_.end());
    ModelLaunchConfig modelLaunchConfig;
    ASSERT_TRUE(executor_->ParseFromModelConfig(config, modelLaunchConfig, /*isMultiNodesInfer=*/true));

    // For rankIdx = 0, the remote DP rank index should be 0
    EXPECT_EQ(executor_->GetRemoteDPRankIdx(modelLaunchConfig, 0, true), 0);

    modelLaunchConfig.isMasterNode = false; // Now switch to slave node
    // For rankIdx = 0, the remote DP rank index should be 0
    EXPECT_EQ(executor_->GetRemoteDPRankIdx(modelLaunchConfig, 0, true), 0);
}

} // namespace mindie_llm