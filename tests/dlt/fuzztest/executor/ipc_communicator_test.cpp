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
#include <thread>
#include <string>
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/MockObject.h>
#include <semaphore.h>
#include "basic_types.h"
#include "ipc_communicator.h"
#include "executor_interface.h"
#include "secodeFuzz.h"

using model_execute_data::ExecuteRequest;
using model_execute_data::ExecuteResponse;
using model_execute_data::ExecuteModelRequest;
using model_execute_data::SequenceGroupMetadata;
using namespace mindie_llm;
using namespace std;
using namespace mockcpp;

class IPCCommunicatorTest : public::testing::Test {
protected:
    int seed;
    int repeat;
    void SetUp() override
    {
        seed = 0;
        repeat = 50;
    }
};

TEST_F(IPCCommunicatorTest, SetupChannelFail1)
{
    GlobalMockObject::reset();

    MOCKER_CPP(&sem_init, int (*)(sem_t *__sem, int __pshared, unsigned int __value))
            .stubs()
            .will(returnValue(1)); //non zero means fail
    IPCCommunicator ipcCommunicator("/test", 1);
    ipcCommunicator.SetupChannel();


    GlobalMockObject::reset();
    //non zero means fail
    MOCKER_CPP(&sem_init, int (*)(sem_t *__sem, int __pshared, unsigned int __value))
            .stubs()
            .will(returnValue(0))
            .then(returnValue(1));
    IPCCommunicator ipcCommunicator2("/test2", 1);
    ipcCommunicator2.SetupChannel();
}

TEST_F(IPCCommunicatorTest, SendMessageViaSMSucc)
{
    GlobalMockObject::reset();
    IPCCommunicator ipcCommunicator("/test", 1);
    bool ret = ipcCommunicator.SetupChannel();
    ExecuteRequest request;
    request.set_execute_type(MODEL_INIT);
    ipcCommunicator.SendMessageViaSM(request);
}

TEST_F(IPCCommunicatorTest, SendMessageViaSMFail1)
{
    GlobalMockObject::reset();
    IPCCommunicator ipcCommunicator("/test", 1);
    bool ret = ipcCommunicator.SetupChannel();

    MOCKER_CPP(&SerializeExecuteMessage, bool (*)(ExecuteRequest &request, std::string &buf))
            .stubs()
            .will(returnValue(false)); //true means fail, weird.

    ExecuteRequest request;
    request.set_execute_type(MODEL_INIT);
    ipcCommunicator.SendMessageViaSM(request);
}

TEST_F(IPCCommunicatorTest, SendMessageViaSMFail2)
{
    GlobalMockObject::reset();
    IPCCommunicator ipcCommunicator("/test", 1);
    bool ret = ipcCommunicator.SetupChannel();

    MOCKER_CPP(&IPCCommunicator::WriteMessage, bool (*)(const char *message, uint32_t length))
            .stubs()
            .will(returnValue(false)); //true means fail, weird.

    ExecuteRequest request;
    request.set_execute_type(MODEL_INIT);
    ipcCommunicator.SendMessageViaSM(request);
}


TEST_F(IPCCommunicatorTest, ResponseHandlingFuzz)
{
    GlobalMockObject::reset();
    IPCCommunicator ipcCommunicator("/test", 1);
    bool ret = ipcCommunicator.SetupChannel();
    ResponseHandler asyncResponseHandler = [](ExecuteResponse& response) {
        return;
    };
    ipcCommunicator.StartHandleResponseThread();
    ipcCommunicator.RegisterResponseHandler(asyncResponseHandler);
    ipcCommunicator.RegisterResponseHandler(asyncResponseHandler);
    ipcCommunicator.StartHandleResponseThread();
    ipcCommunicator.StartHandleResponseThread();
    ipcCommunicator.CleanUp();
}

TEST_F(IPCCommunicatorTest, ReceiveResponse)
{
    GlobalMockObject::reset();
    IPCCommunicator ipcCommunicator("/test", 1);
    bool ret = ipcCommunicator.SetupChannel();
    MOCKER_CPP(&IPCCommunicator::WaitOnAllSemaphores, void (*)(std::vector<sem_t *> &semaphoreList))
            .stubs();
    ExecuteResponse response;
    ipcCommunicator.ReceiveResponse(response);
    MOCKER(model_execute_data::ExecuteType_IsValid).expects(once()).will(returnValue(false));
    ipcCommunicator.ReceiveResponse(response);
    MOCKER_CPP(&ExecuteResponse::status, int32_t (*)())
            .stubs()
            .will(returnValue(1));
    ipcCommunicator.ReceiveResponse(response);
    ipcCommunicator.CleanUp();
}

TEST_F(IPCCommunicatorTest, UnlinkCloseSemaphoresFailed)
{
    IPCCommunicator ipcCommunicator("/test", 1);
    ipcCommunicator.SetupChannel();
    GlobalMockObject::reset();
    MOCKER_CPP(&sem_unlink, int (*)(const char *__name))
            .stubs()
            .will(returnValue(1)); //non zero means fail
    ipcCommunicator.CleanUp();

    IPCCommunicator ipcCommunicator2("/test2", 1);
    ipcCommunicator2.SetupChannel();
    GlobalMockObject::reset();
    MOCKER_CPP(&sem_unlink, int (*)(const char *__name))
            .stubs()
            .will(returnValue(0))
            .then(returnValue(1));
    ipcCommunicator2.CleanUp();

    IPCCommunicator ipcCommunicator3("/test3", 1);
    ipcCommunicator3.SetupChannel();
    GlobalMockObject::reset();
    MOCKER_CPP(&sem_close, int (*)(sem_t *__sem))
            .stubs()
            .will(returnValue(1));
    ipcCommunicator3.CleanUp();

}

TEST_F(IPCCommunicatorTest, ReceiveInitResponseTest)
{
    IPCCommunicator ipcCommunicator("/test", 1);
    ipcCommunicator.SetupChannel();
    *reinterpret_cast<uint32_t*>(ipcCommunicator.responseSharedMemory_.sharedMemory->GetBuf()) = 4;
    ipcCommunicator.SignalAllSemaphores(ipcCommunicator.responseSharedMemory_.semConsumeVec);
    std::vector<ExecuteResponse> reponses;
    ipcCommunicator.ReceiveInitResponses(reponses);
    MOCKER_CPP(&IPCCommunicator::WaitOnAllSemaphores, void (*)(std::vector<sem_t *> &semaphoreList))
            .stubs();
    MOCKER_CPP(&IPCCommunicator::ParseResponse, bool (*)(ExecuteResponse &executeResponse, char *sharedBuf))
            .stubs()
            .will(returnValue(true));
    ipcCommunicator.ReceiveInitResponses(reponses);
}

TEST_F(IPCCommunicatorTest, WriteMessageFuzz)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, "WriteMessageFuzz write fail or succ", 0)
    {
        GlobalMockObject::reset();
        IPCCommunicator ipcCommunicator("/test", 1);
        bool ret = ipcCommunicator.SetupChannel();
        EXPECT_TRUE(ret);
        const string msg = DTSetGetString(&g_Element[0], 1, 100, "start str", "no desc");
        bool succ = ipcCommunicator.WriteMessage(msg.c_str(), msg.size());
        EXPECT_TRUE(succ);

        GlobalMockObject::reset();
        MOCKER_CPP(&SharedMemory::Write, bool (*)(uint32_t dstOffset, const char *src, uint32_t size))
            .stubs()
            .will(returnValue(true)); //true means fail, weird.

        const string msg1 = DTSetGetString(&g_Element[0], 1, 100, "start str", "no desc");
        bool succ1 = ipcCommunicator.WriteMessage(msg1.c_str(), msg1.size());
        EXPECT_TRUE(!succ1);

        GlobalMockObject::reset();
        // first write succ, second failed
        MOCKER_CPP(&SharedMemory::Write, bool (*)(uint32_t dstOffset, const char *src, uint32_t size))
            .stubs()
            .will(returnValue(false))
            .then(returnValue(true));
        const string msg2 = DTSetGetString(&g_Element[0], 1, 100, "start str", "no desc");
        bool succ2 = ipcCommunicator.WriteMessage(msg.c_str(), msg.size());
        EXPECT_TRUE(!succ2);

    }
    DT_FUZZ_END()
    SUCCEED();
}
TEST_F(IPCCommunicatorTest, InitSemaphoresFuzz)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, "InitSemaphoresFuzz succ", 0)
    {
        IPCCommunicator ipcCommunicator("/test", 1);
        bool ret = ipcCommunicator.SetupChannel();
        EXPECT_TRUE(ret);
    }
    DT_FUZZ_END()

    DT_FUZZ_START(seed, repeat, "InitSemaphoresFuzz failure", 0)
    {
        IPCCommunicator ipcCommunicator("test", 1);
        bool ret = ipcCommunicator.SetupChannel();
        EXPECT_TRUE(!ret);
    }
    DT_FUZZ_END()

    SUCCEED();
}

TEST_F(IPCCommunicatorTest, SerializeExecuteMessageFuzz)
{
    DT_SetEnableFork(0);
    DT_FUZZ_START(seed, repeat, "SerializeExecuteMessage message size > MAX_REQUEST_BUF_SIZE", 0)
    {

        ExecuteModelRequestPtr batchInferRequest = std::make_unique<model_execute_data::ExecuteModelRequest>();
        model_execute_data::SequenceGroupMetadata metadata;

        int init_blocks = 1024 * 1024 * 4;
        int min_blocks = 1024 * 1024 * 8; // will exceed the buffer size
        int max_blocks = 1024 * 1024 * 16;
        int block_num = *reinterpret_cast<int*>(DT_SetGetNumberRange(&g_Element[0], init_blocks, min_blocks, max_blocks));
        std::vector<BlockId> blockIds(block_num, 0);
        metadata.add_block_tables(
            std::string(reinterpret_cast<const char *>(blockIds.data()), blockIds.size() * sizeof(BlockId)));
        batchInferRequest->mutable_seq_group_metadata_list()->Add(std::move(metadata));

        ExecuteRequest execRequest;
        execRequest.set_execute_type(MODEL_INFER);
        execRequest.mutable_execute_model_request()->CopyFrom(*batchInferRequest);
        string buf;
        bool ret = SerializeExecuteMessage(execRequest, buf);
        EXPECT_TRUE(!ret);
    }

    DT_FUZZ_END()

    DT_FUZZ_START(seed, repeat, "SerializeExecuteMessage serialze failure", 0)
    {
        ExecuteModelRequestPtr batchInferRequest = std::make_unique<model_execute_data::ExecuteModelRequest>();
        model_execute_data::SequenceGroupMetadata metadata;
        
        int init_blocks = 1024 * 1;
        int min_blocks =  1024 * 2; // will exceed the buffer size
        int max_blocks =  1024 * 4;
        int block_num = *reinterpret_cast<int*>(DT_SetGetNumberRange(&g_Element[0], init_blocks, min_blocks, max_blocks));
        std::vector<BlockId> blockIds(block_num, 0);
        metadata.add_block_tables(
            std::string(reinterpret_cast<const char *>(blockIds.data()), blockIds.size() * sizeof(BlockId)));
        batchInferRequest->mutable_seq_group_metadata_list()->Add(std::move(metadata));

        ExecuteRequest execRequest;
        execRequest.set_execute_type(MODEL_INFER);
        execRequest.mutable_execute_model_request()->CopyFrom(*batchInferRequest);
        
        GlobalMockObject::reset();
        MOCKER_CPP(&model_execute_data::ExecuteRequest::SerializeToArray, bool (*)(void* data, int size))
            .stubs()
            .will(returnValue(false));

        string buf;
        bool ret = SerializeExecuteMessage(execRequest, buf);
        EXPECT_TRUE(!ret);
    }
    DT_FUZZ_END()
    SUCCEED();
}