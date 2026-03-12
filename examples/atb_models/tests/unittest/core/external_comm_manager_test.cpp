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
#include "atb_speed/base/external_comm_manager.h"
#include "atb_speed/utils/singleton.h"


HcclResult HcclGetCommName(HcclComm comm, char* commName)
{
    HcclComm hcclComm = commName;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommDestroy(HcclComm comm)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCreateSubCommConfig(HcclComm *comm, uint32_t rankNum, uint32_t *rankIds,
    uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig *config, HcclComm *subComm)
{
    return HcclResult::HCCL_SUCCESS;
}

class ExternalCommManagerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        char hcclCommName[128] = "hcclCommTemp";
        HcclComm hcclComm = hcclCommName;
        MOCKER(atb::Comm::CreateHcclCommByRankTableFile).stubs()
            .will(returnValue(hcclComm));
        MOCKER(HcclGetCommName).stubs()
            .with(any(), outBoundP(hcclCommName, sizeof(hcclCommName)))
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(HcclCommConfigInit).stubs();
        MOCKER(HcclCreateSubCommConfig).stubs()
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        atb_speed::GetSingleton<atb_speed::ExternalCommManager>().SetLcclCommDomainRange(0, 65535);
    }

    void TearDown() override
    {
        MOCKER(HcclCommDestroy).stubs()
            .with(any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Reset();
    }

    uint32_t worldSize = 4;
    uint32_t subCommRankId = 0;
    std::string backend = "hccl";
    std::string rankTableFile = "";
    uint32_t streamId = 0;
};

// 单例测试，验证返回的实例地址是否相同
TEST_F(ExternalCommManagerTest, SingletonTest)
{
    atb_speed::ExternalCommManager& instance1 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>();
    atb_speed::ExternalCommManager& instance2 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>();
    EXPECT_EQ(&instance1, &instance2);
}

// 测试 Init 接口
TEST_F(ExternalCommManagerTest, InitTest)
{
    uint32_t worldSizeTemp = 1;
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFile, streamId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().worldSize_, worldSize);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().rank_, subCommRankId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().rankTableFile_, rankTableFile);

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, "lcoc",
        rankTableFile, streamId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().worldSize_, worldSize);

    // worldSizeTemp
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSizeTemp, subCommRankId, backend,
        rankTableFile, streamId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().worldSize_, worldSizeTemp);

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSizeTemp, subCommRankId, "lccl",
        rankTableFile, streamId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().worldSize_, worldSizeTemp);

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSizeTemp, subCommRankId, "lcoc",
        rankTableFile, streamId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().worldSize_, worldSizeTemp);
}

// 测试 Ranktablefile Init 接口
TEST_F(ExternalCommManagerTest, RanktablefileInitTest)
{
    std::string rankTableFileTemp = "filepath";

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFileTemp, streamId);
    // 多次调用直接返回
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFileTemp, streamId);

    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().worldSize_, worldSize);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().rank_, subCommRankId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().rankTableFile_, rankTableFileTemp);
}

// 测试 LwdGlobalComm Init 接口
TEST_F(ExternalCommManagerTest, InitLwdGlobalCommTest)
{
    std::string lwdGlobalComm = std::to_string(1);

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFile, streamId, lwdGlobalComm);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().worldSize_, worldSize);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().rank_, subCommRankId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().rankTableFile_, rankTableFile);
    EXPECT_EQ((uint64_t)(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().globalComm_), (uint64_t)1);
}

// 测试 GetHcclSubCommDomain 接口
TEST_F(ExternalCommManagerTest, GetHcclSubCommDomainTest)
{
    std::string rankTableFileTemp = "filepath";
    std::string backendTemp = "lccl";
    std::vector<uint32_t> rankIds = {0, 1};
    uint32_t bufferSize = 128;

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFileTemp, streamId);
    std::string commDomain_hccl = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backend, bufferSize, 0, false);
    std::string commDomain_lccl = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backendTemp, bufferSize, 0, false);
    EXPECT_NE(commDomain_lccl, commDomain_hccl);
}

// 测试 GetCommDomain 接口
TEST_F(ExternalCommManagerTest, GetCommDomainlcclTest)
{
    std::string backendTemp = "lccl";
    std::vector<uint32_t> rankIds = {0, 1};
    uint32_t bufferSize = 128;

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backendTemp,
        rankTableFile, streamId);
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], {0}, subCommRankId, backendTemp, bufferSize, 0, false);
    std::string commDomain1 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backendTemp, bufferSize, 0, false);
    std::string commDomain2 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backendTemp, bufferSize, 0, true);
    std::string commDomain3 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backendTemp, 200, 0, true);
    std::string commDomain4 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backendTemp, bufferSize, 1, true);
    EXPECT_NE(commDomain1, commDomain2);
    EXPECT_NE(commDomain2, commDomain3);
    EXPECT_NE(commDomain2, commDomain4);
}

// 测试 GetCommDomain 接口
TEST_F(ExternalCommManagerTest, GetCommDomainlcocTest)
{
    std::string backendTemp = "lcoc";
    std::vector<uint32_t> rankIds = {0, 1};
    uint32_t bufferSize = 128;

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backendTemp,
        rankTableFile, streamId);
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], {0}, subCommRankId, backendTemp, bufferSize, 0, false);
    std::string commDomain1 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backendTemp, bufferSize, 0, false);
    std::string commDomain2 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backendTemp, bufferSize, 0, true);
    EXPECT_NE(commDomain1, commDomain2);
}

// 测试 GetCommDomain 接口
TEST_F(ExternalCommManagerTest, GetCommDomainHcclTest)
{
    std::vector<uint32_t> rankIds = {0, 1};
    uint32_t bufferSize = 128;

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFile, streamId);
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], {0}, subCommRankId, backend, bufferSize, 0, false);
    std::string commDomain1 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backend, bufferSize, 0, false);
    std::string commDomain2 = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backend, bufferSize, 0, true);
    EXPECT_NE(commDomain1, commDomain2);
    std::string commDomain_null = atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
            rankIds[0], {0}, subCommRankId, backend, bufferSize, 0, true);
    EXPECT_EQ(commDomain_null, "");
}

// 测试 异常抛出 接口
TEST_F(ExternalCommManagerTest, ThrowErroTest)
{
    std::vector<uint32_t> rankIds = {0, 1};
    uint32_t bufferSize = 128;

    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFile, streamId);
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().SetLcclCommDomainRange(0, 0);
    EXPECT_THROW(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backend, bufferSize, 0, true), std::runtime_error);
        atb_speed::GetSingleton<atb_speed::ExternalCommManager>().SetLcclCommDomainRange(UINT32_MAX, UINT32_MAX);
    EXPECT_THROW(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
                rankIds[0], rankIds, subCommRankId, backend, bufferSize, 0, true), std::runtime_error);
}

// 测试 GetCommPtr 接口
TEST_F(ExternalCommManagerTest, GetCommPtrTest)
{
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFile, streamId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr("0"), nullptr);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr(""), nullptr);
    EXPECT_THROW(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr("100"), std::out_of_range);
}

// 测试 GetCommInfo 接口
TEST_F(ExternalCommManagerTest, GetCommInfoTest)
{
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFile, streamId);
    EXPECT_NE(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommInfo("0"), nullptr);
    EXPECT_THROW(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommInfo(""), std::runtime_error);
    EXPECT_THROW(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommInfo("100"), std::out_of_range);
}

// 测试 IsInitialized 接口
TEST_F(ExternalCommManagerTest, IsInitializedTest)
{
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFile, streamId);
    EXPECT_EQ(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().IsInitialized(), true);
}

// 测试 PrintCommInfo 接口
TEST_F(ExternalCommManagerTest, PrintCommInfoTest)
{
    atb_speed::GetSingleton<atb_speed::ExternalCommManager>().Init(worldSize, subCommRankId, backend,
        rankTableFile, streamId);
    EXPECT_NE(atb_speed::GetSingleton<atb_speed::ExternalCommManager>().PrintCommInfo(), "");
}