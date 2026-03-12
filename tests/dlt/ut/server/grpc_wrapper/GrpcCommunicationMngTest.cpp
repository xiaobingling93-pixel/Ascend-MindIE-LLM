/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <libgen.h>
#include <gtest/gtest.h>
#include <string>
#include <cstring>
#include "mockcpp/mockcpp.hpp"
#include <openssl/pem.h>
#include <openssl/evp.h>
#define private public
#include "env_util.h"
#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "grpc_communication_mng.h"
#include "common_util.h"
#include "memory_utils.h"
#include "mock_util.h"

using namespace prefillAndDecodeCommunication;

MOCKER_CPP_OVERLOAD_EQ(ModelDeployConfig)
MOCKER_CPP_OVERLOAD_EQ(ScheduleConfig)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(BackendConfig)

namespace mindie_llm {

class GrpcCommunicationMngTest : public testing::Test {
protected:
    void SetUp() override
    {
        mng = &GrpcCommunicationMng::GetInstance();
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE",
                                         GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_grpc.json");
        MockServerConfig();
    }

    void TearDown() override
    {
        EnvUtil::GetInstance().ClearEnvVar("RANK_TABLE_FILE");
        EnvUtil::GetInstance().ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
        EnvUtil::GetInstance().ClearEnvVar("MIES_CONTAINER_IP");
        EnvUtil::GetInstance().ClearEnvVar("HOST_IP");
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

    GrpcCommunicationMng *mng;
    DecodeParameters params;
    std::string errMsg;
    std::string reqId = "req123";

    ServerConfig serverConfig_;

    std::string GetParentDirectory()
    {
        char buffer[1024];
        try {
            return std::filesystem::current_path().string();
        } catch (const std::filesystem::filesystem_error &e) {
            std::cerr << "Error getting current directory: " << e.what() << std::endl;
            return "";
        }

        std::unique_ptr<char, decltype(&std::free)> temp(strdup(buffer), &std::free);

        if (temp == nullptr) {
            std::cerr << "Memory allocation failed" << std::endl;
            return "";
        }

        char *parent = dirname(temp.get());
        std::string result(parent);
        return result;
    }

    void MockServerConfig()
    {
        serverConfig_.allowAllZeroIpListening = false;
        serverConfig_.httpsEnabled = false;
        serverConfig_.ipAddress = "127.0.0.1";
        serverConfig_.managementIpAddress = "127.0.0.2";
        serverConfig_.port = 1025;
        serverConfig_.managementPort = 1026;
        serverConfig_.metricsPort = 1027;
        serverConfig_.maxLinkNum = 1000;
        serverConfig_.fullTextEnabled = false;
        serverConfig_.tlsCaPath = "../../config_manager/conf/cert/";
        serverConfig_.tlsCaFile = {"ca.pem"};
        serverConfig_.tlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.tlsPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.tlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.tlsCrlFiles = {"server_crl.pem"};
        serverConfig_.managementTlsCaFile = {"management_ca.pem"};
        serverConfig_.managementTlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.managementTlsPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.managementTlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.managementTlsCrlFiles = {"server_crl.pem"};
        serverConfig_.inferMode = "standard";
        serverConfig_.interCommTLSEnabled = true;
        serverConfig_.interCommPort = 1121;
        serverConfig_.interCommTlsCaPath = "../../config_manager/conf/cert/";
        serverConfig_.interCommTlsCaFiles = {"ca.pem"};
        serverConfig_.interCommTlsCert = "../../config_manager/conf/cert/server.pem";
        serverConfig_.interCommPk = "../../config_manager/conf/cert/server.key.pem";
        serverConfig_.interCommTlsCrlPath = "../../config_manager/conf/cert/";
        serverConfig_.interCommTlsCrlFiles = {"server_crl.pem"};
        serverConfig_.tokenTimeout = 5;
        serverConfig_.e2eTimeout = 5;
        serverConfig_.distDPServerEnabled = false;
        MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(serverConfig_));
    }
};

TEST_F(GrpcCommunicationMngTest, GetInstanceReturnsSingleton)
{
    GrpcCommunicationMng &instance1 = GrpcCommunicationMng::GetInstance();
    GrpcCommunicationMng &instance2 = GrpcCommunicationMng::GetInstance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(GrpcCommunicationMngTest, InitWithInvalidAddress) { EXPECT_FALSE(mng->Init(true, "invalid_addr", "50051")); }

TEST_F(GrpcCommunicationMngTest, InitWithValidAddress)
{
    EXPECT_FALSE(mng->Init(true, "127.0.0.1", "50051"));
    MOCKER_CPP(&GrpcCommunicationMng::SetEnvForSecurity, bool (*)()).stubs().will(returnValue(false));
    EXPECT_FALSE(mng->Init(true, "127.0.0.1", "50051"));
    MOCKER_CPP(&GrpcCommunicationMng::IsValidServerAddr, bool (*)(const std::string &, const std::string &))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(mng->Init(false, "127.0.0.1", "50051"));
    MOCKER_CPP(sem_init, int (*)(sem_t *, int, unsigned int)).stubs().will(returnValue(1));
    EXPECT_FALSE(mng->Init(true, "127.0.0.1", "50051"));
    GlobalMockObject::verify();
    GlobalMockObject::reset();
    MockServerConfig();
    MOCKER_CPP(sem_init, int (*)(sem_t *, int, unsigned int)).stubs().will(returnValue(0));
    MOCKER_CPP(&GrpcCommunicationMng::IsValidServerAddr, bool (*)(const std::string &, const std::string &))
        .stubs()
        .will(returnValue(true));
    EXPECT_TRUE(mng->Init(false, "127.0.0.1", "50051"));
    mng->StopServerThread();
}

TEST_F(GrpcCommunicationMngTest, SetTlsOpsSuccess)
{
    grpc::experimental::TlsCredentialsOptions tlsOpts;
    mng->crlPath_ = {};
    mng->caPath_ = {};
    EXPECT_FALSE(mng->SetTlsOps(tlsOpts));
    mng->caPath_ = {GetParentDirectory() + "/../../config_manager/conf/cert/server.ca_err"};
    EXPECT_FALSE(mng->SetTlsOps(tlsOpts));
    mng->caPath_ = {GetParentDirectory() + "/../../config_manager/conf/cert/server.pem"};
    EXPECT_TRUE(mng->SetTlsOps(tlsOpts));
    mng->crlPath_ = {GetParentDirectory() + "/conf/server.crl_err"};
    EXPECT_FALSE(mng->SetTlsOps(tlsOpts));
}

TEST_F(GrpcCommunicationMngTest, GetFileContentSuccess)
{
    std::string content;
    EXPECT_FALSE(mng->GetFileContent("/valid/path", content));
    EXPECT_EQ(content, "");
    EXPECT_TRUE(mng->GetFileContent(GetParentDirectory() + "/../../config_manager/conf/cert/server.pem", content));
    EXPECT_EQ(content, "test");
}

TEST_F(GrpcCommunicationMngTest, CreateDecodeRequestSenderAndSendDecodeRequest)
{
    EXPECT_FALSE(mng->Init(true, "127.0.0.1", "50051"));
    EXPECT_FALSE(mng->CreateDecodeRequestSender("127.0.0.1"));
    EXPECT_FALSE(mng->SendDecodeRequest(params, "127.0.0.1", reqId, errMsg));
    MOCKER_CPP(&GrpcCommunicationMng::GetClientTlsOpts,
               bool (*)(std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions> &))
        .stubs()
        .will(returnValue(true));
    EXPECT_FALSE(mng->CreateDecodeRequestSender("127.0.0.1"));
    MOCKER_CPP(&DecodeRequestSender::Init, bool (*)()).stubs().will(returnValue(true));
    EXPECT_TRUE(mng->CreateDecodeRequestSender("127.0.0.1"));
    EXPECT_FALSE(mng->SendDecodeRequest(params, "valid_node", reqId, errMsg));
    EXPECT_FALSE(mng->SendDecodeRequest(params, "127.0.0.1", reqId, errMsg));
    MOCKER_CPP(&DecodeRequestSender::SendDecodeRequestMsg,
               bool (*)(const DecodeParameters &, const std::string &, std::string &))
        .stubs()
        .will(returnValue(true));
    EXPECT_TRUE(mng->SendDecodeRequest(params, "127.0.0.1", reqId, errMsg));
}

TEST_F(GrpcCommunicationMngTest, CreateKvReleaseSenderAndSendKvReleaseMsg)
{
    prefillAndDecodeCommunication::RequestId requestId;
    EXPECT_FALSE(mng->SendKvReleaseMsg(requestId, "127.0.0.1"));
    EXPECT_FALSE(mng->Init(true, "127.0.0.1", "50051"));
    EXPECT_FALSE(mng->CreateKvReleaseSender("127.0.0.1"));
    MOCKER_CPP(&GrpcCommunicationMng::GetClientTlsOpts,
               bool (*)(std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions> &))
        .stubs()
        .will(returnValue(true));
    EXPECT_FALSE(mng->CreateKvReleaseSender("127.0.0.1"));
    MOCKER_CPP(&KvReleaseSender::Init, bool (*)()).stubs().will(returnValue(true));
    EXPECT_TRUE(mng->CreateKvReleaseSender("127.0.0.1"));
    EXPECT_FALSE(mng->SendKvReleaseMsg(requestId, "valid_node"));
    EXPECT_FALSE(mng->SendKvReleaseMsg(requestId, "127.0.0.1"));
    MOCKER_CPP(&KvReleaseSender::SendKvReleaseMsg, bool (*)(const prefillAndDecodeCommunication::RequestId &))
        .stubs()
        .will(returnValue(true));
    EXPECT_TRUE(mng->SendKvReleaseMsg(requestId, "127.0.0.1"));
}

TEST_F(GrpcCommunicationMngTest, GetKeyContentSuccess)
{
    SensitiveInfoManager keyContent{nullptr, 0, MAX_PRIVATE_KEY_CONTENT_BYTE_LEN, MIN_PRIVATE_KEY_CONTENT_BYTE_LEN};
    EXPECT_FALSE(mng->GetKeyContent(keyContent));
    EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "0");
    EXPECT_FALSE(mng->GetKeyContent(keyContent));
    EXPECT_FALSE(mng->GetKeyContent(keyContent));
    EXPECT_FALSE(mng->GetKeyContent(keyContent));
    MOCKER_CPP(&SensitiveInfoManager::CopySensitiveInfo, bool (*)(char *, int)).stubs().will(returnValue(true));
    EXPECT_FALSE(mng->GetKeyContent(keyContent));
    MOCKER_CPP(&memset_s, errno_t (*)(PVOID, size_t, int, size_t)).stubs().will(returnValue(0));
    EXPECT_FALSE(mng->GetKeyContent(keyContent));
    MOCKER_CPP(&SensitiveInfoManager::IsValid, bool (*)()).stubs().will(returnValue(true));
    EXPECT_FALSE(mng->GetKeyContent(keyContent));
    mng->certKeyPath_ = GetParentDirectory() + "/../../config_manager/conf/cert/server.key.pem";
    EXPECT_FALSE(mng->GetKeyContent(keyContent));
}

TEST_F(GrpcCommunicationMngTest, SetEnvForSecuritySuccess) { EXPECT_TRUE(mng->SetEnvForSecurity()); }

TEST_F(GrpcCommunicationMngTest, GetSecFilePathSuccess) { EXPECT_TRUE(mng->GetSecFilePath()); }

TEST_F(GrpcCommunicationMngTest, FillIpAddressAddsPort)
{
    std::string ip = "127.0.0.1";
    std::string ip2 = "1270:0012:1111:8081";
    std::string ip3 = "127.0.0.1;8082";
    mng->FillIpAddress(ip);
    EXPECT_EQ(ip, "127.0.0.1:1121");
    mng->FillIpAddress(ip2);
    EXPECT_EQ(ip2, "[1270:0012:1111:8081]:1121");
    mng->FillIpAddress(ip3);
    EXPECT_EQ(ip3, "127.0.0.1:8082");
}

TEST_F(GrpcCommunicationMngTest, RunServer)
{
    mng->useTls_ = true;
    EXPECT_FALSE(mng->RunServer());
}

TEST_F(GrpcCommunicationMngTest, StopServerThreadHandlesErrors)
{
    if (!mng->serverThread_.joinable()) {
        mng->serverThread_ = std::thread([]() { std::this_thread::sleep_for(std::chrono::milliseconds(50)); });
    }
    mng->terminateSemInitialized_ = false;
    mng->isRunning_.store(true);
    pthread_cancel(mng->serverThread_.native_handle());
    mng->serverThread_.detach();
    mng->StopServerThread();
    EXPECT_FALSE(mng->isRunning_);
}

auto decodeHandler = [](const auto &, auto &) {};
auto kvHandler = [](const auto &) {};

TEST_F(GrpcCommunicationMngTest, RegisterHandlersAndStartServiceSuccess)
{
    EXPECT_TRUE(mng->RegisterDecodeRequestHandler(decodeHandler));
    EXPECT_TRUE(mng->RegisterKvReleaseHandler(kvHandler));
    mng->isRunning_.store(true);
    EXPECT_FALSE(mng->RegisterDecodeRequestHandler(decodeHandler));
    EXPECT_FALSE(mng->RegisterKvReleaseHandler(kvHandler));
    mng->StopServerThread();
}

TEST_F(GrpcCommunicationMngTest, BindIPv6AddressSuccess)
{
    int testSocket = socket(AF_INET6, SOCK_STREAM, 0);
    ASSERT_GE(testSocket, 0) << "Failed to create test socket";

    std::string ipv6Address = "::1";
    uint16_t port = 8080;

    bool result = mng->BindIPv6Address(testSocket, ipv6Address, port);

    EXPECT_TRUE(result) << "Failed to bind IPv6 address";

    close(testSocket);
}

TEST_F(GrpcCommunicationMngTest, BindIPv6AddressInvalidIP)
{
    int testSocket = socket(AF_INET6, SOCK_STREAM, 0);
    ASSERT_GE(testSocket, 0) << "Failed to create test socket";

    std::string invalidIp = "invalid-ipv6-address";
    uint16_t port = 8080;

    bool result = mng->BindIPv6Address(testSocket, invalidIp, port);

    EXPECT_FALSE(result) << "Should have failed with invalid IPv6 address";

    close(testSocket);
}

TEST_F(GrpcCommunicationMngTest, BindIPv6AddressPortConflict)
{
    int firstSocket = socket(AF_INET6, SOCK_STREAM, 0);
    ASSERT_GE(firstSocket, 0) << "Failed to create first socket";

    std::string ipv6Address = "::1";
    uint16_t port = 8081;

    bool firstBind = mng->BindIPv6Address(firstSocket, ipv6Address, port);
    EXPECT_TRUE(firstBind) << "First bind should succeed";

    int secondSocket = socket(AF_INET6, SOCK_STREAM, 0);
    ASSERT_GE(secondSocket, 0) << "Failed to create second socket";

    bool secondBind = mng->BindIPv6Address(secondSocket, ipv6Address, port);
    EXPECT_FALSE(secondBind) << "Second bind should fail due to port conflict";

    close(firstSocket);
    close(secondSocket);
}

TEST_F(GrpcCommunicationMngTest, BindIPv6AddressError)
{
    int testSocket = socket(AF_INET6, SOCK_STREAM, 0);
    ASSERT_GE(testSocket, 0) << "Failed to create test socket";

    std::string ipv6Address = "::1";
    uint16_t randomPort = 0;
    MOCKER_CPP(&memset_s, errno_t (*)(PVOID, size_t, int, size_t)).stubs().will(returnValue(-1));
    bool result = mng->BindIPv6Address(testSocket, ipv6Address, randomPort);

    EXPECT_FALSE(result) << "Should Fail with random port allocation";

    close(testSocket);
}

TEST_F(GrpcCommunicationMngTest, GetCertificateProviderBasic)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider;

    bool result = mng->GetCertificateProvider(provider);
    EXPECT_NO_THROW(mng->GetCertificateProvider(provider));
}

TEST_F(GrpcCommunicationMngTest, GetCertificateProviderMultipleCalls)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider;

    for (int i = 0; i < 3; i++) {
        bool result = mng->GetCertificateProvider(provider);
        EXPECT_NO_THROW(mng->GetCertificateProvider(provider));
        if (provider) {
            EXPECT_NE(provider, nullptr);
        }
    }
}

TEST_F(GrpcCommunicationMngTest, GetCertificateProviderNullCheck)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider = nullptr;
    EXPECT_NO_THROW(mng->GetCertificateProvider(provider));
}

TEST_F(GrpcCommunicationMngTest, CertificateProviderLifecycle)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider;

    bool result = mng->GetCertificateProvider(provider);

    if (provider) {
        auto useCount = provider.use_count();
        EXPECT_GE(useCount, 1);

        auto providerCopy = provider;
        EXPECT_GE(provider.use_count(), 2);
    }
}

TEST_F(GrpcCommunicationMngTest, GetCertificateProviderWithMockConfig)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider;

    bool result = mng->GetCertificateProvider(provider);

    EXPECT_NO_THROW(mng->GetCertificateProvider(provider));
}

TEST_F(GrpcCommunicationMngTest, GetCertificateProviderErrorHandling)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider;

    EXPECT_NO_THROW(mng->GetCertificateProvider(provider));

    auto originalCertPath = serverConfig_.tlsCert;
    serverConfig_.tlsCert = "/invalid/path/cert.pem";

    MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(serverConfig_));

    EXPECT_NO_THROW(mng->GetCertificateProvider(provider));

    serverConfig_.tlsCert = originalCertPath;
    MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(serverConfig_));
}

TEST_F(GrpcCommunicationMngTest, GetCertificateProviderThreadSafety)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider;

    std::thread t1([this, &provider]() { EXPECT_NO_THROW(mng->GetCertificateProvider(provider)); });

    std::thread t2([this, &provider]() { EXPECT_NO_THROW(mng->GetCertificateProvider(provider)); });

    t1.join();
    t2.join();
}

TEST_F(GrpcCommunicationMngTest, GetCertificateProviderAfterReconfiguration)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider;

    bool result1 = mng->GetCertificateProvider(provider);

    serverConfig_.tlsCert = "../../config_manager/conf/cert/updated_server.pem";
    MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(serverConfig_));

    bool result2 = mng->GetCertificateProvider(provider);

    EXPECT_NO_THROW(mng->GetCertificateProvider(provider));

    serverConfig_.tlsCert = "../../config_manager/conf/cert/server.pem";
    MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(serverConfig_));
}

TEST_F(GrpcCommunicationMngTest, GetCertificateProviderMemoryManagement)
{
    std::shared_ptr<grpc::experimental::CertificateProviderInterface> provider;

    for (int i = 0; i < 5; i++) {
        bool result = mng->GetCertificateProvider(provider);

        if (provider) {
            auto useCount = provider.use_count();
            EXPECT_GE(useCount, 1);

            {
                auto tempProvider = provider;
                EXPECT_EQ(provider.use_count(), useCount + 1);
            }
            EXPECT_EQ(provider.use_count(), useCount);
        }
    }
}
} // namespace mindie_llm