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

#ifndef PD_COMMUNICATION_MGR_H
#define PD_COMMUNICATION_MGR_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/security/tls_certificate_provider.h>
#include <grpcpp/security/tls_credentials_options.h>
#include <grpcpp/server.h>
#include <semaphore.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "SensitiveInfoManager.h"
#include "dmi_msg_receiver.h"
#include "dmi_msg_sender.h"
#include "endpoint_def.h"
#include "prefillAndDecodeCommunication.grpc.pb.h"

namespace mindie_llm {
using grpc::Server;

class GrpcCommunicationMng {
   public:
    static GrpcCommunicationMng& GetInstance();

    bool Init(bool useTls, const std::string& localIp, const std::string& port);

    bool SendDecodeRequest(prefillAndDecodeCommunication::DecodeParameters& decodeParameters,
                           const std::string& decodeNodeIp, const std::string& reqId, std::string& errMsg);
    bool SendKvReleaseMsg(const prefillAndDecodeCommunication::RequestId& requestId, const std::string& prefillNodeIp);

    bool RegisterDecodeRequestHandler(DecodeRequestHandler decodeRequestHandler);
    bool RegisterKvReleaseHandler(KVReleaseHandler kvReleaseHandler);

   private:
    GrpcCommunicationMng() = default;

    ~GrpcCommunicationMng();

    GrpcCommunicationMng(const GrpcCommunicationMng&) = delete;

    GrpcCommunicationMng& operator=(const GrpcCommunicationMng&) = delete;

    void StopServerThread();

    bool IsValidNetAddr(const std::string& ip) const;

    bool IsValidNetPort(const std::string& port) const;

    bool IsValidServerAddr(const std::string& ip, const std::string& port) const;

    bool CreateAndConfigureSocket(int& sockfd, bool isIPv6) const;

    uint16_t ParsePortFromServerAddr(const std::string& serverAddr) const;

    bool BindIPv4Address(int sockfd, const std::string& ipPart, uint16_t port) const;

    bool BindIPv6Address(int sockfd, const std::string& ipPart, uint16_t port) const;

    bool CreateDecodeRequestSender(const std::string& decodeRequestReceiveNode);

    bool CreateKvReleaseSender(const std::string& kvReleaseReceiveNode);

    bool RunServer();

    bool RegisterAndStartService(const std::string& serverAddr, grpc::ServerBuilder& builder);

    // 安全相关配置方法
    bool SetTlsOps(grpc::experimental::TlsCredentialsOptions& tlsOpts);

    bool GetClientTlsOpts(std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions>& tlsChannelOpts);

    bool GetCertificateProvider(std::shared_ptr<grpc::experimental::CertificateProviderInterface>& certificateProvider);

    bool GetFileContent(const std::string& filePath, std::string& content);

    bool GetKeyContent(SensitiveInfoManager& keyContent);

    bool GetKeyPassWordContent(SensitiveInfoManager& keyPwContent);

    bool SetEnvForSecurity();

    bool GetSecFilePath();

    void FillIpAddress(std::string& ipAddress) const;

    std::atomic<bool> isRunning_{false};
    bool terminateSemInitialized_ = false;
    sem_t terminateSem_ = {};
    std::thread serverThread_;
    std::mutex decodeReqSenderMutex_;
    std::mutex kvReleaseSenderMutex_;

    // 目标地址:发送对象
    std::unordered_map<std::string, std::shared_ptr<DecodeRequestSender>> decodeRequestSenders_;
    std::unordered_map<std::string, std::shared_ptr<KvReleaseSender>> kvReleaseSenders_;

    std::string localAddr_;
    std::string port_;
    DecodeRequestHandler decodeRequestHandler_;
    KVReleaseHandler kvReleaseHandler_;

    // 安全相关配置成员
    bool useTls_{true};
    std::vector<std::string> caPath_;
    std::string certPath_;
    std::string certKeyPath_;
    std::string keyPassPath_;
    std::vector<std::string> crlPath_;
    std::string kfsMasterPath_;
    std::string kfsStandbyPath_;
};
}  // namespace mindie_llm

#endif  // PD_COMMUNICATION_MGR_H
