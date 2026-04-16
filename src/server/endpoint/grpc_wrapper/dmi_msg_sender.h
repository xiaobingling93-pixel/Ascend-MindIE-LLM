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

#ifndef PD_MSG_SENDER_H
#define PD_MSG_SENDER_H

#include <grpcpp/security/tls_credentials_options.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include "prefillAndDecodeCommunication.grpc.pb.h"

namespace mindie_llm {
// GrpcMsgSender class
class GrpcMsgSender {
   public:
    GrpcMsgSender(const std::string &receiverAddr, const std::string &localAddr, bool useTls,
                  std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions> tlsChannelOpt)
        : receiverAddr_(receiverAddr),
          localAddr_(localAddr),
          TlsChannelOpt_(std::move(tlsChannelOpt)),
          useTls_(useTls) {}

    virtual ~GrpcMsgSender() = default;
    // Initialize the sender
    bool Init();

   protected:
    virtual void CreateStub(std::shared_ptr<grpc::Channel> &channel) = 0;
    std::chrono::system_clock::time_point CalDeadLine(uint32_t milliseconds) const {
        auto steadyNow = std::chrono::steady_clock::now();
        auto steadyDeadline = steadyNow + std::chrono::milliseconds(milliseconds);
        auto systemNow = std::chrono::system_clock::now();
        auto systemDeadline = systemNow + (steadyDeadline - steadyNow);
        return systemDeadline;
    }
    std::string receiverAddr_;
    std::string localAddr_;
    std::mutex lock_;
    std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions> TlsChannelOpt_{nullptr};

   private:
    bool useTls_;
};

class DecodeRequestSender final : public GrpcMsgSender {
   public:
    DecodeRequestSender(const std::string &receiverAddr, const std::string &localAddr, bool useTls,
                        std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions> tlsChannelOpt)
        : GrpcMsgSender(receiverAddr, localAddr, useTls, std::move(tlsChannelOpt)) {}
    ~DecodeRequestSender() override = default;
    bool SendDecodeRequestMsg(const prefillAndDecodeCommunication::DecodeParameters &message, const std::string &reqId,
                              std::string &errMsg);

   private:
    void CreateStub(std::shared_ptr<grpc::Channel> &channel) override;

    std::unique_ptr<prefillAndDecodeCommunication::DecodeService::Stub> stub_{nullptr};
};

class KvReleaseSender final : public GrpcMsgSender {
   public:
    KvReleaseSender(const std::string &receiverAddr, const std::string &localAddr, bool useTls,
                    std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions> tlsChannelOpt)
        : GrpcMsgSender(receiverAddr, localAddr, useTls, std::move(tlsChannelOpt)) {}
    ~KvReleaseSender() override = default;
    bool SendKvReleaseMsg(const prefillAndDecodeCommunication::RequestId &message);

   private:
    void CreateStub(std::shared_ptr<grpc::Channel> &channel) override;

    std::unique_ptr<prefillAndDecodeCommunication::PrefillService::Stub> stub_{nullptr};
};

}  // namespace mindie_llm

#endif  // PD_MSG_SENDER_H
