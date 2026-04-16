/**
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

#ifndef PD_MSG_RECEIVER_H
#define PD_MSG_RECEIVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/server.h>
#include <grpcpp/server_context.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "prefillAndDecodeCommunication.grpc.pb.h"

using DecodeRequestHandler = std::function<void(const prefillAndDecodeCommunication::DecodeParameters& request,
                                                prefillAndDecodeCommunication::DecodeRequestResponse& response)>;

using KVReleaseHandler = std::function<void(const std::string& requestID)>;

namespace mindie_llm {
class DecodeRequestReceiver : public prefillAndDecodeCommunication::DecodeService::Service {
   public:
    explicit DecodeRequestReceiver(std::string localAddr) : localAddr_(localAddr) {}

    grpc::Status DecodeRequestChannel(grpc::ServerContext* context,
                                      const prefillAndDecodeCommunication::DecodeParameters* request,
                                      prefillAndDecodeCommunication::DecodeRequestResponse* response) override;

    bool RegisterMsgHandler(DecodeRequestHandler callback);

   private:
    bool isValidRequest(const prefillAndDecodeCommunication::DecodeParameters* request,
                        prefillAndDecodeCommunication::DecodeRequestResponse* response, std::string& errMsg);

    std::string localAddr_;

    DecodeRequestHandler decodeRequestHandler_{nullptr};
};

class KvReleaseReceiver : public prefillAndDecodeCommunication::PrefillService::Service {
   public:
    explicit KvReleaseReceiver(std::string localAddr) : localAddr_(localAddr) {}

    grpc::Status ReleaseKVCacheChannel(grpc::ServerContext* context,
                                       const prefillAndDecodeCommunication::RequestId* request,
                                       google::protobuf::Empty* response) override;

    bool RegisterMsgHandler(KVReleaseHandler callback);

   private:
    bool isValidRequest(const prefillAndDecodeCommunication::RequestId* request);

    std::string localAddr_;

    KVReleaseHandler kvReleaseHandler_{nullptr};
};

}  // namespace mindie_llm

#endif  // PD_MSG_RECEIVER_H
