/*
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

#ifndef BUFFERED_RESPONSER_H
#define BUFFERED_RESPONSER_H
#include <chrono>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>

#include "concurrent_map.h"
#include "request_response/request_id.h"
#include "response_buffer.h"

namespace mindie_llm {
using ForwardRespToManagerCall = std::function<void(ResponseSPtr response)>;
using ResponseBufferPtr = std::shared_ptr<ResponseBuffer>;

struct BufferResponseConfig {
    bool bufferResponseEnabled = false;
    uint32_t prefillExpectedTime;
    uint32_t decodeExpectedTime;
};

class BufferedResponser {
   public:
    explicit BufferedResponser(ForwardRespToManagerCall cb, BufferResponseConfig &config);

    ~BufferedResponser();

    void RespBufferThread();

    void TryRespond(const ResponseSPtr &response);

    void RecordArriveTime(RequestIdNew inferReqId,
                          std::chrono::time_point<std::chrono::high_resolution_clock> arriveTime);

   private:
    std::thread respBufferThread_;
    ConcurrentMap<std::string, ResponseBufferPtr> respBufferMap_;  // request id to response buffer
    double sloBufferRatio = 0.95;
    const uint32_t changeNsToMs = 1000000;
    ForwardRespToManagerCall forwardRespToManagerCall_;  // from llm manager
    BufferResponseConfig bufferResponseConfig_;
    std::atomic<bool> stop_ = {false};

    void SendEndResponse(ResponseBufferPtr &responseBuffer);

    void MaybeSendContinueResponse(ResponseBufferPtr &responseBuffer, double prefillExpectedTime,
                                   double decodeExpectedTime);
};
}  // namespace mindie_llm

#endif  // BUFFERED_RESPONSER_H
