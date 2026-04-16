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

#ifndef RESPONSE_BUFFER_H
#define RESPONSE_BUFFER_H
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "concurrent_deque.h"
#include "request_response/response.h"
#include "utils/data_type.h"

namespace mindie_llm {

// response buffer for the same request
class ResponseBuffer {
   public:
    explicit ResponseBuffer(InferReqType inferStage, std::int64_t lastRespArriveTime)
        : lastRespArriveTime_(lastRespArriveTime), inferStage_(inferStage) {}

    ~ResponseBuffer() = default;

    void AddResponse(const ResponseSPtr &response) { responseQueue4UniqueReq.PushBack(response); }

    void SetlastRespArriveTime(std::int64_t lastRespArriveTime) { lastRespArriveTime_ = lastRespArriveTime; }

    void SetInferStage(InferReqType inferStage) { inferStage_ = inferStage; }

    InferReqType GetInferStage() const { return inferStage_; }

    void SetReqEnded() { isEnded_ = true; }

    bool IsEnded() const { return isEnded_; }

    std::int64_t GetlastRespArriveTime() const { return lastRespArriveTime_; }

    bool IsEmpty() { return responseQueue4UniqueReq.Empty(); }

    ResponseSPtr PopFront() {
        ResponseSPtr result;
        if (responseQueue4UniqueReq.PopFront(result)) {
            return result;
        }
        return nullptr;
    }

   private:
    ConcurrentDeque<ResponseSPtr> responseQueue4UniqueReq;
    std::int64_t lastRespArriveTime_;
    InferReqType inferStage_;
    std::atomic<bool> isEnded_ = {false};
};
}  // namespace mindie_llm

#endif  // RESPONSE_BUFFER_H
