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

#include "simulate_request_executor.h"

#include "infer_instances.h"
#include "log.h"

namespace mindie_llm {

constexpr uint32_t SIMULATE_WAIT_TIME_SECONDS = 5;

SimulateRequestExecutor::SimulateRequestExecutor(ConstructTag, InferReqType reqType) : reqType_(reqType) {
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
              "SimulateRequestExecutor: Created with reqType=" << static_cast<int>(reqType));
}

std::shared_ptr<SimulateRequestExecutor> SimulateRequestExecutor::Create(InferReqType reqType) {
    return std::make_shared<SimulateRequestExecutor>(ConstructTag{}, reqType);
}

SimulateResult SimulateRequestExecutor::RunSimulateOnce() { return RunSimulateOnce(SIMULATE_WAIT_TIME_SECONDS); }

SimulateResult SimulateRequestExecutor::RunSimulateOnce(uint32_t waitTime) {
    isFinish_.store(false);
    {
        std::unique_lock<std::mutex> locker(mutex_);
        while (!responseQueue_.empty()) {
            responseQueue_.pop();
        }
    }

    RequestSPtr request = CreateSimulateRequest();
    if (!request) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "SimulateRequestExecutor: Failed to create simulate request");
        return {SimulateResult::Status::ERROR, "Failed to create simulate request"};
    }

    std::string requestId = request->requestId;
    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER,
               "SimulateRequestExecutor: Running. requestId=" << requestId << ", waitTime=" << waitTime << "s");

    SetSimulateCallback(request);

    Status status = GetInferInstance()->Process(request);
    if (!status.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "SimulateRequestExecutor: Failed to submit. requestId=" << requestId);
        return {SimulateResult::Status::ERROR, "Failed to submit simulate request"};
    }

    SimulateResult result = WaitForSimulateResult(requestId, waitTime);
    if (result.status == SimulateResult::Status::TIMEOUT) {
        OnSimulateTimeout(requestId);
    }

    return result;
}

RequestSPtr SimulateRequestExecutor::CreateSimulateRequest() {
    auto request = std::make_shared<Request>();

    // 使用正常方式生成 requestId，通过 isSimulateRequest 字段标识虚推
    request->requestId = GetNextInferRequestId();
    request->isSimulateRequest = true;
    request->input_ids = {1};
    request->input_token_num = 1;
    request->reqType = reqType_;
    request->priority = 0;
    request->maxOutputLen = 1;
    request->temperature = 1.0;
    request->topK = 1;
    request->topP = 1.0;
    request->doSample = false;
    request->ignoreEos = false;

    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "SimulateRequestExecutor: Created. requestId="
                                                << request->requestId
                                                << ", isSimulateRequest=true, reqType=" << static_cast<int>(reqType_));

    return request;
}

void SimulateRequestExecutor::SetSimulateCallback(RequestSPtr request) {
    // 使用 weak_ptr 安全捕获，避免悬空指针
    std::weak_ptr<SimulateRequestExecutor> weakSelf = shared_from_this();

    request->serverResponseCallback_ = [weakSelf](ResponseSPtr response) {
        auto self = weakSelf.lock();
        if (!self || response == nullptr || self->isFinish_.load()) {
            return;
        }

        // 跳过 PD 分离模式特有的中间状态
        if (response->inferStatusFlag == InferStatusType::RELEASE_KV_COMPLETE ||
            response->inferStatusFlag == InferStatusType::ILLEGAL_INPUT ||
            response->transferStatusFlag == TransferStatusType::RECOMPUTED_TRIGGERED) {
            return;
        }

        std::unique_lock<std::mutex> locker(self->mutex_);
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                  "SimulateRequestExecutor: Callback. requestId=" << response->reqId << ", isEos=" << response->isEos);

        self->responseQueue_.push(response);
        self->cv_.notify_one();
    };
}

SimulateResult SimulateRequestExecutor::WaitForSimulateResult(const std::string& requestId, uint32_t waitTime) {
    std::cv_status status = std::cv_status::no_timeout;
    auto lastTimePoint = std::chrono::steady_clock::now() + std::chrono::seconds(waitTime);
    ResponseSPtr response = nullptr;

    std::unique_lock<std::mutex> locker(mutex_);
    while (responseQueue_.empty() && status != std::cv_status::timeout) {
        status = cv_.wait_until(locker, lastTimePoint);
    }

    if (!responseQueue_.empty()) {
        response = responseQueue_.front();
        responseQueue_.pop();
    }
    isFinish_.store(true);
    locker.unlock();

    if (response == nullptr) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
                  "SimulateRequestExecutor: Timeout. requestId=" << requestId);
        return {SimulateResult::Status::TIMEOUT, "Engine callback timeout"};
    }

    if (!HealthManager::GetHealth()) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
                  "SimulateRequestExecutor: Health status changed. requestId=" << requestId);
        return {SimulateResult::Status::ERROR, "Health status changed during health detector"};
    }

    if (!ParseTokensFromResponse(response)) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "SimulateRequestExecutor: Failed to parse tokens. requestId=" << requestId);
        return {SimulateResult::Status::ERROR, "Failed to get engine response"};
    }

    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "SimulateRequestExecutor: Success. requestId=" << requestId);
    return {SimulateResult::Status::SUCCESS, "healthy"};
}

bool SimulateRequestExecutor::ParseTokensFromResponse(const ResponseSPtr& response) {
    if (response->responseContents.empty()) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "SimulateRequestExecutor: Response contents is empty");
        return false;
    }

    for (const auto& content : response->responseContents) {
        if (content.seqId == 0 || content.outTokenIds.empty()) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                       GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                       "SimulateRequestExecutor: Invalid response content, seqId="
                           << content.seqId << ", outTokenIds.size=" << content.outTokenIds.size());
            return false;
        }
    }

    return true;
}

void SimulateRequestExecutor::OnSimulateTimeout(const std::string& requestId) {
    RequestIdNew reqId{requestId};
    Status stopResult = GetInferInstance()->ControlRequest(reqId, OperationV2::STOP);
    if (stopResult.StatusCode() != Error::Code::OK) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
                  "SimulateRequestExecutor: Failed to stop simulate inference. requestId=" << requestId);
    } else {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                  "SimulateRequestExecutor: Stopped simulate inference. requestId=" << requestId);
    }
}

}  // namespace mindie_llm
