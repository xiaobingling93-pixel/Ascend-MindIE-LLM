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

#include "http_metrics.h"

#include "log.h"

namespace mindie_llm {

HttpMetrics &HttpMetrics::GetInstance() {
    static HttpMetrics instance;
    return instance;
}

HttpMetrics::HttpMetrics() {
    const std::string dynamicAverageWindowSizeEnv = EnvUtil::GetInstance().Get("DYNAMIC_AVERAGE_WINDOW_SIZE");
    size_t dynamicAverageWindowSizeEnvInt = 0u;
    if (!dynamicAverageWindowSizeEnv.empty()) {
        try {
            int dynamicAverageWindowSizeEnvToInt = std::stoi(dynamicAverageWindowSizeEnv);
            if (dynamicAverageWindowSizeEnvToInt < 0) {
                ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                          GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_WARNING),
                          "Invalid DYNAMIC_AVERAGE_WINDOW_SIZE, use default value: 1000");
                return;
            }

            dynamicAverageWindowSizeEnvInt = static_cast<size_t>(dynamicAverageWindowSizeEnvToInt);
            if (dynamicAverageWindowSizeEnvInt == 0) {
                ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                          GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_WARNING),
                          "Invalid DYNAMIC_AVERAGE_WINDOW_SIZE, use default value: 1000");
            } else {
                dynamicAverageWindowSize = dynamicAverageWindowSizeEnvInt;
            }
        } catch (const std::exception &e) {
            ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                      GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_WARNING),
                      "Invalid DYNAMIC_AVERAGE_WINDOW_SIZE, use default value: 1000");
        }
    }
}

void HttpMetrics::CollectStatisticsRequest(const std::shared_ptr<SingleReqInferInterfaceBase> &inferRequest) {
    TTFTAdd(inferRequest);
    TBTAdd(inferRequest);
}

void HttpMetrics::TTFTAdd(const std::shared_ptr<SingleReqInferInterfaceBase> &inferRequest) {
    if (inferRequest == nullptr) {
        return;
    }

    auto currReqTTFTCost = inferRequest->GetMetrics().firstTokenCost;

    // 使用 std::lock_guard 自动管理锁的生命周期
    std::lock_guard<std::mutex> lock(TTFTMutex);
    if (TTFTQueue_.size() >= dynamicAverageWindowSize) {
        ttftSum_ -= TTFTQueue_.front();
        TTFTQueue_.pop();
    }

    TTFTQueue_.push(currReqTTFTCost);
    ttftSum_ += currReqTTFTCost;
}

void HttpMetrics::TBTAdd(const std::shared_ptr<SingleReqInferInterfaceBase> &inferRequest) {
    if (inferRequest == nullptr) {
        return;
    }

    auto currReqTBTCostVec = inferRequest->GetMetrics().decodeTime;
    if (currReqTBTCostVec.empty()) {
        return;
    }
    uint64_t currReqTBTCostVecSum =
        std::accumulate(currReqTBTCostVec.begin(), currReqTBTCostVec.end(), static_cast<uint64_t>(0));
    size_t currReqTBTCostVecAverage = currReqTBTCostVecSum / currReqTBTCostVec.size();

    // 使用 std::lock_guard 自动管理锁的生命周期
    std::lock_guard<std::mutex> lock(TBTMutex);
    if (TBTQueue_.size() >= dynamicAverageWindowSize) {
        tbtSum_ -= TBTQueue_.front();
        TBTQueue_.pop();
    }

    TBTQueue_.push(currReqTBTCostVecAverage);
    tbtSum_ += currReqTBTCostVecAverage;
}

size_t HttpMetrics::TTFTSize() noexcept {
    std::lock_guard<std::mutex> lock(TTFTMutex);
    return TTFTQueue_.size();
}

size_t HttpMetrics::TBTSize() noexcept {
    std::lock_guard<std::mutex> lock(TBTMutex);
    return TBTQueue_.size();
}

size_t HttpMetrics::DynamicAverageTTFT() {
    std::lock_guard<std::mutex> lock(TTFTMutex);
    if (TTFTQueue_.empty()) {
        return 0;
    }
    return ttftSum_ / TTFTQueue_.size();
}

size_t HttpMetrics::DynamicAverageTBT() {
    std::lock_guard<std::mutex> lock(TBTMutex);
    if (TBTQueue_.empty()) {
        return 0;
    }
    return tbtSum_ / TBTQueue_.size();
}
}  // namespace mindie_llm
