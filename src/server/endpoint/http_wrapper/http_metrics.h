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

#ifndef ENDPOINT_HTTP_METRICS_H
#define ENDPOINT_HTTP_METRICS_H

#include <cstdint>
#include <cstdlib>
#include <queue>
#include <vector>

#include "single_req_infer_interface_base.h"

namespace mindie_llm {
class HttpMetrics {
   public:
    HttpMetrics(const HttpMetrics &) = delete;
    ~HttpMetrics() = default;

    static HttpMetrics &GetInstance();
    HttpMetrics &operator=(const HttpMetrics &) = delete;

    void CollectStatisticsRequest(const std::shared_ptr<SingleReqInferInterfaceBase> &inferRequest);
    void TTFTAdd(const std::shared_ptr<SingleReqInferInterfaceBase> &inferRequest);
    void TBTAdd(const std::shared_ptr<SingleReqInferInterfaceBase> &inferRequest);

    size_t TTFTSize() noexcept;
    size_t TBTSize() noexcept;

    size_t DynamicAverageTTFT();
    size_t DynamicAverageTBT();

   private:
    explicit HttpMetrics();
    std::queue<size_t> TTFTQueue_{};
    std::queue<size_t> TBTQueue_{};
    uint64_t ttftSum_ = 0;
    uint64_t tbtSum_ = 0;
    size_t dynamicAverageWindowSize = 1000;

    std::mutex TTFTMutex;
    std::mutex TBTMutex;
};
}  // namespace mindie_llm

#endif  // ENDPOINT_HTTP_METRICS_H
