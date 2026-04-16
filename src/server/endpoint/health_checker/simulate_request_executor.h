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

#ifndef SIMULATE_REQUEST_EXECUTOR_H
#define SIMULATE_REQUEST_EXECUTOR_H

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "endpoint_def.h"
#include "request_response/request.h"
#include "simulate_task_runner.h"

namespace mindie_llm {

// @brief 虚推请求执行器，实现 ISimulateExecutor 接口
// @note 必须通过 Create() 工厂方法创建，确保回调生命周期安全
class SimulateRequestExecutor : public ISimulateExecutor, public std::enable_shared_from_this<SimulateRequestExecutor> {
   public:
    // @brief 构造标签，限制只能通过 Create() 工厂方法构造
    struct ConstructTag {
        explicit ConstructTag() = default;
    };

    ~SimulateRequestExecutor() override = default;

    SimulateRequestExecutor(const SimulateRequestExecutor&) = delete;
    SimulateRequestExecutor& operator=(const SimulateRequestExecutor&) = delete;

    // @brief 创建执行器实例（唯一创建方式）
    static std::shared_ptr<SimulateRequestExecutor> Create(InferReqType reqType = InferReqType::REQ_STAND_INFER);

    // @brief 供 make_shared 调用的构造函数，外部请勿直接使用
    explicit SimulateRequestExecutor(ConstructTag, InferReqType reqType);

    // @brief 执行一次虚推，使用默认超时（5秒）
    SimulateResult RunSimulateOnce() override;

    // @brief 执行一次虚推，支持自定义超时
    SimulateResult RunSimulateOnce(uint32_t waitTime);

    RequestSPtr CreateSimulateRequest();
    void SetSimulateCallback(RequestSPtr request);
    SimulateResult WaitForSimulateResult(const std::string& requestId, uint32_t waitTime);
    bool ParseTokensFromResponse(const ResponseSPtr& response);
    void OnSimulateTimeout(const std::string& requestId);

    InferReqType reqType_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<ResponseSPtr> responseQueue_;
    std::atomic<bool> isFinish_{false};
};

}  // namespace mindie_llm

#endif  // SIMULATE_REQUEST_EXECUTOR_H
