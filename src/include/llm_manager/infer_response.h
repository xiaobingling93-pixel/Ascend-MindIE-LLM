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

#ifndef LLM_MANAGER_UTILS_INFER_RESPONSE_H
#define LLM_MANAGER_UTILS_INFER_RESPONSE_H
#include "infer_request_id.h"
#include "infer_tensor.h"
#include "status.h"

namespace mindie_llm {
// InferResponse GetFlags接口获取到的标志定义
enum class InferResponseEndFlagEnum {
    // 请求继续迭代执行
    INFER_RESPONSE_CONTINUE = 0,
    // 请求正常结束
    INFER_RESPONSE_EOS = 1,
    // 请求被主动CANCEL或STOP，用户不感知，丢弃响应
    INFER_RESPONSE_CANCEL = 2,
    // 请求执行中出错，响应输出为空，err_msg非空
    INFER_RESPONSE_EXEC_ERROR = 3,
    // 请求输入校验异常，响应输出为空，err_msg非空
    INFER_RESPONSE_ILLEGAL_INPUT = 4,
    // 请求因达到最大序列长度而结束，响应为最后一轮迭代输出
    INFER_RESPONSE_REACH_MAX_SEQ_LEN = 5,
    // 请求因达到最大输出长度（包括请求和模型粒度）而结束，响应为最后一轮迭代输出
    INFER_RESPONSE_REACH_MAX_OUTPUT_LEN = 6,
    // 请求因pull kv失败而结束，响应为最后一轮迭代输出
    INFER_RESPONSE_PULL_KVCACHE_ERROR = 7,
};

enum class IbisSchedulerResponseTransferFlagEnum {
    // not transfer
    IBISSCHEDULER_NOT_TRANSFER = 0,
    // just publish
    IBISSCHEDULER_PUBLISH_COMPLETED = 1,
    // just pull
    IBISSCHEDULER_PULL_COMPELETED = 2,
    // recompute
    IBISSCHEDULER_RECOMPUTE_REQ = 3,

    IBISSCHEDULER_PUBLISH_WAITING = 4,
    IBISSCHEDULER_PUBLISH_FAILED = 5,
    IBISSCHEDULER_PULL_FAILED = 6
};

class InferResponse {
   public:
    explicit InferResponse(const InferRequestId &reqId);

    InferResponse() = default;

    virtual ~InferResponse() = default;

    virtual const InferRequestId &GetRequestId() const = 0;

    virtual bool IsEnd() const = 0;

    virtual uint32_t GetFlags() const = 0;

    virtual Status GetOutput(const std::string &name, std::shared_ptr<InferTensor> &tensor) const = 0;

    virtual uint32_t GetIterTimes() const = 0;

    virtual uint32_t GetTransferFlag() const = 0;
};
}  // namespace mindie_llm
#endif
