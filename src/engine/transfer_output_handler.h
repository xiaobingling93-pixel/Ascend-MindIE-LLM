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

#ifndef MODEL_TRANSFER_OUTPUT_HANDLER_H
#define MODEL_TRANSFER_OUTPUT_HANDLER_H
#include <semaphore.h>

#include <thread>

#include "concurrent_deque.h"
#include "engine/illm_engine.h"

namespace mindie_llm {
class TransferOutputHandler {
   public:
    explicit TransferOutputHandler(ForwardRespToManagerCall cb, size_t localDPRank = 0);

    void Entry4Executor(PullKVResponseSPtr pullKvResponse);  // used by D instance

    ConcurrentDeque<RequestId> &GetPulledReqIds();

   private:
    ForwardRespToManagerCall forwardRespToManagerCall_;

    ConcurrentDeque<RequestId> kvPulledReqIds_;

    size_t localDPRank_{0};
};

}  // namespace mindie_llm
#endif
