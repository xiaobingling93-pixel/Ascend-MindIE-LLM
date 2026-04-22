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
#include "grpc_wrapper.h"

#include "grpc_handler.h"
#include "log.h"

namespace mindie_llm {
int32_t GrpcWrapper::Start() {
    if (started_) {
        return EP_OK;
    } else {
        if (!GrpcHandler::GetInstance().InitDmiBusiness()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, INIT_ERROR),
                       "Failed to register init dmi business");
            return EP_ERROR;
        }
        if (!GrpcHandler::GetInstance().InitGrpcService()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, INIT_ERROR),
                       "Failed to register init grpc service");
            return EP_ERROR;
        }
        started_.store(true);
        return EP_OK;
    }
}

void GrpcWrapper::Stop() {
    if (!started_) {
        return;
    }
    started_.store(false);
}
}  // namespace mindie_llm
