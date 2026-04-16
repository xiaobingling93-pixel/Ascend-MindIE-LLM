/*
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
#include "http_wrapper.h"

#include "endpoint_def.h"
#include "http_server.h"

namespace mindie_llm {

bool HttpWrapper::Start() {
    std::lock_guard<std::mutex> guard(mMutex);
    if (mStarted) {
        return true;
    }
    auto res = HttpServer::HttpServerInit();
    if (res == 0) {
        mStarted = true;
        return true;
    }
    return false;
}

void HttpWrapper::Stop() {
    std::lock_guard<std::mutex> guard(mMutex);
    if (!mStarted) {
        return;
    }

    HttpServer::HttpServerDeInit();
    mStarted = false;
}
}  // namespace mindie_llm
