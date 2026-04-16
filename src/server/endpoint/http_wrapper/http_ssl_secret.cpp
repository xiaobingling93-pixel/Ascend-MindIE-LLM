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
#include "http_ssl_secret.h"

#include "common_util.h"
#include "config_manager_impl.h"
#include "log.h"

using namespace mindie_llm;

static const int MASTER_KEY_CHECK_AHEAD_TIME = 30;
static const int MASTER_KEY_CHECK_PERIOD = 7 * 24;

void HttpSslSecret::Start() {
    boost::unique_lock<boost::mutex> guard(mMutex);
    mCheckExpiredRunning = true;
    mCheckExpiredThread = std::thread([this]() { CheckKeyExpiredTask(); });
}

void HttpSslSecret::Stop() {
    {
        boost::unique_lock<boost::mutex> guard(mMutex);
        mCheckExpiredRunning = false;
        mCond.notify_one();
    }
    if (mCheckExpiredThread.joinable()) {
        mCheckExpiredThread.join();
    }
}

void HttpSslSecret::CheckKeyExpiredTask() {
    while (true) {
        {
            boost::unique_lock<boost::mutex> lockGuard{mMutex};
            if (!mCheckExpiredRunning) {
                return;
            }
            // check every week
            mCond.wait_until(lockGuard,
                             boost::chrono::steady_clock::now() + boost::chrono::hours(MASTER_KEY_CHECK_PERIOD));
            if (!mCheckExpiredRunning) {
                return;
            }
        }
    }
}
