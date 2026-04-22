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

#ifndef MIES_HTTP_SSL_SECRET_H
#define MIES_HTTP_SSL_SECRET_H

#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <condition_variable>
#include <thread>

#include "config_manager.h"
#include "endpoint_def.h"

namespace mindie_llm {
class HttpSslSecret {
   public:
    void Start();
    void Stop();

   private:
    void CheckKeyExpiredTask();

    boost::mutex mMutex;
    ServerConfig serverConfig{};
    bool mCheckExpiredRunning = false;
    boost::condition_variable mCond;
    std::thread mCheckExpiredThread;
};
}  // namespace mindie_llm

#endif  // MIES_HTTP_SSL_SECRET_H
