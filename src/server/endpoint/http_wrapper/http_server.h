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

#ifndef OCK_ENDPOINT_HTTP_SERVER_H
#define OCK_ENDPOINT_HTTP_SERVER_H

#include <map>
#include <memory>

#include "http_ssl.h"
#include "httplib.h"

namespace mindie_llm {
class HttpServer {
   public:
    static uint32_t HttpServerInit();
    static uint32_t HttpServerDeInit();
};
}  // namespace mindie_llm

#endif  // OCK_ENDPOINT_HTTP_SERVER_H
