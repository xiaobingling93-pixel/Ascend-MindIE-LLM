/*
 * Copyright (c) 2024 Yuji Hirose. All rights reserved.
 * MIT License.
 *
 * Implement server supporting both http/https by switch based on httplib::Server from cpp-httplib.
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef MIES_HTTPS_SERVER_HELPER_H
#define MIES_HTTPS_SERVER_HELPER_H
#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "http_rest_resource.h"
#include "httplib.h"
#include "threadpool_monitor.h"

namespace mindie_llm {
class HttpsServerHelper : public httplib::Server {
   public:
    HttpsServerHelper(bool openSSL, const std::function<bool(SSL_CTX &sslCtx)> &setupSslCtxCallback,
                      uint32_t maxLinkNum)
        : openSSL(openSSL) {
        ctx_ = SSL_CTX_new(TLS_method());
        if (ctx_) {
            if (!setupSslCtxCallback(*ctx_)) {
                SSL_CTX_free(ctx_);
                ctx_ = nullptr;
            }
        }
        const int KEEP_ALIVE_TIMEOUT = 180;
        this->set_keep_alive_max_count(maxLinkNum);
        this->set_keep_alive_timeout(KEEP_ALIVE_TIMEOUT);
    };

    ~HttpsServerHelper() override;

    bool is_valid() const override;

    SSL_CTX *SslContext() const;

    void AddRequestToMonitor(std::shared_ptr<RequestContext> reqContextPtr) const;

    void RemoveMonitorRequest(const std::string &requestUuid) {
        if (threadpool_ != nullptr) {
            threadpool_->RemoveMonitorRequest(requestUuid);
        }
    }

    ThreadPoolMonitor *threadpool_{nullptr};

   private:
    SSL_CTX *ctx_;
    bool process_and_close_socket(socket_t sock) override;
    bool ProcessHttpSocket(socket_t sock);
    bool ProcessHttpsSocket(socket_t sock);
    void ShutdownAndCloseSocket(socket_t sock);
    std::mutex ctxMutex_;
    bool openSSL{false};
};
}  // namespace mindie_llm

#endif  // MIES_HTTPS_SERVER_HELPER_H
