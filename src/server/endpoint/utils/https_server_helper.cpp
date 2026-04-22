/*
 * Copyright (c) 2024 Yuji Hirose. All rights reserved.
 * MIT License.
 *
 * Implement server supporting both http/https by switch based on httplib::Server from cpp-httplib
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

#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "https_server_helper.h"

#include "httplib.h"
#include "log.h"

using namespace mindie_llm;
inline HttpsServerHelper::~HttpsServerHelper() {
    // threadpool_会在httplib server中被销毁，不需要手动销毁
    if (ctx_) {
        SSL_CTX_free(ctx_);
        ctx_ = nullptr;
    }
}

inline bool HttpsServerHelper::is_valid() const { return ctx_; }

inline SSL_CTX *HttpsServerHelper::SslContext() const { return ctx_; }

void HttpsServerHelper::AddRequestToMonitor(std::shared_ptr<RequestContext> reqContextPtr) const {
    // add req to threadpool monitor
    if (this->threadpool_ != nullptr) {
        this->threadpool_->AddRequestToMonitor(reqContextPtr);
    }
}

bool HttpsServerHelper::ProcessHttpSocket(socket_t sock) {
    std::string remote_addr;
    int remote_port = 0;
    httplib::detail::get_remote_ip_and_port(sock, remote_addr, remote_port);

    std::string local_addr;
    int local_port = 0;
    httplib::detail::get_local_ip_and_port(sock, local_addr, local_port);

    auto ret = httplib::detail::process_server_socket(
        svr_sock_, sock, keep_alive_max_count_, keep_alive_timeout_sec_, read_timeout_sec_, read_timeout_usec_,
        write_timeout_sec_, write_timeout_usec_,
        [&](httplib::Stream &strm, bool close_connection, bool &connection_closed) {
            return process_request(strm, remote_addr, remote_port, local_addr, local_port, close_connection,
                                   connection_closed, nullptr);
        });

    ShutdownAndCloseSocket(sock);
    return ret;
}

bool HttpsServerHelper::process_and_close_socket(socket_t sock) {
    if (!openSSL) {  // http
        return ProcessHttpSocket(sock);
    } else {  // https
        return ProcessHttpsSocket(sock);
    }
}

inline void HttpsServerHelper::ShutdownAndCloseSocket(socket_t sock) {
    httplib::detail::shutdown_socket(sock);
    httplib::detail::close_socket(sock);
}

bool HttpsServerHelper::ProcessHttpsSocket(socket_t sock) {
    auto ssl = httplib::detail::ssl_new(
        sock, ctx_, ctxMutex_,
        [&](SSL *ssl2) {
            int ssl_error = 0;
            return httplib::detail::ssl_connect_or_accept_nonblocking(sock, ssl2, SSL_accept, read_timeout_sec_,
                                                                      read_timeout_usec_, &ssl_error);
        },
        [](SSL *) { return true; });
    if (!ssl) {
        ShutdownAndCloseSocket(sock);
        return false;
    }

    std::string remote_addr;
    int remote_port = 0;
    httplib::detail::get_remote_ip_and_port(sock, remote_addr, remote_port);

    std::string local_addr;
    int local_port = 0;
    httplib::detail::get_local_ip_and_port(sock, local_addr, local_port);

    auto ret = httplib::detail::process_server_socket_ssl(
        svr_sock_, ssl, sock, keep_alive_max_count_, keep_alive_timeout_sec_, read_timeout_sec_, read_timeout_usec_,
        write_timeout_sec_, write_timeout_usec_,
        [&](httplib::Stream &strm, bool close_connection, bool &connection_closed) {
            return process_request(strm, remote_addr, remote_port, local_addr, local_port, close_connection,
                                   connection_closed, [&](httplib::Request &req) { req.ssl = ssl; });
        });

    // Shutdown gracefully if the result seemed successful, non-gracefully if the connection appeared to be closed.
    const bool shutdown_gracefully = ret;
    httplib::detail::ssl_delete(ctxMutex_, ssl, sock, shutdown_gracefully);

    ShutdownAndCloseSocket(sock);
    return ret;
}
