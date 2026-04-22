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

#ifndef OCK_HTTP_SSL_H
#define OCK_HTTP_SSL_H

#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

#include <condition_variable>
#include <thread>

#include "config_manager.h"
#include "endpoint_def.h"

namespace mindie_llm {
enum SSLCertCategory : uint16_t { MANAGEMENT_CERT, BUSINESS_CERT, METRICS_CERT };

enum ServerGroupType : uint16_t { ALL_SAME, BUSINESS_MANAGEMENT_SAME, MANAGEMENT_METRICS_SAME, ALL_DIFFERENT };

class HttpSsl {
   public:
    EpCode Start(SSL_CTX *sslCtx, SSLCertCategory certCategory);

   private:
    EpCode InitWorkDir();
    EpCode InitTlsPath(ServerConfig &serverConfig);
    EpCode InitSSL(SSL_CTX *sslCtx);

    static int CaVerifyCallback(X509_STORE_CTX *x509ctx, void *arg);
    EpCode LoadCaFileList(std::vector<std::string> &caFileList);
    EpCode LoadCaCert(SSL_CTX *sslCtx);
    EpCode LoadServerCert(SSL_CTX *sslCtx);
    EpCode LoadPrivateKey(SSL_CTX *sslCtx);
    EpCode CertVerify(X509 *cert);

   private:
    std::string workDir;
    std::string crlFullPath;
    // 证书相关路径
    std::string tlsCaPath;
    std::set<std::string> tlsCaFile;
    std::string tlsCrlPath;
    std::set<std::string> tlsCrlFile;
    std::string tlsCert;
    std::string tlsPk;
    SSLCertCategory sslCertCategory;
};
}  // namespace mindie_llm

#endif  // OCK_HTTP_SSL_H
