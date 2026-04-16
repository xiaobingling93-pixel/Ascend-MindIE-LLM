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

#include "grpc_tls_function_expansion.h"

#include <openssl/pem.h>
#include <openssl/x509.h>

#include <cstdio>
#include <memory>
#include <string>

#include "common_util.h"
#include "endpoint_def.h"
#include "log.h"

namespace mindie_llm {
bool GrpcTlsFunctionExpansion::CheckTlsOption(const std::vector<std::string>& caPath, const std::string& cert,
                                              const std::vector<std::string>& crlPath) {
    for (const auto& ca : caPath) {
        if (!CheckCert(ca)) {
            return false;
        }
    }
    if (!CheckCert(cert)) {
        return false;
    }
    for (const auto& crl : crlPath) {
        if (!CheckCrl(crl)) {
            return false;
        }
    }
    return true;
}

bool GrpcTlsFunctionExpansion::CheckCert(const std::string& realCertPath) {
    // 打开证书文件
    std::unique_ptr<FILE, FileDeleter> certFile(fopen(realCertPath.data(), "r"));
    if (certFile == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
                   "Failed to open cert file, path: " << realCertPath);
        return false;
    }
    // 导入证书
    std::unique_ptr<X509, X509CertDeleter> cert(PEM_read_X509(certFile.get(), nullptr, nullptr, nullptr));
    if (cert == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
                   "Certificate file is empty or invalid, path: " << realCertPath);
        return false;
    }

    // check x509 version
    if (X509_get_version(cert.get()) != X509_VERSION_3) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
                   "Failed to parse X509 certificate, path: " << realCertPath << ", format may be invalid");
        return false;
    }

    // Validity period of the proofreading certificate
    if (X509_cmp_current_time(X509_getm_notAfter(cert.get())) < 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
                   "Cert has expired! current time after cert time, path: " << realCertPath);
        return false;
    }

    if (X509_cmp_current_time(X509_getm_notBefore(cert.get())) > 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
                   "Cert has expired! current time before cert time.");
        return false;
    }

    return true;
}

bool GrpcTlsFunctionExpansion::CheckCrl(const std::string& realCrlPath) {
    // read crl file
    std::unique_ptr<FILE, FileDeleter> crlFile(fopen(realCrlPath.data(), "r"));
    if (crlFile == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
                   "CRL file is empty or invalid, path: " << realCrlPath);
        return false;
    }
    std::unique_ptr<X509_CRL, X509CrlDeleter> crl(PEM_read_X509_CRL(crlFile.get(), nullptr, nullptr, nullptr));
    if (crl == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
                   "Failed to parse X509 CRL, path: " << realCrlPath);
        return false;
    }

    // check crl time
    if (X509_cmp_current_time(X509_CRL_get0_nextUpdate(crl.get())) < 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, CHECK_ERROR),
                   "Crl has expired! current time after next update time, path: " << realCrlPath);
        return false;
    }
    return true;
}
}  // namespace mindie_llm
