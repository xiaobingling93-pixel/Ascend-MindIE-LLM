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
#include "http_ssl.h"

#include <limits>

#include "common_util.h"
#include "config_info.h"
#include "config_manager_impl.h"
#include "file_utils.h"
#include "log.h"
#include "memory_utils.h"

#ifdef UT_ENABLED
#define LOCAL_API
#else
#define LOCAL_API static
#endif

using namespace mindie_llm;

static const int MASTER_KEY_CHECK_AHEAD_TIME = 30;
static const int MASTER_KEY_CHECK_PERIOD = 7 * 24;

#define SSL_LAYER_CHECK_RET(_condition, _msg)                                                                 \
    do {                                                                                                      \
        if (_condition) {                                                                                     \
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,                                                                \
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), _msg); \
            return EP_ERROR;                                                                                  \
        }                                                                                                     \
    } while (0)

static std::vector<spdlog::level::level_enum> g_SECEASY_LOG_LEVEL_ULOG{
    spdlog::level::trace, spdlog::level::level_enum::info, spdlog::level::level_enum::warn,
    spdlog::level::level_enum::err};

static spdlog::level::level_enum SeceasyLogToUlogLevel(uint32_t level) {
    if (level >= g_SECEASY_LOG_LEVEL_ULOG.size()) {
        return spdlog::level::level_enum::warn;
    }

    return g_SECEASY_LOG_LEVEL_ULOG[level];
}

namespace mindie_llm {

LOCAL_API bool GetInstallPath(std::string &configPath) noexcept {
    std::string linkedPath = "/proc/" + std::to_string(getpid()) + "/exe";
    std::string realPath;
    try {
        realPath.resize(PATH_MAX);
    } catch (const std::bad_alloc &e) {
        std::cout << "[%s] " << GetCurTime() << "Failed to alloc mem" << std::endl;
        return false;
    } catch (...) {
        std::cout << "[%s] " << GetCurTime() << "Failed to resize" << std::endl;
        return false;
    }
    auto size = readlink(linkedPath.c_str(), &realPath[0], realPath.size());
    if (size < 0 || size >= PATH_MAX) {
        return false;
    }
    realPath[size] = '\0';
    std::string path{realPath};

    std::string::size_type position = path.find_last_of('/');
    if (position == std::string::npos) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Get lib path failed : invalid folder path.");
        return false;
    }
    // get bin dir
    path = path.substr(0, position);

    position = path.find_last_of('/');
    if (position == std::string::npos) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Get lib path failed : invalid folder path.");
        return false;
    }
    // get install dir
    path = path.substr(0, position);
    path.append("/");

    configPath = path;
    return true;
}

static int32_t SetEnvForSecurity(std::string &workDir) {
    std::string path = workDir;
    path.append("/lib");
    if (!CanonicalPath(path)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Get lib path failed");
        return EP_ERROR;
    }
    if (::setenv("EP_OPENSSL_PATH", path.c_str(), 1) != 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Set ep openssl path failed. " << strerror(errno));
        return EP_ERROR;
    }
    return EP_OK;
}

EpCode HttpSsl::Start(SSL_CTX *sslCtx, SSLCertCategory certCategory = BUSINESS_CERT) {
    auto ret = EP_OK;
    ServerConfig serverConfig = GetServerConfig();
    sslCertCategory = certCategory;
    if (!serverConfig.httpsEnabled) {
        return EP_OK;
    }

    ret = InitWorkDir();
    if (ret != EP_OK) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Load init ssl workDir failed");
        return EP_ERROR;
    }

    ret = InitTlsPath(serverConfig);
    if (ret != EP_OK) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Load init ssl tls paths failed");
        return EP_ERROR;
    }
    ret = InitSSL(sslCtx);
    if (ret != EP_OK) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Load init ssl failed");
        return EP_ERROR;
    }

    return EP_OK;
}

EpCode HttpSsl::InitWorkDir() {
    std::string workDirTemp;
    auto success = GetInstallPath(workDirTemp);
    if (!success) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Get install path failed");
        return EP_ERROR;
    }

    if (SetEnvForSecurity(workDirTemp) != 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Set env for security failed");
        return EP_ERROR;
    }
    workDir = workDirTemp;
    return EP_OK;
}

EpCode HttpSsl::InitTlsPath(ServerConfig &serverConfig) {
    switch (sslCertCategory) {
        case MANAGEMENT_CERT:
            tlsCaPath = serverConfig.tlsCaPath;
            tlsCaFile = serverConfig.managementTlsCaFile;
            tlsCrlPath = serverConfig.managementTlsCrlPath;
            tlsCrlFile = serverConfig.managementTlsCrlFiles;
            tlsCert = serverConfig.managementTlsCert;
            tlsPk = serverConfig.managementTlsPk;
            break;
        case BUSINESS_CERT:
            tlsCaPath = serverConfig.tlsCaPath;
            tlsCaFile = serverConfig.tlsCaFile;
            tlsCrlPath = serverConfig.tlsCrlPath;
            tlsCrlFile = serverConfig.tlsCrlFiles;
            tlsCert = serverConfig.tlsCert;
            tlsPk = serverConfig.tlsPk;
            break;
        case METRICS_CERT:
            tlsCaPath = serverConfig.tlsCaPath;
            tlsCaFile = serverConfig.metricsTlsCaFile;
            tlsCert = serverConfig.metricsTlsCert;
            tlsCrlPath = serverConfig.metricsTlsCrlPath;
            tlsCrlFile = serverConfig.metricsTlsCrlFiles;
            tlsPk = serverConfig.metricsTlsPk;
            break;
        default:
            SSL_LAYER_CHECK_RET(true, "Failed match tlsCategory");
    }
    return EP_OK;
}

EpCode HttpSsl::InitSSL(SSL_CTX *sslCtx) {
    /* SSL_library_init() */
    auto ret = OPENSSL_init_ssl(0, nullptr);
    SSL_LAYER_CHECK_RET((ret <= 0), "Failed to init openssl");

    /* SSL_load_error_strings() */
    ret = OPENSSL_init_ssl(OPENSSL_INIT_LOAD_SSL_STRINGS | OPENSSL_INIT_LOAD_CRYPTO_STRINGS, nullptr);
    SSL_LAYER_CHECK_RET((ret <= 0), "Failed to load error strings");

    SSL_CTX_set_session_cache_mode(sslCtx, SSL_SESS_CACHE_SERVER);
    const std::string sidCtx = "mindie_tls1.3_server";
    ret = SSL_CTX_set_session_id_context(
        sslCtx, static_cast<const unsigned char *>(static_cast<const void *>(sidCtx.c_str())), sidCtx.size());
    SSL_LAYER_CHECK_RET((ret <= 0), "Failed to set session_id_context");

    SSL_CTX_ctrl(sslCtx, SSL_CTRL_SET_MIN_PROTO_VERSION, TLS1_3_VERSION, nullptr);

    ret = SSL_CTX_set_ciphersuites(sslCtx,
                                   "TLS_AES_128_GCM_SHA256:"
                                   "TLS_AES_256_GCM_SHA384:"
                                   "TLS_CHACHA20_POLY1305_SHA256:"
                                   "TLS_AES_128_CCM_SHA256");
    SSL_LAYER_CHECK_RET(ret <= 0, "Failed to set cipher suites to TLS context");

    ret = LoadCaCert(sslCtx);
    SSL_LAYER_CHECK_RET(ret != EP_OK, "Failed to load ca cert");

    ret = LoadServerCert(sslCtx);
    SSL_LAYER_CHECK_RET(ret != EP_OK, "Failed to load server cert");

    ret = LoadPrivateKey(sslCtx);
    SSL_LAYER_CHECK_RET(ret != EP_OK, "Failed to load private key");
    return EP_OK;
}

EpCode HttpSsl::LoadCaFileList(std::vector<std::string> &caFileList) {
    std::string path = workDir;
    path.append(tlsCaPath);
    caFileList.clear();
    for (auto &file : tlsCaFile) {
        auto tmpPath = path + file;
        if (!CanonicalPath(tmpPath)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Failed to check ca path with ca file " << file);
            return EP_ERROR;
        }
        caFileList.emplace_back(tmpPath);
    }
    return EP_OK;
}

EpCode HttpSsl::LoadCaCert(SSL_CTX *sslCtx) {
    // 设置校验函数
    SSL_CTX_set_verify(sslCtx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
    if (!tlsCrlPath.empty() && !tlsCrlFile.empty()) {
        crlFullPath = "";
        std::string crlDirPath = workDir + tlsCrlPath;
        bool isFirstFile = true;
        for (auto &file : tlsCrlFile) {
            std::string tmpPath = crlDirPath + file;
            if (!isFirstFile) {
                crlFullPath += ",";
            } else {
                isFirstFile = false;
            }
            crlFullPath += tmpPath;
        }
        auto crlStr = const_cast<char *>(crlFullPath.c_str());
        SSL_CTX_set_cert_verify_callback(sslCtx, HttpSsl::CaVerifyCallback, reinterpret_cast<void *>(crlStr));
    }

    std::vector<std::string> caFileList;
    SSL_LAYER_CHECK_RET(LoadCaFileList(caFileList) != EP_OK, "Failed to load ca file list");

    for (auto &caFile : caFileList) {
        auto ret = SSL_CTX_load_verify_locations(sslCtx, caFile.c_str(), nullptr);
        SSL_LAYER_CHECK_RET(ret <= 0, "TLS load verify file failed");
    }

    return EP_OK;
}

EpCode HttpSsl::LoadServerCert(SSL_CTX *sslCtx) {
    auto tmpPath = workDir + tlsCert;
    SSL_LAYER_CHECK_RET(!CanonicalPath(tmpPath), "Get invalid cert path");

    /* load cert */
    auto ret = SSL_CTX_use_certificate_file(sslCtx, tmpPath.c_str(), SSL_FILETYPE_PEM);
    SSL_LAYER_CHECK_RET(ret <= 0, "TLS use certification file failed");

    X509 *cert = SSL_CTX_get0_certificate(sslCtx);
    return CertVerify(cert);
}

EpCode HttpSsl::LoadPrivateKey(SSL_CTX *sslCtx) {
    auto tmpPath = workDir + tlsPk;
    if (!CanonicalPath(tmpPath)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Failed to get private key path");
        return EP_ERROR;
    }

    int ret = 0;
    /* load private key */
    ret = SSL_CTX_use_PrivateKey_file(sslCtx, tmpPath.c_str(), SSL_FILETYPE_PEM);
    if (ret <= 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Failed to set use private key file");
        return EP_ERROR;
    }

    /* check private key */
    ret = SSL_CTX_check_private_key(sslCtx);
    if (ret <= 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Failed to set use private key file");
        return EP_ERROR;
    }
    return EP_OK;
}

static X509_CRL *LoadCertRevokeListFile(const char *crlFile) {
    // check whether file is exist
    char *realCrlPath = realpath(crlFile, nullptr);
    if (realCrlPath == nullptr) {
        return nullptr;
    }

    // load crl file
    BIO *in = BIO_new(BIO_s_file());
    if (in == nullptr) {
        free(realCrlPath);
        realCrlPath = nullptr;
        return nullptr;
    }

    int result = BIO_ctrl(in, BIO_C_SET_FILENAME, BIO_CLOSE | BIO_FP_READ, const_cast<char *>(realCrlPath));
    if (result <= 0) {
        (void)BIO_free(in);
        free(realCrlPath);
        realCrlPath = nullptr;
        return nullptr;
    }

    X509_CRL *crl = PEM_read_bio_X509_CRL(in, nullptr, nullptr, nullptr);
    if (crl == nullptr) {
        (void)BIO_free(in);
        free(realCrlPath);
        realCrlPath = nullptr;
        return nullptr;
    }

    (void)BIO_free(in);
    free(realCrlPath);
    realCrlPath = nullptr;

    return crl;
}

int HttpSsl::CaVerifyCallback(X509_STORE_CTX *x509ctx, void *arg) {
    if (x509ctx == nullptr || arg == nullptr) {
        return 0;
    }

    const char *crlPath = reinterpret_cast<const char *>(arg);
    std::vector<std::string> paths;
    if (crlPath != nullptr) {
        std::string crlListStr(crlPath);
        std::stringstream crlStream(crlListStr);
        std::string item;
        while (std::getline(crlStream, item, ',')) {
            if (!item.empty()) {
                paths.push_back(item);
            }
        }
    }
    const int checkSuccess = 1;
    const int checkFailed = -1;

    if (!paths.empty()) {
        X509_STORE *x509Store = X509_STORE_CTX_get0_store(x509ctx);
        if (x509Store == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Failed to get cert in store");
            return checkFailed;
        }
        unsigned long flags = X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL;
        X509_STORE_CTX_set_flags(x509ctx, flags);
        for (auto singleCrlPath : paths) {
            X509_CRL *crl = LoadCertRevokeListFile(singleCrlPath.c_str());
            if (crl == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                           GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                           "Failed to load cert revocation list");
                return checkFailed;
            }
            if (X509_cmp_current_time(X509_CRL_get0_nextUpdate(crl)) <= 0) {
                ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                          GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_WARNING),
                          "The crl [cert revocation list] has expired! current time after next update time.");
            }
            auto result = X509_STORE_add_crl(x509Store, crl);
            if (result != 1U) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                           GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                           "Store add crl failed. Status code is " << result);
                X509_CRL_free(crl);
                return checkFailed;
            }
            X509_CRL_free(crl);
        }
    }

    auto verifyResult = X509_verify_cert(x509ctx);
    if (verifyResult != 1U) {
        ULOG_ERROR(
            SUBMODLE_NAME_ENDPOINT,
            GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
            "Verify failed in callback. Error: " << X509_verify_cert_error_string(X509_STORE_CTX_get_error(x509ctx)));
        return checkFailed;
    }

    return checkSuccess;
}

EpCode HttpSsl::CertVerify(X509 *cert) {
    if (cert == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), "Get cert failed");
        return EP_ERROR;
    }

    // Validity period of the proofreading certificate
    if (X509_cmp_current_time(X509_getm_notAfter(cert)) < 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Certificate has expired! current time after cert time.");
        return EP_ERROR;
    }
    if (X509_cmp_current_time(X509_getm_notBefore(cert)) > 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Certificate has expired! current time before cert time.");
        return EP_ERROR;
    }

    // The length of the private key of the verification certificate
    EVP_PKEY *pkey = X509_get_pubkey(cert);
    int keyLength = EVP_PKEY_get_bits(pkey);
    if (keyLength < MIN_PRIVATE_KEY_CONTENT_BIT_LEN) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Certificate key length is too short, key length < " << MIN_PRIVATE_KEY_CONTENT_BIT_LEN);
        return EP_ERROR;
    }
    EVP_PKEY_free(pkey);

    return EP_OK;
}
}  // namespace mindie_llm
