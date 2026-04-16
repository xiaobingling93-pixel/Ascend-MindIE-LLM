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

#include "base64_util.h"

#include <openssl/buffer.h>
#include <openssl/evp.h>

#include "endpoint_def.h"
#include "log.h"

namespace mindie_llm {
const uint32_t MAX_BUFFER_SIZE = 1024;

std::string Base64Util::Encode(const std::string &input) {
    try {
        BIO *b64 = nullptr;
        BIO *bio = nullptr;

        BUF_MEM *bptr;

        b64 = BIO_new(BIO_f_base64());
        if (b64 == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, ENCODE_DECODE_ERROR),
                       "Base64 encode failed.");
            return "";
        }
        bio = BIO_new(BIO_s_mem());
        if (bio == nullptr) {
            CleanBio(b64);
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, ENCODE_DECODE_ERROR),
                       "Base64 encode failed.");
            return "";
        }
        bio = BIO_push(b64, bio);
        long ret = BIO_write(bio, input.data(), input.size());
        if (ret < 0) {
            CleanBio(bio);
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, ENCODE_DECODE_ERROR),
                       "Base64 encode failed.");
            return "";
        }
        ret = BIO_flush(bio);
        if (ret <= 0) {
            CleanBio(bio);
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, ENCODE_DECODE_ERROR),
                       "Base64 encode failed.");
            return "";
        }
        ret = BIO_get_mem_ptr(bio, &bptr);
        if (ret <= 0) {
            CleanBio(bio);
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, ENCODE_DECODE_ERROR),
                       "Base64 encode failed.");
            return "";
        }

        std::string encoded(bptr->data, bptr->length);

        BIO_free_all(bio);

        return encoded;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, ENCODE_DECODE_ERROR),
                   "Base64 encode failed: " << input);
    }
    return "";
}

void Base64Util::CleanBio(BIO *bio) {
    if (bio != nullptr) {
        // 获取bio中的数据和长度
        char *data = nullptr;
        auto dataLen = BIO_get_mem_data(bio, &data);
        if (dataLen > 0) {
            OPENSSL_cleanse(data, dataLen);
        }
        BIO_free_all(bio);
    }
}
}  // namespace mindie_llm
