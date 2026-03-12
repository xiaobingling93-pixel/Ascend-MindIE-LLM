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

#ifndef GRPC_CONTEXT_H
#define GRPC_CONTEXT_H

#include <atomic>
#include <vector>
#include <cstdint>
#include <string>
#include <sstream>
#include "data_type.h"
#include "endpoint_def.h"
#include "prefillAndDecodeCommunication.grpc.pb.h"
#include "common_util.h"

namespace mindie_llm {
struct KvCacheInfo {
    std::vector<std::vector<int64_t>> blockTable;
    std::vector<uint64_t> dpInstanceIds;

    // toString 方法使用 vectorToString
    std::string ToString() const
    {
        std::ostringstream oss;
        oss << "KvCacheInfo { blockTable: [";
        for (size_t i = 0; i < blockTable.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "m" << i << "=" << VectorToString(blockTable[i]);
        }
        oss << "], dpInstanceIds: " << VectorToString(dpInstanceIds) << " }";
        return oss.str();
    }
};

// dmi接口依赖信息
struct DmiServerInfo {
    std::string reqId;
    std::string pNodeAddr;
    std::string dtargetAddr;
    KvCacheInfo kvCacheInfo;
    InferReqType reqType;
    // 构造函数
    DmiServerInfo(const std::string& reqId,
                  const std::string& pNodeAddr,
                  const std::string& dtargetAddr,
                  KvCacheInfo kvCacheInfo,
                  InferReqType reqType)
        : reqId(reqId),
          pNodeAddr(pNodeAddr),
          dtargetAddr(dtargetAddr),
          kvCacheInfo(kvCacheInfo), // 使用拷贝构造函数
          reqType(reqType) {}
    // 拷贝构造函数
    DmiServerInfo(const DmiServerInfo &other)
        : reqId(other.reqId),
          pNodeAddr(other.pNodeAddr),
          dtargetAddr(other.dtargetAddr),
          kvCacheInfo(other.kvCacheInfo),
          reqType(other.reqType) {}
    DmiServerInfo& operator=(const DmiServerInfo&) = default;

    // ToString method
    std::string ToString() const
    {
        std::ostringstream oss;
        oss << "DmiServerInfo { "
            << "reqId: " << reqId << ", "
            << "pNodeAddr: " << pNodeAddr << ", "
            << "kvCacheInfo: " << kvCacheInfo.ToString() << ", "
            << "reqType: " << static_cast<int>(reqType) // Assuming reqType can be cast to int
            << " }";
        return oss.str();
    }
};

// 协议
struct TritonTextInfo {
    std::string userSepcId;
    TritonTextInfo() = default;
    // 构造函数
    TritonTextInfo(const std::string& userSepcId) : userSepcId(userSepcId) {}
    // 拷贝构造函数
    TritonTextInfo(const TritonTextInfo &other) : userSepcId(other.userSepcId) {}
    // 拷贝赋值操作符
    TritonTextInfo& operator=(const TritonTextInfo &other)
    {
        if (this != &other) {  // 防止自赋值
            this->userSepcId = other.userSepcId;
        }
        return *this;
    }
};
// 给grpc扩展接口预留的上下文信息
class GrpcContext {
public:
    GrpcContext() = default;
    ~GrpcContext() = default;

    explicit GrpcContext(const DmiServerInfo& info): serverInfo_(info) {}

    GrpcContext(const DmiServerInfo& info, const TritonTextInfo& tritonText): serverInfo_(info),
        protocalType_(MsgType::MSG_TYPE_TRITON), tritonTextInfo_(tritonText) {}

    const DmiServerInfo& GetDmiServerInfo() const
    {
        return serverInfo_;
    }

    void SetTritonTextInfo(const TritonTextInfo& info)
    {
        tritonTextInfo_ = info;
        protocalType_ = MsgType::MSG_TYPE_TRITON;
    }

    const TritonTextInfo& GetTritonTextInfo() const
    {
        return tritonTextInfo_;
    }

    MsgType GetProtocalTyep() const
    {
        return protocalType_;
    }

    void SetDecodeParams(const prefillAndDecodeCommunication::DecodeParameters& para)
    {
        para_ = para;
    }

    const prefillAndDecodeCommunication::DecodeParameters& GetDecodeParams() const
    {
        return para_;
    }

private:
    DmiServerInfo serverInfo_;
    MsgType protocalType_ = MsgType::MSG_TYPE_TRITON;
    TritonTextInfo tritonTextInfo_;
    prefillAndDecodeCommunication::DecodeParameters para_;
};
} // namespace mindie_llm

#endif // GRPC_CONTEXT_H