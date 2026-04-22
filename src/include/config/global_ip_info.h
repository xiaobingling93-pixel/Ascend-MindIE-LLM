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

#pragma once
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "request_response/callback.h"
#include "utils/common_util.h"

namespace mindie_llm {

struct DeviceInfo {
    std::string deviceIp;
    int64_t devicePhysicalId;
    int64_t superDeviceId = -1;  // default value -1 means not set
};

inline std::ostream &operator<<(std::ostream &os, const DeviceInfo &info) {
    os << "DeviceInfo{" << "deviceIp: \"" << info.deviceIp << "\", "
       << "devicePhysicalId: " << info.devicePhysicalId << ", "
       << "superDeviceId: ";
    if (info.superDeviceId == -1) {
        os << "null";
    } else {
        os << info.superDeviceId;
    }
    os << "}";
    return os;
}
struct GlobalIpInfo {
    std::string role = "none";
    bool needInit = false;
    bool needSwitch = false;
    uint32_t localInstanceId;  // localInstanceId
    uint32_t instanceIdxInPod;
    uint32_t numInstancesPerPod;
    uint32_t flexPrefillPercentage{0};  // flex node prefill percentage
    bool isSingleContainer = false;
    // std::string localHostIp; // localHostIp
    std::vector<std::string> localHostIpList;  // localHostIp
    std::string localSuperPodId;
    std::vector<uint64_t> localDpInstanceIds;

    std::vector<std::string> localDeviceIps;          // <localDeviceIp1, localDeviceIp2, ....>
    std::vector<std::string> localDeviceLogicalIds;   // <localDeviceLogicalId1, localDeviceLogicalId2, ....>
    std::vector<std::string> localDevicePhysicalIds;  // <localDevicePhysicalId1, localDevicePhysicalId2, ....>
    std::vector<std::string> localSuperDeviceIds;     // <localSuperDeviceId1, localSuperDeviceId2, ....>
    std::vector<std::string> localDeviceRankIds;      // <localDevicePhysicalId1, localDevicePhysicalId2, ....>
    // ToDo: use unordered_map instead
    std::map<uint64_t, std::string> superPodIdInfo;           // key ---> dpInstanceId, value ---> superPodId
    std::map<uint64_t, int64_t> failLinkInstanceIDAndReason;  // key ---> dpInstanceId, value ---> failReason
    std::map<uint64_t, int64_t> spInfo;                       // key ---> dpInstanceId, value ---> sp_size
    std::map<uint64_t, int64_t> cpInfo;                       // key ---> dpInstanceId, value ---> cp_size

    // key ---> dpInstanceId, value ---> [device_ip.device_id, ...]
    std::map<uint64_t, std::vector<DeviceInfo>> linkIpInfo;
    std::map<uint64_t, std::vector<DeviceInfo>> unlinkIpInfo;
    // in each map, key --> dpInstanceId, value --> <deviceIp1, physical_device_id1, deviceIp2, ...>
    std::map<uint64_t, std::vector<std::string>> hostIpInfo;  // key ---> dpInstanceId, value ---> [hostIp1,...]
    std::map<uint64_t, std::vector<std::string>> unlinkHostIpInfo;
    void ClearIpInfo() {
        linkIpInfo.clear();
        unlinkIpInfo.clear();
    }

    void ResetRetryState() {
        needSwitch = false;
        unlinkIpInfo.clear();
    }

    std::string ToString() const {
        std::ostringstream oss;

        oss << "role: " << role << "\n"
            << "needInit: " << std::boolalpha << needInit << "\n"
            << "needSwitch: " << std::boolalpha << needSwitch << "\n"
            << "localInstanceId: " << localInstanceId << "\n"
            << "localDpInstanceIds: " << VectorToString(localDpInstanceIds) << "\n"
            << "localHostIp: " << VectorToString(localHostIpList) << "\n"
            << "localDeviceIps: " << VectorToString(localDeviceIps) << "\n"
            << "localDeviceLogicalIds: " << VectorToString(localDeviceLogicalIds) << "\n"
            << "localDevicePhysicalIds: " << VectorToString(localDevicePhysicalIds) << "\n"
            << "localDeviceRankIds: " << VectorToString(localDeviceRankIds) << "\n"
            << "hostIpInfo: " << MapToString(hostIpInfo) << "\n"
            << "failLinkInstanceIDAndReason: " << MapToString(failLinkInstanceIDAndReason) << "\n"
            << "spSize: " << MapToString(spInfo) << "\n"
            << "cpSize: " << MapToString(cpInfo) << "\n"
            << "linkIpInfo: " << MapToString(linkIpInfo) << "\n"
            << "unlinkIpInfo: " << MapToString(unlinkIpInfo) << "\n";

        return oss.str();
    }
};
}  // namespace mindie_llm
