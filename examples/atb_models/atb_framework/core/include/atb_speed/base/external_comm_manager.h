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

#ifndef ATB_SPEED_EXTERNAL_COMM_MANAGER_H
#define ATB_SPEED_EXTERNAL_COMM_MANAGER_H

#include <atb/types.h>
#include "hccl/hccl.h"
#include "atb/comm.h"
#include "atb_speed/log.h"

namespace atb_speed {

const std::string LCCL = "lccl";
const std::string HCCL = "hccl";
const std::string LCOC = "lcoc";

/// A cache object contains information of a communication group
class CommInfo {
public:
    ~CommInfo();

    uint64_t cacheId_ = 0;
    uint32_t subCommRankId_ = 0;
    std::vector<uint32_t> rankIds_ = {};
    std::string backend_ = "";
    HcclComm hcclComm_ = nullptr;
    uint32_t bufferSize_ = 0;
    uint32_t streamId_ = 0;
    bool enableReuse_ = true;

    std::string ToString() const;
};

/// A class manages all the communication group (including commDomain and hcclComm ptr)
class ExternalCommManager {
public:
    void Init(uint32_t worldSize, uint32_t subCommRankId,
        std::string backend, std::string rankTableFile, uint32_t streamId);

    void Init(uint32_t worldSize, uint32_t subCommRankId,
        std::string backend, std::string rankTableFile, uint32_t streamId, std::string lwdGlobalComm);

    void SetLcclCommDomainRange(int32_t lowerBound, int32_t upperBound);

    void Reset();

    std::string GetCommDomain(uint32_t groupId, const std::vector<uint32_t> &rankIds,
        uint32_t subCommRankId, std::string backend, uint32_t bufferSize, uint32_t streamId,
        bool enableReuse = true);

    HcclComm GetCommPtr(std::string commDomain);

    std::shared_ptr<CommInfo> GetCommInfo(std::string commDomain);

    bool IsInitialized();

    std::string PrintCommInfo();

    void ResumeHcclComm();

    uint32_t worldSize_ = 0;
    uint32_t rank_;
    std::string rankTableFile_ = "";
    std::string lwdGlobalComm_ = "";

private:
    std::string GetCommDomainFromCache(
        const std::vector<uint32_t> &rankIds, std::string backend, uint32_t bufferSize, uint32_t streamId);
    std::string GetSelfAssignedCommDomain(std::shared_ptr<CommInfo> &commInfo, uint32_t groupId);
    std::string GetHcclSubCommDomain(std::shared_ptr<CommInfo> &commInfo, uint32_t groupId);
    std::string GetHcclGlobalCommDomain(std::shared_ptr<CommInfo> &commInfo);

    std::map<std::string, std::shared_ptr<CommInfo>> commInfoCache_ = {};
    HcclComm globalComm_ = nullptr;
    uint32_t commDomainCounter_ = 0;
    uint32_t lcclCommDomainLowerBound_ = 0;
    uint32_t lcclCommDomainUpperBound_ = 0;
};

}  // namespace atb_speed

#endif  // ATB_SPEED_EXTERNAL_COMM_MANAGER_H