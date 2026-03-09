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
#include "dmi_role.h"
#include <unordered_map>

#include "parse_protocol.h"
#include "grpc_communication_mng.h"
#include "log.h"
#include "safe_io.h"

using namespace std;
using OrderedJson = nlohmann::ordered_json;
constexpr uint32_t MAX_INSTANCE_PER_POD = 64; // max instances per pod
constexpr uint32_t MAX_P_PERCENTAGE = 100; // max p percentage is 100
constexpr uint32_t MIN_P_PERCENTAGE = 0; // min p percentage is 0

namespace mindie_llm {
static const std::unordered_map<int64_t, std::string> LINK_FAILED_MAP = {
    {203005, "Timeout"},
    {203004, "Engine error"},
};

DmiRole::DmiRole()
{}

DmiRole::~DmiRole()
{
    try {
        keepAlive.store(false); // 关闭保活响应
        retryTerminate_.store(true);
        taskQueue_.Push(nullptr);
        if (retryThread_.joinable()) {
            retryThread_.join();
        }
    } catch (...) {
        // safe destruct
    }
}
std::shared_ptr<DmiRole> DmiRole::GetInstance()
{
    static std::shared_ptr<DmiRole> dmiRoleInstance = std::make_shared<DmiRole>();
    return dmiRoleInstance;
}

void DmiRole::RunThread()
{
#ifndef UT_ENABLED
        retryThread_ = std::thread([this]() { this->RetryThread(); });
#endif
}

bool DmiRole::PDParseRequestBodyToJson(const ReqCtxPtr &reqCtx, ordered_json &body) const noexcept
{
    try {
        std::string msgBody = reqCtx->MsgBody();
        if (!ordered_json::accept(msgBody)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                JSON_PARSE_ERROR), "Convert string to json object failed, CallbackId is " << reqCtx->CallbackId());
            return false;
        }
        body = ordered_json::parse(msgBody, CheckOrderedJsonDepthCallback);
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Convert string to json object exception, CallbackId is " << reqCtx->CallbackId());
        return false;
    }
    return true;
}

bool DmiRole::UpdatePDInfo(const std::string &roleName, const std::string &preRole, const ordered_json &body,
    GlobalIpInfo &globalIpInfo)
{
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Previous role is " << preRole << ". role in request is " << roleName);
    bool res = true;
    if (preRole == "none") { // 初始化
        res = UpdatePDSwitchInfo(roleName, body, globalIpInfo, true);
    // flex 节点不初始化
    } else if (preRole != roleName && preRole != "flex" && roleName != "flex") {
        res = UpdatePDSwitchInfo(roleName, body, globalIpInfo, false);
    } else if (preRole == roleName) {
        res = UpdatePDNotSwitchInfo(roleName, body, globalIpInfo);
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Whether the node's role needs to be switched: "
        << globalIpInfo.needSwitch);
    return res;
}

bool DmiRole::UpdatePDInfoV2(const std::string &roleName, const std::string &preRole, const ordered_json &body,
    GlobalIpInfo &globalIpInfo)
{
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Previous role is : " << preRole << ". role in request is : " << roleName);
    bool res = true;
    if (preRole == "none") { // 初始化
        res = UpdatePDSwitchInfoV2(roleName, body, globalIpInfo, true);
    } else if (preRole != roleName) {
        res = UpdatePDSwitchInfoV2(roleName, body, globalIpInfo, false);
    } else if (preRole == roleName) {
        res = UpdatePDNotSwitchInfoV2(roleName, body, globalIpInfo);
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Whether the node's role needs to be switched: "
        << globalIpInfo.needSwitch);
    return res;
}

void DmiRole::ProcessInitInfo(const ordered_json &body, GlobalIpInfo &globalIpInfo)
{
    try {
        globalIpInfo.needInit = true;
        globalIpInfo.localInstanceId = body["local"]["id"];
        globalIpInfo.localHostIpList.emplace_back(body["local"]["host_ip"]);
        if (body["local"].contains("super_pod_id")) {
            globalIpInfo.localSuperPodId = body["local"]["super_pod_id"];
        }
        if (body["local"].contains("instance_idx_in_pod")) {
            globalIpInfo.instanceIdxInPod = body["local"]["instance_idx_in_pod"];
        }
        if (body["local"].contains("num_instances_per_pod")) {
            globalIpInfo.numInstancesPerPod = body["local"]["num_instances_per_pod"];
        }
        if (body["local"].contains("is_single_container")) {
            globalIpInfo.isSingleContainer = body["local"]["is_single_container"];
        }
        globalIpInfo.hostIpInfo[globalIpInfo.localInstanceId] = globalIpInfo.localHostIpList;
        localInstanceId_ = globalIpInfo.localInstanceId;
        for (const auto &deviceInfo : body["local"]["device"]) {
            globalIpInfo.localDeviceIps.emplace_back(deviceInfo["device_ip"]);
            globalIpInfo.localDeviceLogicalIds.emplace_back(deviceInfo["device_logical_id"]);
            globalIpInfo.localDevicePhysicalIds.emplace_back(deviceInfo["device_id"]);
            if (deviceInfo.contains("super_device_id")) {
                globalIpInfo.localSuperDeviceIds.emplace_back(deviceInfo["super_device_id"]);
            }
        }
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Error occurred while processing dmi role init information: " << e.what());
        throw std::runtime_error("DmiRole::ProcessInitInfo" + std::string(e.what()));
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Error occurred while processing dmi role init information.");
        throw std::runtime_error("DmiRole::ProcessInitInfo Error");
    }
}

// update local ip info
void DmiRole::ProcessInitInfoV2(const ordered_json &body, GlobalIpInfo &globalIpInfo)
{
    try {
        // update local info
        globalIpInfo.needInit = true;
        auto firstNodeInfo = body["local"][0];
        auto dpInstanceId = firstNodeInfo["dp_inst_list"][0]["dp_inst_id"];
        globalIpInfo.numInstancesPerPod = MAX_INSTANCE_PER_POD;
        auto ret = ReverseDpInstId(dpInstanceId);
        globalIpInfo.localInstanceId = ret.first;
        localInstanceId_ = globalIpInfo.localInstanceId;
        if (firstNodeInfo.contains("super_pod_id")) {
            globalIpInfo.localSuperPodId = firstNodeInfo["super_pod_id"];
        }

        for (const auto &nodeInfo : body["local"]) {
            globalIpInfo.localHostIpList.emplace_back(nodeInfo["host_ip"]);
            for (const auto& dpGroupInfo : nodeInfo["dp_inst_list"]) {
                globalIpInfo.localDpInstanceIds.emplace_back(dpGroupInfo["dp_inst_id"]);
                for (const auto &deviceInfo : dpGroupInfo["device"]) {
                    globalIpInfo.localDeviceIps.emplace_back(deviceInfo["device_ip"]);
                    globalIpInfo.localDeviceLogicalIds.emplace_back(deviceInfo["device_logical_id"]);
                    globalIpInfo.localDevicePhysicalIds.emplace_back(deviceInfo["device_id"]);
                    globalIpInfo.localDeviceRankIds.emplace_back(deviceInfo["rank_id"]);
                    if (deviceInfo.contains("super_device_id")) {
                        globalIpInfo.localSuperDeviceIds.emplace_back(deviceInfo["super_device_id"]);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Error occurred : " << e.what());
        throw std::runtime_error("DmiRole::ProcessInitInfoV2" + std::string(e.what()));
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Unknown error occurred.");
        throw std::runtime_error("DmiRole::ProcessInitInfoV2 Unknown Error");
    }
}

bool DmiRole::UpdatePDSwitchInfo(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo,
    bool needInit)
{
    std::lock_guard<std::mutex> lock(mtx_);
    globalIpInfo.role = roleName;
    globalIpInfo.needSwitch = true;
    if (needInit) { // 初始化
        ProcessInitInfo(body, globalIpInfo);
         // 初始化的时候P、D或flex的p_percentage为0、100时，peers不允许为空
        if (body["peers"].size() == 0 && (roleName != "flex" ||
            globalIpInfo.flexPrefillPercentage == MIN_P_PERCENTAGE ||
            globalIpInfo.flexPrefillPercentage == MAX_P_PERCENTAGE)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                JSON_PARSE_ERROR), "[DmiRole::UpdatePDSwitchLinkInfo] Parse req json failed, while role is P, D, "
                "or flex's p_percentage is " << MIN_P_PERCENTAGE <<
                " or " << MAX_P_PERCENTAGE << ", peers can't be empty.");
            return false;
        }
        if (body["local"].contains("p_percentage")) {
            globalIpInfo.flexPrefillPercentage = body["local"]["p_percentage"];
            FlexPPercentageProcessor::GetInstance().SetPdRoleFlexPPercentage(globalIpInfo.flexPrefillPercentage);
        }
    }

    // 根据peers信息更新linkIpInfo, 更新链接信息时，根据当前链接信息，只更新新增的链接 和 需要断开的链接
    try {
        std::map<uint64_t, std::vector<DeviceInfo>> currentLinkIpInfo{};
        std::string superPodId;
        for (const auto &serverInfo : body["peers"]) {
            if (serverInfo.contains("super_pod_id")) {
                superPodId = serverInfo["super_pod_id"];
                globalIpInfo.superPodIdInfo[serverInfo["id"]] = serverInfo["super_pod_id"];
            }
            uint32_t instanceId = serverInfo["id"];
            globalIpInfo.localHostIpList.emplace_back(body["local"]["host_ip"]);
            globalIpInfo.hostIpInfo[instanceId] = {serverInfo["host_ip"]};
            instanceIdToServerIp_[instanceId] = serverInfo["server_ip"];
            std::vector<DeviceInfo> linkDeviceIp;
            for (const auto &deviceInfo : serverInfo["device"]) {
                DeviceInfo device;
                device.deviceIp = deviceInfo["device_ip"].get<std::string>();
                device.devicePhysicalId = std::stoi(deviceInfo["device_id"].get<std::string>());
                if (deviceInfo.contains("super_device_id")) {
                    device.superDeviceId = std::stoi(deviceInfo["super_device_id"].get<std::string>());
                }
                linkDeviceIp.emplace_back(device);
            }
            currentLinkIpInfo.insert({instanceId, linkDeviceIp});
        }
        UpdateIpInfo(currentLinkIpInfo, globalIpInfo, superPodId);
        return true;
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Error occurred. " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Unknown error occurred.");
        return false;
    }
}

bool DmiRole::UpdatePDSwitchInfoV2(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo,
    bool needInit)
{
    std::lock_guard<std::mutex> lock(mtx_);
    globalIpInfo.role = roleName;
    globalIpInfo.needSwitch = true;
    if (needInit) { // 初始化
        ProcessInitInfoV2(body, globalIpInfo);
    } else {  // PD 转换
        globalIpInfo.unlinkIpInfo = this->successLinkIP_;
    }
    try {
        for (const auto &nodeInfo : body["local"]) {
            for (const auto& dpGroupInfo : nodeInfo["dp_inst_list"]) {
                globalIpInfo.localDpInstanceIds.emplace_back(dpGroupInfo["dp_inst_id"]);
            }
        }
        for (const auto &peerInfo : body["peers"]) {
            for (const auto &nodeInfo : peerInfo) {
                auto ret = ReverseDpInstId(nodeInfo["dp_inst_list"][0]["dp_inst_id"]);
                uint32_t instanceId = ret.first;
                globalIpInfo.spInfo[instanceId] = nodeInfo.value("sp_size", 1);
                globalIpInfo.cpInfo[instanceId] = nodeInfo.value("cp_size", 1);
                instanceIdToServerIp_[instanceId] = nodeInfo["server_ip"];
                for (const auto &dpGroupInfo : nodeInfo["dp_inst_list"]) {
                    auto dpInstanceId = dpGroupInfo["dp_inst_id"];
                    if (globalIpInfo.hostIpInfo.count(dpInstanceId) != 0) {
                        globalIpInfo.hostIpInfo[dpInstanceId].emplace_back(nodeInfo["host_ip"]);
                    } else {
                        globalIpInfo.hostIpInfo[dpInstanceId] = {nodeInfo["host_ip"]};
                    }
                    if (nodeInfo.contains("super_pod_id")) {
                        globalIpInfo.superPodIdInfo[dpInstanceId] = nodeInfo["super_pod_id"];
                    }
                    remoteNodeLinkStatus_[dpInstanceId] = {"None", false};
                    std::vector<DeviceInfo> linkDeviceIp;
                    for (const auto &deviceInfo : dpGroupInfo["device"]) {
                        DeviceInfo device;
                        device.deviceIp = deviceInfo["device_ip"].get<std::string>();
                        device.devicePhysicalId = std::stoi(deviceInfo["device_id"].get<std::string>());
                        if (deviceInfo.contains("super_device_id")) {
                            device.superDeviceId = std::stoi(deviceInfo["super_device_id"].get<std::string>());
                        }
                        linkDeviceIp.emplace_back(device);
                    }
                    if (globalIpInfo.linkIpInfo.find(dpInstanceId) != globalIpInfo.linkIpInfo.end()) {
                        // 如果dpInstanceId已存在，则合并设备信息
                        auto& existingDevices = globalIpInfo.linkIpInfo[dpInstanceId];
                        existingDevices.insert(existingDevices.end(), linkDeviceIp.begin(), linkDeviceIp.end());
                    } else {
                        globalIpInfo.linkIpInfo.insert({dpInstanceId, linkDeviceIp});
                    }
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Error occurred : " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Unknown error occurred.");
        return false;
    }
}

bool DmiRole::UpdatePDNotSwitchInfo(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo)
{
    std::lock_guard<std::mutex> lock(mtx_);
    globalIpInfo.role = roleName;
    globalIpInfo.needSwitch = false;
    std::string superPodId = "";
    std::map<uint64_t, std::vector<DeviceInfo>> currentLinkIpInfo{};
    try {
        for (const auto &serverInfo : body["peers"]) {
            uint32_t instanceId = serverInfo["id"];
            globalIpInfo.localHostIpList.emplace_back(body["local"]["host_ip"]);
            globalIpInfo.hostIpInfo[instanceId] = {serverInfo["host_ip"]};
            instanceIdToServerIp_[instanceId] = serverInfo["server_ip"];
            std::vector<DeviceInfo> linkDeviceIp;
            for (const auto &deviceInfo : serverInfo["device"]) {
                DeviceInfo device;
                device.deviceIp = deviceInfo["device_ip"].get<std::string>();
                device.devicePhysicalId = std::stoi(deviceInfo["device_id"].get<std::string>());
                linkDeviceIp.emplace_back(device);
            }
            currentLinkIpInfo.insert({instanceId, linkDeviceIp});
        }
        if (body["local"].contains("p_percentage")) {
            globalIpInfo.flexPrefillPercentage = body["local"]["p_percentage"];
            FlexPPercentageProcessor::GetInstance().SetPdRoleFlexPPercentage(globalIpInfo.flexPrefillPercentage);
        }
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Error occurred. " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Unknown error occurred.");
        return false;
    }

    this->UpdateIpInfo(currentLinkIpInfo, globalIpInfo, superPodId);
    return true;
}

void DmiRole::UpdateIpInfo(std::map<uint64_t, std::vector<DeviceInfo>>& currentLinkIpInfo,
    GlobalIpInfo &globalIpInfo, std::string &superPodId)
{
    // this->successLinkIP_ - currentLinkIpInfo = unlink
    for (auto it = this->successLinkIP_.cbegin(); it != this->successLinkIP_.cend(); ++it) {
        auto key = it->first;
        if (currentLinkIpInfo.find(key) == currentLinkIpInfo.cend()) {
            globalIpInfo.unlinkIpInfo[key] = this->successLinkIP_[key];
            if (superPodId != "") {
                globalIpInfo.superPodIdInfo[key] = superPodId;
            }
        }
    }

    // currentLinkIpInfo - this->successLinkIP_ = link
    for (auto it = currentLinkIpInfo.cbegin(); it != currentLinkIpInfo.cend(); ++it) {
        auto key = it->first;
        if (this->successLinkIP_.find(key) == this->successLinkIP_.cend()) {
            globalIpInfo.linkIpInfo[key] = currentLinkIpInfo[key];
        }
    }
}

void DmiRole::UpdateHostIpInfo(
    std::map<uint64_t, std::vector<std::string>>& currentLinkHostIpInfo,
    GlobalIpInfo &globalIpInfo)
{
    // this->successHostIP_ - currentLinkIpInfo = unlink
    for (auto it = this->successHostIP_.cbegin(); it != this->successHostIP_.cend(); ++it) {
        auto key = it->first;
        if (currentLinkHostIpInfo.find(key) == currentLinkHostIpInfo.cend()) {
            globalIpInfo.unlinkHostIpInfo[key] = this->successHostIP_[key];
        }
    }

    // currentLinkIpInfo - this->successHostIP_ = link
    for (auto it = currentLinkHostIpInfo.cbegin(); it != currentLinkHostIpInfo.cend(); ++it) {
        auto key = it->first;
        if (this->successHostIP_.find(key) == this->successHostIP_.cend()) {
            globalIpInfo.hostIpInfo[key] = currentLinkHostIpInfo[key];
        }
    }
}

// v2 interface
bool DmiRole::UpdatePDNotSwitchInfoV2(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo)
{
    std::lock_guard<std::mutex> lock(mtx_);
    globalIpInfo.role = roleName;
    globalIpInfo.needSwitch = false;
    std::string superPodId = "";
    (void)roleName;  // 显式使用一次roleName，以消除编译器警告（编译器误报）

    // 首先读“peers”字段的值存在tempLinkIpInfo中
    std::map<uint64_t, std::vector<DeviceInfo>> tempLinkIpInfo{};
    std::map<uint64_t, std::vector<std::string>> tempHostIpInfo{};
    try {
        for (const auto &nodeInfo : body["local"]) {
            for (const auto& dpGroupInfo : nodeInfo["dp_inst_list"]) {
                globalIpInfo.localDpInstanceIds.emplace_back(dpGroupInfo["dp_inst_id"]);
            }
        }
        for (const auto &peerInfo : body["peers"]) {
            for (const auto &nodeInfo : peerInfo) {
                auto ret = ReverseDpInstId(nodeInfo["dp_inst_list"][0]["dp_inst_id"]);
                uint32_t instanceId = ret.first;
                globalIpInfo.spInfo[instanceId] = nodeInfo.value("sp_size", 1);
                globalIpInfo.cpInfo[instanceId] = nodeInfo.value("cp_size", 1);
                instanceIdToServerIp_[instanceId] = nodeInfo["server_ip"];

                for (const auto &dpGroupInfo : nodeInfo["dp_inst_list"]) {
                    std::vector<DeviceInfo> linkDeviceIp;
                    auto dpInstanceId = dpGroupInfo["dp_inst_id"];
                    if (nodeInfo.contains("super_pod_id")) {
                        superPodId = nodeInfo["super_pod_id"];
                        globalIpInfo.superPodIdInfo[dpInstanceId] = superPodId;
                    }
                    for (const auto &deviceInfo : dpGroupInfo["device"]) {
                        DeviceInfo device;
                        device.deviceIp = deviceInfo["device_ip"].get<std::string>();
                        device.devicePhysicalId = std::stoi(deviceInfo["device_id"].get<std::string>());
                        if (deviceInfo.contains("super_device_id")) {
                            device.superDeviceId = std::stoi(deviceInfo["super_device_id"].get<std::string>());
                        }
                        linkDeviceIp.emplace_back(device);
                    }
                    if (tempLinkIpInfo.find(dpInstanceId) != tempLinkIpInfo.end()) {
                        // 如果dpInstanceId已存在，则合并设备信息
                        auto& existingDevices = tempLinkIpInfo[dpInstanceId];
                        existingDevices.insert(existingDevices.end(), linkDeviceIp.begin(), linkDeviceIp.end());
                    } else {
                        tempLinkIpInfo.insert({dpInstanceId, linkDeviceIp});
                    }
                    if (tempHostIpInfo.find(dpInstanceId) != tempHostIpInfo.end()) {
                        tempHostIpInfo[dpInstanceId].emplace_back(nodeInfo["host_ip"]);
                    } else {
                        tempHostIpInfo[dpInstanceId] = {nodeInfo["host_ip"]};
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Error occurred : " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Unknown error occurred.");
        return false;
    }
    this->UpdateIpInfo(tempLinkIpInfo, globalIpInfo, superPodId);
    this->UpdateHostIpInfo(tempHostIpInfo, globalIpInfo);
    return true;
}

void DmiRole::UpdateSuccessLinkIp(GlobalIpInfo &globalIpInfo)
{
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& pair : globalIpInfo.unlinkIpInfo) {
        this->successLinkIP_.erase(pair.first);
        remoteNodeLinkStatus_.erase(pair.first);
    }
    for (const auto& pair : globalIpInfo.linkIpInfo) {
        this->successLinkIP_[pair.first] = pair.second;
        remoteNodeLinkStatus_[pair.first] = {"ok", true};
    }
    for (const auto& pair : globalIpInfo.retryLinkIpInfo) {
        const int64_t failLinkReasonInt = globalIpInfo.failLinkInstanceIDAndReason[pair.first];
        std::string failedReason = "Unknown reason";
        if (LINK_FAILED_MAP.find(failLinkReasonInt) != LINK_FAILED_MAP.end()) {
            failedReason = LINK_FAILED_MAP.at(failLinkReasonInt);
        }
        std::string failedString = "failed : " + failedReason;
        // key remoteDpInstanceId
        remoteNodeLinkStatus_[pair.first] = {failedString, true};
    }
}

void DmiRole::UpdateSuccessHostIp(GlobalIpInfo &globalIpInfo)
{
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& pair : globalIpInfo.unlinkHostIpInfo) {
        this->successHostIP_.erase(pair.first);
    }
    for (const auto& pair : globalIpInfo.hostIpInfo) {
        this->successHostIP_[pair.first] = pair.second;
    }
}

void DmiRole::HandlePDRoleV1(const ReqCtxPtr &ctx, const std::string &roleName)
{
    ordered_json body;
    GlobalIpInfo globalIpInfo;
    if (!PDParseRequestBodyToJson(ctx, body) || !JsonParse::CheckPDRoleReqJson(body)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
            "[DmiRole::HandlePDRole] Req body converts to json fail. Reset to previous node status.");
        HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson("Req body converts to json fail. Reset to previous node status.",
            "Input validation error"));
        return;
    }

    std::string preRole = GetInferInstance()->GetPDRole();
    if (!UpdatePDInfo(roleName, preRole, body, globalIpInfo)) {
        HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::ServiceUnavailable_503,
            HttpRestResource::WrapperJson("Update pd info failed. Reset to previous node status.",
            "Service Unavailable"));
        return;
    }
    ProcessPDRoleSwitch(ctx, roleName, globalIpInfo);
}

void DmiRole::HandlePDRoleV2(const ReqCtxPtr &ctx, const std::string &roleName)
{
    ordered_json body;
    GlobalIpInfo globalIpInfo;
    try {
        if (!PDParseRequestBodyToJson(ctx, body) || !JsonParse::CheckPDRoleV2ReqJson(body)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_PREFIX_CACHE, STATUS_WARNING),
                "[DmiRole::HandlePDRole] Req body converts to json fail. Reset to previous node status. ");
            HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::UnprocessableContent_422,
                HttpRestResource::WrapperJson("Req body converts to json fail. Reset to previous node status.",
                "Input validation error"));
            return;
        }

        std::string preRole = GetInferInstance()->GetPDRole();
        if (!UpdatePDInfoV2(roleName, preRole, body, globalIpInfo)) { // Json -> globalIpInfo
            HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::ServiceUnavailable_503,
                HttpRestResource::WrapperJson("Parse req json failed. Reset to previous node status.",
                "Service Unavailable"));
            return;
        }
        ProcessPDRoleSwitch(ctx, roleName, globalIpInfo);
    }
    catch (const std::exception& e) {
        // 捕获标准异常
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "Standard exception caught: " << e.what());
        return;
    }
    catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "Unknown exception caught.");
        return;
    }
}

void DmiRole::ProcessPDRoleSwitch(const ReqCtxPtr &ctx, const std::string &roleName,
    GlobalIpInfo &globalIpInfo)
{
    GetInferInstance()->SetPDRoleStatus(PDRoleStatus::SWITCHING);
    GetInferInstance()->UpdatePDRole(roleName);
    auto relinkCallback = std::bind(&DmiRole::RetryLinkCallback, this, std::placeholders::_1);

    OrderedJson jsonObj;
    jsonObj["result"] = "ok";

    if (roleName == "prefill") {
        keepAlive.store(false); // 关闭保活响应
    }

    // only retry when need to link/unlink/switch_role/set_percentage
    if (!globalIpInfo.linkIpInfo.empty() || !globalIpInfo.unlinkIpInfo.empty() || globalIpInfo.needSwitch ||
        globalIpInfo.role == "flex") {
        ioContext_ = std::make_shared<boost::asio::io_context>();
        auto retryTask = std::make_unique<RetryTask>(relinkCallback, ioContext_, globalIpInfo);
        taskQueue_.Push(std::move(retryTask));
        HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, jsonObj.dump());
        return;
    }
    GetInferInstance()->SetPDRoleStatus(PDRoleStatus::READY);
    HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, jsonObj.dump());
}

void DmiRole::ResetContext(boost::system::error_code ec)
{
    if (this->ioContext_ != nullptr && !ec) {
        this->ioContext_->stop();
        this->ioContext_ = nullptr;
    } else {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "Error occurred in context.");
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Reset context success.");
}

void DmiRole::RetryThread()
{
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Start retry thread.");
    while (!retryTerminate_.load()) {
        auto retryTask = taskQueue_.Take();
        if (retryTask == nullptr) {
            break;
        }
        retryTask->Execute();
        ResetContext(boost::system::error_code());
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Finish a retry task.");
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "RetryTask has been destructed.");
}

// this callback function will be used in RetryTask to establish links between devices
void DmiRole::RetryLinkCallback(GlobalIpInfo &globalIpInfo)
{
    if (!GetInferInstance()->AssignDmiRole(globalIpInfo).IsOk()) {
        globalIpInfo.ClearIpInfo();
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), "Update PD role failed.");
        return;
    }
    UpdateSuccessLinkIp(globalIpInfo);
    UpdateSuccessHostIp(globalIpInfo);
    globalIpInfo.linkIpInfo = globalIpInfo.retryLinkIpInfo;
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Retry one step done.");
}

void DmiRole::ModifyPullKVFailId(const uint32_t &instanceId)
{
    std::lock_guard<std::mutex> lock(mtx_);
    this->successLinkIP_.erase(instanceId);
    this->successHostIP_.erase(instanceId);
    remoteNodeLinkStatus_[instanceId].first = "failed : pull kv failed.";
    this->abnormalLink_ = true;
}

const std::map<uint64_t, std::vector<DeviceInfo>> &DmiRole::GetSuccessLinkIp()
{
    return successLinkIP_;
}

const std::map<uint64_t, std::vector<std::string>> &DmiRole::GetSuccessHostIp()
{
    return successHostIP_;
}

const std::map<uint64_t, std::pair<std::string, bool>> &DmiRole::GetRemoteNodeLinkStatus()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return this->remoteNodeLinkStatus_;
}


std::map<uint64_t, std::pair<std::string, bool>> GetInstanceStatus(
    const std::map<uint64_t, std::pair<std::string, bool>>& dpInstanceStatusMap)
{
    std::map<uint64_t, std::pair<std::string, bool>> instanceStatus;
    // 遍历所有dpInstance的状态
    for (const auto& entry : dpInstanceStatusMap) {
        auto dpInstanceId = entry.first;
        bool dpInstanceStatus = entry.second.second;
        std::string statusString = entry.second.first;

        // 反推instanceId
        auto instanceId = dpInstanceId / 10000;
        // 如果该instance状态还没有被设置，先设置为true
        if (instanceStatus.find(instanceId) == instanceStatus.end()) {
            instanceStatus[instanceId] = {"ok", true};
        }

        if (!dpInstanceStatus) {
            instanceStatus[instanceId] = {"dp instance id : " + std::to_string(dpInstanceId) + statusString, false};
        }
    }

    return instanceStatus;
}

std::map<uint64_t, std::pair<std::string, bool>> DmiRole::GetRemoteNodeLinkStatusV2()
{
    std::lock_guard<std::mutex> lock(mtx_);
    // summarize status for each dp group -> instance status
    auto ret = GetInstanceStatus(this->remoteNodeLinkStatus_);
    return ret;
}

const std::map<uint32_t, std::string> &DmiRole::GetInstanceIdToServerIp()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return this->instanceIdToServerIp_;
}

const uint32_t &DmiRole::GetLocalInstanceId()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return this->localInstanceId_;
}

// Unhealthy P node: all links ought to be linked, so it is always healthy
// Unhealthy D node: any link is abnormal and no success linked peers
bool DmiRole::IsHealthy()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return !(this->abnormalLink_ && this->successLinkIP_.size() == 0);
}
} // namespace mindie_llm