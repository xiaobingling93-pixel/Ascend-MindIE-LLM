/**
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

#ifndef MINDIE_LLM_MANAGER_H
#define MINDIE_LLM_MANAGER_H

#include <map>
#include <memory>
#include <set>
#include <string>

#include "callback.h"
#include "data_type.h"
#include "status.h"

namespace mindie_llm {
class LlmManagerImpl;
/// A component called the llmmanager, to support continuous batching of requests
///
/// This class is a manager to to support continuous batching of requests, provides basic functions such as
/// initialize the LlmManager，get running params，shutdown the LlmManager and functions of PD Separation.
class LlmManager {
   public:
    /// This Constructor initializes a LlmManager object with the following parameters
    ///
    /// \param llmConfigPath The path of the LLM configuration file
    /// \param getRequest The callback function for getting requests
    /// \param sendResponse The callback function for retrieving response tensor from the llmmanger
    /// \param controlCallback The callback function for acquiring requests with control operations
    /// \param statusCallback The callback function for retrieving status information from the llmmanger
    /// \param statusResponseCallback The callback function for obtaining the status of requests being queued and the
    /// \param ipInfo The map saved params need to be set in modelConfig and the
    /// execution status of requests with control operations
    LlmManager(const std::string &llmConfigPath, mindie_llm::GetRequestsCallback getRequest,
               mindie_llm::SendResponsesCallback sendResponse, mindie_llm::ControlSignalCallback controlCallback,
               mindie_llm::LlmManagerStatsCallback statusCallback,
               mindie_llm::SendStatusResponseCallback statusResponseCallback,
               std::map<std::string, std::string> ipInfo = std::map<std::string, std::string>());

    uint32_t GetMaxPositionEmbeddings() const;

    /// Get model's parameters,
    /// this function is used to get the parameters of the model, such as the maximum number of tokens. etc
    ///
    /// \return map<std::string, std::string> format, which stores configuration information
    std::map<std::string, std::string> GetModelParams() const;

    void Shutdown();

    /// Assign the DMI role to the specified request,
    /// this function is used to assign the DMI role to the specified request
    ///
    /// \param runtimeRequest The request to assign the DMI role to
    /// \param isForceRelease Whether to force release the DMI role
    ///
    /// \return true if the DMI role was successfully assigned, false otherwise
    bool UpdateEngineInfo(std::shared_ptr<mindie_llm::InferRequest> &runtimeRequest, bool isForceRelease);

    bool ExecuteRecoverCommand(RecoverCommandInfo &commandInfo);

    /// This function initializes the LLM manager with the specified model instance ID and NPU device IDs.
    ///
    /// \param modelInstanceId The model instance ID used to initialize the LLM manager
    /// \param npuDeviceIds The NPU device IDs used to initialize the LLM manager
    ///
    /// \return Status indicating whether the initialization is successful
    Status Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds);

    Status Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds, std::map<std::string, std::string> extendInfo);

    Status InitModelForMultiPd(std::map<std::string, std::string> pdInfo, uint32_t modelInstanceId);

    ~LlmManager();

   private:
    std::shared_ptr<LlmManagerImpl> impl_;
};
}  // namespace mindie_llm
#endif  // MINDIE_LLM_MANAGER_H
