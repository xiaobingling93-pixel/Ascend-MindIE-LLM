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
#include "lora/loraops_mixin.h"

#include "lora_manager.h"

namespace mindie_llm {
Status LoraOpsMixin::LoraLoad(const std::vector<LoraParamSPtr> &loraInfo, size_t dpSize) {
    if (loraInfo.size() != 1) {
        return Status(Error::Code::ERROR, "The number of load loraInfo is invalid");
    }
    Status ret;
    for (size_t i = 0; i < dpSize; ++i) {
        auto loraManager = mindie_llm::LoraManager::GetInstance(i);
        if (loraManager != nullptr) {
            ret = loraManager->Load(loraInfo.front());
        } else {
            return Status(Error::Code::ERROR, "LoraManager instance for index " + std::to_string(i) + " is null");
        }
    }
    return ret;
}

Status LoraOpsMixin::LoraUnLoad(const std::vector<LoraParamSPtr> &loraInfo, size_t dpSize) {
    if (loraInfo.size() != 1) {
        return Status(Error::Code::ERROR, "The number of unload loraInfo is invalid");
    }
    std::string loadLoraId = loraInfo.front()->loraName;
    Status ret;
    for (size_t i = 0; i < dpSize; ++i) {
        auto loraManager = mindie_llm::LoraManager::GetInstance(i);
        if (loraManager != nullptr) {
            ret = loraManager->StartToUnload(loadLoraId);
        } else {
            return Status(Error::Code::ERROR, "LoraManager instance for index " + std::to_string(i) + " is null");
        }
    }
    return ret;
}

Status LoraOpsMixin::LoraGetLoaded(std::vector<LoraParamSPtr> &loraInfo, size_t dpSize) {
    Status ret;
    for (size_t i = 0; i < dpSize; ++i) {
        auto loraManager = mindie_llm::LoraManager::GetInstance(i);
        if (loraManager != nullptr) {
            ret = loraManager->GetLoadedLoras(loraInfo);
        } else {
            return Status(Error::Code::ERROR, "LoraManager instance for index " + std::to_string(i) + " is null");
        }
    }
    return ret;
}

void LoraOpsMixin::InitStaticLoras(const std::vector<ModelParam> &modelParamVec, size_t dpSize) {
    for (size_t i = 0; i < dpSize; ++i) {
        auto loraManager = mindie_llm::LoraManager::GetInstance(i);
        if (loraManager != nullptr) {
            loraManager->InitLoadedLoras(modelParamVec);
        }
    }
}
}  // namespace mindie_llm
