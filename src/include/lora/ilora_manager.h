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

#ifndef ILORA_MANAGER_H
#define ILORA_MANAGER_H

#include <memory>

#include "basic_types.h"
#include "concurrent_map.h"
#include "config/config_info.h"
#include "llm_manager_v2/llm_manager_v2.h"

namespace mindie_llm {
class ILoraManager {
   public:
    virtual ~ILoraManager() = default;
    // llm_engine使用，可直接下发加载
    virtual Status Load(const LoraParamSPtr loraInfo) = 0;

    // llm_engine使用，开始准备卸载
    virtual Status StartToUnload(const std::string &loraName) = 0;

    // llm_engine使用，查询可用lora
    virtual Status GetLoadedLoras(std::vector<LoraParamSPtr> &loraInfo) = 0;

    // llm_engine使用，尝试卸载并等待
    virtual void TryUnLoadWaiting() = 0;

    // sequence使用，如果loraId不可用则将sequence的loraId设置为None
    virtual bool ValidateLoraId(const std::optional<std::string> &loraId) = 0;

    // 初始化静态lora
    virtual void InitLoadedLoras(const std::vector<ModelParam> &modelParamVec) = 0;

    // seqgrp构造使用, 增加lora引用计数
    virtual void IncLoraRef(const std::optional<std::string> &loraId) = 0;

    // seqgrp析构使用, 减少lora引用计数
    virtual void DecLoraRef(const std::optional<std::string> &loraId) = 0;
};
}  // namespace mindie_llm

#endif
