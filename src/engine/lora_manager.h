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

#ifndef MINDIE_LLM_LORA_MANAGER_H
#define MINDIE_LLM_LORA_MANAGER_H

#include <mutex>
#include <string>

#include "executor/executor_interface.h"
#include "lora/ilora_manager.h"

namespace mindie_llm {
// Lora加载卸载返回状态索引
enum class LoraStatus {
    LOAD_SUCCESS = 0,               // 加载成功
    DUPLICATED_LORA_ID = 1,         // 该loraId已被占用，loraId正在被使用
    UNLOADING = 2,                  // 该loraId已被占用，loraId正在卸载
    INVALID_LORA_ID = 3,            // loraId不合法
    INVALID_LORA_PATH = 4,          // 无效的Lora路径
    INVALID_LORA_RANK = 5,          // 超出最大的lorarank
    SLOTS_FULL = 6,                 // 槽位已满，当前lora个数已达到max_loras，无正在被卸载的lora
    SLOTS_FULL_WITH_UNLOADING = 7,  // 槽位已满，有正在卸载的lora
    UNLOAD_SUCCESS = 8,             // 异步接口返回卸载成功
    LORA_NOT_FOUND = 9,             // 未找到该lora,该lora未被加载或者等待卸载
    UNSUPPORT_CMD = 10              // 未采用python组图
};

class LoraManager : public ILoraManager {
   public:
    // 初始化一次
    static void Initialize(std::vector<IExecutorSPtr> executors, uint32_t maxLoras);

    static std::shared_ptr<LoraManager> GetInstance(size_t localDPRank);

    LoraManager(IExecutorSPtr executor, uint32_t maxLoras);

    ~LoraManager() override = default;
    // 同步接口可直接下发加载
    Status Load(const LoraParamSPtr loraInfo) override;

    // 异步接口此处开始卸载
    Status StartToUnload(const std::string &loraName) override;

    // 查询可用的Lora信息
    Status GetLoadedLoras(std::vector<LoraParamSPtr> &loraInfo) override;

    // 异步卸载lora接口
    void TryUnLoadWaiting() override;

    // sequence使用，如果loraId不可用则将sequence的loraId设置为None
    bool ValidateLoraId(const std::optional<std::string> &loraId) override;

    // 初始化静态lora信息
    void InitLoadedLoras(const std::vector<ModelParam> &modelParamVec) override;

    // 增加lora引用计数
    void IncLoraRef(const std::optional<std::string> &loraId) override;

    // 减少lora引用计数
    void DecLoraRef(const std::optional<std::string> &loraId) override;

    // 加载判断LoRA状态
    LoraStatus GetLoraStatus(const LoraParamSPtr loraInfo, bool &loraIsInvalid);

   private:
    ConcurrentMap<std::string, LoraParamSPtr> loaded_;         // 已经加载的
    ConcurrentMap<std::string, LoraParamSPtr> wait2Unloaded_;  // 等待被卸载的
    ConcurrentMap<std::string, uint32_t> loraIdRef_;           // lora引用计数
    IExecutorSPtr executor_;
    uint32_t maxLoras_ = 0;
    static std::once_flag initFlag_;
    static std::vector<std::shared_ptr<LoraManager>> instances_;
};
using LlmLoraPtr = std::shared_ptr<LoraManager>;
}  // namespace mindie_llm
#endif  // MINDIE_LLM_LORA_MANAGER_H
