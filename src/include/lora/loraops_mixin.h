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

#ifndef LORAOPS_MIXIN_H
#define LORAOPS_MIXIN_H

#include "config/config_info.h"
#include "llm_manager_v2/llm_manager_v2.h"

namespace mindie_llm {
class LoraOpsMixin {
   public:
    virtual Status LoraLoad(const std::vector<LoraParamSPtr> &loraInfo, size_t dpSize);

    virtual Status LoraUnLoad(const std::vector<LoraParamSPtr> &loraInfo, size_t dpSize);

    virtual Status LoraGetLoaded(std::vector<LoraParamSPtr> &loraInfo, size_t dpSize);

    virtual void InitStaticLoras(const std::vector<ModelParam> &modelParamVec, size_t dpSize);
};
}  // namespace mindie_llm

#endif
