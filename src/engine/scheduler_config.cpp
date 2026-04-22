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
#include "config_info.h"

namespace mindie_llm {

bool SchedulerConfig::ChooseV2BlockManager() const {
    return (dpSize == 1) && (!enablePrefixCache) && (!enableSplit) && (spSize == 1) && (cpSize == 1);

    // TBC_workFlowTemplateType workFlowModelConfigGamma (!enableAsyncScheduling)
}
}  // namespace mindie_llm
