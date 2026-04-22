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

#ifndef PD_ROLE_H
#define PD_ROLE_H

namespace mindie_llm {
enum class PDRole {
    UNKNOWN = 0,
    PREFILL = 1,
    DECODE = 2,
    Flex = 3,  // 配比微调特性使用，在pd分离场景可以弹性执行prefill和decode请求
};

enum class PDRoleStatus {
    UNKNOWN = 0,
    READY = 1,
    SWITCHING = 2,
};
}  // namespace mindie_llm

#endif  // PD_ROLE_H
