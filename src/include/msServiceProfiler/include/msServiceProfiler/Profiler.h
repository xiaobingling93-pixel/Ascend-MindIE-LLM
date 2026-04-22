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

#ifndef PROFILER_H
#define PROFILER_H

// 定义日志级别枚举
namespace msServiceProfiler {
enum class Level { INFO, WARNING, ERROR };
}

// 定义 Profiler 类模板
namespace msServiceProfiler {
template <Level level = Level::INFO>
class Profiler {};
}  // namespace msServiceProfiler

#endif  // PROFILER_H
