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

#ifndef MINDIE_LLM_MEMORY_BRIDGE_SWAPPER_H
#define MINDIE_LLM_MEMORY_BRIDGE_SWAPPER_H

#include <memory>
#include <vector>

namespace hmm {
class Swapper {
   public:
    virtual ~Swapper() noexcept = default;
    virtual void H2dSwap(const uint64_t &cpuBlock, const uint64_t &npuBlock,
                         const std::vector<uint64_t> &params) const = 0;
    virtual void D2hSwap(const uint64_t &cpuBlock, const uint64_t &npuBlock,
                         const std::vector<uint64_t> &params) const = 0;
    static std::shared_ptr<Swapper> &Create();
};
}  // namespace hmm
#endif
