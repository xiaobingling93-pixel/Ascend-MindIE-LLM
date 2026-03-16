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
 
#pragma once

#include "hashless_block_obj.h"

namespace mindie_llm {

class PrefixCachingBlockObj : public HashLessBlockObj, public std::enable_shared_from_this<PrefixCachingBlockObj> {
public:
    PrefixCachingBlockObj() = default;

    ~PrefixCachingBlockObj() override = default;

    void InitBlockObj(const BlockObjSPtr prevBlock, const std::vector<TokenId> &tokenIds,
                      BlockSharedAttr blockSharedAttr, BlockId blockId, HashValue extraHash) override;

    void ResetBlockObj() override;

    void AppendTokenIds(const std::vector<TokenId> &tokenIds) override;

    size_t GetNumTokensTotal() const override;

    HashValue ExtraHash() override;

    HashValue PrefixHash() override;

    bool IsFull() const override;

    HashValue GetHashValue() override;

private:
    size_t blockSize_ = 0;

    HashValue cachedPrefixHash_ = INVALID_HASH_VALUE;

    HashValue extraHash_ = INVALID_HASH_VALUE;

    size_t numTokensTotal_ = 0;

    void UpdateNumTokensTotal();

    bool IsLastTokenPlaceholder() const;
    
    bool IsReadyToCalcPrefixHash() const;
};

using PrefixCachingBlockObjSPtr = std::shared_ptr<PrefixCachingBlockObj>;
} // namespace mindie_llm