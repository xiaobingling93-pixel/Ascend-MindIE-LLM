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
 
#ifndef BLOCK_OBJ_H
#define BLOCK_OBJ_H

#include <memory>
#include <vector>

#include "basic_types.h"


namespace mindie_llm {

struct BlockSharedAttr;

class BlockObj {
public:
    explicit BlockObj() = default;

    virtual ~BlockObj() = default;

    virtual void AppendTokenIds(const std::vector<TokenId> &tokenIds) = 0;

    virtual BlockId GetBlockId() const = 0;

    virtual size_t GetRankIdx() const = 0;

    // this should be promoted to a private method in the future...
    virtual void ResetBlockId() = 0;

    virtual void ResetBlockObj() = 0;

    // this should be promoted to a private method in the future...
    virtual void SetBlockId(BlockId blockId) = 0;

    virtual const std::vector<TokenId> &GetTokenIds() const = 0;

    virtual size_t GetNumTokensTotal() const = 0;

    virtual bool IsFull() const = 0;

    virtual std::shared_ptr<BlockObj> &GetPrevBlock() = 0;

    virtual bool IsComputed() const = 0; // Should be only used by PrefixCacingAllocator

    virtual float LastAccessed() const = 0; // block 最后一次被访问的时间

    virtual size_t GetBlockSize() const = 0;

    virtual void SetComputed(bool isCompute) = 0;

    virtual void SetLastAccessed(TimeStamp lastAccess) = 0;

    virtual HashValue GetHashValue() = 0;

    virtual HashValue ExtraHash() = 0; // 用于Lora

    virtual HashValue PrefixHash() = 0; // 前缀block和当前block的hash

    virtual void InitBlockObj(const std::shared_ptr<BlockObj> prevBlock, const std::vector<TokenId> &tokenIds,
                              BlockSharedAttr blockSharedAttr, BlockId blockId, HashValue extraHash = 0) = 0;

    virtual void ReplaceToken(size_t startIndex, TokenId newToken) = 0;

    virtual void SetRankIdx(size_t rankIdx) = 0;
};
using BlockObjSPtr = std::shared_ptr<BlockObj>;
} // namespace mindie_llm

#endif
