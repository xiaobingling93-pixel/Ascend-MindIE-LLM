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
 
#ifndef HASHLESS_BLOCK_OBJ_H
#define HASHLESS_BLOCK_OBJ_H

#include "basic_types.h"
#include "block_allocator.h"
#include "block_obj.h"

namespace mindie_llm {

/*
    class HashLessBlockObj inherits class BlockObj,
    It implements interfaces for naive logical blocks without Prefix Caching.
    The core function of this class is to manage token ids appended to the block.
*/
class HashLessBlockObj : public BlockObj {
public:
    HashLessBlockObj() = default;

    ~HashLessBlockObj() override = default;

    void InitBlockObj(const BlockObjSPtr prevBlock, const std::vector<TokenId> &tokenIds,
                      BlockSharedAttr blockSharedAttr, BlockId blockId, HashValue) override;

    void ResetBlockObj() override;

    void AppendTokenIds(const std::vector<TokenId> &tokenIds) override;

    void AppendTokenIdsNoCow(const std::vector<TokenId> &tokenIds);

    BlockId GetBlockId() const override;

    size_t GetRankIdx() const override;

    // this should be promoted to a private method in the future...
    void ResetBlockId() override;

    // this should be promoted to a private method in the future...
    void SetBlockId(BlockId blockId) override;

    const std::vector<TokenId> &GetTokenIds() const override;

    BlockObjSPtr &GetPrevBlock() override;

    size_t GetNumTokensTotal() const override;

    size_t GetBlockSize() const override;

    size_t GetNumEmptySlots() const;

    bool IsFull() const override;
   
    bool IsComputed() const override; // not used in HashLessBlock

    float LastAccessed() const override; // not used in HashLessBlock

    void SetComputed(bool isCompute) override; // not used in HashLessBlock

    void SetLastAccessed(TimeStamp lastAccess) override; // not used in HashLessBlock

    HashValue GetHashValue() override; // not used in HashLessBlock

    HashValue ExtraHash() override; // not used in HashLessBlock

    HashValue PrefixHash() override; // not used in HashLessBlock

    void ReplaceToken(size_t startIndex, TokenId newToken) override;

    void SetRankIdx(size_t rankIdx) override;

private:
    BlockObjSPtr prevBlock_ = nullptr;

    std::vector<TokenId> tokenIds_ = {};

    size_t blockSize_ = 0;

    BlockId blockId_ = INVALID_BLOCKID;

    size_t rankIdx_ = 0;
};
} // namespace mindie_llm

#endif
