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
 
#include "hashless_block_obj.h"

#include <iostream>


namespace mindie_llm {

/*
    Init a block.
    This func handles everything for initialization except for construction.
    Normally used in an ObjectPool to avoid frequent object creation/deletion.
*/
void HashLessBlockObj::InitBlockObj(const BlockObjSPtr prevBlock, const std::vector<TokenId> &tokenIds,
                                    BlockSharedAttr blockSharedAttr, BlockId blockId, HashValue)
{
    ResetBlockObj();
    prevBlock_ = prevBlock;
    tokenIds_ = std::vector<TokenId>{}; // first init and append token ids later
    blockSize_ = blockSharedAttr.blockSize;
    blockId_ = blockId;
    AppendTokenIdsNoCow(tokenIds);
}

void HashLessBlockObj::ResetBlockObj()
{
    prevBlock_ = nullptr;

    tokenIds_ = {};

    blockSize_ = 0;

    blockId_ = INVALID_BLOCKID;
}

/*
    Key function for appending token ids to the block.
    It first appends tokens to block's token id container,
    and then call the allocator to do CoW if necessary.
*/
void HashLessBlockObj::AppendTokenIds(const std::vector<TokenId> &tokenIds)
{
    if (blockId_ == INVALID_BLOCKID) {
        throw std::runtime_error("block id invalid!");
    }
    
    AppendTokenIdsNoCow(tokenIds);
}

/*
Append token ids to the token id container. Only used inside this class.
Caller must ensure there is enough empty slots in this block!
*/
void HashLessBlockObj::AppendTokenIdsNoCow(const std::vector<TokenId> &tokenIds)
{
    if (tokenIds.empty()) {
        return;
    }

    if (tokenIds.size() > GetNumEmptySlots()) {
        throw std::runtime_error("space on block is not enough for appending given tokens");
    }

    tokenIds_.insert(tokenIds_.end(), tokenIds.begin(), tokenIds.end()); // append tokens
}

BlockId HashLessBlockObj::GetBlockId() const { return blockId_; }

size_t HashLessBlockObj::GetRankIdx() const { return rankIdx_; }

// not used in HashLessBlock
HashValue HashLessBlockObj::GetHashValue() { return INVALID_HASH_VALUE; }

// this should be promoted to a private method in the future...
void HashLessBlockObj::ResetBlockId()
{
    if (blockId_ == INVALID_BLOCKID) {
        throw std::runtime_error("block id is null!");
    }
    blockId_ = INVALID_BLOCKID;
}

// this should be promoted to a private method in the future...
void HashLessBlockObj::SetBlockId(BlockId blockId)
{
    // should check whether it is safe to do so in the future...
    blockId_ = blockId;
}

bool HashLessBlockObj::IsFull() const { return GetNumEmptySlots() == 0; }

size_t HashLessBlockObj::GetNumEmptySlots() const
{
    if (blockSize_ < tokenIds_.size()) {
        throw std::runtime_error("number of empty slots invalid!");
    }
    return blockSize_ - tokenIds_.size();
}

size_t HashLessBlockObj::GetNumTokensTotal() const { return tokenIds_.size(); }

const std::vector<TokenId> &HashLessBlockObj::GetTokenIds() const { return tokenIds_; }

size_t HashLessBlockObj::GetBlockSize() const { return blockSize_; }

BlockObjSPtr &HashLessBlockObj::GetPrevBlock() { return prevBlock_; }

// not used in HashLessBlock
bool HashLessBlockObj::IsComputed() const { throw std::runtime_error("not implemented"); };

// not used in HashLessBlock
float HashLessBlockObj::LastAccessed() const { throw std::runtime_error("not implemented"); };

// not used in HashLessBlock
void HashLessBlockObj::SetComputed([[maybe_unused]] bool isCompute) { throw std::runtime_error("not implemented"); };

// not used in HashLessBlock
void HashLessBlockObj::SetLastAccessed([[maybe_unused]] float lastAccess)
{
    throw std::runtime_error("not implemented");
};

// not used in HashLessBlock
HashValue HashLessBlockObj::ExtraHash() { throw std::runtime_error("not implemented"); };

// not used in HashLessBlock
HashValue HashLessBlockObj::PrefixHash() { throw std::runtime_error("not implemented"); };

void HashLessBlockObj::ReplaceToken(size_t startIndex, TokenId newToken)
{
    if (startIndex >= tokenIds_.size()) {
        throw std::out_of_range("startIndex is out of range of tokenIds_");
    }

    if (startIndex >= blockSize_) {
        throw std::out_of_range("startIndex is out of range of blockSize_");
    }

    tokenIds_[startIndex] = newToken;
}

void HashLessBlockObj::SetRankIdx(size_t rankIdx) { rankIdx_ = rankIdx; }

} // namespace mindie_llm