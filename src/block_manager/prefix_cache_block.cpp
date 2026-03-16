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

#include "prefix_cache_block.h"

#include <stdexcept>

#include "math_utils.h"


using namespace std;

namespace mindie_llm {

void PrefixCachingBlockObj::InitBlockObj(const BlockObjSPtr prevBlock, const std::vector<TokenId> &tokenIds,
                                         BlockSharedAttr blockSharedAttr, BlockId blockId, HashValue extraHash)
{
    HashLessBlockObj::InitBlockObj(prevBlock, tokenIds, blockSharedAttr, blockId, 0);
    blockSize_ = blockSharedAttr.blockSize;
    extraHash_ = extraHash;
    UpdateNumTokensTotal();
}

void PrefixCachingBlockObj::ResetBlockObj()
{
    blockSize_ = 0;
    cachedPrefixHash_ = INVALID_HASH_VALUE;
    extraHash_ = INVALID_HASH_VALUE;
    numTokensTotal_ = 0;
}

// Append可能会引起BlockId变化,也可能引起引用计数变化
void PrefixCachingBlockObj::AppendTokenIds(const vector<TokenId> &tokenIds)
{
    if (PrefixHash() != INVALID_HASH_VALUE) {
        throw runtime_error("AllocateBLockId Error:block is cached!");
    }

    // 检查bLock剩余空间是否足够
    if (tokenIds.size() > blockSize_ || tokenIds.size() > blockSize_ - GetTokenIds().size()) {
        throw runtime_error("AllocateBlockId Error:block is not enough!");
    }

    HashLessBlockObj::AppendTokenIds(tokenIds);
    UpdateNumTokensTotal();
    PrefixHash();
}

HashValue PrefixCachingBlockObj::ExtraHash() { return extraHash_; }

bool PrefixCachingBlockObj::IsLastTokenPlaceholder() const
{
    if (GetTokenIds().empty()) {
        return false;
    }
    return GetTokenIds().back() == PLACEHOLDER_TOKEN;
}

bool PrefixCachingBlockObj::IsFull() const
{
    return GetNumEmptySlots() == 0;
}

HashValue PrefixCachingBlockObj::GetHashValue() { return cachedPrefixHash_; }

bool PrefixCachingBlockObj::IsReadyToCalcPrefixHash() const
{
    // 只有token存满才计算hash
    if (!IsFull()) {
        return false;
    }

    // Decode阶段append的token是占位符，所以即使block放满也不认为可以计算hash
    if (IsLastTokenPlaceholder()) {
        return false;
    }

    return true;
}

HashValue PrefixCachingBlockObj::PrefixHash()
{
    // 避免重复计算
    if (cachedPrefixHash_ != INVALID_HASH_VALUE) {
        return cachedPrefixHash_;
    }

    if (!IsReadyToCalcPrefixHash()) {
        return INVALID_HASH_VALUE;
    }

    // 如果preBLock没有计算hash，本block也不能计算hash
    BlockObjSPtr preBlock = GetPrevBlock();
    if (preBlock != nullptr) {
        if (preBlock->PrefixHash() == INVALID_HASH_VALUE) {
            return INVALID_HASH_VALUE;
        }
    }

    HashValue seed = 0;
    if (GetPrevBlock()) {
        HashCombine(seed, GetPrevBlock()->PrefixHash());
    }

    for (const TokenId token : GetTokenIds()) {
        HashCombine(seed, token);
    }

    HashCombine(seed, extraHash_);
    cachedPrefixHash_ = seed;
    return seed;
}

size_t PrefixCachingBlockObj::GetNumTokensTotal() const { return numTokensTotal_; }

void PrefixCachingBlockObj::UpdateNumTokensTotal()
{
    numTokensTotal_ = 0;

    auto prevBLock = GetPrevBlock();
    if (prevBLock != nullptr) {
        numTokensTotal_ += prevBLock->GetNumTokensTotal();
    }
    numTokensTotal_ += GetTokenIds().size();
}

} // namespace mindie_llm