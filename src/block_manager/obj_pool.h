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

#ifndef OBJ_POOL_H
#define OBJ_POOL_H

#include <functional>
#include <limits>
#include <memory>
#include <stack>
#include <stdexcept>

#include "block_obj.h"

namespace mindie_llm {

/// 预先创建并初始化后放入对象池中 避免频繁的创建释放对象
///
/// \param poolSize 对象池的大小以及每次扩增的大小
template <typename T>
class ObjPool {
   public:
    explicit ObjPool<T>(uint64_t poolSize, std::function<std::shared_ptr<T>()> createFunc)
        : poolSize_(poolSize), createFunction_(createFunc) {
        for (uint64_t i = 0; i < poolSize_; i++) {
            objPool_.push(createFunction_());
        }
    }

    ~ObjPool() = default;

    // 从池中获取一个对象
    std::shared_ptr<T> AcquireObj() {
        if (objPool_.empty()) {
            IncreaseCapcity();
        }

        std::shared_ptr<T> blockObj = objPool_.top();
        objPool_.pop();
        return blockObj;
    }

    // 将对象重新入池
    void FreeObj(std::shared_ptr<T> &blockObj) {
        if (blockObj == nullptr) {
            throw std::runtime_error("FreeObj Error: attempt to free a null blockObj.");
        }
        if (GetFreeObjNum() == poolSize_) {
            throw std::runtime_error("no ObjPool is used, but caller holds an object to free. must be a bug!");
        }

        blockObj->ResetBlockObj();
        objPool_.push(blockObj);
        // reset shared pointer blockObj to nullptr
        blockObj.reset();
    }

    // 获取可用对象的个数
    uint64_t GetFreeObjNum() const { return objPool_.size(); }

    uint64_t GetPoolSize() const { return poolSize_; }

   private:
    // 池中没有可用的对象时，创建更多的对象(增加个数为 poolSize_)
    void IncreaseCapcity() {
        if (poolSize_ > std::numeric_limits<uint64_t>::max() / capcityMultiplier_) {
            throw std::overflow_error("Cannot increase capacity: size would overflow.");
        }
        for (uint64_t i = poolSize_; i < capcityMultiplier_ * poolSize_; i++) {
            objPool_.push(createFunction_());
        }
        poolSize_ *= capcityMultiplier_;
    }

    uint64_t poolSize_;

    std::function<std::shared_ptr<T>()> createFunction_;

    std::stack<std::shared_ptr<T>> objPool_;  // 对象池

    const uint64_t capcityMultiplier_ = 2;
};
using BlockObjPoolSPtr = std::shared_ptr<ObjPool<BlockObj>>;
}  // namespace mindie_llm

#endif
