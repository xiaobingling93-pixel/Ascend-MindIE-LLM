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

#ifndef ATB_SPEED_BASE_MAPPING_H
#define ATB_SPEED_BASE_MAPPING_H
#include <hccl/hccl.h>
#include "atb_speed/log.h"
#include "models/base/param/param_utils.h"
#include "operations/fusion/parallel_info.h"

namespace atb_speed {
namespace base {

/// The different realizations of expert parallelism strategies
enum class MoeExpertParallelDegree : uint32_t {
    /// The expert parallelism where experts are deterministic on each of the devices
    STATIC = 0,
    /// The expert parallelism where experts on each device are chosen upon expected
    /// "workload" of each expert to ideally even out the amount of calculation on each device
    DYNAMIC,
    /// The mixture of static and dynamic Ep strategies, where static Ep is applied for Prefill Stage
    /// and dynamic Ep is applied for Decode Stage
    DUO_GRAPH
};

/// The different realizations of data parallelism strategies
enum class AttnDataParallelDegree : uint32_t {
    /// The data parallelism where weights are duplicated across devices in the same Dp communication group
    OUTER = 0,
    /// The data parallelism where weights are loaded just as if tensor parallelism were applied, yet data parallelism
    /// mechanism is applied in calculation
    INNER
};

/// Defines different types of parallelism and corresponding modules
enum ParallelType : uint32_t {
    WORD_EMBED_TP = 0,
    WORD_EMBED_DP,
    LM_HEAD_TP,
    LM_HEAD_DP,
    ATTN_TP,
    ATTN_DP,
    ATTN_CP,
    ATTN_PREFIX_CACHE_CP,
    ATTN_INNER_SP,
    ATTN_O_PROJ_TP,
    ATTN_O_PROJ_DP,
    MLP_TP,
    MLP_DP,
    MOE_TP,
    MOE_EP,
    MOE_EP_INTER_NODE,
    MOE_EP_INTRA_NODE,
    DENSE_TP,
    PARALLEL_TYPE_END,
    DYNAMIC_EPLB,
};

class Mapping {
public:
    /// Global world size
    uint32_t worldSize_;
    /// An indicator that shows whether commDomains are assigned to each parallel strategy
    bool isInitialized_ = false;

    /// Convert and `nlohmann::json` object to a `Mapping` object
    /// \param paramJson An `nolhmann::json` object holds all the required parameters.
    void ParseParam(const nlohmann::json &paramJson);
    /// Add a `ParallelInfo` strategy into `parallelStrategies_` with key `parallelType`
    /// \param parallelType The key of the strategy
    /// \param parallelInfo A `ParallelInfo` object that holds info of the communication group
    void Register(ParallelType parallelType, atb_speed::common::ParallelInfo parallelInfo);
    /// Get a `ParallelInfo` object from `parallelStrategies_` by key `parallelType`
    /// \param parallelType The key of the strategy
    /// \throw Throws out of range error if key `parallelType` is not in `parallelStrategies_`
    /// \return a `ParallelInfo` object corresponding to the parallelism strategy of the target module
    const atb_speed::common::ParallelInfo Get(ParallelType parallelType) const;
    /// Initialize the communication group of each parallelism strategy
    /// \param defaultBackend The communication bacekdn
    /// \return A flag indicating whether the communication domain is created successfully
    void InitGlobalCommDomain(std::string defaultBackend);

private:
    /// A map holds a `ParallelInfo` object and corresponding module
    std::map<ParallelType, atb_speed::common::ParallelInfo> parallelStrategies_;
    /// Number of devices in the current node
    uint32_t localWorldSize_ = 0;
    /// Global rank
    uint32_t rank_;
    /// The default communication backend, currently support `hccl` and `lccl`
    std::string defaultBackend_ = "";
    /// Path of the file contains devices' Ip and rank info to construct communication groups
    std::string rankTableFile_ = "";
    /// The global communication in layerwise-disaggregated
    std::string lwdGlobalComm_ = "";
};

} // namespace base
} // namespace atb_speed


#endif