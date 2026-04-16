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

#ifndef SEQUENCE_GROUP_FROM_INFER
#define SEQUENCE_GROUP_FROM_INFER

#include "config_info.h"
#include "id_utils.h"
#include "request_response/request.h"
#include "sequence_group.h"
namespace mindie_llm {

class SeqGroupBuilderFromInferReq {
   public:
    // Build a sequence group from an infer request.
    static SequenceGroupSPtr BuildSequenceGroup(RequestSPtr request, SchedulerConfigSPtr schedulerConfig,
                                                size_t rankId);

   private:
    static SamplingParamsSPtr GetSampleParamFromInferRequest(RequestSPtr request, bool chooseV2BlockManager);

    static SequenceSPtr InitSeqFromInferRequest(RequestSPtr request, SchedulerConfigSPtr schedulerConfig);

    static void InitSeqGrpFromInferRequest(RequestSPtr request, RequestIdNew inferReqId, SequenceGroupSPtr &seqGroup,
                                           SchedulerConfigSPtr schedulerConfigSptr);
};
}  // namespace mindie_llm
#endif
