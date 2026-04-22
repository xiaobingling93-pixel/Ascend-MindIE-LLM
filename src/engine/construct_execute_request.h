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

#ifndef CONSTRUCT_EXECUTE_REQUEST_H
#define CONSTRUCT_EXECUTE_REQUEST_H

#include "executor/executor_interface.h"
#include "sequence_group.h"
#include "sequence_group_meta_data.h"

namespace mindie_llm {
class ConstructExecuteRequest {
   public:
    static void ConstructExecuteModelRequest(ExecuteModelRequestPtr &modelRequest, SequenceGroupMetaDatas &metadatas,
                                             SchedulerOutputs &scOut, size_t dpRankId);

    static PullKVRequestPtr ConstructPullKVRequest(SequenceGroupMetaDatas &seqGroupMetadata);

   private:
    static model_execute_data::SequenceData MakeSequenceData(SequenceData &metaData);
    static void ConstructProtoMeta(const SequenceGroupMetaData &metaData,
                                   model_execute_data::SequenceGroupMetadata &protoMeta, bool isPrefill);
    static void LwdConstructCloudProtoMeta(const SequenceGroupMetaData &metaData,
                                           model_execute_data::SequenceGroupMetadata &protoMeta, bool isPrefill);
    static model_execute_data::ForwardType ConvertToProtoForwardType(ForwardMode fMode);
    static void ConstructSampleParam(model_execute_data::SamplingParams &sampleParams,
                                     const SequenceGroupMetaData &metaData);
    static void ConstructPrefillData(const SequenceGroupMetaData &metaData,
                                     model_execute_data::SequenceGroupMetadata &protoMeta);
    static void ConstructChunkedPrefillParam(const SequenceGroupMetaData &metaData,
                                             model_execute_data::SequenceGroupMetadata &protoMeta);
    static void ClearBeamParam4ChunkedPrefill(model_execute_data::SamplingParams &sampleParams,
                                              model_execute_data::SequenceGroupMetadata &protoMeta);
};

}  // namespace mindie_llm

#endif
