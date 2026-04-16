/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "single_req_self_develop_infer_interface.h"

#include "common_util.h"
#include "endpoint_def.h"
#include "http_rest_resource.h"
#include "infer_tokenizer.h"
#include "parameters_checker.h"
#include "parse_protocol.h"
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
SingleReqSelfDevelopInferInterface::SingleReqSelfDevelopInferInterface(
    const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute,
    const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : SingleReqInferInterfaceBase{singleLLMReqHandlerBase, isReCompute, loraConfigs} {}

bool SingleReqSelfDevelopInferInterface::CheckTokenInput(nlohmann::ordered_json &body, std::string &msg) {
    if (body["input_id"].is_null()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "input_id not found. The requestId is " << requestId_);
        msg = "Input_id should not be empty";
        return false;
    }
    if (!body["input_id"].is_array()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "input_id must be an array.");
        msg = "Parameter input_id must be an array";
        return false;
    }
    const int maxToken = 1024 * 1024;
    auto &dataJson = body["input_id"];
    for (size_t i = 0; i < dataJson.size(); ++i) {
        if (!dataJson[i].is_number_integer()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Data element must be integer type. The requestId is " << requestId_);
            msg = "Parameter input_id must be integer";
            return false;
        }
        if (dataJson[i] < 0 || dataJson[i] > maxToken) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Data element must be in range [0, " << maxToken << "], but got " << dataJson[i]
                                                            << ". The requestId is " << requestId_);
            msg = "Data element in param input_id is invalid";
            return false;
        }
        reqTokens_.push_back(dataJson[i]);
    }
    return true;
}

bool SingleReqSelfDevelopInferInterface::CheckTextInput(nlohmann::ordered_json &body, std::string &msg,
                                                        uint64_t &timestamp) {
    if (body["inputs"].is_null()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Inputs not found. The requestId is " << requestId_);
        msg = "Inputs must not be null";
        return false;
    }
    if (body["inputs"].type() == OrderedJson::value_t::string) {
        inputParam->textInput = body["inputs"];
    } else if (body["inputs"].type() == OrderedJson::value_t::array) {
        if (body["inputs"].empty()) {
            msg = "Inputs arr is empty";
            return false;
        }
        if (!CheckMultimodalUrlFromJson(body["inputs"], msg)) {
            return false;
        }
        inputParam->textInput = body["inputs"].dump();
    } else {
        msg = "The type of inputs is abnormal";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
        return false;
    }
    std::string errorMsg = "";
    std::u16string utf16 = GetU16Str(inputParam->textInput, &errorMsg);
    if (!errorMsg.empty()) {
        msg = "Failed to check the input text. ";
        msg += errorMsg;
        return false;
    }
    if (utf16.length() == 0 || utf16.length() > GetMaxInputLen()) {
        msg = "Inputs must be necessary and data type must be string and length in (0, " +
              std::to_string(GetMaxInputLen()) + "]";
        msg += ", but the length of inputs is " + std::to_string(utf16.length());
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
        return false;
    }

    if (this->isReCompute_) {
        // when recompute, tokenids can be directly parsed from input, no need to do tokenize
        if (!this->GetTokensFromInput(inputParam->textInput, reqTokens_, this->respTokenMap[SPECIAL_SEQ_ID_PRESET],
                                      msg)) {
            msg = "Failed to get token from input: " + msg;
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, ABNORMAL_TRANSMISSION_ERROR), msg);
            return false;
        }
    } else {
        auto encodeSpan = PROF(INFO, Domain("Request").Resource(requestId_.c_str()).SpanStart("encode"));
        auto status =
            TokenizerProcessPool::GetInstance().Encode(inputParam->textInput, reqTokens_, ENCODE_FLAG, timestamp);
        if (!status.IsOk()) {
            msg = status.StatusMsg();
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, LOCAL_INVOKING_ERROR),
                       msg << ". The requestId is " << requestId_);
            return false;
        }
        PROF(encodeSpan.Metric("recvTokenSize", reqTokens_.size()));
    }
    return true;
}

bool SingleReqSelfDevelopInferInterface::ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg,
                                                                    uint64_t &timestamp) {
    try {
        // inputs
        if (body.contains("inputs") && body.contains("input_id")) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Could not have inputs an input_id at the same time " << requestId_);
            msg = std::string("Input conflict.");
            return false;
        }
        if (body.contains("inputs") || this->isReCompute_) {
            if (!CheckTextInput(body, msg, timestamp)) {
                return false;
            }
        } else if (body.contains("input_id")) {
            if (!CheckTokenInput(body, msg)) {
                return false;
            }
        } else {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Could not find input data. The requestId is " << requestId_);
            msg = std::string("Not found inputs.");
            return false;
        }
        if (reqTokens_.size() == 0 || reqTokens_.size() > MAX_TOKENS_NUM) {
            msg = "Inputs token length must be in (0, " + std::to_string(MAX_TOKENS_NUM) + "]";
            msg += ", but got " + std::to_string(reqTokens_.size());
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
            return false;
        }
        return true;
    } catch (...) {
        msg = "Check self develop req parameter error";
        return false;
    }
}
bool SingleReqSelfDevelopInferInterface::SetupInferParams(RequestSPtr tmpReq, std::string &msg) {
    auto paramCheckRet =
        JsonParse::CheckOptionalItemType(reqJsonBody_, "parameters", OrderedJson::value_t::object, msg);
    if (!paramCheckRet.isCorrectType) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "value type of \"parameters\" is invalid, requestId is " << requestId_);
        return false;
    }
    if (!(AssignTemperature(reqJsonBody_["parameters"], tmpReq, msg, false) &&
          AssignTopK(reqJsonBody_["parameters"], tmpReq, msg) &&
          AssignTopP(reqJsonBody_["parameters"], tmpReq, msg, false) &&
          AssignTypicalP(reqJsonBody_["parameters"], tmpReq, msg) &&
          AssignDoSample(reqJsonBody_["parameters"], tmpReq, msg) &&
          AssignSeed(reqJsonBody_["parameters"], tmpReq, msg) &&
          AssignRepetitionPenalty(reqJsonBody_["parameters"], tmpReq, msg) &&
          AssignWatermark(reqJsonBody_["parameters"], tmpReq, msg) &&
          AssignPriority(reqJsonBody_["parameters"], tmpReq, msg) &&
          AssignMaxNewTokens(reqJsonBody_["parameters"], inputParam, msg) &&
          AssignTimeout(reqJsonBody_["parameters"], inputParam, msg) &&
          AssignDetails(reqJsonBody_["parameters"], inputParam, msg) && AssignStream(reqJsonBody_, inputParam, msg))) {
        return false;
    }
    return true;
}

bool SingleReqSelfDevelopInferInterface::BuildResponseJson(ResponseSPtr response,
                                                           const std::vector<BestNTokens> &tempTokens,
                                                           RespBodyQueue &jsonStrings, const uint64_t &timestamp) {
    bool res = true;
    if (inputParam->streamMode) {
        if (!ProcessResponseStream(response, tempTokens, jsonStrings, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       "Failed to process selfDevelop response stream");
            return false;
        }
    } else {
        if (!ProcessResponseSingle(response, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       "Failed to process selfDevelop response single");
            return false;
        }
        res = EncodeSelfDevelopResponse(jsonStrings);
    }
    return res;
}

std::string SingleReqSelfDevelopInferInterface::ChangeUtf8Str(std::string &input) const {
    try {
        return CleanStringForJson(input);
    } catch (const std::exception &e) {
        // 处理转换错误
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                   "Failed to change str to utf8. " << e.what());
        return " ";
    }
}

bool SingleReqSelfDevelopInferInterface::EncodeSelfDevelopResponse(RespBodyQueue &jsonStrs) {
    uint64_t seqId;
    if (!GetUniqueSequenceId(seqId)) {
        return false;
    }
    try {
        OrderedJson jsonObj;
        jsonObj["generated_text"] = ChangeUtf8Str(fullTextMap[seqId]);
        ;
        if (inputParam->showDetails) {
            jsonObj["details"]["finish_reason"] = finishReasonMap[seqId];
            jsonObj["details"]["generated_tokens"] = postTokenIdMap[seqId].size();
            jsonObj["details"]["generated_token_ids"] = postTokenIdMap[seqId];
            if (request_->seed.has_value()) {
                jsonObj["details"]["seed"] = request_->seed.value();
            } else {
                jsonObj["details"]["seed"] = nullptr;
            }
        }
        jsonStrs.push(jsonObj.dump());
        return true;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ENCODE_DECODE_ERROR),
                   "Failed to encode infer response");
        return false;
    }
}

bool SingleReqSelfDevelopInferInterface::EncodeSelfDevelopStreamResponse(RespBodyQueue &jsonStrings) noexcept {
    uint64_t seqId;
    if (!GetUniqueSequenceId(seqId)) {
        return false;
    }
    try {
        OrderedJson jsonObj;
        auto &metrics = singleLLMReqHandlerBase_->GetMetrics();
        if (!metrics.callbackIndexQue.empty()) {
            uint8_t callbackMetricsStage = metrics.callbackIndexQue.front();
            metrics.callbackIndexQue.pop();
            if (callbackMetricsStage == PREFILL_CALLBACK_METRICS_TAG) {
                jsonObj["prefill_time"] = metrics.firstTokenCost;
                jsonObj["decode_time"] = nullptr;
            } else if (callbackMetricsStage == DECODE_CALLBACK_METRICS_TAG &&
                       metrics.decodeTime.size() > metrics.callbackIndex) {
                jsonObj["prefill_time"] = nullptr;
                jsonObj["decode_time"] = metrics.decodeTime[metrics.callbackIndex++];
            }
        }

        if (!isEnd) {
            jsonObj["token"]["id"] = postSingleTokenMap[seqId];
            jsonObj["token"]["text"] = inputParam->postSingleText[seqId];
            std::string outputStr = "data: " + jsonObj.dump() + "\n\n";
            jsonStrings.push(outputStr);
            return true;
        }
        jsonObj["generated_text"] = ChangeUtf8Str(fullTextMap[seqId]);
        if (inputParam->showDetails) {
            jsonObj["details"]["finish_reason"] = finishReasonMap[seqId];
            jsonObj["details"]["generated_tokens"] = postTokenIdMap[seqId].size();
            if (request_->seed.has_value()) {
                jsonObj["details"]["seed"] = request_->seed.value();
            } else {
                jsonObj["details"]["seed"] = nullptr;
            }
        } else {
            jsonObj["details"] = nullptr;
        }
        jsonObj["token"]["id"] = postSingleTokenMap[seqId];
        jsonObj["token"]["text"] = nullptr;

        std::string outputStr = "data: " + jsonObj.dump() + "\n\n";
        jsonStrings.push(outputStr);
        return true;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ENCODE_DECODE_ERROR),
                   "Failed to encode infer response");
        return false;
    }
}

void SingleReqSelfDevelopInferInterface::SendStreamResponse(RespBodyQueue &jsonStrings) {
    if (!EncodeSelfDevelopStreamResponse(jsonStrings)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to encode buffer");
        return;
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Response has ended is " << isEnd << ", requestId is " << requestId_);
}

void SingleReqSelfDevelopInferInterface::SetDMIReComputeBuilder() {
    singleLLMReqHandlerBase_->SetDMIReComputeBuildCallBack(
        std::bind(&SingleReqSelfDevelopInferInterface::BuildSelfDevelopReComputeBody, this, std::placeholders::_1));
}

std::string SingleReqSelfDevelopInferInterface::BuildSelfDevelopReComputeBody(const std::vector<BestNTokens> &tokens) {
    OrderedJson newReqJsonObj;
    // Get tokens in non-stream mode
    if (tokens.size() != 0) {
        ConvertTokenToMap(tokens);
    }
    newReqJsonObj["inputs"] = this->BuildReComputeInput();
    newReqJsonObj["stream"] = this->inputParam->streamMode;
    newReqJsonObj["parameters"]["priority"] = request_->priority;
    newReqJsonObj["parameters"]["timeout"] = this->inputParam->timeout;
    if (this->inputParam->maxNewTokens != MAX_NEW_TOKENS_DFT) {
        newReqJsonObj["parameters"]["max_new_tokens"] = this->inputParam->maxNewTokens;
    }
    if (request_->temperature.has_value()) {
        newReqJsonObj["parameters"]["temperature"] = request_->temperature.value();
    }
    if (request_->topK.has_value()) {
        newReqJsonObj["parameters"]["top_k"] = request_->topK.value();
    }
    if (request_->topP.has_value()) {
        newReqJsonObj["parameters"]["top_p"] = request_->topP.value();
    }
    if (request_->doSample.has_value()) {
        newReqJsonObj["parameters"]["do_sample"] = request_->doSample.value();
    }
    if (request_->seed.has_value()) {
        newReqJsonObj["parameters"]["seed"] = request_->seed.value();
    }
    if (request_->repetitionPenalty.has_value()) {
        newReqJsonObj["parameters"]["repetition_penalty"] = request_->repetitionPenalty.value();
    }
    if (request_->watermark.has_value()) {
        newReqJsonObj["parameters"]["watermark"] = request_->watermark.value();
    }
    newReqJsonObj["parameters"]["details"] = this->inputParam->showDetails;
    if (request_->typicalP.has_value()) {
        newReqJsonObj["parameters"]["typical_p"] = request_->typicalP.value();
    }
    return newReqJsonObj.dump();
}
}  // namespace mindie_llm
