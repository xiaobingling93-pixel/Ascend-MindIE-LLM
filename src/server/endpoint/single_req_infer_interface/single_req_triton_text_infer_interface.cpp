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
#include "single_req_triton_text_infer_interface.h"

#include "common_util.h"
#include "config_manager_impl.h"
#include "endpoint_def.h"
#include "http_rest_resource.h"
#include "infer_tokenizer.h"
#include "parameters_checker.h"
#include "parse_protocol.h"

using OrderedJson = nlohmann::ordered_json;
namespace mindie_llm {
SingleReqTritonTextInferInterface::SingleReqTritonTextInferInterface(
    const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool stream, std::string model,
    bool isReCompute, const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : SingleReqInferInterfaceBase{singleLLMReqHandlerBase, isReCompute, loraConfigs} {
    inputParam->streamMode = stream;
    this->model = model;
}

static bool AssignFirstTokenCost(const OrderedJson &jsonObj, Metrics &reMetrics, std::string &error) noexcept {
    const std::string key = "firstTokenCost";
    auto res = JsonParse::CheckOptionalItemType(jsonObj, key, OrderedJson::value_t::number_unsigned, error);
    if (!res.isCorrectType || !res.isPresent) {
        return false;
    }
    uint64_t value = jsonObj[key];
    if (value > std::numeric_limits<size_t>::max()) {
        std::stringstream ss;
        ss << "Parameter firstTokenCost exceeds the max value of type size_t, which is "
           << std::numeric_limits<size_t>::max();
        error = ss.str();
        return false;
    }

    reMetrics.firstTokenCost = value;
    return true;
}

static bool AssignDecodeTime(const OrderedJson &jsonObj, Metrics &reMetrics, std::string &error,
                             uint64_t respTokensSize) noexcept {
    const std::string key = "decodeTime";
    auto res = JsonParse::CheckOptionalItemType(jsonObj, key, OrderedJson::value_t::array, error);
    if (!res.isCorrectType || !res.isPresent) {
        return false;
    }
    uint64_t decodeTimeNum = jsonObj[key].size();
    if (respTokensSize > 0 && decodeTimeNum != respTokensSize - 1) {
        std::stringstream ss;
        ss << "Parameter decodeTime length is invalid.";
        error = ss.str();
        return false;
    }
    for (uint64_t i = 0; i < decodeTimeNum; i++) {
        if (!jsonObj[key][i].is_number_unsigned() || jsonObj[key][i] > std::numeric_limits<size_t>::max()) {
            std::stringstream ss;
            ss << "Parameter decodeTime exceeds the max value of size_t type. ";
            error = ss.str();
            return false;
        }
        reMetrics.decodeTime.push_back(jsonObj[key][i]);
    }
    return true;
}

void SingleReqTritonTextInferInterface::UpdateResponseBody(const uint64_t &curSeqId, OrderedJson &jsonObj,
                                                           std::string &textOutput, std::string &jsonString) {
    FillResponseBody(curSeqId, jsonObj, textOutput, respTokenMap[curSeqId].size());
    if (inputParam->streamMode && textOutput != "") {
        inputParam->prevDecodeIndex[curSeqId] = inputParam->currentDecodeIndex[curSeqId];
        inputParam->currentDecodeIndex[curSeqId] = respTokenMap[curSeqId].size();
    }
    jsonString = jsonObj.dump(-1, ' ', false, OrderedJson::error_handler_t::replace);
}

bool SingleReqTritonTextInferInterface::BuildResponseJson(ResponseSPtr response,
                                                          [[maybe_unused]] const std::vector<BestNTokens> &tokenIdList,
                                                          RespBodyQueue &jsonStrings, const uint64_t &timestamp) {
    if (!response->isEos) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Infer response not finished. The requestId is " << requestId_);
        if (!inputParam->streamMode) {
            return true;
        }
    } else {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Infer response finished. The requestId is " << requestId_);
    }
    uint64_t seqId;
    if (!GetUniqueSequenceId(seqId)) {
        return false;
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "RespTokens.size is " << respTokenMap[seqId].size() << ".");

    auto lastResponse = response->isEos;
    if (lastResponse) {
        finishReasonMap[seqId] = GetFinishReasonStr(eosMap[seqId]);
    }
    std::string textOutput = "";
    std::string jsonString;
    auto &serverConfig = GetServerConfig();
    Status status = DecodeOutputText(seqId, textOutput, respTokenMap[seqId], serverConfig.fullTextEnabled, timestamp);
    if (status.IsOk()) {
        ParseDetokenizedOutput(textOutput, seqId, false);
        TruncateOutputText(response, textOutput);
        OrderedJson jsonObj;
        UpdateResponseBody(seqId, jsonObj, textOutput, jsonString);
        if (inputParam->streamMode) {
            std::string respStr = "data:" + std::move(jsonString) + "\n\n";
            jsonString = std::move(respStr);
        }
        jsonStrings.push(jsonString);
    } else {
        ULOG_ERROR(
            SUBMODLE_NAME_ENDPOINT,
            GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ENCODE_DECODE_ERROR),
            "Convert token to text output failed. The requestId is " << requestId_ << ". " << status.StatusMsg());
        return false;
    }
    return true;
}
bool SingleReqTritonTextInferInterface::SetupInferParams(RequestSPtr tmpReq, std::string &msg) {
    auto paramCheckRet =
        JsonParse::CheckOptionalItemType(reqJsonBody_, "parameters", OrderedJson::value_t::object, msg);
    if (!paramCheckRet.isCorrectType) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Parameters in request param invalid for requestId is " << requestId_);
        return false;
    }
    const auto &params = reqJsonBody_["parameters"];
    if (!(AssignTemperature(params, tmpReq, msg, false) && AssignTopK(params, tmpReq, msg, true) &&
          AssignTopP(params, tmpReq, msg) && AssignTypicalP(params, tmpReq, msg) &&
          AssignDoSample(params, tmpReq, msg) && AssignSeed(params, tmpReq, msg) &&
          AssignRepetitionPenalty(params, tmpReq, msg) && AssignWatermark(params, tmpReq, msg) &&
          AssignPriority(params, tmpReq, msg) && AssignMaxNewTokens(params, inputParam, msg) &&
          AssignDetails(params, inputParam, msg) && AssignBatchSize(params, inputParam, msg) &&
          AssignTimeout(params, inputParam, msg))) {
        return false;
    }
    // 重计算时，需要从请求体里拿firstTokenCost和decodeTime
    if (this->isReCompute_) {
        Metrics reMetrics;
        if (!AssignFirstTokenCost(reqJsonBody_["parameters"], reMetrics, msg) ||
            !AssignDecodeTime(reqJsonBody_["parameters"], reMetrics, msg,
                              this->respTokenMap[SPECIAL_SEQ_ID_PRESET].size())) {
            return false;
        }
        singleLLMReqHandlerBase_->SetRecomputeMetrics(reMetrics);
    }
    return true;
}

bool SingleReqTritonTextInferInterface::ParseRequestId(std::string &error) noexcept {
    auto idCheckRet = JsonParse::CheckOptionalItemType(reqJsonBody_, "id", OrderedJson::value_t::string, error);
    std::regex pattern("^[a-zA-Z0-9_-]{1,256}$");
    if (!idCheckRet.isCorrectType) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                   "Parse id in request failed. The requestId is " << requestId_);
        return false;
    }

    if (idCheckRet.isPresent) {
        std::string requestId = reqJsonBody_["id"];
        inputParam->userInputId = requestId;
        if (!std::regex_match(requestId, pattern)) {
            error = std::string("The id can contain only digits, letters, underscores(_), hyphens(-) and ")
                        .append("no more than ")
                        .append(std::to_string(MAX_INPUT_ID_LENGTH))
                        .append(" words in length.");
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       error << ". The requestId is " << requestId_);
            return false;
        }
        RequestIdNew requestIdTmp = requestId_;
        if (!singleLLMReqHandlerBase_->GetContextRequestId(requestId_)) {
            requestId_ = requestIdTmp;
            userInputIdMap_.Insert(inputParam->userInputId.value(), requestId_);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
                      "Receive user id " << inputParam->userInputId.value() << ", request id " << requestId_);
        }
    }

    return true;
}

bool SingleReqTritonTextInferInterface::ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg,
                                                                   uint64_t &timestamp) {
    try {
        if (!ParseRequestId(msg)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                       "Parse id in request failed");
            return false;
        }
        if (!body.contains("text_input") || body["text_input"].is_null()) {
            msg = std::string("The text_input not found");
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
            return false;
        }
        if (body["text_input"].type() == OrderedJson::value_t::string) {
            inputParam->textInput = body["text_input"];
        } else if (body["text_input"].type() == OrderedJson::value_t::array) {
            if (body["text_input"].empty()) {
                msg = "Parameter text_input arr is empty";
                return false;
            }
            if (!CheckMultimodalUrlFromJson(body["text_input"], msg)) {
                return false;
            }
            inputParam->textInput = body["text_input"].dump();
        } else {
            msg = "The type of text_input is abnormal. The requestId is " + requestId_;
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
            msg = "text_input must be necessary and data type must be string and length in (0, " +
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
                           GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, ABNORMAL_TRANSMISSION_ERROR),
                           msg);
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

        if (reqTokens_.size() == 0 || reqTokens_.size() > MAX_TOKENS_NUM) {
            msg = "text_input token length must be in (0, " + std::to_string(MAX_TOKENS_NUM) + "]";
            msg += ", but got " + std::to_string(reqTokens_.size());
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
            return false;
        }
        return true;
    } catch (...) {
        msg = "Check triton text req parameter error";
        return false;
    }
}

void SingleReqTritonTextInferInterface::TruncateOutputText(const ResponseSPtr &response, std::string &textOutput) {
    if (!response->isEos) {
        return;
    }

    int64_t truncationIndex = response->responseContents[0].truncationIndex;  // only access item 0 ?
    if (truncationIndex == 0) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_WARNING),
                  "Failed to get truncationIndices");
        return;
    }

    std::u16string utf16Output = GetU16Str(textOutput);
    std::string errorMsg = "";
    if (inputParam->streamMode) {
        if (truncationIndex < 0 && utf16Output.length() >= static_cast<size_t>(std::abs(truncationIndex)) &&
            request_->includeStopStrInOutput.value_or(true)) {
            textOutput = TransformTruncation(utf16Output, 0, utf16Output.length() + truncationIndex, &errorMsg);
        }
    } else if (truncationIndex < 0 && utf16Output.length() >= static_cast<size_t>(std::abs(truncationIndex))) {
        textOutput = TransformTruncation(utf16Output, 0, utf16Output.length() + truncationIndex, &errorMsg);
    }
    if (!errorMsg.empty()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, LOCAL_INVOKING_ERROR), errorMsg);
        return;
    }
}

Status SingleReqTritonTextInferInterface::DecodeOutputText(const uint64_t &curSeqId, std::string &textOutput,
                                                           std::vector<int64_t> &tokens, bool showFullText,
                                                           const uint64_t &timestamp) {
    Status status;
    if (inputParam->streamMode && !showFullText) {
        status =
            TokenizerProcessPool::GetInstance().DecodeOne(tokens, textOutput, inputParam->prevDecodeIndex[curSeqId],
                                                          inputParam->currentDecodeIndex[curSeqId], timestamp);
    } else {
        status = TokenizerProcessPool::GetInstance().Decode(tokens, textOutput, timestamp, false);
    }
    return status;
}

void SingleReqTritonTextInferInterface::FillResponseBody(const uint64_t &curSeqId, OrderedJson &respBody,
                                                         const std::string &textOutput, uint32_t tokenCnt) noexcept {
    respBody["id"] = requestId_;
    if (inputParam->userInputId.has_value()) {
        respBody["id"] = inputParam->userInputId.value();
    }

    respBody["model_name"] = model;
    respBody["model_version"] = nullptr;
    respBody["text_output"] = textOutput;
    if (inputParam->showDetails) {
        if (!finishReasonMap[curSeqId].empty()) {
            respBody["details"]["finish_reason"] = finishReasonMap[curSeqId];
        }
        respBody["details"]["generated_tokens"] = tokenCnt;
        respBody["details"]["first_token_cost"] = nullptr;
        respBody["details"]["decode_cost"] = nullptr;

        auto &metrics = singleLLMReqHandlerBase_->GetMetrics();
        if (inputParam->streamMode && !metrics.callbackIndexQue.empty()) {
            respBody["details"]["batch_size"] = metrics.batchSize.back();
            respBody["details"]["queue_wait_time"] = metrics.queueWaitTime.back();
            uint8_t callbackMetricsStage = metrics.callbackIndexQue.front();
            metrics.callbackIndexQue.pop();
            if (callbackMetricsStage == PREFILL_CALLBACK_METRICS_TAG) {
                respBody["prefill_time"] = metrics.firstTokenCost;
                respBody["decode_time"] = nullptr;
            } else if (callbackMetricsStage == DECODE_CALLBACK_METRICS_TAG &&
                       metrics.decodeTime.size() > metrics.callbackIndex) {
                respBody["prefill_time"] = nullptr;
                respBody["decode_time"] = metrics.decodeTime[metrics.callbackIndex++];
            }
        }
        if (!inputParam->streamMode) {
            respBody["details"]["batch_size"] = metrics.batchSize;
            respBody["details"]["queue_wait_time"] = metrics.queueWaitTime;
        }
    }
}

void SingleReqTritonTextInferInterface::SetDMIReComputeBuilder() {
    singleLLMReqHandlerBase_->SetDMIReComputeBuildCallBack(
        std::bind(&SingleReqTritonTextInferInterface::BuildTritonTextReComputeBody, this, std::placeholders::_1));
}

std::string SingleReqTritonTextInferInterface::BuildTritonTextReComputeBody(const std::vector<BestNTokens> &tokens) {
    OrderedJson newReqJsonObj;
    // Get tokens in non-stream mode
    if (tokens.size() != 0) {
        ConvertTokenToMap(tokens);
    }
    newReqJsonObj["text_input"] = this->BuildReComputeInput();
    if (this->inputParam->userInputId.has_value()) {
        newReqJsonObj["id"] = this->inputParam->userInputId.value();
    }
    newReqJsonObj["parameters"]["details"] = this->inputParam->showDetails;
    if (this->inputParam->maxNewTokens != MAX_NEW_TOKENS_DFT) {
        newReqJsonObj["parameters"]["max_new_tokens"] = this->inputParam->maxNewTokens;
    }
    if (request_->doSample.has_value()) {
        newReqJsonObj["parameters"]["do_sample"] = request_->doSample.value();
    }
    if (request_->repetitionPenalty.has_value()) {
        newReqJsonObj["parameters"]["repetition_penalty"] = request_->repetitionPenalty.value();
    }
    if (request_->seed.has_value()) {
        newReqJsonObj["parameters"]["seed"] = request_->seed.value();
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
    if (this->inputParam->batchSize > 0) {
        newReqJsonObj["parameters"]["batch_size"] = this->inputParam->batchSize;
    }
    if (request_->typicalP.has_value()) {
        newReqJsonObj["parameters"]["typical_p"] = request_->typicalP.value();
    }
    if (request_->watermark.has_value()) {
        newReqJsonObj["parameters"]["watermark"] = request_->watermark.value();
    }
    newReqJsonObj["parameters"]["priority"] = request_->priority;
    newReqJsonObj["parameters"]["timeout"] = this->inputParam->timeout;
    newReqJsonObj["parameters"]["firstTokenCost"] = singleLLMReqHandlerBase_->GetMetrics().firstTokenCost;
    newReqJsonObj["parameters"]["decodeTime"] = singleLLMReqHandlerBase_->GetMetrics().decodeTime;
    return newReqJsonObj.dump();
}
}  // namespace mindie_llm
