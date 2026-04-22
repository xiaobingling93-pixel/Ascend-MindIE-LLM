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
#include "single_req_triton_token_infer_interface.h"

#include <regex>
#include <sstream>
#include <utility>

#include "common_util.h"
#include "httplib.h"
#include "parameters_checker.h"
#include "parse_protocol.h"
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
const std::string SingleReqTritonTokenInferInterface::defOutputName{"output0"};

SingleReqTritonTokenInferInterface::SingleReqTritonTokenInferInterface(
    const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute,
    const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : SingleReqInferInterfaceBase{singleLLMReqHandlerBase, isReCompute, loraConfigs} {}

bool SingleReqTritonTokenInferInterface::CheckReqId(OrderedJson &body, std::string &msg) {
    const std::string key = "id";
    auto res = JsonParse::CheckOptionalItemType(body, key, OrderedJson::value_t::string, msg);
    if (!res.isCorrectType) {
        return false;
    }
    if (!res.isPresent) {
        return true;
    }

    std::string requestId = body["id"];
    inputParam->userInputId = requestId;
    std::regex pattern("^[a-zA-Z0-9_-]{1,256}$");
    if (!std::regex_match(requestId, pattern)) {
        msg = std::string("The id can contain only digits, letters, underscores(_), hyphens(-) and no more than ")
                  .append(std::to_string(MAX_INPUT_ID_LENGTH))
                  .append(" words in length.");
        return false;
    }
    RequestIdNew requestIdTmp = requestId_;
    if (!singleLLMReqHandlerBase_->GetContextRequestId(requestId_)) {
        requestId_ = requestIdTmp;
        userInputIdMap_.Insert(inputParam->userInputId.value(), requestId_);
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
                  "Receive user id " << inputParam->userInputId.value() << ", request id " << requestId_);
    }
    return true;
}

bool SingleReqTritonTokenInferInterface::ValidateAndPrepareReqToken(OrderedJson &body, std::string &msg,
                                                                    [[maybe_unused]] uint64_t &timestamp) {
    try {
        if (!body.contains("inputs") || body["inputs"].is_null() || !body["inputs"].is_array()) {
            msg = "Req should contain 'inputs' and the type should be array.";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
            return false;
        }
        auto &inputs = body["inputs"];
        auto inputSize = inputs.size();
        if (inputSize != 1UL) {
            msg = "Inputs count must be 1.";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Parse input only support 1 input. The requestId is " << requestId_);
            return false;
        }
        inputNames.reserve(inputSize);
        inputShape.reserve(inputSize);
        inputDataType.reserve(inputSize);
        outputNames.reserve(inputSize);
        if (!CheckReqInputName(inputs[0], msg) || !CheckReqInputShape(inputs[0], msg) ||
            !CheckReqInputDataType(inputs[0], msg) || !CheckReqInputData(inputs[0], msg)) {
            return false;
        }
        if (!CheckReqId(body, msg) || !CheckOutputs(body, msg)) {
            return false;
        }

        auto &inputsData = inputs[0]["data"];
        if (inputsData.size() > 0) {
            if (this->isReCompute_) {
                oriReqTokenLen_ = inputsData[0];
            } else {
                reqTokens_.push_back(inputsData[0]);
            }
            for (size_t i = 1; i < inputsData.size(); i++) {
                reqTokens_.push_back(inputsData[i]);
            }
        }

        return true;
    } catch (...) {
        msg = "Check triton token req parameter error";
        return false;
    }
}

bool SingleReqTritonTokenInferInterface::CheckOutputs(OrderedJson &body, std::string &msg) {
    if (!body.contains("outputs") || body["outputs"].is_null() || !body["outputs"].is_array()) {
        msg = "Req should contain 'outputs' and the type should be array.";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
        return false;
    }
    auto &outputs = body["outputs"];
    if (outputs.size() != inputDataType.size()) {
        std::stringstream ss;
        ss << "Outputs size should be " << inputDataType.size() << ", got " << outputs.size() << ".";
        msg = ss.str();
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   msg << " in request for requestId is " << requestId_);
        return false;
    }
    for (auto i = 0U; i < outputs.size(); i++) {
        if (!outputs[i].contains("name") || outputs[i]["name"].is_null() || !outputs[i]["name"].is_string()) {
            std::stringstream ss;
            ss << "Outputs name must be non-empty string.";
            msg = ss.str();
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Name in outputs(" << i << ") is not contained and the type should be string.");
            return false;
        }
        inputParam->outputNames.emplace_back(outputs[i]["name"]);
    }
    return true;
}
bool SingleReqTritonTokenInferInterface::SetupInferParams(RequestSPtr tmpReq, std::string &msg) {
    auto paramCheckRet =
        JsonParse::CheckOptionalItemType(reqJsonBody_["inputs"][0], "parameters", OrderedJson::value_t::object, msg);
    if (!paramCheckRet.isCorrectType) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Parameters in request param invalid for requestId is " << requestId_);
        return false;
    }
    const auto &params = reqJsonBody_["inputs"][0]["parameters"];
    if (!(AssignTemperature(params, tmpReq, msg, false) && AssignTopK(params, tmpReq, msg, true) &&
          AssignTopP(params, tmpReq, msg) && AssignDoSample(params, tmpReq, msg) && AssignSeed(params, tmpReq, msg) &&
          AssignRepetitionPenalty(params, tmpReq, msg) && AssignWatermark(params, tmpReq, msg) &&
          AssignPriority(params, tmpReq, msg) && AssignMaxNewTokens(params, inputParam, msg) &&
          AssignTimeout(params, inputParam, msg) && AssignDetails(params, inputParam, msg) &&
          AssignBatchSize(params, inputParam, msg))) {
        return false;
    }
    return true;
}

bool SingleReqTritonTokenInferInterface::BuildResponseJson(ResponseSPtr response,
                                                           [[maybe_unused]] const std::vector<BestNTokens> &tempTokens,
                                                           RespBodyQueue &jsonObjs,
                                                           [[maybe_unused]] const uint64_t &timestamp) {
    if (!response->isEos) {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Infer resp for requestId is " << requestId_ << " not finished.");
        return true;
    }

    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Infer resp for requestId is " << requestId_ << " finished.");
    OrderedJson respBody;
    auto localOutputNames = inputParam->outputNames;
    if (localOutputNames.empty()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "OutputNames of inputParam is empty");
        return false;
    }
    uint64_t seqId;
    if (!GetUniqueSequenceId(seqId)) {
        return false;
    }
    try {
        respBody["id"] = requestId_;
        if (inputParam->userInputId.has_value()) {
            respBody["id"] = inputParam->userInputId.value();
        }
        respBody["outputs"][0]["name"] = localOutputNames[0].empty() ? "output0" : localOutputNames[0];
        respBody["outputs"][0]["shape"] = {1, respTokenMap[seqId].size()};
        respBody["outputs"][0]["datatype"] = "UINT32";
        for (long &token : respTokenMap[seqId]) {
            respBody["outputs"][0]["data"].emplace_back(token);
        }
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Infer resp for requestId is " << requestId_ << " set response.");
        jsonObjs.push(respBody.dump());
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Error failed to get respBody outputs info");
        return false;
    }
    return true;
}

bool SingleReqTritonTokenInferInterface::CheckReqInputName(OrderedJson &body, std::string &msg) {
    if (!body.contains("name") || body["name"].is_null() || !body["name"].is_string()) {
        msg = "Cannot find name in inputs or its type is not string";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   msg << " for requestId is " << requestId_);
        return false;
    }
    std::string name = body["name"];
    if (name.size() > 256U) {
        msg = "The length of name set in inputs exceeds 256.";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   msg << " for requestId is " << requestId_);
        return false;
    }
    inputNames.emplace_back(name);
    return true;
}

bool SingleReqTritonTokenInferInterface::CheckReqInputShape(OrderedJson &body, std::string &msg) {
    if (!body.contains("shape") || body["shape"].is_null() || !body["shape"].is_array()) {
        msg = "Cannot find shape in inputs or its type is not array";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   msg << " for requestId is " << requestId_);
        return false;
    }
    auto &shapeObj = body["shape"];
    const auto count = shapeObj.size();
    for (auto i = 0U; i < count; i++) {
        if (!shapeObj[i].is_number_unsigned()) {
            msg = "Shape element must be integer type.";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       msg << " for requestId is " << requestId_);
            return false;
        }
    }

    auto validShape = (count == 1UL ||
                       ((count - 1U) == 1UL && shapeObj[0] == 1 && shapeObj[1] > 0 && (shapeObj[1] <= MAX_TOKENS_NUM)));
    if (!validShape) {
        msg = "Shape must be at most two-dimensional, and the last element's value must be in (0,";
        msg += std::to_string(MAX_TOKENS_NUM) + "].";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Only support shape with dim_count=1 for requestId is " << requestId_);
        return false;
    }

    std::vector<int64_t> localShape{};
    if (count == 1UL) {
        localShape.emplace_back(1UL);
        localShape.emplace_back(shapeObj[0]);
    } else {
        localShape.emplace_back(shapeObj[0]);
        localShape.emplace_back(shapeObj[1]);
    }
    inputShape.emplace_back(localShape);
    return true;
}

bool SingleReqTritonTokenInferInterface::CheckReqInputDataType(OrderedJson &body, std::string &msg) {
    if (!body.contains("datatype") || body["datatype"].is_null() || !body["datatype"].is_string()) {
        msg = "Cannot find datatype in inputs or its type is not string";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   msg << " for requestId is " << requestId_);
        return false;
    }

    std::string typeName = body["datatype"];
    auto pos = DATA_TYPE_MAPPING.stringToType.find(typeName);
    if (pos == DATA_TYPE_MAPPING.stringToType.end()) {
        msg = std::string("Unsupported datatype, got ").append(typeName).append(".");
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Datatype in inputs is invalid");
        return false;
    }
    auto dataType = pos->second;
    if (dataType != InferDataType::TYPE_UINT32) {
        msg = std::string("Unsupported datatype, got ").append(typeName).append(".");
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Datatype in inputs only support U32 for requestId is " << requestId_);
        return false;
    }
    inputDataType.emplace_back(dataType);
    return true;
}

bool SingleReqTritonTokenInferInterface::CheckReqInputData(OrderedJson &body, std::string &msg) {
    if (!body.contains("data") || body["data"].is_null() || !body["data"].is_array()) {
        msg = "Cannot find data in inputs or its type is not array";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   msg << " for requestId is " << requestId_);
        return false;
    }

    if (body["data"].size() > MAX_TOKENS_NUM) {
        std::stringstream ss;
        ss << "Data element must be no more than max token id length, which is " << MAX_TOKENS_NUM << ".";
        msg = ss.str();
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   msg << " for requestId is " << requestId_);
        return false;
    }

    if (body["data"].size() != static_cast<uint64_t>(inputShape[0][1])) {
        std::stringstream ss;
        ss << "Data size should be " << inputShape[0][1] << ", got " << body["data"].size() << ".";
        msg = ss.str();
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   msg << " for requestId is " << requestId_);
        return false;
    }

    for (const auto &i : body["data"]) {
        if (!i.is_number_integer()) {
            msg = "Data element must be integer type.";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       msg << " for requestId is " << requestId_);
            return false;
        }
    }

    return true;
}

void SingleReqTritonTokenInferInterface::SetDMIReComputeBuilder() {
    singleLLMReqHandlerBase_->SetDMIReComputeBuildCallBack(
        std::bind(&SingleReqTritonTokenInferInterface::BuildTritonTokenReComputeBody, this, std::placeholders::_1));
}

std::string SingleReqTritonTokenInferInterface::BuildTritonTokenReComputeBody(const std::vector<BestNTokens> &tokens) {
    OrderedJson newReqJsonObj;
    if (inputParam->userInputId.has_value()) {
        newReqJsonObj["id"] = inputParam->userInputId.value();
    }
    // Get tokens in non-stream mode
    if (tokens.size() != 0) {
        ConvertTokenToMap(tokens);
    }
    OrderedJson input;
    std::vector<int64_t> inputTokens = {};
    this->BuildReComputeInput(inputTokens);
    input["data"] = inputTokens;
    input["shape"] = {1, inputTokens.size()};
    input["name"] = "input0";
    input["datatype"] = "UINT32";
    newReqJsonObj["inputs"] = {input};

    OrderedJson output;
    output["name"] = "output0";
    newReqJsonObj["outputs"] = {output};

    OrderedJson parameters;
    uint64_t seqId;
    if (GetUniqueSequenceId(seqId)) {
        BuildReComputeBodySampling(seqId, parameters);
    }
    parameters["details"] = this->inputParam->showDetails;
    if (this->inputParam->batchSize > 0) {
        parameters["batch_size"] = this->inputParam->batchSize;
    }
    parameters["priority"] = request_->priority;
    parameters["timeout"] = this->inputParam->timeout;
    newReqJsonObj["parameters"] = parameters;

    return newReqJsonObj.dump();
}

void SingleReqTritonTokenInferInterface::BuildReComputeBodySampling(const uint64_t &curSeqId, OrderedJson &parameters) {
    if (request_->temperature.has_value()) {
        parameters["temperature"] = request_->temperature.value();
    }
    if (request_->topK.has_value()) {
        parameters["top_k"] = request_->topK.value();
    }
    if (request_->topP.has_value()) {
        parameters["top_p"] = request_->topP.value();
    }
    if (request_->doSample.has_value()) {
        parameters["do_sample"] = request_->doSample.value();
    }
    if (request_->seed.has_value()) {
        parameters["seed"] = request_->seed.value();
    }
    if (request_->repetitionPenalty.has_value()) {
        parameters["repetition_penalty"] = request_->repetitionPenalty.value();
    }
    if (this->inputParam->maxNewTokens != MAX_NEW_TOKENS_DFT) {
        parameters["max_new_tokens"] =
            static_cast<size_t>(this->inputParam->maxNewTokens) -
            (this->oriReqTokenLen_ + this->respTokenMap[curSeqId].size() - this->reqTokens_.size());
    }
    if (request_->watermark.has_value()) {
        parameters["watermark"] = request_->watermark.value();
    }
}

void SingleReqTritonTokenInferInterface::BuildReComputeInput(std::vector<int64_t> &inputTokens) {
    inputTokens.push_back(oriReqTokenLen_);
    for (size_t i = 0; i < oriReqTokenLen_; i++) {
        inputTokens.push_back(reqTokens_[i]);
    }
    uint64_t seqId;
    if (!GetUniqueSequenceId(seqId)) {
        return;
    }
    for (const auto &item : this->respTokenMap[seqId]) {
        inputTokens.push_back(item);
    }
}
}  // namespace mindie_llm
