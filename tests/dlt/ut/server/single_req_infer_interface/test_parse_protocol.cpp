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

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <nlohmann/json.hpp>
#include "parse_protocol.h"
#include "check_utils.h"

using OrderedJson = nlohmann::ordered_json;
using Json = nlohmann::json;
using namespace nlohmann;
using namespace mindie_llm;

namespace mindie_llm {

extern "C++" {
uint32_t IsTgiFormat(const OrderedJson &jsonData);
uint32_t IsVllmFormat(const OrderedJson &jsonData);
uint32_t GetInferTypeFromJson(const OrderedJson &jsonData, uint16_t &inferType);
std::string GetLastPartOfPath(std::string &path);
}

class ParseProtocolTest : public testing::Test {
protected:
    ParseProtocolTest() = default;
    void SetUp()
    {
        jsonStr = "";
        json.clear();
    }
    void TearDown()
    {
        GlobalMockObject::verify();
    }

    OrderedJson json;
    std::string jsonStr, errorMsg;
    JsonParse jsonParse;
};

TEST_F(ParseProtocolTest, testCheckOptionalItemType)
{
    OptionalItemResult expectedResult;
    
    // JSON contains this optional item
    json["key-of-optional-item"] = "value-of-optional-item";
    expectedResult = JsonParse::CheckOptionalItemType(json, "key-of-optional-item", OrderedJson::value_t::string, errorMsg);
    EXPECT_EQ(expectedResult.isPresent, true);
    EXPECT_EQ(expectedResult.isCorrectType, true);

    // JSON does not contain this optional item
    json.clear();
    expectedResult = JsonParse::CheckOptionalItemType(json, "key-of-optional-item", OrderedJson::value_t::string, errorMsg);
    EXPECT_EQ(expectedResult.isPresent, false);
    EXPECT_EQ(expectedResult.isCorrectType, true);
}

TEST_F(ParseProtocolTest, testJsonIsTgiFormat)
{
    ASSERT_EQ(IsTgiFormat(json), EP_ERROR);

    json["inputs"] = "";
    ASSERT_EQ(IsTgiFormat(json), EP_OK);

    json["inputs"] = "test";
    ASSERT_EQ(IsTgiFormat(json), EP_OK);
}

TEST_F(ParseProtocolTest, IsVllmFormat)
{
    ASSERT_EQ(IsVllmFormat(json), EP_PARSE_NO_PARAM_ERR);

    json["prompt"] = "";
    ASSERT_EQ(IsVllmFormat(json), EP_OK);

    json["prompt"] = "test";
    ASSERT_EQ(IsVllmFormat(json), EP_OK);
}

TEST_F(ParseProtocolTest, testCheckIPV4)
{
    std::string ipAddress;
    std::string inputName = "mockIpAddress";
    bool allowAllZeroIp = true;

    // CheckIPV4 should return true given valid IP address
    ipAddress = "127.0.0.1";
    EXPECT_EQ(CheckIPV4(ipAddress, inputName, allowAllZeroIp), true);

    // CheckIPV4 should return false given invalid IP address
    ipAddress = "256.0.0.1";
    EXPECT_EQ(CheckIPV4(ipAddress, inputName, allowAllZeroIp), false);

    // CheckIPV4 return true/false depends on whether allowAllZeroIp is true/false
    // given input IP address is 0.0.0.0
    ipAddress = "0.0.0.0";
    allowAllZeroIp = true;
    EXPECT_EQ(CheckIPV4(ipAddress, inputName, allowAllZeroIp), true);
    allowAllZeroIp = false;
    EXPECT_EQ(CheckIPV4(ipAddress, inputName, allowAllZeroIp), false);
}

TEST_F(ParseProtocolTest, testJsonContainItemWithType)
{
    std::string key = "temperature";
    std::string error = "default error";
    ordered_json::value_t type = OrderedJson::value_t::boolean;

    // JsonContainItemWithType should return false if the key is not found
    ASSERT_EQ(jsonParse.JsonContainItemWithType(json, key, type, error), false);
    EXPECT_EQ(error, std::string("Not found ").append(key).append("."));

    // JsonContainItemWithType should return false if the key is found but its value is null
    json[key] = nullptr;
    ASSERT_EQ(jsonParse.JsonContainItemWithType(json, key, type, error), false);
    EXPECT_EQ(error, std::string(key).append(" must not be null."));

    // JsonContainItemWithType should return true if the key is found and its value type corresponds to type
    // float
    type = ordered_json::value_t::number_float;
    json[key] = 1.2;
    ASSERT_EQ(jsonParse.JsonContainItemWithType(json, key, type, error), true);
    json[key] = "1.2";
    EXPECT_EQ(jsonParse.JsonContainItemWithType(json, key, type, error), false);
    EXPECT_EQ(error, std::string(key).append(" must be float type."));
    // integer
    type = ordered_json::value_t::number_integer;
    json[key] = 1;
    ASSERT_EQ(jsonParse.JsonContainItemWithType(json, key, type, error), true);
    json[key] = 1.0;
    EXPECT_EQ(jsonParse.JsonContainItemWithType(json, key, type, error), false);
    EXPECT_EQ(error, std::string(key).append(" must be integer type."));
}

TEST_F(ParseProtocolTest, testGetInferTypeFromJsonStr)
{
    uint16_t inferType;

    // GetInferTypeFromJsonStr should return EP_INVALID_PARAM if jsonData does not keys for vLLM or TGI formats
    EXPECT_EQ(jsonParse.GetInferTypeFromJsonStr(json.dump(), inferType), EP_INVALID_PARAM);

    json["prompt"] = "testVllm";
    json["inputs"] = nullptr;
    EXPECT_EQ(jsonParse.GetInferTypeFromJsonStr(json.dump(), inferType), EP_OK);
    EXPECT_EQ(inferType, MSG_TYPE_VLLM);

    json["prompt"] = nullptr;
    json["inputs"] = "testTgi";
    EXPECT_EQ(jsonParse.GetInferTypeFromJsonStr(json.dump(), inferType), EP_OK);
    EXPECT_EQ(inferType, MSG_TYPE_TGI);

    json["prompt"] = "testVllm";
    json["inputs"] = "testTgi";
    EXPECT_EQ(jsonParse.GetInferTypeFromJsonStr(json.dump(), inferType), EP_OK);
    EXPECT_EQ(inferType, MSG_TYPE_VLLM); // vLLM format is prior to TGI format
}

TEST_F(ParseProtocolTest, testHandleGetInfo)
{
    ScheduleConfig scheduleParam;
    scheduleParam.maxPrefillTokens = 2;
    scheduleParam.maxBatchSize = 5;
    scheduleParam.maxIterTimes = 3;
    json["docker_label"] = nullptr;
    json["max_batch_total_tokens"] = 2;
    json["max_best_of"] = 1;
    json["max_concurrent_requests"] = 5;
    json["max_stop_sequences"] = nullptr;
    json["max_waiting_tokens"] = nullptr;
    json["sha"] = nullptr;
    json["validation_workers"] = nullptr;
    json["version"] = "1.0.0";
    json["waiting_served_ratio"] = nullptr;

    std::vector<ModelDeployConfig> modelParams;
    ModelDeployConfig modelDeployParam1;
    modelDeployParam1.torchDtype = "torchD1";
    modelDeployParam1.modelName = "model1";
    modelDeployParam1.maxSeqLen = 10;
    modelDeployParam1.maxInputTokenLen = 5;
    modelParams.emplace_back(modelDeployParam1);
    json["models"][0]["model_device_type"] = "npu";
    json["models"][0]["model_dtype"] = "torchD1";
    json["models"][0]["model_id"] = "model1";
    json["models"][0]["model_pipeline_tag"] = "text-generation";
    json["models"][0]["model_sha"] = nullptr ;
    json["models"][0]["max_total_tokens"] = 10;
    ModelDeployConfig modelDeployParam2;
    modelDeployParam2.torchDtype = "torchD2";
    modelDeployParam2.modelName = "model2";
    modelDeployParam2.maxSeqLen = 20;
    modelDeployParam2.maxInputTokenLen = 10;
    modelParams.emplace_back(modelDeployParam2);
    json["models"][1]["model_device_type"] = "npu";
    json["models"][1]["model_dtype"] = "torchD2";
    json["models"][1]["model_id"] = "model2";
    json["models"][1]["model_pipeline_tag"] = "text-generation";
    json["models"][1]["model_sha"] = nullptr ;
    json["models"][1]["max_total_tokens"] = 20;
    json["max_input_length"] = 10;
    
    std::string infoStr;
    jsonParse.HandleGetInfo(scheduleParam, modelParams, infoStr);
    EXPECT_EQ(infoStr, json.dump());
}

TEST_F(ParseProtocolTest, testEncodeTritonModel)
{
    ModelDeployConfig modelParam;
    modelParam.modelName = "llama_65b";
    json["name"] = "llama_65b";
    json["platform"] = "MindIE Server";
    json["inputs"][0]["name"] = "input0";
    json["inputs"][0]["shape"] = {-1};
    json["inputs"][0]["datatype"] = "UINT32";
    json["outputs"][0]["name"] = "output0";
    json["outputs"][0]["shape"] = {-1};
    json["outputs"][0]["datatype"] = "UINT32";

    jsonParse.EncodeTritonModel(modelParam, jsonStr);
    EXPECT_EQ(jsonStr, json.dump());
}

TEST_F(ParseProtocolTest, testEncodeTritonModelConfig)
{
    ModelDeployConfig modelParam;
    modelParam.modelName = "llama_65b";
    modelParam.inputDatatype = 8; // TYPE_INT32 = 8,
    modelParam.outputDatatype = 9; // TYPE_INT64 = 9,
    modelParam.maxSeqLen = 5;
    modelParam.npuMemSize = 1;
    modelParam.cpuMemSize = 2;
    modelParam.worldSize = 3;
    modelParam.modelWeightPath = "/data/weights/llama1-65b-safetensors";
    modelParam.modelInstanceType = "Standard";

    json["model_name"] = "llama_65b";
    json["input_datatype"] = "INT32";
    json["output_datatype"] = "INT64";
    json["max_seq_len"] = 5;
    json["npu_mem_size"] = 1;
    json["cpu_mem_size"] = 2;
    json["world_size"] = 3;
    json["model_weight_path"] = "llama1-65b-safetensors"; // GetLastPartOfPath is called in EncodeTritonModelConfig
    json["model_instance_type"] = "Standard";

    jsonParse.EncodeTritonModelConfig(modelParam, jsonStr);
    EXPECT_EQ(jsonStr, json.dump());
}

TEST_F(ParseProtocolTest, testEncodeTritonEngine)
{
    ScheduleConfig config;
    config.maxIterTimes = 5;
    config.prefillPolicyType = 0;
    config.decodePolicyType = 1;
    config.maxPrefillBatchSize = 3;
    config.maxPrefillTokens = 4;
    json["name"] = "MindIE Server";
    json["version"] = "1.0.0";
    json["extensions"]["max_iter_times"] = 5;
    json["extensions"]["prefill_policy_type"] = 0;
    json["extensions"]["decode_policy_type"] = 1;
    json["extensions"]["max_prefill_batch_size"] = 3;
    json["extensions"]["max_prefill_tokens"] = 4;
    
    jsonParse.EncodeTritonEngine(config, jsonStr);
    EXPECT_EQ(jsonStr, json.dump());
}

TEST_F(ParseProtocolTest, testEncodeSlotCount)
{
    ScheduleConfig scheduleParam;
    scheduleParam.maxBatchSize = 200;
    uint64_t freeSlot = 50;
    uint64_t availableTokensLen = 15;
    json["total_slots"] = 200;
    json["free_slots"] = 50;
    json["available_tokens_length"] = 15;

    jsonParse.EncodeSlotCount(scheduleParam, freeSlot, availableTokensLen, jsonStr);
    EXPECT_EQ(jsonStr, json.dump());
}

TEST_F(ParseProtocolTest, testEncodeOpenAiModel)
{
    ModelDeployConfig modelParam;
    modelParam.modelName = "llama_65b";
    json["id"] = "llama_65b";
    json["object"] = "model";
    json["created"] = 0;
    json["owned_by"] = "MindIE Server";

    jsonParse.EncodeOpenAiModel(modelParam, 0, jsonStr);
    ASSERT_EQ(jsonStr, json.dump());
}

TEST_F(ParseProtocolTest, testEncodeOpenAiModel2)
{
    LoraConfig loraParam;
    loraParam.loraName = "lora";
    loraParam.baseModel = "loraBase";
    json["id"] = "lora";
    json["object"] = "model";
    json["created"] = 0;
    json["owned_by"] = "MindIE Server";
    json["parent"] = "loraBase";

    jsonParse.EncodeOpenAiModel(loraParam, 0, jsonStr);
    ASSERT_EQ(jsonStr, json.dump());
}


TEST_F(ParseProtocolTest, testEncodeOpenAiModels)
{
    json["object"] = "list";
    auto i = 0;

    std::vector<ModelDeployConfig> modelParams;
    ModelDeployConfig modelParam1, modelParam2;
    modelParam1.modelName = "model1";
    modelParam2.modelName = "model2";
    modelParams.emplace_back(modelParam1);
    modelParams.emplace_back(modelParam2);
    for (auto& singleModelParams : modelParams) {
        json["data"][i]["id"] = singleModelParams.modelName;
        json["data"][i]["object"] = "model";
        json["data"][i]["created"] = 0;
        json["data"][i]["owned_by"] = "MindIE Server";
        ++i;
    }

    std::vector<LoraConfig> loraParams;
    LoraConfig loraParam1, loraParam2;
    loraParam1.loraName = "lora1";
    loraParam1.baseModel = "loraBase1";
    loraParam2.loraName = "lora2";
    loraParam2.baseModel = "loraBase2";
    loraParams.emplace_back(loraParam1);
    loraParams.emplace_back(loraParam2);
    for (auto& singleLoraParam : loraParams) {
        json["data"][i]["id"] = singleLoraParam.loraName;
        json["data"][i]["object"] = "model";
        json["data"][i]["created"] = 0;
        json["data"][i]["owned_by"] = "MindIE Server";
        json["data"][i]["parent"] = singleLoraParam.baseModel;
        ++i;
    }

    jsonParse.EncodeOpenAiModels(modelParams, loraParams, 0, jsonStr);
    ASSERT_EQ(jsonStr, json.dump());
}

TEST_F(ParseProtocolTest, testEncodeUnhealthyNodeInfo)
{
    std::map<std::string, NodeHealthStatus> slavesStatus;
    slavesStatus["192.168.3.125"] = NodeHealthStatus::ABNORMAL;
    json["message"] = "Abnormal node detected";
    json["no_contact_node"] = OrderedJson::array();
    json["no_contact_node"].emplace_back("node(192.168.3.125) is abnormal.");

    jsonParse.EncodeAbnormalNodeInfo(slavesStatus, jsonStr);
    EXPECT_EQ(json.dump(), jsonStr);
}

TEST_F(ParseProtocolTest, testGetJsonDataType)
{
    ordered_json::value_t type = nlohmann::ordered_json::value_t::discarded;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "invalid_type");
    type = nlohmann::ordered_json::value_t::null;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "null");
    type = nlohmann::ordered_json::value_t::object;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "object");
    type = nlohmann::ordered_json::value_t::array;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "array");
    type = nlohmann::ordered_json::value_t::string;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "string");
    type = nlohmann::ordered_json::value_t::boolean;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "boolean");
    type = nlohmann::ordered_json::value_t::number_integer;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "integer");
    type = nlohmann::ordered_json::value_t::number_unsigned;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "unsigned");
    type = nlohmann::ordered_json::value_t::number_float;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "float");
    type = nlohmann::ordered_json::value_t::binary;
    ASSERT_EQ(jsonParse.GetJsonDataType(type), "binary");
}

// NOTE: the following unit tests come from the code repo before refactor

TEST_F(ParseProtocolTest, CheckPDIPInfo)
{
    bool ret = jsonParse.CheckPDIPInfo(json);
    ASSERT_EQ(ret, false);

    json["server_ip"] = "7.223.36.250";
    json["device"] = 1;
    ret = jsonParse.CheckPDIPInfo(json);
    ASSERT_EQ(ret, false);

    OrderedJson deviceInfo = {
        {"device_logical_id", 1}
    };
    json["device"] = OrderedJson::array({deviceInfo});
    ret = jsonParse.CheckPDIPInfo(json);
    ASSERT_EQ(ret, false);

    deviceInfo = {
        {"device_logical_id", "mockId"}
    };
    json["device"] = OrderedJson::array({deviceInfo});
    ret = jsonParse.CheckPDIPInfo(json);
    ASSERT_EQ(ret, false);

    deviceInfo = {
        {"device_logical_id", "1"}
    };
    json["device"] = OrderedJson::array({deviceInfo});
    ret = jsonParse.CheckPDIPInfo(json);
    ASSERT_EQ(ret, false);

    deviceInfo = {
        {"device_id", "1"},
        {"device_logical_id", "1"},
        {"device_ip", "1"}
    };
    json["device"] = OrderedJson::array({deviceInfo});
    ret = jsonParse.CheckPDIPInfo(json);
    ASSERT_EQ(ret, false);

    deviceInfo = {
        {"device_id", "1"},
        {"device_logical_id", "1"},
        {"device_ip", "10.20.1.26"}
    };
    json["device"] = OrderedJson::array({deviceInfo});
    ret = jsonParse.CheckPDIPInfo(json);
    ASSERT_EQ(ret, true);
}

TEST_F(ParseProtocolTest, PrometheusFormat)
{
    std::string name = "prometheus";
    std::string value = "format";
    std::string managementIpAddress = "127.0.0.2";
    std::string managementPort = "3007";

    auto jsonObj = jsonParse.PrometheusFormat(name, value, managementIpAddress, managementPort);
    ASSERT_EQ(jsonObj["value"], "format");

    auto metricsJsonObj = jsonObj["metric"][0];
    ASSERT_EQ(metricsJsonObj["__name__"], "prometheus");
    ASSERT_EQ(metricsJsonObj["job"], "node");
    ASSERT_EQ(metricsJsonObj["instance"], "127.0.0.2:3007");
}

TEST_F(ParseProtocolTest, GetContextJsonBody)
{
    httplib::Request req;
    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    bool ret = jsonParse.GetContextJsonBody(ctx, json);
    ASSERT_EQ(ret, false);
}

TEST_F(ParseProtocolTest, CheckHostIP)
{
    OrderedJson json;
    bool ret = jsonParse.CheckHostIP(json);
    ASSERT_EQ(ret, true);

    json["host_ip"] = nullptr;
    ret = jsonParse.CheckHostIP(json);
    ASSERT_EQ(ret, false);

    json["host_ip"] = 12345; // 这里使用了难以理解的字面量 12345，表示一个无效的 IP 地址
    ret = jsonParse.CheckHostIP(json);
    ASSERT_EQ(ret, false);

    json["host_ip"] = "invalid_ip";
    ret = jsonParse.CheckHostIP(json);
    ASSERT_EQ(ret, false);

    json["host_ip"] = "256.256.256.256";
    ret = jsonParse.CheckHostIP(json);
    ASSERT_EQ(ret, false);

    json["host_ip"] = "192.168.1.1";
    ret = jsonParse.CheckHostIP(json);
    ASSERT_EQ(ret, true);
}

TEST_F(ParseProtocolTest, CheckPDV2IPInfo)
{
    OrderedJson json;
    bool ret = jsonParse.CheckPDV2IPInfo(json);
    ASSERT_EQ(ret, false);

    json["server_ip"] = "7.223.36.250";
    ret = jsonParse.CheckPDV2IPInfo(json);
    ASSERT_EQ(ret, false);

    json["dp_inst_list"] = 1;
    ret = jsonParse.CheckPDV2IPInfo(json);
    ASSERT_EQ(ret, false);

    OrderedJson dpInstInfo = {
        {"dp_inst_id", 1}
    };
    json["dp_inst_list"] = OrderedJson::array({dpInstInfo});
    ret = jsonParse.CheckPDV2IPInfo(json);
    ASSERT_EQ(ret, false);

    unsigned dpInstIdNum = 1;
    dpInstInfo = {
        {"dp_inst_id", dpInstIdNum},
        {"device", 1}
    };
    json["dp_inst_list"] = OrderedJson::array({dpInstInfo});
    ret = jsonParse.CheckPDV2IPInfo(json);
    ASSERT_EQ(ret, false);

    OrderedJson deviceInfo = {
        {"device_id", "1"},
        {"device_logical_id", "1"},
        {"device_ip", "10.20.1.26"}
    };
    dpInstInfo["device"] = OrderedJson::array({deviceInfo});
    json["dp_inst_list"] = OrderedJson::array({dpInstInfo});
    ret = jsonParse.CheckPDV2IPInfo(json);
    ASSERT_EQ(ret, true);
}

TEST_F(ParseProtocolTest, testDecodeGeneralTGIStreamMode)
{
    bool streamMode;
    std::string error;
    // case 1: contain "stream"
    std::string jsonStr = R"({"stream": true})";
    EXPECT_EQ(jsonParse.DecodeGeneralTGIStreamMode(jsonStr, streamMode, error), EP_OK);
    EXPECT_EQ(streamMode, true);
    // case 2: not contain "stream"
    jsonStr = R"({"other_field": "value"})";
    EXPECT_EQ(jsonParse.DecodeGeneralTGIStreamMode(jsonStr, streamMode, error), EP_OK);
    EXPECT_EQ(streamMode, false);
    // case 3: "stream" is not bool
    jsonStr = R"({"stream": 123})";
    EXPECT_EQ(jsonParse.DecodeGeneralTGIStreamMode(jsonStr, streamMode, error), EP_PARSE_JSON_ERR);
    // case 4: invalid json
    jsonStr = "{123";
    EXPECT_EQ(jsonParse.DecodeGeneralTGIStreamMode(jsonStr, streamMode, error), EP_PARSE_JSON_ERR);
    // case 5: json contain invalid charactor
    jsonStr = "{\xFF\"stream\": true}";
    EXPECT_EQ(jsonParse.DecodeGeneralTGIStreamMode(jsonStr, streamMode, error), EP_PARSE_JSON_ERR);
}

TEST_F(ParseProtocolTest, testCheckPDRoleReqJson)
{
    OrderedJson json;
    EXPECT_FALSE(JsonParse::CheckPDRoleReqJson(json));
    json["local"] = OrderedJson::parse(R"({"abc": "123"})");
    EXPECT_FALSE(JsonParse::CheckPDRoleReqJson(json));
    json["local"] = OrderedJson::parse(R"({
        "server_ip": "192.168.1.100",
        "id": 100,
        "host_ip": "10.0.0.5",
        "device": [
            {
                "device_logical_id": "101",
                "device_id": "201",
                "device_ip": "172.16.0.2",
                "rank_id": "0"
            },
            {
                "device_logical_id": "102",
                "device_id": "202",
                "device_ip": "172.16.0.3"
            }
        ]
    })");
    EXPECT_TRUE(JsonParse::CheckPDRoleReqJson(json));
    OrderedJson validPeer = OrderedJson::parse(R"({
        "server_ip": "192.168.1.200",
        "id": 200,
        "host_ip": "10.0.0.6",
        "device": [
            {
                "device_logical_id": "301",
                "device_id": "401",
                "device_ip": "172.16.0.4"
            }
        ]
    })");
    OrderedJson peersArr = OrderedJson::array();
    peersArr.push_back(validPeer);
    json["peers"] = peersArr;
    EXPECT_TRUE(JsonParse::CheckPDRoleReqJson(json));
}

TEST_F(ParseProtocolTest, testJsonHttpMetrics)
{
    std::map<std::string, uint64_t> batchSchedulerMetrics{};
    std::string jsonStr = "";
    EXPECT_TRUE(JsonParse::JsonHttpMetrics(HttpMetrics::GetInstance(), batchSchedulerMetrics, jsonStr));
}

TEST_F(ParseProtocolTest, testCheckPDRoleV2ReqJson)
{
    OrderedJson json;
    EXPECT_FALSE(JsonParse::CheckPDRoleV2ReqJson(json));
    OrderedJson validLocalNode = OrderedJson::parse(R"({
        "server_ip": "192.168.1.100",
        "host_ip": "10.0.0.5",
        "dp_inst_list": [
            {
                "dp_inst_id": 100,
                "device": [
                    {
                        "device_logical_id": "101",
                        "device_id": "201",
                        "device_ip": "172.16.0.2"
                    }
                ]
            }
        ]
    })");
    json["local"] = OrderedJson::array({validLocalNode});
    EXPECT_TRUE(JsonParse::CheckPDRoleV2ReqJson(json));
    OrderedJson validPeerNode = OrderedJson::parse(R"({
        "server_ip": "192.168.1.200",
        "host_ip": "10.0.0.6",
        "dp_inst_list": [
            {
                "dp_inst_id": 200,
                "device": [
                    {
                        "device_logical_id": "301",
                        "device_id": "401",
                        "device_ip": "172.16.0.4"
                    }
                ]
            }
        ]
    })");
    OrderedJson peerInfo = OrderedJson::array({validPeerNode});
    OrderedJson peersArr = OrderedJson::array({peerInfo});
    json["peers"] = peersArr;
    EXPECT_TRUE(JsonParse::CheckPDRoleV2ReqJson(json));
}


TEST_F(ParseProtocolTest, DecodeFaultRecoveryCmd)
{
    FaultRecoveryCmd cmdType;
    std::string cmdStr;

    // Test valid JSON with CMD_PAUSE_ENGINE
    std::string jsonStr = R"({"cmd": 0})";
    uint32_t ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_OK);
    ASSERT_EQ(cmdType, FaultRecoveryCmd::CMD_PAUSE_ENGINE);
    ASSERT_EQ(cmdStr, "CMD_PAUSE_ENGINE");

    // Test valid JSON with CMD_REINIT_NPU
    jsonStr = R"({"cmd": 1})";
    ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_OK);
    ASSERT_EQ(cmdType, FaultRecoveryCmd::CMD_REINIT_NPU);
    ASSERT_EQ(cmdStr, "CMD_REINIT_NPU");

    // Test valid JSON with CMD_START_ENGINE
    jsonStr = R"({"cmd": 2})";
    ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_OK);
    ASSERT_EQ(cmdType, FaultRecoveryCmd::CMD_START_ENGINE);
    ASSERT_EQ(cmdStr, "CMD_START_ENGINE");

    // Test valid JSON with CMD_PAUSE_ENGINE_ROCE
    jsonStr = R"({"cmd": 3})";
    ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_OK);
    ASSERT_EQ(cmdType, FaultRecoveryCmd::CMD_PAUSE_ENGINE_ROCE);
    ASSERT_EQ(cmdStr, "CMD_PAUSE_ENGINE_ROCE");

    // Test invalid JSON string
    jsonStr = R"({"cmd": invalid})";
    ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_PARSE_JSON_ERR);

    // Test missing cmd field
    jsonStr = R"({"other_field": 0})";
    ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_PARSE_NO_PARAM_ERR);

    // Test null cmd field
    jsonStr = R"({"cmd": null})";
    ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_PARSE_NO_PARAM_ERR);

    // Test empty JSON string
    jsonStr = "";
    ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_PARSE_JSON_ERR);

    // Test malformed JSON
    jsonStr = R"({cmd: 0})";
    ret = jsonParse.DecodeFaultRecoveryCmd(jsonStr, cmdType, cmdStr);
    ASSERT_EQ(ret, EP_PARSE_JSON_ERR);
}
}