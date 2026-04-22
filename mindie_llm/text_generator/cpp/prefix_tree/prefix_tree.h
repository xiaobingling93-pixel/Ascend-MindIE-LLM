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

#ifndef MINDIE_LLM_PREFIX_TREE_H
#define MINDIE_LLM_PREFIX_TREE_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mindie_llm {
namespace prefix_tree {
constexpr float MAX_FREQ = 1e9;
constexpr float FREQ_HALF = 0.5;
constexpr float DEFAULT_FREQ = 1.0;

struct Node {
    std::map<int, std::shared_ptr<Node>> children = {};
    std::map<int, float> freqs = {};
};

class PrefixTree {
   public:
    explicit PrefixTree(int rootTokenId, unsigned int maxNode = 65536, unsigned int maxOutputNode = 512)
        : rootTokenId(rootTokenId), maxNode(maxNode), maxOutputNode(maxOutputNode) {}
    void Put(std::vector<int>& tokenIds, const std::string& mode = "output", int batchId = 0);
    std::pair<std::vector<int>, int> GetOneDraft(std::vector<int>& tokenIds, int batchId, int decodingLength);
    void ResetInputFreq(int batchId);
    void Trim();

   private:
    void AddNode(std::vector<int>& tokenIds, std::map<int, std::shared_ptr<Node>>& nodes,
                 const std::string& mode = "output", int batchId = 0, unsigned int tokenIdIndex = 0);
    void Pack(std::vector<int>& tokenIds, std::map<int, std::shared_ptr<Node>>& nodes, int batchId,
              unsigned int tokenIdIndex = 0);
    void AddNodeFreq(std::shared_ptr<Node>& node, int batchId) const;
    std::pair<std::vector<int>, int> SearchBestDraft(std::map<int, std::shared_ptr<Node>> nodes, int batchId,
                                                     std::vector<int>& draftIds, int decodingLength) const;
    void ClearInput(std::map<int, std::shared_ptr<Node>>& nodes, int batchId);
    void TrimNode(std::map<int, std::shared_ptr<Node>>& nodes);
    void CountNode(std::map<int, std::shared_ptr<Node>>& nodes, unsigned int& size);

    int rootTokenId;
    unsigned int maxNode = 65536;
    unsigned int maxOutputNode = 512;
    unsigned int nNode = 0;
    unsigned int nOutputNode = 0;
    std::map<int, std::shared_ptr<Node>> fullNodes = {};
};
}  // namespace prefix_tree
}  // namespace mindie_llm

#endif
