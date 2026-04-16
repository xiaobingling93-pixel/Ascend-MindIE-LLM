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

#include "prefix_tree.h"

namespace mindie_llm {
namespace prefix_tree {
void PrefixTree::Put(std::vector<int>& tokenIds, const std::string& mode, int batchId) {
    batchId = (mode == "output") ? -1 : batchId;
    AddNode(tokenIds, this->fullNodes, mode, batchId);
}

std::pair<std::vector<int>, int> PrefixTree::GetOneDraft(std::vector<int>& tokenIds, int batchId, int decodingLength) {
    auto tempNodes = this->fullNodes;
    int matchTokenId = -1;
    if (tokenIds.size() > 0) {
        for (auto& tokenId : tokenIds) {
            auto iter = tempNodes.find(tokenId);
            if (iter == tempNodes.end()) {
                tempNodes.clear();
                break;
            }
            if ((iter->second->freqs.find(batchId) != iter->second->freqs.end()) ||
                (iter->second->freqs.find(-1) != iter->second->freqs.end())) {
                tempNodes = iter->second->children;
            }
            matchTokenId = tokenId;
        }
    }

    std::vector<int> draftIds;
    int size = 0;

    if (tempNodes.empty()) {
        int draftId = (tokenIds.size() > 0) ? tokenIds.back() : this->rootTokenId;
        draftIds.push_back(draftId);
        size += 1;
        return std::make_pair(draftIds, size);
    }

    draftIds.push_back((matchTokenId == -1) ? this->rootTokenId : matchTokenId);
    return SearchBestDraft(tempNodes, batchId, draftIds, decodingLength);
}

void PrefixTree::ResetInputFreq(int batchId) {
    if (this->fullNodes.empty()) {
        return;
    }
    ClearInput(this->fullNodes, batchId);
}

void PrefixTree::Trim() {
    if (this->nNode > this->maxNode || this->nOutputNode > this->maxOutputNode) {
        TrimNode(this->fullNodes);
        unsigned int size = 0;
        CountNode(this->fullNodes, size);
        this->nNode = size;
        this->nOutputNode = size;
    }
}

void PrefixTree::AddNode(std::vector<int>& tokenIds, std::map<int, std::shared_ptr<Node>>& nodes,
                         const std::string& mode, int batchId, unsigned int tokenIdIndex) {
    if (tokenIds.empty() || tokenIdIndex >= tokenIds.size()) {
        return;
    }
    int currentToken = tokenIds[tokenIdIndex];
    auto iter = nodes.find(currentToken);
    if (iter == nodes.end()) {
        Pack(tokenIds, nodes, batchId, tokenIdIndex);
        this->nNode += (tokenIds.size() - tokenIdIndex);
        if (mode == "output") {
            this->nOutputNode += (tokenIds.size() - tokenIdIndex);
        }
        return;
    }
    auto node = iter->second;
    AddNodeFreq(node, batchId);
    AddNode(tokenIds, node->children, mode, batchId, tokenIdIndex + 1);
}

void PrefixTree::Pack(std::vector<int>& tokenIds, std::map<int, std::shared_ptr<Node>>& nodes, int batchId,
                      unsigned int tokenIdIndex) {
    if (tokenIdIndex >= tokenIds.size()) {
        return;
    }
    int currentToken = tokenIds[tokenIdIndex];
    nodes[currentToken] = std::make_shared<Node>();
    nodes[currentToken]->freqs[batchId] = DEFAULT_FREQ;
    Pack(tokenIds, nodes[currentToken]->children, batchId, tokenIdIndex + 1);
}

void PrefixTree::AddNodeFreq(std::shared_ptr<Node>& node, int batchId) const {
    if (node->freqs.find(batchId) == node->freqs.end()) {
        node->freqs[batchId] = DEFAULT_FREQ;
    } else if (node->freqs[batchId] < MAX_FREQ) {
        node->freqs[batchId] += DEFAULT_FREQ;
    }
}

std::pair<std::vector<int>, int> PrefixTree::SearchBestDraft(std::map<int, std::shared_ptr<Node>> nodes, int batchId,
                                                             std::vector<int>& draftIds, int decodingLength) const {
    int size = 0;
    int tempFreq = 0;
    int maxFreq = 0;
    int bestId = 0;
    std::shared_ptr<Node> bestNode;

    for (int i = 0; i < decodingLength; i++) {
        maxFreq = 0;
        bestId = 0;
        bestNode = nullptr;
        for (auto iter = nodes.cbegin(); iter != nodes.cend(); iter++) {
            if (iter->second->freqs.find(batchId) != iter->second->freqs.end()) {
                tempFreq += iter->second->freqs[batchId];
            }
            if (iter->second->freqs.find(-1) != iter->second->freqs.end()) {
                tempFreq += iter->second->freqs[-1];
            }
            if (tempFreq > maxFreq) {
                maxFreq = tempFreq;
                bestId = iter->first;
                bestNode = iter->second;
            }
            tempFreq = 0;
        }
        if (bestNode == nullptr) {
            break;
        }
        draftIds.push_back(bestId);
        size += 1;
        nodes = bestNode->children;
    }
    return std::make_pair(draftIds, size);
}

void PrefixTree::ClearInput(std::map<int, std::shared_ptr<Node>>& nodes, int batchId) {
    for (auto iter = nodes.cbegin(); iter != nodes.cend(); iter++) {
        auto it = iter->second->freqs.find(batchId);
        if (it == iter->second->freqs.end()) {
            continue;
        }
        iter->second->freqs.erase(it);
        if (!iter->second->children.empty()) {
            ClearInput(iter->second->children, batchId);
        }
    }
}

void PrefixTree::TrimNode(std::map<int, std::shared_ptr<Node>>& nodes) {
    for (auto iter = nodes.begin(); iter != nodes.end();) {
        float outputFreq = 0.0;
        if (iter->second->freqs.find(-1) != iter->second->freqs.end()) {
            outputFreq = iter->second->freqs[-1];
        }
        if (outputFreq > 1.0) {
            iter->second->freqs[-1] *= FREQ_HALF;
            if (!iter->second->children.empty()) {
                TrimNode(iter->second->children);
            }
            iter++;
        } else {
            nodes.erase(iter++);
        }
    }
}

void PrefixTree::CountNode(std::map<int, std::shared_ptr<Node>>& nodes, unsigned int& size) {
    size += nodes.size();
    for (auto iter = nodes.cbegin(); iter != nodes.cend(); iter++) {
        if (!iter->second->children.empty()) {
            CountNode(iter->second->children, size);
        }
    }
}
}  // namespace prefix_tree
}  // namespace mindie_llm
