# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from collections import defaultdict
from typing import Union
import numpy as np
from _prefix_tree import _PrefixTree
from ....utils.env import ENV

PATTEN_INPUT = "input"
PATTEN_OUTPUT = "output"


class TreeNode:
    def __init__(self, next_node, freqs_dict):
        self.next_node = next_node
        self.freqs_dict = freqs_dict


class Tree:
    def __init__(self, start_token_id, max_size=65536, max_output_size=1024):
        self.tree_nodes = {}
        self.start_token_id = start_token_id
        self.n = 0
        self.n_output = 0
        self.max_size = max_size
        self.max_output_size = max_output_size
        self.n_output_node = None

    def search_one_draft(self, pre_token_ids, max_size=8, use_batch=0):
        temp_nodes = self.tree_nodes
        match_token_id = None
        if len(pre_token_ids) != 0:
            for match_token_id in pre_token_ids:
                temp_node = temp_nodes.get(match_token_id, None)
                temp_nodes = {}
                if temp_node is None:
                    break
                if temp_node.freqs_dict.get(use_batch, 0.0) > 0 or temp_node.freqs_dict.get(-1, 0.0) > 0:
                    temp_nodes = temp_node.next_node

        if len(temp_nodes) == 0:
            token_id = pre_token_ids[-1] if len(pre_token_ids) > 0 else self.start_token_id
            return [token_id], np.ones((1, 1), dtype=np.int64), [0, 0]

        ids = [match_token_id or self.start_token_id]
        size = 0
        while size < max_size:
            if len(temp_nodes) == 0:
                break
            max_freq = 0.0
            best_node = None
            best_id = None
            for token_id in temp_nodes.keys():
                temp_node = temp_nodes[token_id]
                freqs_dict = temp_node.freqs_dict
                in_freq = freqs_dict.get(use_batch, 0.0)
                out_freq = freqs_dict.get(-1, 0.0)
                if in_freq > 0 or out_freq > 0:
                    temp_freq = out_freq + in_freq
                    if temp_freq > max_freq:
                        max_freq = temp_freq
                        best_node = temp_node
                        best_id = token_id

            if best_node is None:
                break
            ids.append(best_id)
            temp_nodes = best_node.next_node
            size += 1

        return ids, np.tril(np.ones((size + 1, size + 1), dtype=np.int64), 0), size

    def dfs_match(self, nodes, freqs, idx, output_weight):
        for node in nodes.values():
            fo = node.freqs_dict.get(-1, 0.0)
            fi = node.freqs_dict.get(idx, 0.0)
            if fo > 0 or fi > 0:
                fm = (1.0 - output_weight) * fi + output_weight * fo
                freqs.append([None, fi, fo, fm])
                if len(node.next_node) > 0:
                    self.dfs_match(node.next_node, freqs, idx, output_weight)

    def get_mask(
        self,
        nodes,
        ids,
        mask,
        pid,
        max_size=64,
        max_length=8,
        min_output_freq=1.0,
        min_input_freq=1.0,
        min_mix_freq=1.0,
        output_weight=1e-4,
        sizes=None,
        use_batch=0,
    ):
        if len(ids) >= max_size or max_length <= 0:
            return

        sorts = [
            (k, v, (1.0 - output_weight) * v.freqs_dict.get(use_batch, 0.0) + output_weight * v.freqs_dict.get(-1, 0.0))
            for k, v in nodes.items()
        ]
        sorts = sorted(sorts, key=lambda x: x[2], reverse=True)
        for tid, node, fm in sorts:
            if len(ids) >= max_size:
                return
            fi = node.freqs_dict.get(use_batch, 0.0)
            fo = node.freqs_dict.get(-1, 0.0)
            if fi < min_input_freq and fo < min_output_freq and fm < min_mix_freq:
                continue
            if fi > 0.0:
                sizes[0] += 1
            if fo > 0.0:
                sizes[1] += 1
            ids.append(tid)
            rid = len(ids) - 1

            if pid > -1:
                mask[rid] = mask[pid]
            mask[rid, rid] = 1
            if len(node.next_node) > 0:
                self.get_mask(
                    node.next_node,
                    ids,
                    mask,
                    rid,
                    max_size=max_size,
                    max_length=max_length - 1,
                    min_output_freq=min_output_freq,
                    min_input_freq=min_input_freq,
                    min_mix_freq=min_mix_freq,
                    output_weight=output_weight,
                    sizes=sizes,
                    use_batch=use_batch,
                )
            self.n_output_node = sizes[0]

    def trim(self):
        if self.n > self.max_size or self.n_output > self.max_output_size:
            temp_nodes = self.tree_nodes
            self.trim_recursive(temp_nodes)
            temp_size = [0]
            self.estimate(temp_nodes, temp_size)
            self.n = temp_size[-1]
            self.n_output = temp_size[-1]

    def estimate(self, temp_nodes, temp_size):
        temp_size[-1] += len(temp_nodes)
        for _, temp_node in temp_nodes.items():
            if len(temp_node.next_node) > 0:
                self.estimate(temp_node.next_node, temp_size)

    def trim_recursive(self, temp_nodes):
        for token_id, temp_node in list(temp_nodes.items()):
            output_freq = temp_node.freqs_dict.get(-1, 0.0)
            if output_freq > 1:
                temp_node.freqs_dict[-1] *= 0.5
                if len(temp_node.next_node) > 0:
                    self.trim_recursive(temp_node.next_node)
            else:
                temp_nodes.pop(token_id)

    def add_node(self, next_token_ids, pattern=PATTEN_OUTPUT, use_batch=0, dynamic_algo=False):
        output_flag = pattern == PATTEN_OUTPUT
        use_batch = -1 if output_flag else use_batch
        if dynamic_algo:
            leaves = self.tree_nodes
            while True:
                if len(next_token_ids) == 0:
                    break
                t = next_token_ids[0]
                node = leaves.get(t, None)
                if node is None:
                    n = {}
                    for token in next_token_ids[::-1]:
                        frequences = {use_batch: 1.0}
                        p = TreeNode(n, frequences)
                        n = {token: p}
                    leaves.update(n)
                    self.n += len(next_token_ids)
                    if pattern == PATTEN_OUTPUT:
                        self.n_output += len(next_token_ids)
                    break
                node.freqs_dict[use_batch] = node.freqs_dict.get(use_batch, 0.0) + 1.0
                leaves = node.next_node
                next_token_ids = next_token_ids[1:]
        else:
            len_tokens = len(next_token_ids)
            if len_tokens == 0:
                return
            start = 0
            temp_nodes = self.tree_nodes
            while start < len_tokens:
                temp_token_id = next_token_ids[start]
                temp_node = temp_nodes.get(temp_token_id, None)
                if temp_node is None:
                    next_node = {}
                    for token in next_token_ids[::-1]:
                        freqs = {use_batch: 1}
                        p = TreeNode(next_node, freqs)
                        next_node = {token: p}
                    temp_nodes.update(next_node)
                    self.n += len_tokens
                    self.n_output += len_tokens * output_flag
                    break
                temp_node.freqs_dict[use_batch] = temp_node.freqs_dict.get(use_batch, 0.0) + 1
                temp_nodes = temp_node.next_node
                start += 1

    def clear_input(self, use_batch):
        temp_nodes = self.tree_nodes
        if len(temp_nodes) < 1:
            return
        self._clear_input(temp_nodes, use_batch)

    def _clear_input(self, temp_nodes, use_batch):
        for _, temp_node in temp_nodes.items():
            temp_freqs = temp_node.freqs_dict
            if temp_freqs.get(use_batch, 0.0) == 0.0:
                continue
            temp_freqs[use_batch] = 0.0
            if len(temp_node.next_node) > 0:
                self._clear_input(temp_node.next_node, use_batch)


class TokensKnowledgeBaseCache:
    def __init__(self, eos=(2,), stop_words=None, dynamic_algo=False, max_size=65536, max_output_size=1024):
        self.max_size = max_size
        self.max_output_size = max_output_size
        self.token2tree = {}
        self.temp_output = defaultdict(list)
        self.changed_prefix_trees = set()
        self.changed_input_prefix_trees = set()
        self.stop_words = stop_words if stop_words is not None else {}
        self.eos = eos if eos is not None else [None]
        self.dynamic_algo = dynamic_algo

    def get_tokens_without_eos(self, input_ids: Union[list, np.ndarray]):
        for token_id in self.eos:
            if token_id in input_ids:
                if isinstance(input_ids, list):
                    # 输入的input_ids是next_token，在使用输出维护前缀树时使用
                    input_ids = input_ids[: input_ids.index(token_id)]
                else:
                    # 输入的input_ids是model_inputs种的input_ids，在prefill阶段使用输入维护前缀树时使用
                    input_ids = input_ids[input_ids != token_id]
        return input_ids

    def get_prefix_tree(self, temp_token_id, window_tokens, pattern=PATTEN_OUTPUT, use_batch=0):
        prefix_tree = self.token2tree.get(temp_token_id, None)
        if ENV.performance_prefix_tree:
            if prefix_tree:
                prefix_tree.put(window_tokens, pattern, use_batch)
            else:
                prefix_tree = _PrefixTree(temp_token_id, self.max_size, self.max_output_size)
                prefix_tree.put(window_tokens, pattern, use_batch)
                self.token2tree[temp_token_id] = prefix_tree
        else:
            if prefix_tree:
                prefix_tree.add_node(
                    window_tokens, pattern=pattern, use_batch=use_batch, dynamic_algo=self.dynamic_algo
                )
            else:
                prefix_tree = Tree(temp_token_id, max_size=self.max_size, max_output_size=self.max_output_size)
                prefix_tree.add_node(
                    window_tokens, pattern=pattern, use_batch=use_batch, dynamic_algo=self.dynamic_algo
                )
                self.token2tree[temp_token_id] = prefix_tree
        return prefix_tree

    def output_add(self, input_ids, search_size=8, final=False, pattern=PATTEN_OUTPUT, use_batch=0):
        input_ids = self.get_tokens_without_eos(input_ids)
        self.temp_output[use_batch].extend(input_ids)
        temp_output_ids = self.temp_output[use_batch]
        temp_size = len(temp_output_ids)
        min_search_size = 1 if final else search_size
        if temp_size > min_search_size:
            for i in range(temp_size - min_search_size):
                temp_token_id = temp_output_ids[i]
                if temp_token_id in self.stop_words:
                    continue
                window_tokens = temp_output_ids[i + 1 : i + search_size + 1]
                prefix_tree = self.get_prefix_tree(temp_token_id, window_tokens, pattern=pattern, use_batch=use_batch)
                self.changed_prefix_trees.add(prefix_tree)
            if not final:
                self.temp_output[use_batch] = temp_output_ids[temp_size - search_size :]
        if final:
            self.temp_output[use_batch] = []
            self.clear_input(use_batch)
            self.trim()

    def add(self, input_ids, search_size=8, pattern=PATTEN_INPUT, use_batch=0):
        input_ids = self.get_tokens_without_eos(input_ids)
        input_length = len(input_ids)
        if input_length > 1:
            for i in range(input_length - 1):
                temp_token_id = input_ids[i]
                window_tokens = input_ids[i + 1 : i + search_size + 1]
                prefix_tree = self.get_prefix_tree(temp_token_id, window_tokens, pattern=pattern, use_batch=use_batch)
                self.changed_prefix_trees.add(prefix_tree)
                if pattern == PATTEN_INPUT:
                    self.changed_input_prefix_trees.add(prefix_tree)

    def get_single_draft(self, token_ids, decoding_length=64, search_size=8, use_batch=0):
        decoding_masks = np.ones((1, 1), dtype=np.int64)
        if decoding_length <= 1 or search_size == 0:
            return token_ids[-1:], decoding_masks, []

        decoding_ids = None
        sizes = 0
        for index, token in enumerate(token_ids):
            prefix_tree = self.token2tree.get(token, None)
            if prefix_tree is not None:
                ids = token_ids[index + 1 :]
                if token in self.stop_words and len(ids) == 0:
                    continue
                if ENV.performance_prefix_tree:
                    decoding_ids, sizes = prefix_tree.get_one_draft(ids, use_batch, int(decoding_length - 1))
                    decoding_masks = np.tril(np.ones((sizes + 1, sizes + 1), dtype=np.int64), 0)
                else:
                    decoding_ids, decoding_masks, sizes = prefix_tree.search_one_draft(
                        ids, max_size=decoding_length - 1, use_batch=use_batch
                    )
                decoding_ids_length = len(decoding_ids)
                if decoding_ids_length >= search_size // 2:
                    break
        if decoding_ids is None:
            decoding_ids = token_ids[-1:]

        return decoding_ids, decoding_masks, [sizes]

    def get_all_batch_draft(self, token_id_list, batch_decoding_length=None, search_size=8, indices=None):
        decoding_id_list = []
        decoding_mask_list = []
        size_list = []
        for sub_idx, token_ids in enumerate(token_id_list):
            update_decoding_length = batch_decoding_length[sub_idx]
            decoding_ids, decoding_masks, sizes = self.get_single_draft(
                token_ids, decoding_length=update_decoding_length, search_size=search_size, use_batch=indices[sub_idx]
            )
            decoding_id_list.append(decoding_ids)
            decoding_mask_list.append(decoding_masks)
            size_list.append(sizes)

        max_size = max([len(x) for x in decoding_id_list])
        decoding_mask_list_temp = []
        if self.dynamic_algo:
            for i, decoding_ids in enumerate(decoding_id_list):
                decoding_size = len(decoding_ids)
                diff_size = max_size - decoding_size
                if diff_size > 0:
                    decoding_ids.extend([0] * diff_size)
                    temp_decoding_mask = np.tril(np.ones((max_size, max_size), dtype=np.int64), 0)
                    temp_decoding_mask[:decoding_size, :decoding_size] = decoding_mask_list[i]
                    decoding_mask_list_temp.append(temp_decoding_mask)
                else:
                    decoding_mask_list_temp.append(decoding_mask_list[i])
            decoding_mask_list = decoding_mask_list_temp

        return decoding_id_list, decoding_mask_list, size_list

    def clear_input(self, use_batch):
        for prefix_tree in self.changed_input_prefix_trees:
            if ENV.performance_prefix_tree:
                prefix_tree.reset_input_freq(use_batch)
            else:
                prefix_tree.clear_input(use_batch)

    def trim(self):
        if len(self.changed_prefix_trees) >= 1024:
            for prefix_tree in self.changed_prefix_trees:
                prefix_tree.trim()
            self.changed_prefix_trees.clear()
