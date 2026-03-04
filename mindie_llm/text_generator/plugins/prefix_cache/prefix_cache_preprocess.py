# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import numpy as np

from ...utils.model_input import ModelInput
from ...utils.input_metadata import InputMetadata


class PrefixCachePreprocess:
    def __init__(self, infer_context, cp_size, scp_size, scp_rank):
        self.infer_context = infer_context
        self.block_size = infer_context.block_size
        self.cp_size = cp_size
        self.scp_size = scp_size
        self.scp_rank = scp_rank

    def update_infer_input(self, model_inputs: ModelInput, metadata: InputMetadata):
        if not metadata.is_prefill or metadata.remote_computed_blocks is None:
            return model_inputs
        remote_computed_blocks = metadata.remote_computed_blocks
        if metadata.batch_is_prefill is not None: # prefix_cache + splitfuse
            decode_len = metadata.batch_is_prefill.shape[0] - np.sum(metadata.batch_is_prefill)
            batch_size = metadata.batch_is_prefill.shape[0]
            prefill_idx = list(range(decode_len, batch_size))
        else:
            decode_len = metadata.mix_decode_bs
            batch_size = metadata.batch_size
            prefill_idx = list(range(metadata.mix_decode_bs, metadata.batch_size))

        computed_blocks = []
        for i, batch_sequence_ids in enumerate(metadata.batch_sequence_ids):
            for _ in batch_sequence_ids:
                computed_blocks.append(remote_computed_blocks[i])

        if self.scp_size > 1:
            batch_computed_blocks = metadata.remote_computed_blocks  # shape: [batch_size, csp_size] 
            computed_blocks = np.sum(batch_computed_blocks, axis=1)  # shape: [batch_size] 
        else:
            batch_computed_blocks = None

        # 需要刷新的参数包括input_ids, position_ids, slots, prefill_head_indices
        # 首先提取需要刷新的参数
        input_ids = model_inputs.input_ids
        position_ids = model_inputs.position_ids        
        slots = model_inputs.slots
        prefill_head_indices = model_inputs.prefill_head_indices

        # 先按照原来的长度初始化，后续再做截断
        new_input_ids = np.zeros(metadata.total_seq_num, dtype=np.int64)
        new_position_ids = np.zeros(metadata.total_seq_num, dtype=np.int64)
        new_slots = np.zeros(metadata.total_seq_num, dtype=np.int64)
        new_prefill_head_indices = np.arange(batch_size, dtype=np.int64)
        q_lens = np.zeros(batch_size, dtype=np.int32)

        # decode的部分可以直接赋值
        new_input_ids[:decode_len] = input_ids[:decode_len]
        new_position_ids[:decode_len] = position_ids[:decode_len]
        new_slots[:decode_len] = slots[:decode_len]
        prefill_head_indices[:decode_len] = new_prefill_head_indices[:decode_len]
        q_lens[:decode_len] = [1] * decode_len

        new_total_seq_num = decode_len
        input_start_idx = decode_len
        batch_end_idx = decode_len
        slots_start_idx = decode_len

        # prefill阶段有缓存时，prefill的input tokens需要减去缓存部分
        for i in prefill_idx:
            seq_len = metadata.batch_seq_len[i]
            if metadata.split_start_position is not None:
                split_start_position_i = metadata.split_start_position[i]
            batch_end_idx += seq_len
            # 不叠加splitfuse 或者 叠加splitfuse时被切的第一块， 要去除 被复用的block
            if metadata.split_start_position is None or split_start_position_i == 0:
                cached_size = computed_blocks[i] * self.block_size
                seq_len -= cached_size  # 减去已有缓存的seq_len
                no_cache_blocks = metadata.batch_block_tables[i][computed_blocks[i]:]
                if seq_len <= 0:  # 若有缓存导致seq_len<=0，少用一个block的缓存，保证输入大于0
                    seq_len += self.block_size
                    no_cache_blocks = metadata.batch_block_tables[i][computed_blocks[i] - 1:]
            else:
                # 和splitfuse 叠加时, 被切分的第2，3, 4...块 的 computed_blocks 不再生效。
                no_cache_blocks = metadata.batch_block_tables[i]

            new_total_seq_num += seq_len
            new_input_ids[input_start_idx:input_start_idx + seq_len] = input_ids[batch_end_idx - seq_len:batch_end_idx]
            new_position_ids[input_start_idx:input_start_idx + seq_len] = \
                                                                    position_ids[batch_end_idx - seq_len:batch_end_idx]
            if metadata.split_start_position is not None:  # 和splitfuse 叠加时，slot的起点不是0
                position_end_idx = split_start_position_i + seq_len
                new_slots[input_start_idx:input_start_idx + seq_len] = self.infer_context.block_table_to_slots(
                    no_cache_blocks
                ).reshape(-1)[split_start_position_i:position_end_idx]
            elif self.scp_size > 1:
                new_slots[input_start_idx:input_start_idx + seq_len] = self.get_slots(
                    model_inputs.slots[slots_start_idx: slots_start_idx + model_inputs.context_length[i]],
                    batch_computed_blocks[i],
                    model_inputs.sp_tokens[i]
                    )
            else:
                new_slots[input_start_idx:input_start_idx + seq_len] = \
                    self.infer_context.block_table_to_slots(no_cache_blocks).reshape(-1)[:seq_len]
            q_lens[i] = seq_len // self.cp_size
            if model_inputs.pad_token_count is not None:
                # scp 场景，需要去掉padding， pad_token_count 是每个请求padding的长度
                new_prefill_head_indices[i] = new_total_seq_num - 1 - model_inputs.pad_token_count[i]
            else:
                new_prefill_head_indices[i] = new_total_seq_num - 1
            input_start_idx += seq_len
            slots_start_idx += model_inputs.context_length[i]
            if self.scp_size > 1:
                model_inputs.context_length[i] -= computed_blocks[i] * self.block_size

        # 最后根据最新的长度做截断
        new_input_ids = new_input_ids[:new_total_seq_num]
        new_position_ids = new_position_ids[:new_total_seq_num]
        new_slots = new_slots[:new_total_seq_num]

        sp_computed_slots_padding_idx = self.get_sp_computed_slots_padding_idx(batch_computed_blocks)
        sp_computed_slots_order = \
            self.get_scp_computed_slots_order(batch_computed_blocks, metadata.batch_computed_block_order)
        all_rank_prefix_lens = self.get_all_rank_prefix_lens(batch_computed_blocks)
        per_rank_prefix_lens = self.get_per_rank_prefix_lens(batch_computed_blocks)

        model_inputs_new = ModelInput(input_ids=new_input_ids,
                                      position_ids=new_position_ids,
                                      block_tables=model_inputs.block_tables,
                                      slots=new_slots,
                                      context_length=model_inputs.context_length,
                                      max_seq_len=model_inputs.max_seq_len,
                                      prefill_head_indices=new_prefill_head_indices,
                                      is_prefill=model_inputs.is_prefill,
                                      query_length=q_lens,
                                      adapter_ids=metadata.adapter_ids,
                                      dp_rank_ids=metadata.batch_dp_rank_ids,
                                      sp_tokens=metadata.sp_tokens,
                                      cp_tokens=model_inputs.cp_tokens,
                                      pad_token_count=model_inputs.pad_token_count,
                                      seq_lens=metadata.seq_lens,
                                      sp_computed_slots_padding_idx=sp_computed_slots_padding_idx,
                                      sp_computed_slots_order=sp_computed_slots_order,
                                      all_rank_prefix_lens=all_rank_prefix_lens,
                                      per_rank_prefix_lens=per_rank_prefix_lens)
        return model_inputs_new

    def get_slots(self, all_rank_slots, rank_computed_blocks, sp_tokens):
        idx = 0

        all_rank_slots = all_rank_slots.tolist()
        for i, sp_token in enumerate(sp_tokens):
            all_rank_slots = all_rank_slots[: idx] + all_rank_slots[idx + rank_computed_blocks[i] * self.block_size:]
            idx += (sp_token - rank_computed_blocks[i] * self.block_size)
        all_rank_slots = np.array(all_rank_slots, dtype=np.int64)
        return all_rank_slots

    def get_sp_computed_slots_padding_idx(self, batch_computed_blocks):
        if batch_computed_blocks is None:
            return None
        rank_computed_block_nums = np.sum(batch_computed_blocks, axis=0)
        max_block_num = rank_computed_block_nums.max()
        sp_computed_slots_padding_idx = np.concatenate([
            np.arange(self.block_size * rank_computed_block_nums[self.scp_rank], dtype=np.int32),
            np.zeros(self.block_size * (max_block_num - rank_computed_block_nums[self.scp_rank]), dtype=np.int32)
        ]).reshape(-1)
        return sp_computed_slots_padding_idx
        
    def get_scp_computed_slots_order(self, scp_rank_computed_blocks, computed_block_order):
        if scp_rank_computed_blocks is None or computed_block_order is None:
            return None
        batch_size = scp_rank_computed_blocks.shape[0]
        computed_blocks_total_num = scp_rank_computed_blocks.sum()
        pre_sum = 0
        for i, pre_rank_computed_order in enumerate(computed_block_order):
            if len(pre_rank_computed_order) == 0:
                continue
            pre_rank_first_order = pre_rank_computed_order[0]
            pre_rank_computed_order = [pre_sum + x - pre_rank_first_order for x in pre_rank_computed_order]
            pre_sum += len(pre_rank_computed_order)
            computed_block_order[i] = pre_rank_computed_order

        all_rank_computed_block_order = [[] for _ in range(self.scp_size)]
        for bs_i in range(batch_size):
            pre_idx = 0
            cur_idx = 0
            for rank_idx in range(self.scp_size):
                cur_idx += scp_rank_computed_blocks[bs_i][rank_idx]
                all_rank_computed_block_order[rank_idx].extend(computed_block_order[bs_i][pre_idx:cur_idx])
                pre_idx = cur_idx

        # 由于复用block有gather补齐的动作， 要增加每个block的偏移量
        offset = [0] * computed_blocks_total_num
        max_computed_blocks = np.sum(scp_rank_computed_blocks, axis=0).max()
        offset_sum = 0
        for per_rank_computed_block_order in all_rank_computed_block_order:
            for block_order in per_rank_computed_block_order:
                offset[block_order] = offset_sum
            offset_sum += max_computed_blocks - len(per_rank_computed_block_order)

        # 4. 计算每个 block 经过all gather之后的调整顺序
        block_idx_after_all_gather = [-1] * computed_blocks_total_num
        idx = 0
        for per_rank_computed_block_order in all_rank_computed_block_order:
            for block_order in per_rank_computed_block_order:
                block_idx_after_all_gather[block_order] = idx + offset[block_order]
                idx += 1

        scp_computed_slots_order = []
        for block_idx in block_idx_after_all_gather:
            if block_idx == -1:
                continue
            scp_computed_slots_order.extend(
                [i for i in range(block_idx * self.block_size, (block_idx + 1) * self.block_size)]
            )

        scp_computed_slots_order = np.array(scp_computed_slots_order, np.int32)
        return scp_computed_slots_order

    def get_per_rank_prefix_lens(self, batch_computed_blocks):
        if batch_computed_blocks is None:
            return None
        cur_rank_computed_block_nums = batch_computed_blocks[:, self.scp_rank]
        per_rank_prefix_lens = cur_rank_computed_block_nums * self.block_size
        if sum(per_rank_prefix_lens) == 0:
            per_rank_prefix_lens[0] = self.block_size
        return per_rank_prefix_lens

    def get_all_rank_prefix_lens(self, batch_computed_blocks):
        if batch_computed_blocks is None:
            return None
        computed_blocks = np.sum(batch_computed_blocks, axis=1)
        all_rank_prefix_lens = computed_blocks * self.block_size
        return all_rank_prefix_lens