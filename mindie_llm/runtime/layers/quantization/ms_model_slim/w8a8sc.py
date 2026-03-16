# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from pathlib import Path

import torch
import torch_npu
from torch import nn

from mindie_llm.runtime.layers.linear.linear_method_base import LinearMethodBase
from mindie_llm.runtime.layers.parameter import (
    BaseParameter,
    BiasParameter,
    ModelWeightParameter,
    ScalerParameter,
    PerTensorScaleParameter,
)
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager


MAPPER_REGISTRY = {
        'qwen2': 'mindie_llm.runtime.models.qwen2.config_qwen2.Qwen2Config',
        'qwen3': 'mindie_llm.runtime.models.qwen3.config_qwen3.Qwen3Config',
    }


def sparse_compressed_weight_loader(param: BaseParameter, loaded_weight: torch.Tensor) -> None:
    if param.data.shape != loaded_weight.shape:
        param.data = torch.empty_like(loaded_weight, dtype=param.data.dtype, device=param.data.device)
    param.load_weight(loaded_weight)


def get_part_directory_for_rank(model_path: str):
    """Get the part directory (part*-of-*) for the current tensor parallel rank.

    Args:
        model_path: Path to the model directory containing part*-of-* subdirectories.

    Returns:
        Path object pointing to the part directory for the current rank.
    """
    part_dirs = sorted(Path(model_path).glob("part*-of-*"))
    parallel_info = get_parallel_info_manager()
    if not parallel_info:
        raise RuntimeError("Parallel info manager is not initialized")

    tp_rank = parallel_info.rank
    total_parts = len(part_dirs)

    for part_path in part_dirs:
        part_name = part_path.name
        try:
            part_num = int(part_name.split("-")[0].replace("part", ""))
            parsed_total = int(part_name.split("-")[2].replace("of", ""))
            if parsed_total > 0:
                total_parts = parsed_total
            if part_num == tp_rank % total_parts:
                return part_path
        except (ValueError, IndexError):
            continue
    raise RuntimeError(f"Could not find part directory for TP rank {tp_rank}")



def get_weight_mapper_cls(hf_config):
    """Get Config class from  model_type"""
    model_type = getattr(hf_config, 'model_type', None)
    

    if model_type in MAPPER_REGISTRY:
        module_path, class_name = MAPPER_REGISTRY[model_type].rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    return None


class W8A8SCLinearMethod(LinearMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight = ModelWeightParameter(
            data=torch.empty(1, dtype=torch.int8),
        )
        weight.add_attrs({
            self.INPUT_DIM: 1,
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs,
        })
        weight.weight_loader = sparse_compressed_weight_loader

        input_scale = ScalerParameter(data=torch.empty(1, dtype=weight_dtype))
        input_scale.add_attrs(extra_weight_attrs)
        input_scale.weight_loader = sparse_compressed_weight_loader

        input_offset = ScalerParameter(data=torch.empty(1, dtype=torch.int8))
        input_offset.add_attrs(extra_weight_attrs)
        input_offset.weight_loader = sparse_compressed_weight_loader

        deq_scale_dtype = weight_dtype
        if weight_dtype == torch.float16:
            deq_scale_dtype = torch.int64
        elif weight_dtype == torch.bfloat16:
            deq_scale_dtype = torch.float32
        else:
            raise ValueError(f"Dtype {weight_dtype} is not supported in `W8A8SCLinearMethod`.")

        deq_scale = PerTensorScaleParameter(data=torch.empty(sum(output_partition_sizes), dtype=deq_scale_dtype))
        deq_scale.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})
        deq_scale.weight_loader = sparse_compressed_weight_loader

        quant_bias = BiasParameter(data=torch.empty(sum(output_partition_sizes), dtype=torch.int32))
        quant_bias.add_attrs({self.INPUT_DIM: 0, self.OUTPUT_DIM: 0, **extra_weight_attrs})
        quant_bias.weight_loader = sparse_compressed_weight_loader

        index = ModelWeightParameter(
            data=torch.empty(1, dtype=torch.int8),
        )
        index.add_attrs(extra_weight_attrs)
        index.weight_loader = sparse_compressed_weight_loader

        layer.register_parameter("weight", weight)
        layer.register_parameter("input_scale", input_scale)
        layer.register_parameter("input_offset", input_offset)
        layer.register_parameter("deq_scale", deq_scale)
        layer.register_parameter("quant_bias", quant_bias)
        layer.register_parameter("index", index)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_tensor_quant = torch_npu.npu_quantize(
            input=x, scales=layer.input_scale.data,
            zero_points=layer.input_offset.data,
            dtype=torch.qint8, axis=-1, div_mode=False)
        out = torch_npu.npu_quant_matmul(
            input_tensor_quant, layer.weight.data, layer.deq_scale.data,
            bias=layer.quant_bias.data, output_dtype=layer.weight_dtype)
        return out

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer.weight.data.numel() > 0:
            layer.weight.data = layer.weight.data.contiguous()
        if layer.index.data.numel() > 0:
            layer.index.data = layer.index.data.contiguous()
