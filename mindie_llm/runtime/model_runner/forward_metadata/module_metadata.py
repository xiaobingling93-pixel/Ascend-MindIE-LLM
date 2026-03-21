# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass, fields
from abc import abstractmethod
from typing import Dict, Any
import torch


@dataclass
class ModuleMetadata:
    """Base class for module metadata.
    
    This abstract base class defines the interface for metadata used during
    model forward passes. Subclasses should implement all abstract methods.
    """
    
    @staticmethod
    @abstractmethod
    def from_model_input(model_input: Any) -> 'ModuleMetadata':
        """Create metadata from model input.
        
        Args:
            model_input: Model input data.
            
        Returns:
            Instance of metadata.
        """
        pass

    @staticmethod
    @abstractmethod
    def is_enabled() -> bool:
        """Check if this metadata type is enabled.
        
        Returns:
            True if enabled, False otherwise.
        """
        pass

    @staticmethod
    @abstractmethod
    def register_buffer(max_num_token: int, device: torch.device) -> None:
        """Register buffer for metadata.
        
        Args:
            max_num_token: Maximum number of tokens.
            device: Target device.
        """
        pass

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'ModuleMetadata':
        """Create metadata from dictionary.
        
        NOTE: This method will be deprecated after `model_runner_exp.py` is in use.
        
        Args:
            dict_data: Dictionary containing metadata fields.
            
        Returns:
            Instance of metadata.
        """
        field_names = {field.name for field in fields(cls)}
        filtered_dict = {k: v for k, v in dict_data.items() if k in field_names}
        return cls(**filtered_dict)

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        """Move metadata tensors to the specified device.
        
        Args:
            device: Target device.
        """
        pass

    @abstractmethod
    def copy(self, num_actual_tokens: int, num_tokens: int) -> None:
        """Copy metadata with new token counts.
        
        Args:
            num_actual_tokens: Actual number of tokens.
            num_tokens: Total number of tokens (including padding).
        """
        pass

    def record_stream(self, stream: torch.npu.Stream) -> None:
        """Record the stream for metadata tensors.

        Subclasses are not allowed to have nested structures.
        
        Args:
            stream: NPU stream to record.
        """
        def _record(obj):
            if isinstance(obj, torch.Tensor) and obj.is_npu:
                obj.record_stream(stream)
            elif isinstance(obj, list) or isinstance(obj, tuple):
                for item in obj:
                    _record(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _record(v)

        for value in self.__dict__.values():
            _record(value)
