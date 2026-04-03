# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
from typing import Any, Dict

from mindie_llm.runtime.utils.helpers.safety.file import safe_open


class LLMConfig:
    """
    Parser for json configuration files.
    Usage:
        config = ConfigParser("config.json")
        value = config.get("section.key", default="default value")
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the config parser with the path to the json file.
        Args:
            config_path: Path to the json configuration file.
        """
        self._config_path = config_path
        self._config_dict: Dict[str, Any] = {}
        self._load_config()

    def __repr__(self):
        """
        Return a detailed string representation of the configuration

        Returns:
            str: Representation showing config path and top-level sections
        """
        sections = {}
        for key, value in vars(self).items():
            if not key.startswith('_'):
                sections[key] = value

        section_reprs = []
        for name, value in sections.items():
            formatted_value = format_value(value, indent_level=1)
            section_reprs.append(f"\n    {name}={formatted_value}")

        return f"LLMConfig(\n    path={repr(str(self._config_path))}," + \
            "".join(section_reprs) + \
            "\n)"

    def update(self, config_dict: Dict[str, Any], allow_new_keys: bool = False, current_path: str = '') -> None:
        """
        Update configuration values with provided keyword arguments
        Args:
            allow_new_keys: update new keys when allow_new_keys is True, otherwise only update old keys
            config_dict: Configuration dictionary to update
            current_path: update path
        """
        if not config_dict:
            return
        self._recursive_update(base=self, update=config_dict, allow_new_keys=allow_new_keys, current_path=current_path)

    def merge_models_config(self, model_name: str) -> None:
        """merge model.model_name config to llm config"""
        model_key = f'{model_name}'
        model_config = self.get(model_key)
        if model_config is None:
            return
        model_config_dict = model_config.to_dict()
        self.update(model_config_dict, allow_new_keys=False)

    def check_config(self, validators: Dict):
        """check config's type and range"""
        sections = {}

        for key, value in vars(self).items():
            if not key.startswith('_'):
                sections[key] = value.to_dict() if isinstance(value, SectionConfig) else value

        try:
            validators.validate(sections)
        except KeyError as ke:
            raise KeyError(f"Validation failed, because some keys are missing: {str(ke)}") from ke
        except Exception as e:
            raise RuntimeError(f"Validation failed: {str(e)}") from e

    def get(self, key) -> Any:
        """
        get LLMConfig key
        Args:
            key: LLMConfig key path
        Returns:
            SectionConfig or value
        """
        current = self
        keys = key.split('.')

        for key in keys:
            if not hasattr(current, key):
                return None
            current = getattr(current, key)

        return current

    def _load_config(self) -> None:
        """
        Load and parse json configuration file.
        """
        try:
            with safe_open(self._config_path, 'r', encoding='utf-8') as f:
                self._config_dict = json.load(f)
                self._apply_config(self._config_dict)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'config file not found: {self._config_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError('Invalid json in config file') from e

    def _apply_config(self, config_dict: Dict[str, Any]):
        """
        Recursively convert dictionary to nested attributes.

        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    self._recursive_update(self, value, allow_new_keys=True, current_path=key)
                else:
                    setattr(self, key, SectionConfig.create(value))
            else:
                if isinstance(value, dict):
                    setattr(self, key, SectionConfig.create(value))
                elif isinstance(value, list):
                    setattr(self, key,
                            [SectionConfig.create(item) if isinstance(item, dict) else self._apply_config(item)
                            for item in value])
                else:
                    setattr(self, key, convert_type(value))

    def _recursive_update(self, base: Any, update: Dict[str, Any], current_path: str, allow_new_keys=False):
        """Recursively update keys"""
        if not isinstance(update, dict):
            raise ValueError(f"Expected dict at {current_path}, got {type(update)}")

        for key, value in update.items():
            new_path = key if not current_path else f"{current_path}.{key}"
            if isinstance(value, dict):
                child = base.get(new_path)
                if child is not None:
                    self._recursive_update(
                        base=child,
                        update=value,
                        current_path='',
                        allow_new_keys=allow_new_keys
                    )
                else:
                    setattr(base if not current_path else base.get(current_path), key, SectionConfig.create(value))
            elif isinstance(value, list):
                if hasattr(base, key) or allow_new_keys:
                    converted = [
                        SectionConfig.create(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                    setattr(base, key, converted)
                else:
                    pass
            else:
                tmp_path = base.get(current_path) if current_path else base
                if hasattr(tmp_path, key) or allow_new_keys:
                    setattr(tmp_path, key, convert_type(value))
                else:
                    pass


def convert_type(value: Any) -> Any:
    """
    Automatically detect and convert basic types
    Args:
        value: Value to convert
    Returns:
        Converted value
    """
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ('true', 'false', 'yes', 'no', 'on', 'off'):
            return value.lower() in ('true', 'yes', 'on')
        if value.lower() in ('none', 'null', ''):
            return None
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            pass # Skip values that can't be converted to float
    return value


def format_value(v, indent_level):
    """Recursively format values with proper indentation"""
    indent = '    ' * indent_level
    if isinstance(v, SectionConfig):
        return f"\n{indent}SectionConfig(" + \
            format_dict(v.__dict__, indent_level + 1) + \
            f"\n{indent})"
    elif isinstance(v, list):
        items = []
        for item in v:
            if isinstance(item, (SectionConfig, dict, list)):
                items.append(format_value(item, indent_level + 1))
            else:
                items.append(repr(item))
        return f"[\n{indent}    " + \
            f",\n{indent}    ".join(items) + \
            f"\n{indent}]"
    elif isinstance(v, dict):
        return f"\n{indent}{{" + \
            format_dict(v, indent_level + 1) + \
            f"\n{indent}}}"
    else:
        return repr(v)


def format_dict(d, indent_level):
    """Format dictionary-like objects with indentation"""
    indent = '    ' * indent_level
    items = []
    for k, v in d.items():
        if not k.startswith('_'):
            formatted_value = format_value(v, indent_level + 1)
            items.append(f"\n{indent}{k}={formatted_value}")
    return ",".join(items) if items else ""


class SectionConfig:
    """
    Helper class for representing nested configurations with automatic type detection
    """
    def __init__(self):
        pass

    def __repr__(self):
        items = [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_')]
        return f"SectionConfig({', '.join(items)})"

    @classmethod
    def create(cls, value: Any):
        """
        Factory method to create appropriate config objects

        Args:
            value: Value to convert (dict, list, or primitive)

        Returns:
            ConfigObject for dicts, processed list, or converted primitive
        """
        if isinstance(value, dict):
            obj = cls()
            for k, v in value.items():
                setattr(obj, k, cls.create(v))
            return obj
        elif isinstance(value, list):
            return [cls.create(item) for item in value]
        else:
            return convert_type(value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert SectionConfig to dict
        """
        result = {}
        for key, value in vars(self).items():
            if key.startswith('_'):
                continue

            if isinstance(value, SectionConfig):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, SectionConfig) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def get(self, key) -> Any:
        """
        get config value from key
        Args:
            key: LLMConfig key path
        Returns:
            SectionConfig or value
        """
        current = self
        keys = key.split('.')

        for key in keys:
            if not hasattr(current, key):
                return None
            current = getattr(current, key)

        return current
