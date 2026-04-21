# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List
from enum import Enum

class SkipAction(Enum):
    CACHE = "cache" # replace result with cached result
    REPEAT = "repeat" # repeat the result
    RANDOM = "random" # random the result


class BaseSkip:
    def __init__(self, local_config, global_config):
        self.action = local_config.action
        self.enable = local_config.enable
        self.global_config = global_config
        self.global_step = -1

    def set_context(self, global_step: int):
        self.global_step = global_step

    def is_enabled(self) -> bool:
        return self.enable
    
    def meet_precondition(self) -> bool:
        raise NotImplementedError("meet_precondition is not implemented")

    def warp_function(self, func: Callable, *args, **kwargs):
        raise NotImplementedError("warp_function is not implemented")

    def prepare_data(self, result, *args, **kwargs):
        raise NotImplementedError("prepare_data is not implemented")


SKIP_REGISTRY: dict[str, type[BaseSkip]] = {}


def register_skip(
    name: str,
) -> Callable[[type[BaseSkip]], type[BaseSkip]]:
    def decorator(cls: type[BaseSkip]) -> type[BaseSkip]:
        SKIP_REGISTRY[name] = cls
        return cls

    return decorator


def get_cluster_parser_cls(name):
    if name not in SKIP_REGISTRY:
        raise ValueError(
            f"Unsupported cluster parser: {name}. Supported cls are: {list(SKIP_REGISTRY.keys())}"
        )
    return SKIP_REGISTRY[name]