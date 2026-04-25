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

from typing import Any, Callable
from enum import Enum


class SkipAction(Enum):
    CACHE = "cache"
    # Replace the result with cached data
    # ensuring that the output of each step is consistent with the non-skip version from the previous run,
    # while relying on cached result in every steps.
    REPEAT = "repeat"  # repeat the result
    # Replace with the latest cached results
    # while relying on one cached result at least
    RANDOM = "random"  # random the result
    EMPTY = "empty"  # do thing, and return empty


class BaseSkip:
    support_actions = []

    def __init__(self, local_config, global_config):
        self.action = SkipAction(local_config.action)
        self.enable = local_config.enable
        self.dump_dir = local_config.dump_dir
        self.steps = local_config.steps
        self.global_config = global_config
        self.global_step = -1
        if self.action not in self.support_actions:
            raise ValueError(f"Unsupported action: {self.action}. Supported actions are: {self.support_actions}")

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
