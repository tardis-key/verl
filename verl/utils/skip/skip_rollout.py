# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from typing import Callable
from verl.protocol import DataProto

from verl.utils.skip.skip_base import BaseSkip, register_skip

@register_skip("rollout")
class SkipRollout(BaseSkip):
    print_mark = "[SkipRollout()] "
    
    def __init__(self,local_config, global_config):
        super().__init__(local_config, global_config)
        self.dump_dir = local_config.dump_dir
        self.max_dump_step = local_config.max_dump_step
    
    def meet_precondition(self) -> bool:
        return True

    def warp_function(self, func: Callable, batch: DataProto, *args, **kwargs):
        func(batch, *args, **kwargs)

    def prepare_data(self, return_batch, batch, *args, **kwargs):
        pass