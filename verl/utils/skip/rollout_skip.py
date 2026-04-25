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
from pathlib import Path
import json

from verl.utils.skip.base_skip import BaseSkip, register_skip, SkipAction


@register_skip("rollout")
class RolloutSkip(BaseSkip):
    support_actions = [SkipAction.CACHE, SkipAction.REPEAT]
    print_mark = "[RolloutSkip()] "
    gen_batch_name = "gen_batch.dp"
    meta_name = "meta.json"

    def __init__(self, local_config, global_config):
        super().__init__(local_config, global_config)
        # prepare experiment info
        self.exp_name = global_config.trainer.get("experiment_name", "default_experiment_name")
        self.project_name = global_config.trainer.get("project_name", "default_project_name")
        self.n = int(getattr(self.global_config, "n", 0))
        self.gbs = int(global_config.data.get("gen_batch_size", global_config.data.get("train_batch_size", 0)))
        self.response_length = global_config.data.get("max_response_length", 0)
        self.prompt_length = global_config.data.get("max_prompt_length", 0)

    def meet_precondition(self) -> bool:
        if self.action == SkipAction.CACHE:
            if not self._check_valid_step_path(self._get_step_dump_dir()):
                print(
                    f"{self.print_mark}\033[33mNo dumped data found at gen_step {self.global_step} "
                    f"from {self._get_project_dump_dir()}. The trainer will generate and dump the data for this gen_step.\033[0m",
                    flush=True,
                )
                return False
            else:
                return True
        elif self.action == SkipAction.REPEAT:
            if self._find_latest_step() == -1:
                print(
                    f"{self.print_mark}\033[33mNo dumped data found "
                    f"from {self._get_project_dump_dir()}. The trainer will generate and dump the data for this gen_step.\033[0m",
                    flush=True,
                )
                return False
            else:
                return True
        return False

    def warp_function(self, func: Callable, *args, **kwargs):
        """Load cached gen batch; ``*args``/``kwargs`` mirror the decorated call (e.g. ``self, prompts``)."""
        if self.action == SkipAction.CACHE:
            step_dir = self._get_step_dump_dir()
        elif self.action == SkipAction.REPEAT:
            step_dir = self._get_step_dump_dir(self._find_latest_step())
        else:
            step_dir = Path()
        gen_batch_path = step_dir.joinpath(self.gen_batch_name)
        return DataProto.load_from_disk(gen_batch_path)

    def prepare_data(self, result, *args, **kwargs):
        step_dir = self._get_step_dump_dir()
        step_dir.mkdir(parents=True, exist_ok=True)
        try:
            result.save_to_disk(step_dir.joinpath(self.gen_batch_name))
            meta_path = step_dir.joinpath(self.meta_name)
            meta_path.write_text(json.dumps({"global_steps": self.global_step}))
        except Exception as e:
            print(
                f"{self.print_mark}\033[31mFailed to dump data in {step_dir}: {e}\033[0m",
                flush=True,
            )

    def _get_project_dump_dir(self) -> Path:
        dumped_dir = Path(self.dump_dir).expanduser().resolve()
        sub_dir = (
            f"{self.exp_name}_{self.project_name}"
            + f"/GBS{self.gbs}_N{self.n}_in{self.prompt_length}_out{self.response_length}"
        )
        dumped_dir = dumped_dir.joinpath(sub_dir).absolute()
        return dumped_dir

    def _get_step_dump_dir(self, step=None) -> Path:
        if step is None:
            step = self.global_step
        return self._get_project_dump_dir().joinpath(f"{step}").absolute()

    def _check_valid_step_path(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        gen_batch_path = path.joinpath(self.gen_batch_name)
        meta_path = path.joinpath(self.meta_name)
        return (
            gen_batch_path.exists()
            and gen_batch_path.is_file()
            and meta_path.exists()
            and meta_path.is_file()
        )

    def _get_available_steps(self) -> list[int]:
        result: list[int] = []
        project_dir = self._get_project_dump_dir()
        if not project_dir.is_dir():
            return result
        for child in project_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                step = int(child.name)
            except ValueError:
                continue
            if not self._check_valid_step_path(child):
                continue
            result.append(step)
        return sorted(result)

    def _find_latest_step(self) -> int:
        """Prefer exact ready step, else max step < current, else min step > current; -1 if none."""
        if self._check_valid_step_path(self._get_step_dump_dir()):
            return int(self.global_step)
        available = self._get_available_steps()
        if not available:
            return -1
        return available[-1]
