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

"""Unit tests for ``verl.utils.skip`` (SkipManager, RolloutSkip, config, registry)."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.skip.base_skip import SKIP_REGISTRY, BaseSkip, SkipAction, register_skip
from verl.utils.skip.config import RolloutSkipConfig, SkipManagerConfig
from verl.utils.skip.rollout_skip import RolloutSkip
from verl.utils.skip.skip_manager import SkipManager


def _reset_skip_manager_class_state() -> None:
    SkipManager.config = None  # type: ignore[attr-defined]
    SkipManager.step = -1  # type: ignore[attr-defined]
    SkipManager.skip_instances = {}  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_skip_manager():
    _reset_skip_manager_class_state()
    yield
    _reset_skip_manager_class_state()


def _minimal_rollout_skip_cfg(
    dump_dir: str,
    *,
    enable: bool = True,
    steps: list[int] | None = None,
    action: str = "cache",
) -> OmegaConf:
    steps = steps if steps is not None else [1]
    return OmegaConf.create(
        {
            "skip": {
                "rollout": {
                    "enable": enable,
                    "dump_dir": dump_dir,
                    "steps": steps,
                    "action": action,
                }
            },
            "actor_rollout_ref": {"rollout": {"skip": {"enable": False}, "n": 2}},
            "trainer": {"experiment_name": "ut_exp", "project_name": "ut_proj"},
            "data": {"gen_batch_size": 4, "max_prompt_length": 8, "max_response_length": 16},
            "n": 2,
        }
    )


def _local_rollout_config(cfg: OmegaConf) -> RolloutSkipConfig:
    """Match ``SkipManager`` / trainer: merge YAML dict into ``RolloutSkipConfig`` (not a plain dict)."""
    return omega_conf_to_dataclass(cfg.skip.rollout, RolloutSkipConfig)


def _project_dump_root(dump_dir: Path, cfg: OmegaConf) -> Path:
    exp = cfg.trainer.experiment_name
    proj = cfg.trainer.project_name
    gbs = cfg.data.gen_batch_size
    n = cfg.n
    inp = cfg.data.max_prompt_length
    out = cfg.data.max_response_length
    sub = f"{exp}_{proj}/GBS{gbs}_N{n}_in{inp}_out{out}"
    return dump_dir.joinpath(sub).resolve()


def _write_valid_step_dump(project_root: Path, step: int, proto: DataProto) -> Path:
    step_dir = project_root.joinpath(str(step))
    step_dir.mkdir(parents=True, exist_ok=True)
    proto.save_to_disk(step_dir.joinpath("gen_batch.dp"))
    step_dir.joinpath("meta.json").write_text(json.dumps({"global_steps": step}), encoding="utf-8")
    return step_dir


class TestRolloutSkipConfig:
    def test_defaults(self):
        c = RolloutSkipConfig()
        assert c.enable is False
        assert c.action == "cache"
        assert c.steps == []

    def test_invalid_action(self):
        with pytest.raises(AssertionError, match="action"):
            RolloutSkipConfig(action="not_an_action")

    def test_steps_must_be_int(self):
        with pytest.raises(AssertionError, match="steps"):
            RolloutSkipConfig(steps=[1, "x"])  # type: ignore[list-item]


class TestSkipRegistryAndBaseSkip:
    def test_rollout_registered(self):
        assert "rollout" in SKIP_REGISTRY

    def test_register_skip_adds_class(self):
        name = f"ut_dummy_skip_{uuid.uuid4().hex[:8]}"

        @register_skip(name)
        class _UtSkip(BaseSkip):
            support_actions = [SkipAction.EMPTY]

            def meet_precondition(self) -> bool:
                return True

            def warp_function(self, func, *args, **kwargs):
                return "warped"

            def prepare_data(self, result, *args, **kwargs):
                pass

        assert SKIP_REGISTRY[name] is _UtSkip
        del SKIP_REGISTRY[name]

    def test_base_skip_rejects_unsupported_action(self):
        class _Bad(BaseSkip):
            support_actions = [SkipAction.CACHE]

        with pytest.raises(ValueError, match="Unsupported action"):
            _Bad(
                RolloutSkipConfig(enable=True, action="repeat", steps=[1]),
                OmegaConf.create({}),
            )


class TestRolloutSkipPaths:
    def test_check_valid_step_path(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)

        assert rs._check_valid_step_path(root.joinpath("99")) is False

        proto = DataProto.from_dict(tensors={"x": torch.zeros(1)})
        _write_valid_step_dump(root, 7, proto)
        assert rs._check_valid_step_path(root.joinpath("7")) is True

    def test_get_available_steps_filters_invalid_dirs(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path))
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)
        root.mkdir(parents=True)

        (root / "not_int").mkdir()
        (root / "2").mkdir()
        proto = DataProto.from_dict(tensors={"x": torch.ones(1)})
        _write_valid_step_dump(root, 1, proto)
        _write_valid_step_dump(root, 10, proto)

        assert rs._get_available_steps() == [1, 10]

    def test_find_latest_step_exact_then_smaller_then_larger(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path))
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)
        proto = DataProto.from_dict(tensors={"x": torch.tensor([2.0])})

        _write_valid_step_dump(root, 5, proto)
        _write_valid_step_dump(root, 20, proto)

        rs.set_context(5)
        assert rs._find_latest_step() == 5

        rs.set_context(12)
        assert rs._find_latest_step() == 5

        rs.set_context(3)
        assert rs._find_latest_step() == 5

        shutil.rmtree(root)
        root.mkdir(parents=True)
        rs.set_context(100)
        assert rs._find_latest_step() == -1


class TestRolloutSkipMeetWarpPrepare:
    def test_meet_precondition_cache_miss_and_hit(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), action="cache")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)

        rs.set_context(1)
        assert rs.meet_precondition() is False

        proto = DataProto.from_dict(tensors={"t": torch.arange(3)})
        _write_valid_step_dump(root, 1, proto)
        assert rs.meet_precondition() is True

    def test_meet_precondition_repeat(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), action="repeat")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        root = _project_dump_root(tmp_path, cfg)
        proto = DataProto.from_dict(tensors={"t": torch.tensor([1.0])})

        rs.set_context(2)
        assert rs.meet_precondition() is False

        _write_valid_step_dump(root, 1, proto)
        assert rs.meet_precondition() is True

    def test_prepare_data_and_warp_roundtrip(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), action="cache")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        rs.set_context(3)
        original = DataProto.from_dict(tensors={"k": torch.tensor([[1.0, 2.0]])})
        rs.prepare_data(original)

        loaded = rs.warp_function(lambda: None)
        assert torch.allclose(loaded.batch["k"], original.batch["k"])


class TestSkipManagerInitAndAnnotate:
    def test_init_builds_rollout_skip_instance(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), steps=[1, 2])
        SkipManager.init(cfg)
        assert SkipManager.config is not None
        assert "rollout" in SkipManager.skip_instances
        inst = SkipManager.skip_instances["rollout"]
        assert inst.is_enabled() is True
        assert inst.steps == [1, 2]

    def test_legacy_skip_enable_warns_and_can_disable_legacy(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), steps=[1])
        cfg.actor_rollout_ref.rollout.skip.enable = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SkipManager.init(cfg)
            kinds = [x.category for x in w]
            assert DeprecationWarning in kinds
        assert cfg.actor_rollout_ref.rollout.skip.enable is False

    def test_annotate_sync_bypass_when_step_not_in_steps(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), enable=True, steps=[99])
        SkipManager.init(cfg)

        @SkipManager.annotate(role="rollout")
        def work(x: int) -> int:
            return x + 1

        SkipManager.set_step(1)
        assert work(40) == 41

    def test_annotate_sync_prepare_when_enabled_and_cache_miss(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        SkipManager.init(cfg)
        root = _project_dump_root(tmp_path, cfg)

        @SkipManager.annotate(role="rollout")
        def gen(_: Any = None) -> DataProto:
            return DataProto.from_dict(tensors={"z": torch.tensor([7.0])})

        SkipManager.set_step(1)
        out = gen()
        assert out.batch["z"].item() == 7.0
        assert (root / "1" / "gen_batch.dp").exists()
        assert (root / "1" / "meta.json").exists()

        SkipManager.set_step(1)

        @SkipManager.annotate(role="rollout")
        def gen2(_: Any = None) -> DataProto:
            raise AssertionError("should not run when cache hit")

        loaded = gen2()
        assert torch.allclose(loaded.batch["z"], torch.tensor([7.0]))

    def test_annotate_async_same_semantics(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        SkipManager.init(cfg)
        root = _project_dump_root(tmp_path, cfg)

        @SkipManager.annotate(role="rollout")
        async def agen() -> DataProto:
            return DataProto.from_dict(tensors={"a": torch.tensor([1, 2, 3])})

        async def _run():
            SkipManager.set_step(1)
            out = await agen()
            assert list(out.batch["a"].tolist()) == [1, 2, 3]
            assert (root / "1" / "gen_batch.dp").exists()

            SkipManager.set_step(1)

            @SkipManager.annotate(role="rollout")
            async def agen2() -> DataProto:
                raise AssertionError("cached path")

            loaded = await agen2()
            assert list(loaded.batch["a"].tolist()) == [1, 2, 3]

        asyncio.run(_run())


class TestSkipManagerConfigDataclass:
    def test_skip_manager_config_merge(self):
        c = SkipManagerConfig()
        assert isinstance(c.rollout, RolloutSkipConfig)
        assert c.rollout.enable is False


class TestSkipManagerRuntimeScenarios:
    def test_annotate_unknown_role_is_noop(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), enable=True, steps=[1])
        SkipManager.init(cfg)
        SkipManager.set_step(1)

        @SkipManager.annotate(role="unknown_role")
        def f(x: int) -> int:
            return x * 2

        assert f(3) == 6

    def test_legacy_only_enabled_warns_but_keeps_legacy_flag(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), enable=False, steps=[1])
        cfg.actor_rollout_ref.rollout.skip.enable = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SkipManager.init(cfg)
        assert any(item.category is DeprecationWarning for item in w)
        assert cfg.actor_rollout_ref.rollout.skip.enable is True

    def test_new_and_legacy_enabled_disables_legacy(self, tmp_path: Path):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), enable=True, steps=[1])
        cfg.actor_rollout_ref.rollout.skip.enable = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SkipManager.init(cfg)
        assert any(item.category is DeprecationWarning for item in w)
        assert cfg.actor_rollout_ref.rollout.skip.enable is False


class TestSkipDumpDiskScenarios:
    def test_dump_dirs_are_isolated_by_config(self, tmp_path: Path):
        dump_a = tmp_path / "disk_a"
        dump_b = tmp_path / "disk_b"
        cfg_a = _minimal_rollout_skip_cfg(str(dump_a), enable=True, steps=[1], action="cache")
        cfg_b = _minimal_rollout_skip_cfg(str(dump_b), enable=True, steps=[1], action="cache")
        local_a = _local_rollout_config(cfg_a)
        local_b = _local_rollout_config(cfg_b)
        rs_a = RolloutSkip(local_a, cfg_a)
        rs_b = RolloutSkip(local_b, cfg_b)

        rs_a.set_context(1)
        rs_b.set_context(1)
        rs_a.prepare_data(DataProto.from_dict(tensors={"x": torch.tensor([1.0])}))
        rs_b.prepare_data(DataProto.from_dict(tensors={"x": torch.tensor([2.0])}))

        loaded_a = rs_a.warp_function(lambda: None)
        loaded_b = rs_b.warp_function(lambda: None)
        assert loaded_a.batch["x"].item() == 1.0
        assert loaded_b.batch["x"].item() == 2.0

    def test_prepare_data_handles_disk_write_error_without_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = _minimal_rollout_skip_cfg(str(tmp_path), enable=True, steps=[1], action="cache")
        local = _local_rollout_config(cfg)
        rs = RolloutSkip(local, cfg)
        rs.set_context(1)

        def _raise_save(*args, **kwargs):
            raise OSError("simulated disk write failure")

        monkeypatch.setattr(DataProto, "save_to_disk", _raise_save)
        rs.prepare_data(DataProto.from_dict(tensors={"x": torch.tensor([1.0])}))
        dump_file = rs._get_step_dump_dir().joinpath("gen_batch.dp")
        assert dump_file.exists() is False
