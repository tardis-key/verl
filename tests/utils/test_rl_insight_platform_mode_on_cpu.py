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

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from verl.utils.tracking import RLInsightLogger


@pytest.fixture
def mock_rl_insight(monkeypatch: pytest.MonkeyPatch):
    module = MagicMock()
    module.metric_gauge = MagicMock()

    @contextmanager
    def _trace_state(*args, **kwargs):
        yield

    module.trace_state.side_effect = _trace_state
    monkeypatch.setattr(RLInsightLogger, "_get_rl_insight", classmethod(lambda cls: module))
    return module


def test_platform_mode_skips_metric_registration(monkeypatch, mock_rl_insight):
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")
    monkeypatch.delenv("RL_INSIGHT_SERVER_URL", raising=False)
    monkeypatch.setenv("RL_INSIGHT_OTLP_ENDPOINT", "http://collector:4318/v1/traces")
    mock_rl_insight.is_platform_mode.return_value = True

    RLInsightLogger.register_metrics(["127.0.0.1:8000"], "vllm", [{"replica": 0}])

    mock_rl_insight.update_prometheus_config.assert_not_called()


def test_managed_mode_wins_when_both_urls_are_set(monkeypatch, mock_rl_insight):
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")
    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", "http://127.0.0.1:18080")
    monkeypatch.setenv("RL_INSIGHT_OTLP_ENDPOINT", "http://collector:4318/v1/traces")
    mock_rl_insight.is_platform_mode.return_value = False

    RLInsightLogger.register_metrics(["127.0.0.1:8000"], "vllm", [{"replica": 0}])

    mock_rl_insight.update_prometheus_config.assert_called_once_with(["127.0.0.1:8000"], "vllm", [{"replica": 0}])
