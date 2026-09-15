# Use RL-Insight to Monitor Training

Last updated: 07/15/2026.

[RL-Insight](https://github.com/verl-project/rl-insight) provides online observability for RL training. In verl, it can receive trainer scalar metrics, async rollout engine metrics, TransferQueue metrics, and rollout state traces, then show them in Grafana dashboards managed by the RL-Insight server.

## When to Use

Use RL-Insight when you want one monitor view for:

- trainer metrics such as rewards, losses, and throughput
- async vLLM or SGLang rollout server metrics
- TransferQueue metrics when TransferQueue is enabled
- RL state timelines around rollout generation
- CPU, memory, network, and Ascend NPU hardware metrics

## Step 1: Install and Start RL-Insight

Install RL-Insight in the environment where the monitor server runs. Prefer the latest source version:

```bash
pip install "git+https://github.com/verl-project/rl-insight.git"
```

Or install a released package:

```bash
pip install "rl-insight>=0.2.0"
```

### Install monitor services

`rl-insight server install` downloads Prometheus, Tempo, and Grafana into `~/.rl-insight/services`. The machine that runs this command needs network access to **GitHub release assets** and **`dl.grafana.com`**.

If that machine can reach those hosts:

```bash
rl-insight server install
rl-insight server start
```

If it cannot (common in air-gapped or restricted clusters), download the archives on a networked machine first, copy them to the RL-Insight host, then install from a local directory that contains all three archives. `/path/to/archives` below is only an example path — use any directory you choose, as long as the three packages are placed together in that directory.

Default download URLs for `linux-amd64` (installer versions):

| Service | Version | Download URL |
| --- | --- | --- |
| Prometheus | `2.54.1` | https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz |
| Tempo | `2.6.1` | https://github.com/grafana/tempo/releases/download/v2.6.1/tempo_2.6.1_linux_amd64.tar.gz |
| Grafana | `13.0.0` | https://dl.grafana.com/oss/release/grafana-13.0.0.linux-amd64.tar.gz |

For `linux-arm64`, replace `amd64` with `arm64` in the filenames and URLs (Tempo uses `linux_arm64` in the archive name). Filenames must match exactly.

```bash
rl-insight server install --local-archive /path/to/archives
rl-insight server start
```

`rl-insight server start` prints the detected server IP, Grafana URL, and related endpoints. Use that printed IP in the steps below. By default, RL-Insight uses:

| Service | Default port | Purpose |
| --- | --- | --- |
| RL-Insight server | `18080` | Receives metrics and trace registrations |
| Prometheus | `9090` | Stores and queries metrics |
| Tempo | `3200` | Stores traces |
| Grafana | `3000` | Shows dashboards |

## Step 2: Enable RL-Insight in verl

Set the RL-Insight server address before submitting the training job. `<server-ip>` must be the IP of the machine where you ran `rl-insight server start` (the address printed by that command), and it must be reachable from the training processes:

```bash
export RL_INSIGHT_SERVER_URL="http://<server-ip>:18080"
```

For a multi-node Ray cluster, add the variable to the runtime environment file submitted with the verl job, typically `verl/trainer/runtime_env.yaml`. This propagates the RL-Insight server address to workers on every node:

```yaml
env_vars:
  RL_INSIGHT_SERVER_URL: "http://<server-ip>:18080"
```

If your launch script passes another file through `ray job submit --runtime-env`, add the variable to that file instead.

Add `rl_insight` to `trainer.logger`. When `rl_insight` is enabled, verl sets `VERL_RL_INSIGHT_ENABLE=1` and initializes the RL-Insight client in each process that uses it.

```bash
python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl \
    trainer.experiment_name=ppo_rl_insight \
    ...
```

Trainer scalar metrics are reported to RL-Insight automatically through the logger backend.

## Step 3: Monitor Rollout and TransferQueue Metrics

For rollout engine metrics and TransferQueue metrics, keep rollout stats enabled and expose the TransferQueue metrics endpoint:

```bash
python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    actor_rollout_ref.rollout.disable_log_stats=False \
    transfer_queue.metrics.enabled=True \
    ...
```

When rollout replicas or TransferQueue metrics endpoints start, verl registers them with RL-Insight. The generation path is also wrapped with RL-Insight state traces for vLLM and SGLang rollout workers.

## Step 4: Add Hardware Metrics (Optional)

To monitor CPU, memory, network, or Ascend NPU metrics, follow the [RL-Insight Hardware Monitoring guide](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/hardware/index.md). The guide explains how to install or reuse the exporters and register their monitoring endpoints with RL-Insight.

## View Dashboards

1. Check the terminal output of `rl-insight server start` and open the printed Grafana URL. By default it is `http://<server-ip>:3000`, where `<server-ip>` is the RL-Insight host.
2. Log in with the default credentials:
   - username: `admin`
   - password: `admin`
3. In the left navigation, open **Dashboards**, then open the **RL-Insight** folder.
4. Select the dashboard that matches your run, for example:
   - `verl_trainer_v1_with_vllm_engine` for vLLM rollout
   - `verl_trainer_v1_with_sglang_engine` for SGLang rollout
5. Set the time range to a recent window such as **Last 5 minutes** / **Last 15 minutes** while training is still running.

The dashboards should include training metrics, rollout metrics, TransferQueue metrics if enabled, and rollout state timelines. Example views:

**RL state timeline (sync mode)**

![sync timeline](https://github.com/mengchengTang/verl-data/raw/master/sync_timeline.png)

**RL state timeline (separate async mode)**

![separate async timeline](https://github.com/mengchengTang/verl-data/raw/master/separate_async_timeline.png)

**Inference engine metrics across replicas**

![infer engine metric of all replicas](https://github.com/mengchengTang/verl-data/raw/master/infer_engine_metric_of_all_replica.png)

**TransferQueue metrics**

![transfer queue metric](https://github.com/mengchengTang/verl-data/raw/master/transfer_queue_metric.png)

**CPU hardware metrics**

![CPU hardware metrics](https://github.com/mengchengTang/verl-data/blob/master/cpu%E6%8C%87%E6%A0%87.png?raw=1)

## Platform Mode Without the RL-Insight Server

Use platform mode when the platform owns the observability stack: it provides an
OTLP-compatible trace endpoint and scrapes Prometheus itself. In this mode:

- **Traces are pushed** by RL-Insight to the platform OTLP endpoint.
- **Metrics are pulled** by the platform Prometheus from the endpoints below.
- **verl skips only the Prometheus target registrations added for RL-Insight**.
- The pre-existing `actor_rollout_ref.rollout.prometheus.enable` registration
  path remains unchanged. Enable it only if you also want verl to update its
  local Prometheus configuration.

### Enable platform mode

Set only the external OTLP endpoint:

```bash
export RL_INSIGHT_OTLP_ENDPOINT=https://collector.platform.example:4318/v1/traces
unset RL_INSIGHT_SERVER_URL
```

Then enable the RL-Insight logger:

```bash
python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl \
    trainer.experiment_name=ppo_platform \
    actor_rollout_ref.rollout.disable_log_stats=False \
    ...
```

`RL_INSIGHT_OTLP_ENDPOINT` must be a complete OTLP/HTTP endpoint, normally ending
in `/v1/traces`. verl and RL-Insight do not append the path.

If both `RL_INSIGHT_SERVER_URL` and `RL_INSIGHT_OTLP_ENDPOINT` are set,
`RL_INSIGHT_SERVER_URL` wins and verl uses the original managed-server mode.

### Scrape RL-Insight trainer metrics

RL-Insight creates one Ray `MonitorHubActor` per Ray job. The actor starts a
Prometheus HTTP server:

| Field | Value |
| --- | --- |
| Scheme | `http` |
| Path | `/metrics` |
| Host | IP of the Ray node that runs the `MonitorHubActor` |
| Port | `prometheus.metrics_report_port`; default `9092` |
| Format | Prometheus text exposition format |

The platform should scrape:

```text
http://<monitorhub-node>:9092/metrics
```

The MonitorHub address is not registered with an RL-Insight server in platform
mode. The platform must use its pre-agreed service-discovery rule to find the
Ray node and port.

Trainer metric names are prefixed with:

```text
rl_insight_monitor_
```

Examples:

| Family | Example metric |
| --- | --- |
| Actor | `rl_insight_monitor_actor_loss` |
| Critic | `rl_insight_monitor_critic_rewards_mean` |
| Length | `rl_insight_monitor_response_length_mean` |
| Timing | `rl_insight_monitor_timing_s_gen` |
| Throughput | `rl_insight_monitor_perf_throughput` |
| Training progress | `rl_insight_monitor_training_global_step` |

Common labels on these metrics are:

| Label | Meaning |
| --- | --- |
| `project` | `trainer.project_name` |
| `experiment_name` | `trainer.experiment_name` |
| `job` | Job name assigned by the platform Prometheus scrape config |
| `instance` | `host:port` assigned by Prometheus |

Additional labels may be present when instrumentation code passes them. The
platform should preserve all exposed labels.

Gauges use the metric name directly. Counters use the Prometheus `_total`
suffix. Histograms expose `_bucket`, `_sum`, and `_count` series.

### Scrape rollout engine metrics

vLLM and SGLang rollout replicas expose their own `/metrics` endpoints on the
rollout HTTP server address:

```text
http://<rollout-host>:<rollout-port>/metrics
```

Rollout ports are assigned dynamically. Platform mode suppresses only the
RL-Insight target registration; it does not change the pre-existing
`actor_rollout_ref.rollout.prometheus.enable` behavior. The platform must
discover rollout endpoints with its pre-agreed mechanism.

Keep rollout stats enabled and leave the pre-existing local Prometheus update
disabled unless you explicitly want that legacy behavior:

```bash
actor_rollout_ref.rollout.disable_log_stats=False
actor_rollout_ref.rollout.prometheus.enable=False
```

Representative metrics include:

| Engine | Metric examples |
| --- | --- |
| vLLM | `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:generation_tokens_total`, `vllm:e2e_request_latency_seconds_bucket` |
| SGLang | `sglang:num_running_reqs`, `sglang:num_queue_reqs`, `sglang:gen_throughput`, `sglang:cache_hit_rate`, `sglang:e2e_request_latency_seconds_bucket` |

Common engine labels include:

| Label | Meaning |
| --- | --- |
| `model_name` | Served model name |
| `engine` | Engine/worker identifier emitted by the rollout engine |
| `replica` | Rollout replica rank added by verl |
| `stage` | Present for prefill/decode disaggregation where applicable |

Do not rewrite engine metric names or remove labels; the bundled RL-Insight
dashboards query these names directly.

### Scrape TransferQueue metrics

If TransferQueue metrics are enabled, the TransferQueue process exposes a
Prometheus `/metrics` endpoint. Enable it with:

```bash
transfer_queue.metrics.enabled=True
```

Representative metrics include:

```text
tq_controller_request_total
tq_controller_request_duration_seconds_bucket
tq_storage_active_keys_total
tq_storage_capacity_total
tq_storage_utilization_ratio
tq_partition_production_progress
tq_partition_consumption_progress
```

Common labels include:

| Label | Meaning |
| --- | --- |
| `op_type` | Storage or controller operation type |
| `task_name` | Producer or consumer task name, where applicable |

The platform must discover the TransferQueue endpoint using its pre-agreed rule.
verl does not register it with RL-Insight in platform mode.

### Example scrape configuration

The exact service-discovery mechanism is platform-specific. Conceptually, the
platform Prometheus needs three jobs:

```yaml
scrape_configs:
  - job_name: rl-insight
    metrics_path: /metrics
    static_configs:
      - targets:
          - "<monitorhub-node>:9092"

  - job_name: rollout
    metrics_path: /metrics
    static_configs:
      - targets:
          - "<rollout-node>:<rollout-port>"

  - job_name: transfer-queue
    metrics_path: /metrics
    static_configs:
      - targets:
          - "<transferqueue-node>:<transferqueue-port>"
```

Replace the static targets with the platform's own discovery source. A scrape
interval of 10–15 seconds matches the existing RL-Insight dashboards.

### Platform data flow

```text
verl trainer
  -> RLInsightLogger
     -> Ray MonitorHubActor
        -> /metrics for platform Prometheus
        -> OTLP POST to RL_INSIGHT_OTLP_ENDPOINT

vLLM / SGLang rollout replica
  -> /metrics for platform Prometheus

TransferQueue
  -> /metrics for platform Prometheus
```

The platform should store these metrics in its own Prometheus and query them
with the metric names and labels documented above.

## Troubleshooting

- If trainer metrics do not appear, check that `trainer.logger` contains `rl_insight` and `RL_INSIGHT_SERVER_URL` points to the machine that runs `rl-insight server start`.
- If rollout metrics do not appear, check that `actor_rollout_ref.rollout.disable_log_stats=False` is set.
- If TransferQueue metrics do not appear, check that `transfer_queue.metrics.enabled=True` is set.
- If platform-mode metrics do not appear, check that the platform can reach the MonitorHub, rollout, and TransferQueue `/metrics` endpoints.
- If `server install` fails to download packages, use the offline `--local-archive` path above.

For more RL-Insight server installation details, see the [RL-Insight server installation guide](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/server_installation.md) and [quick start](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/quick_start.md).

## Related documentation

- [Agent Loop protocol](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/agent_loop_protocol.md)
- [Uni-Agent RL-Insight instrumentation guide](https://github.com/verl-project/uni-agent/blob/main/docs/source/concepts/rl-insight-integration.md)
