# 使用 RL-Insight 监控 verl 训练

RL-Insight 为 RL 训练提供在线观测能力。在 verl 中，它可以采集：

- trainer 标量指标，例如 reward、loss、throughput；
- 异步 vLLM / SGLang rollout 引擎指标；
- TransferQueue 指标；
- rollout 阶段的 RL 状态时间线；
- CPU、内存、网络和 Ascend NPU 硬件指标。

RL-Insight 有两种接入方式：

| 模式 | 适用场景 | Trace 数据 | Metrics 数据 |
| --- | --- | --- | --- |
| Managed 模式 | 使用 `rl-insight server start` 启动完整服务栈 | 发给 RL-Insight server 管理的 Tempo | RL-Insight server 注册 target，本地 Prometheus 抓取 |
| Platform 模式 | 平台已有 OTLP endpoint 和 Prometheus | 直接发给平台 OTLP endpoint | 平台按约定自己抓取 `/metrics` |

## 1. 安装并启动 RL-Insight Server

在运行监控服务的机器上安装 RL-Insight：

```bash
pip install "git+https://github.com/verl-project/rl-insight.git"
```

或安装发布版本：

```bash
pip install "rl-insight>=0.2.0"
```

启动本地服务栈：

```bash
rl-insight server install
rl-insight server start
```

默认端口如下：

| 服务 | 默认端口 | 用途 |
| --- | ---: | --- |
| RL-Insight server | `18080` | 服务发现和 target 注册 |
| Prometheus | `9090` | 指标存储与查询 |
| Tempo | `3200` | Trace 存储与查询 |
| Grafana | `3000` | Dashboard 展示 |

如果机器不能访问 GitHub release 和 `dl.grafana.com`，可以先在有网络的机器上下载安装包，再执行离线安装：

```bash
rl-insight server install --local-archive /path/to/archives
rl-insight server start
```

## 2. 在 verl 中启用 RL-Insight

设置 RL-Insight server 地址：

```bash
export RL_INSIGHT_SERVER_URL="http://<server-ip>:18080"
```

多节点 Ray 任务需要把该环境变量传播到所有 worker。如果使用 `verl/trainer/runtime_env.yaml`，配置如下：

```yaml
env_vars:
  RL_INSIGHT_SERVER_URL: "http://<server-ip>:18080"
```

然后在训练命令中启用 `rl_insight` logger：

```bash
python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl \
    trainer.experiment_name=ppo_rl_insight \
    ...
```

verl 会自动设置：

```text
VERL_RL_INSIGHT_ENABLE=1
```

trainer 标量指标会通过 logger backend 自动上报。

## 3. 开启 Rollout 和 TransferQueue 指标

Rollout 引擎指标需要保持 stats 开启：

```bash
actor_rollout_ref.rollout.disable_log_stats=False
```

TransferQueue 指标需要显式开启：

```bash
transfer_queue.metrics.enabled=True
```

完整示例：

```bash
python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    actor_rollout_ref.rollout.disable_log_stats=False \
    transfer_queue.metrics.enabled=True \
    ...
```

## 4. 查看 Grafana Dashboard

启动 RL-Insight server 后，打开：

```text
http://<server-ip>:3000
```

默认账号密码：

```text
username: admin
password: admin
```

常用 dashboard：

| Dashboard | 适用场景 |
| --- | --- |
| `verl_trainer_v1_with_vllm_engine` | vLLM rollout |
| `verl_trainer_v1_with_sglang_engine` | SGLang rollout |
| `agent_loop_trajectory` | Agent loop 轨迹 |

## 5. 平台模式：不启动 RL-Insight Server

当平台自己提供观测能力时，可以使用平台模式：

```bash
export RL_INSIGHT_OTLP_ENDPOINT=https://collector.platform.example:4318/v1/traces
unset RL_INSIGHT_SERVER_URL
```

然后正常启用 verl 的 RL-Insight logger：

```bash
python3 -m verl.trainer.main_ppo \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl \
    trainer.experiment_name=ppo_platform \
    actor_rollout_ref.rollout.disable_log_stats=False \
    ...
```

### 5.1 URL 优先级

RL-Insight 使用两个环境变量：

| 环境变量 | 含义 |
| --- | --- |
| `RL_INSIGHT_SERVER_URL` | RL-Insight server 控制面地址 |
| `RL_INSIGHT_OTLP_ENDPOINT` | 平台 OTLP/HTTP trace 上报 endpoint |

规则如下：

1. 如果设置了 `RL_INSIGHT_SERVER_URL`，走原有 managed 模式；
2. 如果没有设置 `RL_INSIGHT_SERVER_URL`，但设置了 `RL_INSIGHT_OTLP_ENDPOINT`，走平台模式；
3. 如果两个都没有设置，监控禁用；
4. 如果两个都设置了，`RL_INSIGHT_SERVER_URL` 优先，`RL_INSIGHT_OTLP_ENDPOINT` 会被忽略。

`RL_INSIGHT_OTLP_ENDPOINT` 必须是完整 endpoint，通常以 `/v1/traces` 结尾：

```text
https://collector.platform.example:4318/v1/traces
```

verl 和 RL-Insight 不会自动拼接 `/v1/traces`。

### 5.2 平台模式下的注册行为

平台模式只跳过 PR 6680 为 RL-Insight 适配新增的 target 注册：

- rollout metrics 的 RL-Insight 注册；
- TransferQueue metrics 的 RL-Insight 注册。

PR 6680 之前 verl 已经存在的 `actor_rollout_ref.rollout.prometheus.enable` 注册路径保持不变。

如果平台自己负责抓取 Prometheus 数据，通常应保持：

```bash
actor_rollout_ref.rollout.prometheus.enable=False
```

只有确实希望 verl 继续写本地 Prometheus 配置时才开启它。

## 6. 平台需要抓取的 Prometheus 数据

平台模式下，RL-Insight server 不启动，也不负责注册 Prometheus target。平台需要按提前约定的服务发现规则抓取以下 endpoint。

### 6.1 RL-Insight Trainer Metrics

RL-Insight 会为每个 Ray job 创建一个 `MonitorHubActor`。该 actor 会在 Ray 节点上启动 Prometheus HTTP server。

| 字段 | 值 |
| --- | --- |
| Scheme | `http` |
| Path | `/metrics` |
| Host | 运行 `MonitorHubActor` 的 Ray 节点 IP |
| Port | `prometheus.metrics_report_port`，默认 `9092` |
| 格式 | Prometheus text exposition format |

平台应抓取：

```text
http://<monitorhub-node>:9092/metrics
```

平台模式下，这个地址不会注册到 RL-Insight server。平台需要通过自己的服务发现机制找到 Ray 节点和端口。

Trainer 指标名统一带有前缀：

```text
rl_insight_monitor_
```

示例：

| 指标类别 | 示例 |
| --- | --- |
| Actor | `rl_insight_monitor_actor_loss` |
| Critic | `rl_insight_monitor_critic_rewards_mean` |
| 长度统计 | `rl_insight_monitor_response_length_mean` |
| 耗时统计 | `rl_insight_monitor_timing_s_gen` |
| 吞吐 | `rl_insight_monitor_perf_throughput` |
| 训练进度 | `rl_insight_monitor_training_global_step` |

常见 labels：

| Label | 含义 |
| --- | --- |
| `project` | `trainer.project_name` |
| `experiment_name` | `trainer.experiment_name` |
| `job` | 平台 Prometheus scrape 配置分配的 job 名 |
| `instance` | Prometheus 自动分配的 `host:port` |

指标类型规则：

| 类型 | Prometheus 输出 |
| --- | --- |
| Gauge | 直接使用指标名 |
| Counter | 增加 `_total` 后缀 |
| Histogram | 输出 `_bucket`、`_sum`、`_count` |

平台应保留所有原始 labels，不要重写指标名。

### 6.2 Rollout Engine Metrics

vLLM 和 SGLang rollout replica 会在自己的 HTTP server 上暴露：

```text
http://<rollout-host>:<rollout-port>/metrics
```

Rollout 端口是动态分配的。平台模式下，verl 不会把这些地址注册给 RL-Insight，也不会改变原有 `rollout.prometheus.enable` 行为。平台需要用自己的服务发现机制找到 rollout endpoint。

保持 rollout stats 开启：

```bash
actor_rollout_ref.rollout.disable_log_stats=False
```

如果不希望 verl 写本地 Prometheus 配置：

```bash
actor_rollout_ref.rollout.prometheus.enable=False
```

vLLM 常见指标：

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:generation_tokens_total
vllm:e2e_request_latency_seconds_bucket
```

SGLang 常见指标：

```text
sglang:num_running_reqs
sglang:num_queue_reqs
sglang:gen_throughput
sglang:cache_hit_rate
sglang:e2e_request_latency_seconds_bucket
```

常见 labels：

| Label | 含义 |
| --- | --- |
| `model_name` | served model 名称 |
| `engine` | rollout 引擎输出的 engine/worker 标识 |
| `replica` | verl 添加的 rollout replica rank |
| `stage` | prefill/decode 分离场景下的阶段，存在时保留 |

### 6.3 TransferQueue Metrics

如果开启 TransferQueue metrics：

```bash
transfer_queue.metrics.enabled=True
```

TransferQueue 进程会暴露：

```text
http://<transferqueue-node>:<transferqueue-port>/metrics
```

常见指标：

```text
tq_controller_request_total
tq_controller_request_duration_seconds_bucket
tq_storage_active_keys_total
tq_storage_capacity_total
tq_storage_utilization_ratio
tq_partition_production_progress
tq_partition_consumption_progress
```

常见 labels：

| Label | 含义 |
| --- | --- |
| `op_type` | controller 或 storage 操作类型 |
| `task_name` | producer 或 consumer task 名称，存在时保留 |

平台模式下，verl 不会把 TransferQueue endpoint 注册给 RL-Insight。平台需要按自己的服务发现规则抓取。

## 7. 平台 Prometheus 抓取示例

实际服务发现方式由平台决定。概念上，平台 Prometheus 需要三个 scrape job：

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

建议 scrape interval：

```text
10s - 15s
```

## 8. 平台模式数据流

```text
verl trainer
  -> RLInsightLogger
     -> Ray MonitorHubActor
        -> /metrics，供平台 Prometheus 抓取
        -> OTLP POST 到 RL_INSIGHT_OTLP_ENDPOINT

vLLM / SGLang rollout replica
  -> /metrics，供平台 Prometheus 抓取

TransferQueue
  -> /metrics，供平台 Prometheus 抓取
```

## 9. 常见问题排查

- Trainer 指标没有数据：
  - 检查 `trainer.logger` 是否包含 `rl_insight`；
  - Managed 模式检查 `RL_INSIGHT_SERVER_URL` 是否正确；
  - Platform 模式检查 `RL_INSIGHT_OTLP_ENDPOINT` 是否正确。
- Rollout 指标没有数据：
  - 检查 `actor_rollout_ref.rollout.disable_log_stats=False`；
  - 检查平台是否能访问 rollout `/metrics` endpoint。
- TransferQueue 指标没有数据：
  - 检查 `transfer_queue.metrics.enabled=True`；
  - 检查平台是否能访问 TransferQueue `/metrics` endpoint。
- Platform 模式所有 metrics 都没有数据：
  - 检查平台是否能访问 MonitorHub、rollout 和 TransferQueue 的 `/metrics` endpoint；
  - 检查平台服务发现是否找到了正确的节点和端口。
- Trace 没有数据：
  - 检查 `RL_INSIGHT_OTLP_ENDPOINT` 是否是完整 OTLP/HTTP endpoint；
  - 检查 endpoint 是否以 `/v1/traces` 结尾；
  - 检查训练进程到平台 collector 的网络连通性。
