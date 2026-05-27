SkipManager Usage Documentation
=================================

Last updated: 2026-05-23

1. Overview
-----------

**SkipManager** (``verl.utils.skip.SkipManager``) is a general-purpose framework for **skipping
selected steps** in verl training flows. By bypassing expensive stages on configured steps, it helps
save **time**, **memory**, or other resources and improves **developer iteration speed** during
debugging and experimentation.

Skip behavior is centralized under the top-level Hydra key ``skip``. Modules register by **role**
(for example ``"rollout"`` or ``"async_rollout"``) and are attached with
``@SkipManager.annotate(role=...)``. Each role declares which integer **steps** in config are
eligible for skip logic. **Today only rollout-related roles are implemented**; the same mechanism
can be extended to other pipeline stages (see section 4).

Typical use cases
~~~~~~~~~~~~~~~~~

SkipManager is intended for development workflows where repeating full training is costly:

1. **Faster iteration**: skip heavy stages on chosen steps (e.g. generation) while exercising the
   rest of the pipeline.
2. **Deterministic replay**: cache and reload intermediate results to reproduce a prior run on
   specific steps.
3. **Resource savings**: avoid recomputing or holding large tensors when bisecting bugs or tuning
   downstream logic.

The built-in ``rollout`` / ``async_rollout`` modules apply this to sequence generation; other
roles can follow the same pattern as they are added.

Supported entry points today
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Training entry
     - Skip role / config
     - Status
   * - ``main_ppo.py`` (``RayPPOTrainer``)
     - ``skip.rollout``
     - **Supported**
   * - ``main_ppo_sync.py`` (TransferQueue + ReplayBuffer)
     - ``skip.rollout``
     - **Not supported** (see section 2)
   * - ``fully_async_main`` (``FullyAsyncRollouter``)
     - ``skip.async_rollout``
     - **Supported**

.. important::

   The legacy utility ``verl.utils.rollout_skip.RolloutSkip`` (patching ``generate_sequences`` via
   ``actor_rollout_ref.rollout.skip``) is **deprecated** and will be removed after a compatibility
   window. New code and configs should use SkipManager. See :doc:`rollout_skip` for historical
   reference.


2. Rollout Skip Quick Start (``rollout`` role)
----------------------------------------------

Use ``skip.rollout`` when training with ``main_ppo.py`` / ``RayPPOTrainer`` and the standard
``AgentLoopManager.generate_sequences`` path.

Configuration
~~~~~~~~~~~~~

Trainer YAML reserves a ``skip.rollout`` block (see ``verl/trainer/config/ppo_trainer.yaml``).
Override via Hydra CLI:

.. code-block:: bash

   skip.rollout.enable=True
   skip.rollout.dump_dir=/path/to/rollout_dump
   skip.rollout.steps='[1,2,3,10]'
   skip.rollout.action=cache

Parameters (``skip.rollout`` / ``RolloutSkipConfig``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **enable** (bool): Master switch for rollout skip.
- **dump_dir** (str): Root directory for cached ``DataProto`` shards (``~`` is expanded).
- **steps** (list[int]): Trainer **global_steps** on which skip logic is *eligible*. Outside this
  set, the decorated function always runs normally.
- **action** (``cache`` \| ``repeat``):

  - ``cache``: load or write under the exact step directory for the current trainer step.
  - ``repeat``: reuse the nearest available cached step (see ``RolloutSkip._find_latest_step``).

.. note::

   Only ``cache`` and ``repeat`` are validated in ``RolloutSkipConfig`` today, even though
   ``SkipAction`` in code lists additional enum values for future modules.

On-disk layout
~~~~~~~~~~~~~~

.. code-block:: text

   {dump_dir}/{experiment_name}_{project_name}/
       └── GBS{gbs}_N{n}_in{prompt_len}_out{response_len}/
           ├── {global_step}/
           │   ├── gen_batch.dp
           │   └── meta.json
           └── ...

Relationship to legacy RolloutSkip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If **both** ``skip.rollout.enable`` and legacy ``actor_rollout_ref.rollout.skip.enable`` are true,
SkipManager emits a ``DeprecationWarning`` and **forces** the legacy flag to ``False`` so only one
mechanism runs.

``main_ppo.py`` vs ``main_ppo_sync.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**``main_ppo.py`` (supported)**

- ``RayPPOTrainer.fit()`` calls ``SkipManager.init(self.config)`` and updates
  ``SkipManager.set_step(self.global_steps)`` each training step.
- ``AgentLoopManager.generate_sequences`` is decorated with
  ``@SkipManager.annotate(role="rollout")``.

**``main_ppo_sync.py`` (not supported yet)**

``main_ppo_sync`` replaces the Agent Loop integration with ``AgentLoopManagerTQ``. The main reason
rollout skip is not supported today is **logic coupling** in
``AgentLoopManagerTQ.generate_sequences``: it not only drives sequence generation, but also marks
samples in the ReplayBuffer and **writes generated data into TransferQueue (TQ)**. Skipping
``generate_sequences`` would therefore skip both generation and the TQ handoff, which breaks the
downstream training loop that consumes data from TQ.

Decoupling “generate” from “enqueue to TQ” is non-trivial under the current design, so SkipManager
adaptation for ``main_ppo_sync`` is **deferred** until the TransferQueue-based training path is
further stabilized; a compatible hook can be added then without conflating skip with queue
bookkeeping.


3. Fully Async Quick Start (``async_rollout`` role)
---------------------------------------------------

In :doc:`fully_async`, Trainer and Rollouter run in separate processes. Rollout generation happens
on the Rollouter via streaming single-sample dispatch. Use ``skip.async_rollout`` (not
``skip.rollout``) when launching ``fully_async_main``.

Configuration
~~~~~~~~~~~~~

``verl/trainer/config/ppo_trainer.yaml`` defines ``skip.async_rollout`` with the same fields as
``skip.rollout``. Example overrides:

.. code-block:: bash

   skip.async_rollout.enable=True
   skip.async_rollout.dump_dir=/path/to/rollout_dump
   skip.async_rollout.steps="[$(seq -s, 1 128)]"
   skip.async_rollout.action=cache

Parameters (``skip.async_rollout`` / ``AsyncRolloutSkipConfig``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same semantics as ``skip.rollout`` (``enable``, ``dump_dir``, ``steps``, ``action``), with one
critical difference for **steps**:

- **steps** refers to the integer embedded in each sample's ``sample_id`` (see below), **not**
  Trainer ``global_steps`` or ``current_param_version``.

.. important::

   In ``async_rollout``, a step is **not** the real training-step timeline. It is only the
   **prompt request / feed order** on the Rollouter: the monotonic index assigned when
   ``FullyAsyncRollouter`` enqueues the next prompt (``sample_{epoch}_{index}``). Under concurrent
   rollout, completion order can differ from feed order; do not treat these indices as Trainer step
   numbers or parameter-sync boundaries when configuring ``skip.async_rollout.steps``.

Step key from ``sample_id``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each fed sample carries an id of the form ``sample_{epoch}_{index}`` (for example
``sample_0_42``). The integer matched against ``skip.async_rollout.steps`` and used for on-disk
directories is the **last segment** — Rollouter feed-order index at enqueue time (field name
``global_steps`` in code is historical naming for this counter).

On-disk layout
~~~~~~~~~~~~~~

Uses the same tree as ``rollout``, but ``GBS`` comes from ``data.gen_batch_size``. In fully async
runs this is typically **1** (streaming single-prompt generation). Caches from colocate /
``main_ppo.py`` runs (larger ``GBS``) are generally **not** interchangeable unless batch metadata
matches.

``cache`` vs ``repeat`` notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``cache``: load or write under the exact rollouter step directory.
- ``repeat``: with streaming one-sample rollout, verify the loaded cache matches the intended
  prompt when debugging; nearest-step reuse may not correspond to the current prompt.


4. Design and Implementation
----------------------------

This section describes the SkipManager / BaseSkip contract, which functions are intercepted, and
why ``support_online_step`` exists.

SkipManager API
~~~~~~~~~~~~~~~

``SkipManager`` (``verl.utils.skip.skip_manager``) is a class-level registry:

- **``init(config)``**: Parse ``config.skip`` into ``SkipManagerConfig``, instantiate one skip module
  per registered role (``rollout``, ``async_rollout``, …), and store them in
  ``SkipManager.skip_instances``.
- **``set_step(step: int)``**: Set the shared class attribute ``SkipManager.step``. Used by
  ``RayPPOTrainer`` to publish the current trainer ``global_steps`` before each rollout call.
- **``annotate(role, **kwargs)``**: Decorator factory. Wraps a sync or async function with the skip
  decision flow described below.

Decorator flow (``@SkipManager.annotate``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each call to a decorated function:

1. Look up ``skip_instances[role]``; if missing or disabled, call through unchanged.
2. Resolve **step**:

   - If ``skip_instance.support_online_step`` is ``True``: call
     ``skip_instance.extract_step(*args, **kwargs)`` (per-call, from arguments).
   - Else: use ``SkipManager.step`` (class-level, set by trainer).
3. If ``step not in skip_instance.steps``, call through unchanged.
4. If ``meet_precondition(step, func, *args, **kwargs)`` is true, return
   ``warp_function(step, func, *args, **kwargs)`` (load cache / repeat).
5. Otherwise execute the real function, then ``prepare_data(step, result, *args, **kwargs)`` to
   dump when appropriate.

BaseSkip interface
~~~~~~~~~~~~~~~~~~

Each skip module subclasses ``BaseSkip`` (``verl.utils.skip.base_skip``) and registers via
``@register_skip("role_name")``.

Class attributes:

- **``support_actions``**: Allowed ``SkipAction`` values for this module.
- **``support_online_step``** (default ``False``): When ``True``, step is taken from call arguments
  via ``extract_step`` instead of ``SkipManager.step``.

Instance methods (subclasses implement the rollout-specific logic):

- **``is_enabled()``**: Whether ``enable`` is set in config.
- **``meet_precondition(step, func, *args, **kwargs)``**: Whether cached data exists for the chosen
  action (``cache`` / ``repeat``).
- **``warp_function(step, func, *args, **kwargs)``**: Return cached ``DataProto`` without calling
  the wrapped function.
- **``prepare_data(step, result, *args, **kwargs)``**: Persist ``result`` after a real execution.
- **``extract_step(*args, **kwargs)``**: Required when ``support_online_step`` is ``True``; parse
  step from decorated call arguments.

``RolloutSkip`` / ``AsyncRolloutSkip`` (``verl.utils.skip.rollout_skip``) implement the above for
generation caching. Both register under different role names but share dump layout helpers.

Intercepted functions
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 34 28 22

   * - Role
     - Decorated function
     - Defined in
     - Step source
   * - ``rollout``
     - ``AgentLoopManager.generate_sequences``
     - ``verl/experimental/agent_loop/agent_loop.py``
     - ``SkipManager.set_step`` → trainer ``global_steps``
   * - ``async_rollout``
     - ``FullyAsyncAgentLoopManager.generate_sequences_single``
     - ``verl/experimental/fully_async_policy/fully_async_rollouter.py``
     - ``extract_step`` → ``sample_id`` suffix → **prompt feed order**

**``rollout``**: wraps the full batch Agent Loop RPC — chunk dispatch to workers, concat, and
timing aggregation — as a single skip unit.

**``async_rollout``**: wraps one streaming sample's ``generate_sequences_single(self, prompts,
sample_id)`` RPC (worker selection + remote ``generate_sequences``), including the ``sample_id``
argument required for online step resolution.

Step resolution: ``set_step`` vs ``support_online_step``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skip modules need an integer **step** to match against ``skip.<role>.steps``. verl provides two
ways to supply it; choosing the wrong one leads to incorrect skip behavior.

**Shared step — ``SkipManager.set_step``**

- ``SkipManager.step`` is the **only class-level step slot** in a process. Every ``BaseSkip``
  subclass with ``support_online_step = False`` reads the same value.
- This fits **sequential** loops (e.g. ``main_ppo.py``): the trainer calls ``set_step(global_steps)``
  once per training step, then ``generate_sequences`` runs before the next update.
- **Limitation of the current design**: if two or more skip roles in the **same process** need
  **different** step counters at overlapping times (trainer step vs rollouter feed step, or two
  stages updated on different schedules), they cannot both use ``SkipManager.step`` reliably —
  whichever ``set_step`` ran last wins, and in-flight decorated calls may observe the wrong step.

**Per-call step — ``support_online_step`` + ``extract_step``**

- When ``support_online_step = True``, the decorator ignores ``SkipManager.step`` and calls
  ``extract_step(*args, **kwargs)`` on **each** invocation to derive step from runtime arguments.
- This is better for **concurrent** execution: on the Rollouter, multiple samples can be in flight
  (streaming feed + async tasks). A shared ``SkipManager.step`` would race, and instance fields such
  as ``RolloutSkip.lastest_step`` (used by ``repeat``) can conflict across overlapping requests.
- ``AsyncRolloutSkip`` uses this path: ``extract_step`` parses the feed-order index from
  ``sample_id`` on each ``generate_sequences_single(self, prompts, sample_id)`` call (third
  positional argument or ``sample_id=`` keyword), so concurrent samples resolve step independently.
  That index reflects **prompt submission order**, not Trainer ``global_steps`` or generation
  finish order.

Fully async therefore uses ``async_rollout`` with online step — for concurrency, and because the
skip key is the Rollouter feed counter rather than the Trainer step counter.

Extending with custom skip modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Subclass ``BaseSkip`` from ``verl.utils.skip.base_skip``.
2. Decorate the class with ``@register_skip("your_role_name")``.
3. Add a matching field under ``SkipManagerConfig`` so ``SkipManager.init`` can construct your
   module from ``config.skip``.
4. Attach ``@SkipManager.annotate(role="your_role_name")`` on the function you want to intercept.
   If the training loop is concurrent, prefer ``support_online_step = True`` and pass step identity
   through call arguments rather than ``SkipManager.set_step``.

Further reading
---------------

- Legacy rollout skip (deprecated): :doc:`rollout_skip`
- Fully async training: :doc:`fully_async`
