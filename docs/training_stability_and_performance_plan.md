## Training Stability & Performance Plan

### Scope
- Improve training stability (NaNs/Inf, SSL crashes), throughput, and unification across device/precision.
- Keep features; focus on fixes, guardrails, and streamlining. Consider non-destructive size reductions.

### Non-goals
- No feature removals. No code execution in this plan.

## Fix-first (stability-critical)
- [ ] Respect `cfg.norm` in all blocks
  - [ ] Pass `cfg.norm` to `ResidualBlock` and use `_norm(channels, cfg.norm)` for `bn1`/`bn2`.
  - [ ] Default to GroupNorm for small-batch training in `config.yaml`.
  - [ ] Unit test: residual blocks honor `cfg.norm` (batch vs group).
- [ ] SSL target typing/timeout hygiene
  - [ ] Replace generic `ssl_targets.to(torch.long)` with per-target casting.
  - [ ] Handle both tensor and dict targets explicitly; cast only what’s needed.
  - [ ] Remove `signal.alarm`; implement monotonic time budget with early exit + warning.
  - [ ] Document piece-only SSL format: class map `(B,8,8)` or one-hot `(B,13,8,8)`.
- [ ] Cache clearing policy
  - [ ] Remove unconditional `clear_memory_cache()` before SSL computation and during heartbeats.
  - [ ] Trigger cache clears only via memory monitor thresholds.

## Stability and NaN mitigation
- [ ] Precision policy (centralized)
  - [ ] CUDA: prefer bf16 autocast if available; else fp16 + GradScaler.
  - [ ] MPS: prefer bf16 autocast; keep params/buffers fp32; avoid scalers.
  - [ ] Unify `train_step` autocast dtype selection from a single helper.
- [ ] Loss input hygiene
  - [ ] Row-normalize `pi` with epsilon guard; warn and skip all-zero rows.
  - [ ] Maintain single clamp location for policy logits (in model); remove duplicate clamps.
- [ ] Gradient health
  - [ ] Keep grad clipping; add gradient norm metrics (median/p95) every N steps.
  - [ ] Skip batch on non-finite loss (done); avoid frequent scaler re-init.

## Throughput and performance
- [ ] Data pipeline → PyTorch DataLoader
  - [ ] Implement `Dataset` backed by NPZ shards with memory-mapped reads.
  - [ ] Use `DataLoader` with `pin_memory=True`, `num_workers>0`, `persistent_workers=True`, `prefetch_factor` tuned.
  - [ ] Collate to contiguous float32 arrays; non_blocking H2D moves.
  - [ ] Integrate with `DataManager` metadata (SQLite) to pre-select shard lists per epoch.
  - [ ] Replace ad-hoc `next(batch_generator)` with unified loader.
- [ ] LR scheduler semantics with accumulation
  - [ ] Configure scheduler on update-steps: `num_updates = ceil(total_steps / accum_steps)`.
  - [ ] Log micro-step vs update-step counts for clarity.
- [ ] Optional speed-ups
  - [ ] Enable `torch.compile` on CUDA path (guard + fallback); keep disabled for MPS if regressions.
  - [ ] Use channels_last on CUDA only.
  - [ ] Prefer fused AdamW on CUDA if available.

## SSL unification and cost control
- [ ] Default to piece-only SSL (`ssl_tasks: ["piece"]`) unless heads exist for new tasks.
- [ ] Adaptive SSL chunk size based on memory monitor signal (not fixed).
- [ ] SSL gating schedule: warmup to target `ssl_weight`; auto-freeze for K steps after instability.
- [ ] If enabling enhanced SSL tasks later, add dedicated heads and compute losses conditionally.

## Model architecture and size (non-destructive options)
- [ ] Policy head factorization
  - [ ] Set `policy_factor_rank` to 96–128; validate param reduction and policy quality.
- [ ] Value head simplification
  - [ ] Replace flatten with GAP + MLP (`C -> C/2 -> 1`) to reduce activations and params.
- [ ] Channel/head/depth rebalancing
  - [ ] Evaluate `channels=256, blocks=24–32`, `attention_heads=12–16` for similar strength with fewer params.
- [ ] SE optimization
  - [ ] Keep SE on; consider `se_ratio=0.125` to trim FC params on wider nets.
- [ ] Optional: bottleneck residuals (1×1–3×3–1×1) with expansion 2–4× to reduce params/memory while preserving capacity.

## Data quality and curriculum
- [ ] Encode curriculum schedule in `config.yaml` (not in code):
  - [ ] Early: openings-heavy, light tactics; Mid: balanced; Late: add more self-play.
- [ ] Shard sanity metrics on import/epoch start:
  - [ ] Policy entropy mean/std, legal coverage ratio, WDL skew; warn/quarantine outliers.

## Observability, tests, guardrails
- [ ] Unit tests
  - [ ] Norm selection honored by `ResidualBlock` and heads.
  - [ ] SSL target typing (tensor vs dict), piece-only dtype/path.
  - [ ] `pi` normalization and `policy_masking` behavior; all-zero row handling.
- [ ] Runtime logging
  - [ ] Output stats every 500 steps: policy/value mean/std/max, gradient norm distribution, updates/sec.
  - [ ] DataLoader queue depth/backpressure metrics.

## Config and unification
- [ ] Single source of truth in `config.yaml` for precision, accumulation, scheduler semantics.
- [ ] Device setup lives in `utils.device.setup_device` only; training calls it once.
- [ ] Align docs with current defaults; avoid divergence between code and docs.

## Definition of Done (milestones)
- [ ] M1: Fix-first items merged; NaN/Inf incidence → near-zero; SSL no longer times out/crashes.
- [ ] M2: DataLoader deployed; sustained steps/sec improved; fewer cache clears in logs.
- [ ] M3: Precision policy unified; stable on CUDA and MPS; no scaler churn.
- [ ] M4: Optional model tweaks (policy rank, value head GAP) land with equal/better eval vs baseline.

## Rollout & Validation
- [ ] A/B training dry-run (1k update-steps) with fixed seeds; compare loss curves, stability, steps/sec.
- [ ] Eval-integration sanity check (short self-play + eval games) to confirm no regressions.
- [ ] Telemetry review after 24h training window; adjust thresholds.

## Risks & Mitigations
- Precision policy regressions → Gate behind config flags; fallback to prior path per-device.
- DataLoader worker deadlocks → Start with low workers; enable `persistent_workers` after sanity.
- Model size reductions hurting strength → Stage changes (policy rank first, then value head, then channels).

## Ownership & Tracking
- Tech lead: [assign]
- Implementers: [assign]
- Reviewers: [assign]
- Metrics dashboard owner: [assign]

## References (for implementers)
- Norm usage in `ResidualBlock` needs to pass `cfg.norm`.
- SSL piece-only targets format/dtype contract; enhanced SSL returns dict; compute only existing heads.
- Memory monitor thresholds drive cache clears; remove unconditional clears in training loop.


