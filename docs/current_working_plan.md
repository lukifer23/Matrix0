# Current Working Plan

Last updated: 2026-05-10

This is the living document for the active Matrix0 training/debugging loop. Keep it short, factual, and current. Historical project status belongs in `status.md` and `CURRENT_STATUS_SUMMARY.md`; this page tracks the immediate problem, current best checkpoint, active hypotheses, and promotion criteria.

## Current Goal

Make the model improve reliably from local Apple Silicon self-play without accepting misleading wins from corrupted data, noisy targets, or evaluation artifacts.

The current focus is not raw throughput. The priority is a trustworthy loop:

1. Generate self-play data with valid legal masks, policy targets, value targets, and metadata.
2. Verify label quality and outcome mix before training.
3. Train only when generated data passes acceptance checks or the recipe explicitly compensates with anchor data.
4. Evaluate on stable heldout data before promoting a checkpoint.
5. Re-test candidate generator quality before replacing the parent.

## Current Best Checkpoint

Use this as the parent until a candidate beats it on stable heldout metrics and generator quality:

```text
checkpoints/bootstrap_006_capped_value.pt
```

Do not promote candidates from runs that are mostly capped, mostly early draw-adjudicated, only improve raw unmasked policy metrics while regressing legal policy/value, or generate weaker self-play labels than the parent.

## What We Fixed Recently

The local-loop path now fails hard instead of silently producing bad data:

- self-play checkpoint load failure no longer falls back to an untrained model
- partial checkpoint loads require an explicit migration flag
- legal mask generation failures raise instead of saving all-zero masks
- SSL target generation failures raise instead of saving zero SSL targets
- `ssl_weight=0.0` now disables SSL target creation and model SSL compute for true no-SSL ablations
- training rejects invalid policy rows, zero-legal masks, malformed masks, and missing masks when legal-policy/legal-mass training is enabled
- strict local-loop data mode raises on corrupted shards instead of skipping them
- MCTS expansion raises on invalid logits/priors instead of inventing uniform priors
- MCTS root searches are fresh by default; mutable transposition-table visit stats no longer contaminate self-play labels
- MCTS policy targets are built from the current search visit counts, not stale accumulated node counts
- batched legal-only MCTS expansion softmaxes raw legal logits instead of normalizing positive logits as probabilities
- batched MCTS leaf collection now passes shared per-mini-batch virtual-loss state into selection, so batch collection diversifies before network inference/backpropagation
- root value labels now use alternating MCTS backpropagation instead of overwriting `root.q` with raw mean leaf values
- explicit `mcts.value_from_white` config is respected; auto-detection only runs for old configs without the key
- training raises on NaN/Inf policy, value, SSL logits, and scaler backward errors instead of sanitizing/skipping
- strict source-prefix data loading raises on malformed or missing legal masks
- draw adjudication has explicit controls for claimable repetition and fifty-move draws
- `--legal-policy-weight` adds legal-conditioned policy CE to train ranking among legal moves
- `--train-fresh-max-files` bounds fresh shards kept train-visible and prunes moved shard metadata
- `--selection-jitter 0.0` is now truly deterministic; it no longer adds hidden random jitter
- self-play shards now save final-position metadata: FEN, piece count, halfmove clock, legal count, and draw-claim availability
- `azchess.tools.diagnose_policy_targets` now buckets checkpoint comparisons by target entropy, top probability, and legal count

## Current Findings

### Sharp Search Helps Labels, But Not Enough

Sharp-search settings consistently improved basic policy-label shape:

- `legal_policy_mass` stayed near `1.0`
- `policy_top_prob` rose versus earlier soft-label runs
- `policy_entropy` and `policy_support` fell

This supports lower exploration, lower Dirichlet noise, disabled entropy noise, and clean search accounting for current label-quality experiments.

### Fresh48 Anchor Training Was Not Promotable

The 200-game fresh48 + anchor runs improved full unmasked policy CE, but they did not produce a promotable checkpoint:

- `bootstrap_006_policy_anchor_fresh48_lm002_s600_200g`
  - `policy_ce -0.7428`
  - `policy_legal_ce +0.0113`
  - `value_mse -0.0216`
- `bootstrap_006_policy_anchor_fresh48_lm001_lp025_s600_200g`
  - `policy_ce -0.7362`
  - `policy_legal_ce +0.0030`
  - `value_mse -0.0130`
- `bootstrap_006_policy_anchor_fresh48_lp050_retrain_s600`
  - `policy_ce -0.7028`
  - `policy_legal_ce +0.0085`
  - `value_mse +0.0078`

Interpretation: raw policy CE gains were partly legal-mass gains, not enough true legal-move ranking improvement.

### No-SSL And Anchor-Only Ablations Clarified The Failure Mode

The no-SSL retrain on mixed fresh48 data still regressed legal policy and value:

- `bootstrap_006_policy_anchor_fresh48_nossl_retrain_s600`
  - `policy_ce -0.7261`
  - `policy_legal_ce +0.0074`
  - `value_mse +0.0026`

Anchor-only no-SSL training was more stable on the original anchor heldout:

- `bootstrap_006_anchor_only_nossl_s600`
  - anchor eval: `policy_ce -0.6469`, `policy_legal_ce -0.00017`, `policy_legal_top1_match +0.0168`, `value_mse -0.0078`
  - parent-generated fresh self-play eval: `policy_ce -0.7350`, `policy_legal_ce -0.0896`, `policy_legal_top1_match -0.2273`, `value_mse +0.0026`
  - replay-only eval: `policy_ce -0.6463`, `policy_legal_ce +0.00016`, `policy_legal_top1_match +0.0046`, `value_mse +0.0134`

Diagnostic buckets showed the self-play top-1 collapse was mostly a behavior-policy artifact: the parent generated the target search labels, so the parent had an argmax advantage on its own data. The candidate often assigned more probability to the target move while ranking another move first.

### Candidate Generator Check Failed

The anchor-only candidate was tested as a generator:

```text
logs/local_loop/bootstrap_006_anchor_only_nossl_candidate_generator_64g
```

Result:

- `64/64` games capped
- no tablebase hits, no terminal games, no draw adjudications
- all value labels were capped bootstrap targets at weight `0.25`
- `policy_top_prob 0.1777`
- `policy_entropy 1.8532`
- `policy_support 6.91`

Interpretation: this candidate is not promotable. Even if it can improve some heldout metrics, it generated weaker labels and no useful outcome mix.

### Hidden Jitter Invalidated Clean No-Jitter Claims

Investigation found that MCTS selection still added a small random tie-breaker when `selection_jitter=0.0`.

That is fixed now. The next generator probe must rerun with the same visible command because `--selection-jitter 0.0` now means exact zero jitter. New reports will also include final-position diagnostics to explain capped games.

### Tablebase Wiring Is Not The Current Blocker

The Syzygy path is configured and probeable:

- `config.yaml` has `tablebases.enabled: true`, `path: data/syzygy`, and `max_pieces: 5`
- `data/syzygy` contains WDL files and direct `python-chess` WDL probes work
- current capped generator diagnostics show games reaching the cap with too much material, not near the configured tablebase piece limit

Interpretation: tablebase adjudication is not being missed because the path is broken. The candidate generator is failing to simplify/convert games before the move cap.

### Batched Virtual Loss Was Not Active In The Real Collector

The virtual-loss selector existed, but the production batched leaf collector was not passing the shared in-flight map into `_select`.

That is fixed now. New generator probes should use a fresh process/run directory; already-running generator processes loaded the old code and will not pick up this fix.

### Correct 50-Sim Search Is Still Broad

The fixed batch-size-1 probe completed all 32 games but hit a self-play worker shutdown hang, which is now fixed. Salvaged data showed:

- `31` capped, `1` terminal, `0` tablebase
- `policy_top_prob 0.140`
- `policy_entropy 2.955`
- `policy_support 26.6`

By-ply diagnostics showed policy support nearly equal to legal move count. Interpretation: once virtual loss and per-simulation backprop are correct, 50 simulations with a weak model and broad priors spreads visits across most legal moves. The old `policy_support ~7` signal was probably a repeated-leaf artifact, not a healthy target.

`--policy-target-temperature` now exists to sharpen saved training labels without changing move sampling.

## Active Hypotheses

### H1: Generator Quality Is The Blocker

The model is not yet failing because we lack another 600-step training recipe. It is failing because candidates are not proving they can generate better data than the parent.

Acceptance for any candidate generator:

- not all or nearly all games capped
- policy labels at least as sharp as parent-generated data
- final positions explain caps: low material/tablebase proximity, high halfmove clocks, or draw claims are actionable; high material and low halfmove clocks indicate game-quality/search issues
- no strict-mode data/checkpoint failures

### H2: Policy And Value Need Different Data

If sharp self-play remains mostly capped, train policy on fresh sharp labels but anchor value on older tablebase/terminal-heavy data:

- fresh sharp self-play for policy
- `logs/local_loop/bootstrap_003_capped_value_48g/data` as anchor data for value/tablebase coverage
- heldout eval on `logs/local_loop/bootstrap_003_capped_value_48g/data`
- target-only policy sharpening for low-simulation self-play if raw visit targets remain nearly legal-uniform
- low capped value weight; do not let capped bootstrap values dominate value loss

Acceptance:

- legal policy CE/KL improves or is flat with clear top-prob/rank improvements
- value MSE does not regress materially
- candidate generator check does not worsen outcome mix or label quality

### H3: Outcome Mix Needs Search/Termination Work

If the fixed-jitter plus virtual-loss generator remains mostly capped, the next fix is not more retraining. Investigate:

- final material counts at cap
- final halfmove clocks and claimable draw state
- tablebase path/use and piece-count reachability
- search/value bias causing non-converting play
- temperature schedule and endgame determinism
- resignation/adjudication thresholds

## Current Probe Command

Run this next. It repeats the candidate generator check after the zero-jitter and batched virtual-loss fixes and records final-position diagnostics:

```bash
MATRIX0_MPS_TARGET_BATCH=4 \
.venv/bin/python -m azchess.tools.bench_local_loop \
  --config config.yaml \
  --run-dir logs/local_loop/bootstrap_006_anchor_only_nossl_candidate_generator_64g_fixed_jitter_vloss \
  --games 64 \
  --workers 2 \
  --sims 50 \
  --max-game-len 200 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --mps-target-batch 4 \
  --capped-value-weight 0.25 \
  --cpuct 1.6 \
  --cpuct-start 1.8 \
  --cpuct-end 1.2 \
  --cpuct-plies 32 \
  --dirichlet-frac 0.10 \
  --dirichlet-plies 12 \
  --temperature-start 0.8 \
  --temperature-end 0.15 \
  --temperature-moves 24 \
  --opening-random-plies 8 \
  --selection-jitter 0.0 \
  --disable-entropy-noise \
  --draw-halfmove-cap 100 \
  --draw-material-threshold 0 \
  --draw-min-plies 120 \
  --draw-window 0 \
  --draw-min-unique 0 \
  --draw-claim-min-plies 120 \
  --draw-disable-repetition-claims \
  --init-checkpoint logs/local_loop/bootstrap_006_anchor_only_nossl_s600/checkpoints/local_loop_best.pt \
  --skip-train
```

## Promotion Rules

Promote only after all of these are true:

- generator data has acceptable outcome mix, or the training recipe explicitly compensates with anchor data
- legal policy metrics improve on stable heldout data
- value MSE does not regress materially on heldout data
- post-training generator check does not degrade game outcomes or label quality
- no strict-mode data/checkpoint failures occurred
- candidate checkpoint beats the current parent as both evaluator and generator

## Useful Reports

Each local loop writes:

```text
logs/local_loop/<run_name>/local_loop_report.json
```

The high-signal sections are:

- `fresh_data.game_outcomes`
- `fresh_data.quality.policy_top_prob`
- `fresh_data.quality.policy_entropy`
- `fresh_data.quality.policy_support`
- `fresh_data.quality.legal_policy_mass`
- `fresh_data.quality.source_metrics`
- `fresh_data.quality.source_metrics.<source>.final_piece_count`
- `fresh_data.quality.source_metrics.<source>.final_halfmove_clock`
- `fresh_data.quality.source_metrics.<source>.final_legal_count`
- `fresh_data.quality.source_metrics.<source>.final_can_claim_draw`
- `eval_delta`
