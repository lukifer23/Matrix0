# Local Loop Knob Guide

This guide explains the knobs used by `azchess.tools.bench_local_loop` during Matrix0 self-play, training, and checkpoint comparison. It is optimized for the current Apple Silicon local-loop workflow.

## Purpose

Use the local loop to answer one narrow question at a time:

- did generator data get better?
- did policy labels get sharper?
- did value targets get less biased?
- did training improve heldout metrics?
- did a candidate generate better games than its parent?

Do not use a full training cycle until the generator-only run passes basic data-quality checks. The current `bootstrap_007` path is a guarded fresh self-play loop: small fresh batches, stable anchor data, full heldout source-sliced eval, parent policy distillation, and promotion only when aggregate value improves without source-slice regression.

Current mainline parent:

```text
checkpoints/bootstrap_007_fresh_anchor_best.pt
```

## Generator Quality Metrics

Inspect these in `local_loop_report.json` under `fresh_data.quality`.

- `legal_policy_mass`: target policy mass on legal moves. Should be near `1.0`.
- `policy_top_prob`: average probability assigned to the most visited move. Higher means sharper labels.
- `policy_entropy`: entropy of the MCTS visit distribution. Lower means sharper labels.
- `policy_support`: number of moves with non-trivial target probability. Lower means more focused labels.
- `avg_sims`: actual average simulations per move.
- `source_metrics`: the same metrics split by `capped`, `tablebase`, `terminal`, and `draw_adjudication`.
- `source_metrics.<source>.final_piece_count`: material remaining when the game ended or capped.
- `source_metrics.<source>.final_halfmove_clock`: final halfmove clock; high values indicate fifty-move pressure.
- `source_metrics.<source>.final_legal_count`: final mobility at termination/cap.
- `source_metrics.<source>.final_can_claim_draw`: count of games where a draw claim was available at the final position.

Current rough targets:

- `legal_policy_mass ~= 1.0`
- `policy_top_prob >= 0.25` for sharp-search experiments
- `policy_entropy < 1.9` for sharp-search experiments
- low draw adjudication unless a draw-specific run is intentional
- enough tablebase/terminal data for value learning, or explicit anchor data
- capped games whose final-position metadata explains why they capped
- fresh capped fraction below the active gate, currently `0.67` to `0.75`
- heldout source value deltas no worse than `+5e-7` for `tablebase`, `terminal`, and `capped`

## Search Knobs

### `--sims`

MCTS simulations per move.

- Higher: better search, slower games, possibly more decisive outcomes.
- Lower: faster, noisier labels, often more capped games.
- Current probes: `50`, `75`, `100`.

### `--cpuct`

Base PUCT exploration strength.

- Higher: explores more moves, broader visit targets.
- Lower: exploits policy/value more, sharper labels.
- Current sharp-search baseline: `1.6`.

### `--cpuct-start`, `--cpuct-end`, `--cpuct-plies`

Linear schedule for `cpuct` by ply.

- Higher start helps opening exploration.
- Lower end sharpens later search.
- Current sharp-search baseline: `1.8 -> 1.2` over `32` plies.

### `--dirichlet-frac`, `--dirichlet-alpha`, `--dirichlet-plies`

Root exploration noise.

- `dirichlet-frac`: how much root prior mass is replaced by noise.
- `dirichlet-alpha`: shape of noise distribution.
- `dirichlet-plies`: only apply noise before this ply.

Current sharp-search baseline uses lower noise:

```text
--dirichlet-frac 0.10
--dirichlet-plies 12
```

### `--selection-jitter`

Adds random jitter to child selection scores.

- Useful for diversity.
- Bad for clean label-quality experiments.
- Current sharp-search baseline: `0.0`.

As of the May 10, 2026 local-loop hardening pass, `0.0` is exact: MCTS selection no longer adds hidden random tie-breaking jitter when this is disabled. Rerun old no-jitter generator probes before drawing conclusions from them.

The batched collector also applies virtual loss through the actual leaf-collection path. Restart generator processes after updating this code; an already-running generator will keep the old search behavior.

### `--disable-entropy-noise`

Disables extra prior noise when policy logits look uniform.

For the current model, this is important. The model often has broad priors; adding entropy noise makes already-soft labels softer and noisier.

## Move Sampling Knobs

### `--temperature-start`, `--temperature-end`, `--temperature-moves`

Controls move sampling from MCTS visit counts.

- Higher temperature: more diverse games, more noise.
- Lower temperature: more deterministic/self-consistent games.
- Current sharp-search baseline: `0.8 -> 0.15` over `24` moves.

### `--opening-random-plies`

Uniform random legal moves before MCTS starts.

- More: more opening diversity, lower reproducibility.
- Less: cleaner comparison, less variety.
- Current sharp-search baseline: `8`.

## Target Label Knobs

### `--policy-target-temperature`

Applies temperature only to saved MCTS policy targets. It does not affect move sampling.

- `1.0`: preserve raw visit-count distribution.
- `< 1.0`: sharpen low-simulation targets before saving.
- `0.0`: save one-hot argmax targets.

Use this only after search correctness is verified. With a weak model and 50 simulations, correct MCTS can visit almost every legal move; target-only sharpening is a pragmatic way to train a clearer policy signal while keeping generator behavior unchanged.

## Draw Controls

Draw adjudication can dominate training data. Use it intentionally.

### `--draw-halfmove-cap`

Adjudicate draw when `board.halfmove_clock` reaches this value.

### `--draw-material-threshold`

Adjudicate low-material positions as draws when total material is below this threshold.

- Set `0` to disable material-based early draws.

### `--draw-min-plies`

Minimum plies before heuristic draw rules apply.

This applies to heuristic draw rules, not forced draw conditions.

### `--draw-window`, `--draw-min-unique`

Sliding-window repetition heuristic.

- Set both to `0` to disable this heuristic.

### `--draw-claim-min-plies`

Minimum plies before claimable repetition/fifty-move draws are adjudicated.

### `--draw-disable-repetition-claims`

Do not adjudicate claimable threefold repetition. This is useful for diagnosing whether repetition claims are flooding data with neutral labels.

### `--draw-disable-fifty-move-claims`

Do not adjudicate claimable fifty-move draws. The explicit `--draw-halfmove-cap` can still apply if heuristics are enabled.

## Value Target Knobs

### `--capped-value-weight`

Weight for capped/unfinished games when using search values as weak bootstrap targets.

- `0.0`: ignore capped games for value.
- `0.25`: weakly train value on search estimates for capped games.
- Higher values are risky unless search values are demonstrably reliable.

For capped-heavy datasets, value learning is weak even when policy labels are useful.

### `azchess.tools.reweight_npz_values`

Utility for creating a copied data directory with value weights changed for selected result sources. This is useful when existing capped self-play labels are policy-useful but should not contribute to value loss.

Example:

```bash
.venv/bin/python -m azchess.tools.reweight_npz_values \
  --input-dir logs/local_loop/bootstrap_006_anchor_only_nossl_candidate_generator_32g_ptt050_hbfix/data \
  --output-dir logs/local_loop/bootstrap_006_anchor_only_nossl_candidate_generator_32g_ptt050_vw0/data \
  --source capped \
  --source unfinished \
  --value-weight 0.0
```

This preserves policy targets and terminal/tablebase value labels while setting matching `value_weight`, `meta_value_weight`, and zero-valued bootstrap metadata in the copied shards.

## Training Knobs

### `--train-steps`

Number of training steps in the local loop.

Use generator-only first with `--skip-train`. Only train after data quality passes.

### `--batch-size`

Training batch size.

On Apple Silicon, keep this conservative unless memory metrics are stable.

### `--legal-mass-weight`

Penalty that trains raw policy logits to put probability on legal moves.

Current baseline: `0.05`.

Strict local-loop data mode requires legal masks when this is enabled.

### `--legal-policy-weight`

Extra policy CE term computed after masking logits and targets to legal moves.

Use this when full policy CE improves mostly by increasing legal probability mass,
but `policy_legal_ce` / `policy_legal_kl` are flat or worse. This directly
trains ranking among legal moves.

### `--value-include-source`, `--value-exclude-source`

Source-aware value-loss gates. These operate on shard `meta_result_source` values and only affect value loss.

Recommended local-loop split while capped games dominate:

```bash
--value-include-source terminal \
--value-include-source tablebase \
--value-include-source draw_adjudication \
--value-include-source resignation
```

This keeps capped/unfinished self-play useful for policy targets but prevents weak capped bootstrap values from steering value learning.

Alternative:

```bash
--value-exclude-source capped \
--value-exclude-source unfinished
```

Use include mode for promotion-oriented runs because it fails closed when new/unknown result sources appear.

### `--policy-include-source`, `--policy-exclude-source`

Source-aware policy-loss gates. These operate on shard `meta_result_source` values and affect the main policy CE plus legal-policy CE. Legal-mass regularization remains global.

Use this when teacher data is useful for value but hurts heldout legal move ranking:

```bash
--policy-exclude-source teacher:
```

This keeps teacher positions in the batch for value loss if allowed by the value source filter, while preventing the teacher policy distribution from overriding the current self-play/anchor policy curriculum.

### `--ssl-weight`

Overrides SSL loss weight.

Use `0.0` for no-SSL ablations. With the current training path this disables SSL target creation and SSL forward compute for the training loop, while the checkpoint architecture can still contain SSL heads.

### `--policy-label-smoothing`

Policy target smoothing.

Use `0.0` for current self-play label-quality experiments. Smoothing can hide whether MCTS labels are already too soft.

### `--train-anchor-data-dir`

Copies stable prior shards into the training replay buffer after fresh self-play generation.

Use this when fresh data has useful policy labels but poor value outcome mix.

Example:

```text
--train-anchor-data-dir logs/local_loop/bootstrap_003_capped_value_48g/data
--train-anchor-max-files 16
```

### `--train-fresh-max-files`

Limits how many fresh self-play shards remain visible to the training stage.

The local-loop report still records the full generator output in `data_after_selfplay`. Extra fresh shards are moved under `data/excluded_selfplay`, outside the training scan paths. Use this when a long generator run produces good policy labels but too many capped games compared with the anchor set.

Moved fresh shards are also removed from local-loop metadata so the training data manager does not sample excluded files.

Example:

```text
--train-fresh-max-files 48
--train-anchor-data-dir logs/local_loop/bootstrap_003_capped_value_48g/data
```

## Guarded Cycle Knobs

### `azchess.tools.local_loop_cycle`

Runs repeated `bench_local_loop` cycles and promotes candidates only if the configured gate passes. Current fresh-loop runs use one cycle at a time while the gains are small:

```text
.venv/bin/python -m azchess.tools.local_loop_cycle \
  --cycles 1 \
  --seed-best-checkpoint \
  --stop-on-reject \
  --prune-cycle-checkpoints
```

Use multi-cycle runs only after repeated single-cycle promotions are stable.

### `--max-source-value-mse-delta`

Promotion gate for per-source heldout value regression. The current guard is:

```text
--max-source-value-mse-delta 0.0000005
```

This rejects candidates that improve aggregate heldout value by overfitting one source while damaging another. The active heldout sources are:

```text
--eval-result-source tablebase
--eval-result-source terminal
--eval-result-source capped
```

### `--max-fresh-capped-fraction`

Promotion gate on fresh self-play outcome mix. Capped games are still useful at low value weight, but an all-capped fresh batch has been a bad promotion signal. Current settings:

```text
--max-fresh-capped-fraction 0.67
```

Use `0.75` only for a scout or when the terminal/tablebase source guards are comfortably clean.

### `--prune-cycle-checkpoints`

Deletes generated cycle checkpoint files and TensorBoard event files after each cycle. Keep this on for local Apple Silicon work; otherwise each rejected or accepted cycle can leave multiple 600MB+ `.pt` files.

Promotion archives are intentionally left for safety and should be pruned manually after confirming the promoted best checkpoint exists:

```bash
find logs/local_loop/<run>/archives -type f -name '*.pt' -delete
```

### `--eval-select-max-source-value-mse-delta`

Selection-time source guard inside `bench_local_loop`. This prevents the selected training chunk from being chosen solely by aggregate value improvement when any heldout source slice regresses too much:

```text
--eval-select-max-source-value-mse-delta 0.0000005
```

Use this together with the cycle-level `--max-source-value-mse-delta`.

## Current Fresh Self-Play Recipe

As of May 12, 2026, the active loop is:

```text
--games 12
--sims 50
--max-game-len 240
--train-steps 60
--eval-select-interval 10
--lr 3.0e-8
--warmup-steps 10
--capped-value-weight 0.25
--policy-include-source __none__
--policy-distill-checkpoint {parent}
--policy-distill-weight 1.0
--train-anchor-data-dir logs/local_loop/bootstrap_003_capped_value_48g/data
--train-anchor-source-prefix tablebase
--train-anchor-source-prefix terminal
--train-anchor-source-prefix capped
--value-include-source tablebase
--value-include-source terminal
--value-include-source capped
--value-include-source resignation
```

This was the conservative bridge back toward looped self-play. As of the May 12 broad validation and terminal-repair scouts, treat it as saturated rather than an unattended long-run recipe. Do not continue increasing cycle count or teacher volume until a new stabilizer, such as moves-left auxiliary supervision, has passed source-sliced validation.

### Moves-left auxiliary scout

Moves-left supervision is wired as an optional auxiliary objective. It is disabled unless `model.moves_left` is true and `training.moves_left_weight` or `--moves-left-weight` is positive. Self-play shards save per-position `moves_left`; older single-game shards derive the target from `meta_moves`.

Suggested first scout:

```text
--moves-left-weight 0.05
--moves-left-scale 256
--trainable-scope all
--policy-include-source __none__
--policy-distill-checkpoint {parent}
--policy-distill-weight 1.0
```

Promotion still depends on aggregate/source value gates and policy-drift gates. `moves_left_mse` is diagnostic; do not promote a checkpoint only because the auxiliary loss improves.

## Evaluation Knobs

### `--eval-data-dir`

Use stable heldout data instead of evaluating only on the new run’s data.

This is required for meaningful before/after comparison.

### `--eval-batches`

Number of fixed eval batches sampled before training and reused after training.

Use more than one batch for less noisy comparisons.

### `--eval-source-prefix`

Filter eval data by source prefix when using mixed replay directories.

## Strictness

The local loop sets strict mode for subprocesses:

```text
MATRIX0_STRICT_DATA=1
MATRIX0_STRICT_CHECKPOINT=1
```

This means:

- corrupted shards raise instead of being skipped
- missing legal masks raise where legal masks are required
- decoded curriculum/source-prefix batches raise on legal-mask construction failures instead of substituting all-legal masks
- partial checkpoint loads raise unless explicitly configured as a migration
- bad MCTS priors/logits raise instead of becoming uniform labels
- invalid MCTS visit distributions, stale root-policy counts, and NaN/Inf training outputs raise instead of being sanitized or sampled through

This is intentional. A failed run is better than a plausible but corrupted training signal.

## Common Patterns

### Generator-Only Search Probe

Use `--skip-train` to validate labels and outcome mix before training.

Current active probe:

```text
logs/local_loop/bootstrap_006_anchor_only_nossl_candidate_generator_64g_fixed_jitter_vloss
```

Use it to decide whether the anchor-only no-SSL candidate can generate data at least as good as the parent. If it remains all capped with weak target sharpness, do not train or promote it; inspect final-position metadata and fix generation/search behavior first. The current tablebase check passed, so material-heavy caps point at search/conversion behavior rather than broken Syzygy wiring.

### Heldout Training Probe

Use fresh self-play plus a stable heldout eval directory:

```text
--eval-data-dir logs/local_loop/bootstrap_003_capped_value_48g/data
--eval-batches 16
```

### Policy-Focused Anchor Probe

Use sharp fresh data plus anchor data:

```text
--train-anchor-data-dir logs/local_loop/bootstrap_003_capped_value_48g/data
--train-anchor-max-files 16
```

Accept only if legal policy improves and value does not regress materially.
