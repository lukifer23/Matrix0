# Local Loop Knob Guide

This guide explains the knobs used by `azchess.tools.bench_local_loop` during Matrix0 self-play, training, and checkpoint comparison. It is optimized for the current Apple Silicon local-loop workflow.

## Purpose

Use the local loop to answer one narrow question at a time:

- did generator data get better?
- did policy labels get sharper?
- did value targets get less biased?
- did training improve heldout metrics?
- did a candidate generate better games than its parent?

Do not use a full training cycle until the generator-only run passes basic data-quality checks.

## Generator Quality Metrics

Inspect these in `local_loop_report.json` under `fresh_data.quality`.

- `legal_policy_mass`: target policy mass on legal moves. Should be near `1.0`.
- `policy_top_prob`: average probability assigned to the most visited move. Higher means sharper labels.
- `policy_entropy`: entropy of the MCTS visit distribution. Lower means sharper labels.
- `policy_support`: number of moves with non-trivial target probability. Lower means more focused labels.
- `avg_sims`: actual average simulations per move.
- `source_metrics`: the same metrics split by `capped`, `tablebase`, `terminal`, and `draw_adjudication`.

Current rough targets:

- `legal_policy_mass ~= 1.0`
- `policy_top_prob >= 0.25` for sharp-search experiments
- `policy_entropy < 1.9` for sharp-search experiments
- low draw adjudication unless a draw-specific run is intentional
- enough tablebase/terminal data for value learning, or explicit anchor data

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

### `--ssl-weight`

Overrides SSL loss weight.

Use `0.0` for ablations, but note this does not remove SSL heads from the architecture.

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
- partial checkpoint loads raise unless explicitly configured as a migration
- bad MCTS priors/logits raise instead of becoming uniform labels

This is intentional. A failed run is better than a plausible but corrupted training signal.

## Common Patterns

### Generator-Only Search Probe

Use `--skip-train` to validate labels and outcome mix before training.

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

