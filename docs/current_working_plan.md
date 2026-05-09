# Current Working Plan

Last updated: 2026-05-09

This is the living document for the active Matrix0 training/debugging loop. Keep it short, factual, and current. Historical project status belongs in `status.md` and `CURRENT_STATUS_SUMMARY.md`; this page tracks the immediate problem, current best checkpoint, active hypotheses, and promotion criteria.

## Current Goal

Make the model improve reliably from local Apple Silicon self-play without accepting misleading wins from corrupted data, noisy targets, or evaluation artifacts.

The current focus is not raw throughput. The priority is a trustworthy loop:

1. Generate self-play data with valid legal masks, policy targets, value targets, and metadata.
2. Verify label quality before training.
3. Train only when the generated data passes acceptance checks.
4. Evaluate on stable heldout data before promoting a checkpoint.

## Current Best Checkpoint

Use this as the parent until a candidate beats it on stable heldout metrics and generator quality:

```text
checkpoints/bootstrap_006_capped_value.pt
```

Do not promote candidates from runs that are mostly capped, mostly early draw-adjudicated, or only improve raw unmasked policy metrics while regressing legal policy or value.

## What We Fixed Recently

The local-loop path now fails hard instead of silently producing bad data:

- self-play checkpoint load failure no longer falls back to an untrained model
- partial checkpoint loads require an explicit migration flag
- legal mask generation failures raise instead of saving all-zero masks
- SSL target generation failures raise instead of saving zero SSL targets
- training rejects invalid policy rows, zero-legal masks, malformed masks, and missing masks when legal-mass training is enabled
- strict local-loop data mode raises on corrupted shards instead of skipping them
- MCTS expansion raises on invalid logits/priors instead of inventing uniform priors
- draw adjudication has explicit controls for claimable repetition and fifty-move draws

Relevant commits:

- `72a731d Harden local loop training pipeline`
- `0917a97 Add draw adjudication controls`

## Current Findings

### Sharp Search Helps Policy Labels

The sharp-search generator settings improved policy target quality:

- `policy_top_prob` rose from roughly `0.20-0.22` to about `0.25-0.26`
- `policy_entropy` fell from roughly `2.05-2.16` to about `1.8`
- `policy_support` fell from roughly `10-13` to about `7.5`
- `legal_policy_mass` stayed near `1.0`

This suggests lower exploration, lower Dirichlet noise, and disabled entropy noise are better for producing trainable policy labels at the current model strength.

### Outcome Mix Is Still The Main Blocker

Recent sharp-search generator probes:

- `bootstrap_006_sharp_search_generator_32g`: sharp labels, but `26/32` draw adjudications
- `bootstrap_006_sharp_search_halfmove_generator_32g`: draw override did not block claimable repetition; `25/32` draw adjudications
- `bootstrap_006_sharp_no_repetition_generator_32g`: draw adjudication fixed, but `31/32` capped and only `1/32` tablebase

Interpretation: current self-play can make sharper policy labels, but it is not producing enough decisive/tablebase outcomes for strong value learning.

## Active Hypotheses

### H1: 50 Sims Is Too Shallow For Decisive Outcomes

Test `100` sims on a 16-game generator probe. More search may convert more games to tablebase/terminal instead of capping.

Acceptance:

- tablebase + terminal at least `4/16`
- capped below roughly `10/16`
- policy remains sharp: `policy_top_prob >= 0.25`, `policy_entropy < 1.9`
- throughput remains acceptable for follow-up 32-game probes

If this fails, self-play decisiveness is not solved by modestly higher sims.

### H2: Policy Learning And Value Learning Need Different Data Mixes

If sharp self-play remains mostly capped, train policy on sharp capped data but anchor value on older tablebase/terminal-heavy data.

Candidate recipe:

- fresh sharp self-play for policy
- `logs/local_loop/bootstrap_003_capped_value_48g/data` as anchor data for value/tablebase coverage
- heldout eval on `logs/local_loop/bootstrap_003_capped_value_48g/data`
- low or moderate capped value weight; do not let capped bootstrap values dominate value loss

Acceptance:

- legal policy CE/ KL improves on stable heldout
- value MSE does not regress materially
- generator check from candidate does not worsen cap/draw mix

## Current Probe Command

The active/next probe is the 100-sim 16-game run:

```bash
MATRIX0_MPS_TARGET_BATCH=4 \
.venv/bin/python -m azchess.tools.bench_local_loop \
  --config config.yaml \
  --run-dir logs/local_loop/bootstrap_006_sharp_s100_len200_generator_16g \
  --games 16 \
  --workers 2 \
  --sims 100 \
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
  --init-checkpoint checkpoints/bootstrap_006_capped_value.pt \
  --skip-train
```

## Promotion Rules

Promote only after all of these are true:

- generator data has acceptable outcome mix, or the training recipe explicitly compensates with anchor data
- legal policy metrics improve on heldout data
- value MSE does not regress materially on heldout data
- post-training generator check does not degrade game outcomes
- no strict-mode data/checkpoint failures occurred

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
- `eval.delta`

