# Leela Chess Zero Adoption Opportunities

Last updated: 2026-05-12

This document tracks ideas worth borrowing from Leela Chess Zero (LC0) and how they map to Matrix0. It is not a plan to clone LC0 wholesale. Matrix0 is currently in a fragile guarded self-play recovery phase, so new ideas should enter as small scouts with source-sliced gates.

## Current Matrix0 Baseline

Current mainline checkpoint:

```text
checkpoints/bootstrap_007_fresh_anchor_best.pt
```

Current loop state:

- guarded fresh self-play is producing real but small heldout value gains
- `bootstrap_007_fresh12_len240_guard_cycle5_s60_lr3e8` rejected because aggregate heldout value gain was too small (`value_mse -1.90e-7` vs gate `-3e-7`)
- source guards remained clean: capped `+2.79e-10`, tablebase `-3.47e-7`, terminal `+3.75e-7`
- fresh capped fraction was acceptable at `0.50`
- conclusion: the current bootstrap recipe is saturating; validate broadly before more unattended cycles

## LC0 Ideas Worth Adopting

### 1. Moves-Left Auxiliary Head

Priority: high

Why it matters:

LC0-style networks can expose a moves-left prediction head. Matrix0's current biggest source of uncertainty is capped games: they provide weak search-value labels, but they do not tell the model whether the position is close to resolution. A moves-left target would give the value tower a direct training signal for conversion horizon and game persistence.

Matrix0 fit:

- self-play shards already record `moves` and final-position metadata
- capped games already have a meaningful remaining-horizon ambiguity
- this can be added as an auxiliary head without replacing scalar value

Suggested scout:

- add optional `moves_left` head behind config, disabled by default
- train it only when a shard has a reliable target:
  - terminal/tablebase: exact remaining plies from sampled position to game end
  - capped: either exclude or train as censored/low-weight target
- evaluate whether it improves capped/terminal source value without hurting tablebase

Acceptance:

- aggregate heldout value improves
- tablebase, terminal, capped source deltas stay inside `+5e-7`
- no policy CE drift beyond current bounds
- moves-left loss decreases on exact terminal/tablebase positions

Do not:

- use moves-left to promote by itself
- let censored capped labels dominate the head

### 2. WDL Auxiliary Head

Priority: medium

Why it matters:

LC0 uses WDL-style value outputs to separate win, draw, and loss probabilities. Matrix0 currently trains a scalar tanh value. Scalar value hides draw calibration, which matters for tablebase and terminal/draw-heavy positions.

Matrix0 fit:

- `PolicyValueNet` already has optional WDL-head plumbing
- tablebase and terminal outcomes can produce exact WDL labels
- draw-adjudication data can produce exact draw labels when trusted

Suggested scout:

- keep scalar value as the main value output
- enable WDL as an auxiliary loss only
- include WDL targets only for exact sources first: `tablebase`, `terminal`, trusted `draw_adjudication`, `resignation` if resignation labels are reliable
- exclude capped from WDL until there is a principled soft target

Acceptance:

- scalar heldout value does not regress
- terminal/tablebase source slices improve or remain flat
- draw calibration improves on a draw-specific heldout slice

Do not:

- replace scalar value during the current bootstrap phase
- promote a WDL checkpoint without source-sliced scalar value validation

### 3. Stronger Draw And Game-Length Calibration

Priority: high

Why it matters:

LC0 treats draw probability and game length as first-class search concerns. Matrix0's capped-game pressure suggests the search/value loop still has weak conversion and horizon awareness.

Matrix0 fit:

- final-position metadata already records piece count, halfmove clock, legal count, and draw-claim availability
- draw adjudication controls are already configurable
- tablebase source slices are available for exact endgame validation

Suggested scout:

- build a report that buckets fresh games by final piece count, halfmove clock, result source, and moves
- compare promoted vs rejected runs to identify which fresh mixes correlate with safe heldout gains
- add a validation report before unattended loops:
  - capped fraction
  - tablebase fraction
  - terminal fraction
  - average final piece count for capped games
  - terminal/tablebase value source deltas

Acceptance:

- validation identifies a stable outcome mix threshold
- the threshold predicts rejection before training in most bad runs

### 4. Search Parameter Presets By Phase

Priority: medium

Why it matters:

LC0 distinguishes engine/search behavior from training data production. Matrix0 has one broad `config.yaml` with production-like exploration settings, while the current successful loop uses much more conservative command-line overrides.

Matrix0 fit:

- local-loop commands already override sims, game length, LR, and gating
- docs now define the active guarded recipe

Suggested scout:

- add named local-loop presets:
  - `bootstrap_guarded_fresh`
  - `generator_probe`
  - `broad_validation`
  - `teacher_scout`
- keep these as command templates or YAML fragments before wiring them deeply into code

Acceptance:

- commands become reproducible and less error-prone
- no behavior change unless explicitly selected

### 5. Backend/Inference Boundary Cleanup

Priority: medium-low

Why it matters:

LC0 has a strong separation between search, neural backend, batching, and self-play. Matrix0 has improved but still has Python/MPS-specific details crossing module boundaries.

Matrix0 fit:

- `azchess.selfplay.inference` and MCTS batching already exist
- current bottleneck is correctness and promotion reliability, not max throughput

Suggested scout:

- document and test a minimal inference-backend interface:
  - encode batch
  - evaluate policy/value/optional aux heads
  - return legal-masked policy diagnostics
- only refactor once training is stable enough that performance matters again

Acceptance:

- no training/eval behavior changes
- existing local-loop tests still pass
- benchmark shows no MPS regression

## Ideas Not Worth Copying Yet

### Compact LC0 Policy Map

LC0 has compact policy-head formats. Matrix0 currently uses `4672` policy outputs. Changing policy shape would break checkpoints, replay data, legal-mask assumptions, and current diagnostics.

Decision: defer until a major checkpoint boundary.

### Full WDL Replacement

Replacing scalar value with WDL now would reset the value target interface during a fragile recovery.

Decision: use WDL only as an auxiliary scout.

### Large Teacher Expansion

Teacher games can help, but they can also overwrite the emerging self-play distribution. Matrix0 currently has untracked teacher data under:

```text
data/teacher_games/bootstrap_007_teacher_parent/
```

Decision: do not scale teacher volume yet. If tested, use a tiny scout with `teacher` excluded from policy CE unless heldout legal-policy metrics prove it is safe.

## Recommended Order

1. Stop additional `bootstrap_007` fresh cycles; broad validation found a terminal regression and direct terminal repair did not produce a clean candidate.
2. Add moves-left auxiliary targets and a small auxiliary head as the first LC0-inspired model scout.
3. Validate moves-left on exact terminal/tablebase data and require capped/tablebase/terminal value source guards to remain clean.
4. Add WDL auxiliary head only after moves-left and broad validation are stable.
5. Build named local-loop presets to reduce command drift.
6. Revisit compact policy mapping and deeper backend refactors only at a major checkpoint boundary.

## References

- LC0 technical explanation: https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/
- LC0 neural-network notes: https://draft.lczero.org/dev/old/nn/
- LC0 overview: https://lczero.org/dev/overview/
- LC0 WDL discussion: https://draft.lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/
