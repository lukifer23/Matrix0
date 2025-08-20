# Configuration Guide

Matrix0 uses a single `config.yaml` file to control all components of the system.
Below are common parameters:

```yaml
mcts:
  num_simulations: 200      # Search depth
  cpuct: 2.5                # Exploration constant
  dirichlet_alpha: 0.3      # Noise for exploration
  dirichlet_frac: 0.25      # Noise impact fraction
  fpu: 0.5                  # First-play urgency for unvisited nodes
  draw_penalty: -0.1        # Slight draw penalty in terminal evaluation

selfplay:
  num_workers: 4            # Parallel workers
  shared_inference: true    # GPU optimization
  max_game_len: 140         # Game length limit
  resign_threshold: -0.30   # Resign when evaluation drops below this
  min_resign_plies: 50      # Minimum moves before resignation
  resign_consecutive_bad: 3 # Bad evaluations before resigning

training:
  batch_size: 512           # Training batch size
  ssl_weight: 0.1           # SSL loss weight
  warmup_steps: 500         # Learning rate warmup
```

## Draw Adjudication

- `mcts.draw_penalty` applies a small penalty to drawn outcomes.
- `selfplay.resign_threshold`, `min_resign_plies`, and `resign_consecutive_bad` control when games are resigned to avoid long draws.

## Tips

- For evaluation, set `eval.dirichlet_frac: 0.0` and lower `eval.max_moves` (around 220).
- Increase `mcts.draw_penalty` magnitude (e.g., `-0.2`) to reduce drawish play.
- Lower `selfplay.temperature_moves` (e.g., `20`) to rein in midgame randomness.

For external engine-specific configuration, see [External Engine Integration](../EXTERNAL_ENGINES.md).
