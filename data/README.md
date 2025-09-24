# Matrix0 Training Data

This directory contains processed training data for the Matrix0 chess AI model.

## 📁 Structure

```
data/
├── tactical/                    # Processed tactical curriculum bundle
│   ├── tactical_positions.npz          # Tactical positions (ready for DataManager)
│   └── tactical_metadata.json          # Tactical metadata
├── openings/                    # Processed opening curriculum bundle
│   ├── openings_positions.npz          # Opening positions (ready for DataManager)
│   └── openings_metadata.json          # Opening metadata
├── teacher_games/               # Teacher-guided replay data
│   └── enhanced_teacher/*.npz          # Tagged NPZ shards (auto-ingested)
├── stockfish_games/             # Stockfish-generated replay trees (tagged NPZ/json)
├── selfplay/                    # Recent self-play episodes (compacted into replays/)
├── replays/                     # Compacted training shards consumed by trainer
├── backups/                     # Archived self-play NPZs retained post-compaction
├── syzygy/                      # Installed Syzygy tablebases (currently 3-5 man WDL)
└── validation/                  # Validation splits (if configured)
```

## 🎯 Data Sources

### **Tactical Data** (10,000 samples)
- **Source**: Lichess puzzle database
- **Content**: Winning tactical positions (150+ centipawn advantage)
- **Format**: Board states, best moves, normalized evaluations
- **Purpose**: Teach winning combinations and tactical patterns

### **Openings Data** (5,000 samples)
- **Source**: Chess openings database with performance statistics
- **Content**: Quality openings (2000+ rating, 100+ games)
- **Format**: Board states, opening moves, quality scores
- **Purpose**: Teach proper opening principles and theory

## 🚀 Usage in Training

- The orchestrator/trainer now ingests these sources automatically via `training.extra_replay_dirs` in `config.yaml` (tactical, openings, teacher, and stockfish directories).
- Keep `selfplay/` lean by letting the orchestrator compact runs into `replays/`; archived copies are preserved under `backups/` if you need to restore.

## 📊 Data Notes

- Tactical bundle: Winning motifs curated from Lichess, stored in `data/tactical/`.
- Opening bundle: High-quality lines stored in `data/openings/`.
- Teacher games: Enhanced scenarios in `data/teacher_games/enhanced_teacher/` with source tags for curriculum filters.
- Stockfish trees: Tagged NPZ/JSON pairs under `data/stockfish_games/` for flexible slicing (e.g., openings, tactics, endgames).

## 🔄 Maintenance Tips

- Periodically prune or archive `data/backups/` if disk usage becomes a concern.
- Install additional Syzygy tables in `data/syzygy/` before raising `tablebases.max_pieces` above 5.
- If you add new curated directories, append them to `training.extra_replay_dirs` or import them via `DataManager.import_replay_dir`.
