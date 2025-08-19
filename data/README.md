# Matrix0 Training Data

This directory contains processed training data for the Matrix0 chess AI model.

## 📁 Structure

```
data/
├── training/                    # Main training data
│   ├── tactical_training_data.npz      # 10,000 tactical positions
│   └── openings_training_data.npz      # 5,000 opening positions
├── tactical/                    # Raw tactical data
│   ├── tactical_positions.npz          # Processed tactical positions
│   └── tactical_metadata.json          # Tactical data metadata
├── openings/                    # Raw openings data
│   ├── openings_positions.npz          # Processed opening positions
│   └── openings_metadata.json          # Openings metadata
├── selfplay/                    # Self-play games (existing)
└── validation/                  # Validation data (existing)
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

## 🚀 Usage

### **Simple Integration**
```python
from comprehensive_data_loader import ComprehensiveDataLoader

# Initialize loader
loader = ComprehensiveDataLoader()

# Get mixed training batch
batch = loader.get_mixed_batch(512)

# Get curriculum-specific batch
batch = loader.get_curriculum_batch(512, phase="openings")  # Focus on openings
batch = loader.get_curriculum_batch(512, phase="tactics")   # Focus on tactics
batch = loader.get_curriculum_batch(512, phase="mixed")     # Balanced mix
```

### **Training Integration**
```python
# In your training loop
for epoch in range(num_epochs):
    # Determine curriculum phase
    if epoch < 20:
        phase = "openings"
    elif epoch < 50:
        phase = "tactics"
    else:
        phase = "mixed"
    
    # Get appropriate training batch
    batch = loader.get_curriculum_batch(batch_size, phase=phase)
    
    # Train step
    loss = train_step(model, optimizer, batch, device)
```

## 📊 Data Statistics

| Data Source | Samples | Size | Purpose |
|-------------|---------|------|---------|
| **Tactical** | 10,000 | 745KB | Winning combinations |
| **Openings** | 5,000 | 422KB | Opening knowledge |
| **Total** | **15,000** | **1.2MB** | **Comprehensive training** |

## 🔄 Data Processing

The raw data has been processed and filtered for quality:
- **Tactical**: Minimum 150 centipawn advantage
- **Openings**: Minimum 2000 performance rating, 100+ games
- **Format**: Compatible with Matrix0 encoding (19 planes, 8x8 boards)

## 🎓 Curriculum Learning

The data supports progressive learning:
1. **Phase 1**: Opening mastery (80% openings, 20% tactics)
2. **Phase 2**: Tactical training (80% tactics, 20% openings)  
3. **Phase 3**: Mixed training (30% each + 40% self-play)

## 📝 Notes

- All data is pre-processed and ready for training
- Use `comprehensive_data_loader.py` for access
- Data can be mixed with existing self-play data
- Supports both batch and curriculum-based training
