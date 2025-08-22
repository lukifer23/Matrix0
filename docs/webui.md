# Web UI Guide

Matrix0 includes a comprehensive FastAPI-based web interface for model evaluation, game analysis, and interactive play. The web UI provides both read-only evaluation capabilities and interactive features for testing trained models.

## 🚀 Quick Start

### Prerequisites
- **Python Environment**: Virtual environment with Matrix0 dependencies
- **Model Checkpoints**: At least one trained model in `checkpoints/`
- **Port Availability**: Port 8000 available for the web server

### Running the Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the web server
uvicorn webui.server:app --host 127.0.0.1 --port 8000

# Or with auto-reload for development
uvicorn webui.server:app --host 127.0.0.1 --port 8000 --reload
```

### Accessing the Interface

Once running, access the web interface at:
- **Main Interface**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs (automatic Swagger UI)
- **Alternative API Docs**: http://127.0.0.1:8000/redoc

## 📊 Features Overview

### 🎯 Model Evaluation
- **Interactive Play**: Play against trained Matrix0 models
- **Game Analysis**: Review completed games with move analysis
- **Performance Metrics**: View training statistics and model performance
- **Comparison Tools**: Compare different model checkpoints

### 🔍 Analysis Tools
- **Position Evaluation**: Get model evaluation for any position
- **Move Suggestions**: See top moves ranked by the model
- **Game Statistics**: Win/loss ratios and performance trends
- **Training Progress**: Monitor current training status

### 🎮 Interactive Features
- **Human vs AI**: Play chess against Matrix0 models
- **AI vs AI**: Watch models play against each other
- **Custom Positions**: Set up specific positions for analysis
- **Game Import**: Load and analyze PGN games

## 🛠️ API Endpoints

### Core Endpoints

#### `GET /`
- **Description**: Main web interface
- **Returns**: HTML interface for model evaluation and analysis

#### `GET /health`
- **Description**: System health check
- **Returns**: JSON with system status
```json
{
  "status": "healthy",
  "models_available": ["v2_base.pt", "model_step_1000.pt"],
  "memory_usage": "8.2GB / 14GB",
  "training_status": "active"
}
```

### Model Evaluation Endpoints

#### `POST /api/evaluate`
- **Description**: Evaluate a chess position
- **Request Body**:
```json
{
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "model": "v2_base.pt",
  "num_moves": 5
}
```
- **Returns**: Position evaluation and top moves
```json
{
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "value": 0.234,
  "policy": [
    {"move": "e4e5", "probability": 0.156},
    {"move": "d2d4", "probability": 0.142},
    {"move": "b1c3", "probability": 0.089}
  ]
}
```

#### `POST /api/play`
- **Description**: Make a move with the model
- **Request Body**:
```json
{
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "model": "v2_base.pt",
  "time_limit": 1.0
}
```
- **Returns**: Selected move and evaluation
```json
{
  "move": "e4e5",
  "value": 0.156,
  "search_stats": {
    "nodes": 1247,
    "depth": 8,
    "time": 0.987
  }
}
```

### Game Analysis Endpoints

#### `POST /api/analyze`
- **Description**: Analyze a complete game
- **Request Body**:
```json
{
  "pgn": "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6...",
  "model": "v2_base.pt",
  "depth": 10
}
```
- **Returns**: Game analysis with move evaluations
```json
{
  "moves": [
    {
      "move": "e2e4",
      "value_before": 0.012,
      "value_after": 0.045,
      "best_move": "e2e4",
      "accuracy": 1.0
    }
  ],
  "summary": {
    "total_moves": 35,
    "accuracy": 0.687,
    "average_value_change": 0.023
  }
}
```

#### `GET /api/models`
- **Description**: List available models
- **Returns**: Available model checkpoints
```json
{
  "models": [
    {
      "name": "v2_base.pt",
      "size": "53M parameters",
      "created": "2025-08-25T12:00:00",
      "training_step": 1000
    }
  ]
}
```

### Training Status Endpoints

#### `GET /api/training/status`
- **Description**: Get current training status
- **Returns**: Training progress and metrics
```json
{
  "status": "active",
  "current_step": 1247,
  "total_steps": 10000,
  "loss": 3.215,
  "learning_rate": 0.001,
  "ssl_loss": 0.023,
  "estimated_time_remaining": "2h 15m"
}
```

#### `GET /api/training/metrics`
- **Description**: Get detailed training metrics
- **Returns**: Comprehensive training statistics
```json
{
  "loss_history": [6.5101, 6.1206, 5.6831, 5.0682, 4.4772, 3.4785],
  "ssl_loss_history": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
  "performance": {
    "steps_per_second": 0.498,
    "memory_usage": "12.4GB",
    "gpu_utilization": "87%"
  }
}
```

## 🎮 Using the Web Interface

### Interactive Play Mode
1. **Select Model**: Choose from available checkpoints
2. **Game Settings**: Configure time limits and search parameters
3. **Make Moves**: Play chess against the AI model
4. **Analysis**: View model evaluation for each position

### Game Analysis Mode
1. **Load Game**: Import PGN or set up position manually
2. **Configure Analysis**: Set analysis depth and parameters
3. **Review Results**: Examine move-by-move analysis
4. **Export Data**: Save analysis results for further study

### Training Monitoring
1. **Real-time Stats**: Monitor current training progress
2. **Performance Metrics**: View memory usage and GPU utilization
3. **Loss Trends**: Track training loss over time
4. **Model Comparison**: Compare different checkpoints

## 🔧 Advanced Configuration

### Server Configuration
```python
# webui/server.py - Custom server settings
app = FastAPI(
    title="Matrix0 Web UI",
    description="Chess AI evaluation and analysis interface",
    version="2.0.0"
)

# Model loading configuration
MODEL_CACHE_SIZE = 2  # Number of models to keep loaded
DEFAULT_MODEL = "v2_base.pt"
DEFAULT_SEARCH_TIME = 1.0  # seconds
```

### Performance Optimization
```python
# Memory management settings
MAX_MODEL_MEMORY = 14 * 1024**3  # 14GB limit
MODEL_UNLOAD_TIMEOUT = 300  # 5 minutes

# Search optimization
DEFAULT_MCTS_SIMULATIONS = 300
PARALLEL_SEARCH = True
NUM_SEARCH_THREADS = 4
```

## 🚨 Troubleshooting

### Common Issues

#### Model Loading Errors
```
Error: No checkpoints found in checkpoints/
Solution: Ensure model files exist in the checkpoints directory
```

#### Memory Issues
```
Error: MPS memory limit exceeded
Solution: Reduce batch sizes or increase memory limit in config.yaml
```

#### Port Conflicts
```
Error: Address already in use
Solution: Use different port or kill existing process
```

### Performance Tips
- **Model Caching**: Keep frequently used models loaded
- **Search Limits**: Use appropriate time/depth limits for analysis
- **Batch Processing**: Process multiple positions efficiently
- **Memory Monitoring**: Watch memory usage during analysis

## 📈 Monitoring and Logs

### Server Logs
```bash
# View server logs
tail -f logs/webui.log

# Check for errors
grep "ERROR" logs/webui.log
```

### Performance Monitoring
```bash
# Monitor memory usage
top -p $(pgrep uvicorn)

# Check GPU utilization (MPS)
activity_monitor  # macOS Activity Monitor
```

## 🔄 Development

### Adding New Endpoints
```python
# webui/server.py
@app.post("/api/custom")
async def custom_endpoint(request: CustomRequest):
    # Custom analysis logic
    result = await analyze_custom(request)
    return result
```

### Frontend Development
```bash
# Install frontend dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 📚 API Reference

For complete API documentation, visit `http://127.0.0.1:8000/docs` when the server is running. The interactive Swagger UI provides detailed endpoint documentation with examples and testing capabilities.

---

**Web UI v2.0** - Interactive model evaluation and analysis interface for Matrix0 chess AI.
