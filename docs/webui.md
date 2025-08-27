# Matrix0 Enhanced WebUI Guide

Matrix0 features a **comprehensive FastAPI-based web interface** that serves as a **complete monitoring and analysis platform** for SSL-integrated chess AI development. The enhanced WebUI provides **real-time SSL monitoring, training analytics, model analysis, and interactive evaluation** - all integrated with the advanced SSL architecture.

## 🚀 Quick Start

### Prerequisites
- **Python Environment**: Virtual environment with Matrix0 dependencies
- **Model Checkpoints**: At least one SSL-integrated model in `checkpoints/`
- **Port Availability**: Port 8000 available for the web server
- **FastAPI Dependencies**: `pip install fastapi uvicorn` (if not already installed)

### Running the Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Install FastAPI dependencies (if needed)
pip install fastapi uvicorn

# Start the web server
python webui/server.py

# Or with uvicorn directly
uvicorn webui.server:app --host 127.0.0.1 --port 8000 --reload
```

### Accessing the Interface

Once running, access the enhanced WebUI at:
- **Main Interface**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs (automatic Swagger UI)
- **Alternative API Docs**: http://127.0.0.1:8000/redoc

## 📊 Enhanced Features Overview

### 🎮 Game View - Interactive Chess Interface
- **Interactive Play**: Play against SSL-integrated Matrix0 models
- **Game Analysis**: Real-time position evaluation with SSL-aware analysis
- **Move Visualization**: See model evaluation charts and move suggestions
- **Game Management**: Full game lifecycle with PGN export and analysis

### 📈 Training View - Real-Time Training Monitor
- **Live Training Status**: Real-time step tracking and progress visualization
- **Loss Analytics**: Interactive charts showing total loss, SSL loss, and learning rate
- **Performance Metrics**: Memory usage, training speed, and stability monitoring
- **Training History**: Recent training steps with detailed metrics tracking

### 🔬 SSL View - Advanced SSL Monitoring Dashboard
- **SSL Configuration**: Real-time SSL settings and task status display
- **SSL Heads Analysis**: Detailed view of all 5 SSL heads (piece, threat, pin, fork, control)
- **Parameter Tracking**: SSL parameter counts and architecture validation
- **SSL Performance**: Monitor SSL learning effectiveness and task balancing

### 🧠 Analysis View - Comprehensive Model Analysis
- **Architecture Overview**: Model specifications, channels, blocks, attention heads
- **Parameter Breakdown**: Layer-by-layer parameter distribution analysis
- **SSL Integration Status**: SSL enablement verification and task configuration
- **Model Specifications**: Complete technical architecture documentation

## 🛠️ Enhanced API Endpoints

### Core Endpoints

#### `GET /`
- **Description**: Enhanced multi-view web interface
- **Returns**: HTML interface with Game, Training, SSL, and Analysis views

#### `GET /health`
- **Description**: Comprehensive system health check
- **Returns**: JSON with system status and SSL integration
```json
{
  "stockfish": true,
  "model_params": 53206724,
  "device": "cpu",
  "ssl_enabled": true,
  "ssl_tasks": ["piece", "threat", "pin", "fork", "control"],
  "training_status": "operational"
}
```

#### `GET /ssl/status`
- **Description**: SSL configuration and status monitoring
- **Returns**: Complete SSL integration status
```json
{
  "enabled": true,
  "tasks": ["piece", "threat", "pin", "fork", "control"],
  "ssl_head_count": 5,
  "total_ssl_params": 260320,
  "head_parameters": {
    "piece": 53600,
    "threat": 51680,
    "pin": 51680,
    "fork": 51680,
    "control": 51680
  }
}
```

#### `GET /training/status`
- **Description**: Real-time training monitoring
- **Returns**: Live training metrics and progress
```json
{
  "is_training": true,
  "current_step": 4479,
  "total_steps": 24000,
  "progress": 18.66,
  "latest_metrics": {
    "loss": 3.0847,
    "policy_loss": 2.9533,
    "value_loss": 0.0849,
    "ssl_loss": 1.1628,
    "learning_rate": 0.000856
  },
  "recent_history": [...]
}
```

#### `GET /model/analysis`
- **Description**: Comprehensive model architecture analysis
- **Returns**: Detailed model specifications and parameter breakdown
```json
{
  "total_parameters": 53206724,
  "parameter_breakdown": {
    "Conv2d": 245760,
    "BatchNorm2d": 10240,
    "ReLU": 0,
    "Linear": 260320
  },
  "ssl_heads": {
    "piece": {"parameters": 53600, "structure": "..."},
    "threat": {"parameters": 51680, "structure": "..."},
    "pin": {"parameters": 51680, "structure": "..."},
    "fork": {"parameters": 51680, "structure": "..."},
    "control": {"parameters": 51680, "structure": "..."}
  },
  "architecture": {
    "channels": 320,
    "blocks": 24,
    "attention_heads": 20,
    "ssl_enabled": true,
    "ssl_tasks": ["piece", "threat", "pin", "fork", "control"]
  }
}
```

### Game Management Endpoints

#### `POST /new`
- **Description**: Create a new game with SSL-aware engine selection
- **Request Body**:
```json
{
  "white": "matrix0",
  "black": "human",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "engine_tc_ms": 100
}
```
- **Returns**: New game ID and initial position

#### `POST /move`
- **Description**: Play a move in the current game
- **Request Body**:
```json
{
  "game_id": "g_1725495600000",
  "uci": "e2e4"
}
```
- **Returns**: Updated position and game status

#### `POST /engine-move`
- **Description**: Get engine move with SSL-enhanced evaluation
- **Request Body**:
```json
{
  "game_id": "g_1725495600000",
  "engine": "matrix0"
}
```
- **Returns**: Engine move and evaluation

#### `POST /eval`
- **Description**: Evaluate current position with SSL-aware analysis
- **Request Body**:
```json
{
  "game_id": "g_1725495600000",
  "include_stockfish": true
}
```
- **Returns**: Matrix0 and Stockfish evaluation

#### `POST /batch-match`
- **Description**: Run automated tournaments between engines
- **Request Body**:
```json
{
  "games": 20,
  "engine_tc_ms": 100,
  "start_white": "matrix0"
}
```
- **Returns**: Tournament results and statistics

### Analytics Endpoints

#### `GET /analytics/summary`
- **Description**: Get match analytics and performance statistics
- **Returns**: Win/loss/draw ratios and recent match history

#### `GET /pgn/list`
- **Description**: List available PGN files for analysis
- **Returns**: Available game files for review

## 🎮 Using the Enhanced WebUI

### Multi-View Interface Navigation

The enhanced WebUI features four specialized views accessible via the top navigation:

#### 🎮 Game View - Interactive Chess Interface
1. **Engine Selection**: Choose Matrix0, Stockfish, or Human players for each side
2. **Time Controls**: Set engine thinking time (10ms - 10s range)
3. **Interactive Play**: Click on the board or use move input to play
4. **Real-time Analysis**: View live evaluation charts and SSL-aware analysis
5. **Game Management**: Start new games, review move history, export PGN

#### 📈 Training View - Live Training Monitor
1. **Training Status**: View current step, progress, and estimated completion
2. **Loss Visualization**: Interactive charts showing:
   - Total loss progression
   - SSL loss contribution
   - Learning rate changes
   - Policy/value loss breakdown
3. **Performance Metrics**: Monitor memory usage, training speed, stability
4. **Training History**: Review recent steps with detailed metrics

#### 🔬 SSL View - Advanced SSL Monitoring
1. **SSL Configuration**: View current SSL settings and task status
2. **SSL Heads Analysis**: Examine all 5 SSL heads:
   - **Piece Detection**: 53,600 parameters
   - **Threat Detection**: 51,680 parameters
   - **Pin Detection**: 51,680 parameters
   - **Fork Detection**: 51,680 parameters
   - **Control Detection**: 51,680 parameters
3. **Parameter Tracking**: Monitor SSL learning effectiveness
4. **SSL Performance**: Track task balancing and contribution

#### 🧠 Analysis View - Model Architecture Deep Dive
1. **Model Specifications**: View complete architecture details
2. **Parameter Breakdown**: Layer-by-layer parameter distribution
3. **SSL Integration Status**: Verify SSL enablement and configuration
4. **Architecture Comparison**: Compare model specifications and capabilities

## 🔧 Advanced Configuration

### SSL Monitoring Configuration
```python
# webui/server.py - SSL monitoring settings
SSL_UPDATE_INTERVAL = 60  # Update SSL status every 60 seconds
TRAINING_UPDATE_INTERVAL = 30  # Update training status every 30 seconds

# SSL head analysis configuration
SSL_HEAD_ANALYSIS_DEPTH = 5  # Number of SSL heads to analyze
SSL_PARAMETER_TRACKING = True  # Enable detailed parameter tracking
```

### Real-time Updates Configuration
```python
# Periodic update settings for live monitoring
UPDATE_INTERVALS = {
    'training': 30000,  # 30 seconds
    'ssl': 60000,       # 60 seconds
    'analysis': 120000  # 2 minutes
}

# Memory management for real-time updates
MAX_UPDATE_MEMORY = 512 * 1024**2  # 512MB limit for update operations
UPDATE_CACHE_SIZE = 10  # Cache last 10 update results
```

### Model Analysis Configuration
```python
# Model analysis settings
PARAMETER_BREAKDOWN_DEPTH = 10  # Show top 10 parameter types
SSL_HEAD_DETAIL_LEVEL = 'full'  # Options: 'basic', 'detailed', 'full'

# Architecture visualization settings
ARCHITECTURE_VISUALIZATION = True
PARAMETER_DISTRIBUTION_CHART = True
SSL_INTEGRATION_VERIFICATION = True
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

## 🔍 Current Project Status

### Training Pipeline
- **Status**: ✅ **FULLY OPERATIONAL** with complete SSL integration
- **SSL Integration**: ✅ **COMPLETE** - All 5 SSL tasks working simultaneously
- **Multi-Task Learning**: ✅ **ACTIVE** - Policy, value, and SSL training combined
- **Real-time Monitoring**: ✅ **ENHANCED** - WebUI provides comprehensive monitoring

### Model Architecture
- **Parameters**: 53.2M (ResNet-24 with complete SSL integration)
- **SSL Heads**: **5 specialized SSL heads** (piece, threat, pin, fork, control)
- **SSL Parameters**: 260,320 dedicated SSL parameters
- **Memory Usage**: 14GB MPS limit with SSL processing optimization
- **Performance**: ~3-4 seconds per training step with SSL

### WebUI Capabilities
- **Multi-View Interface**: Game, Training, SSL, and Analysis views
- **Real-time Monitoring**: Live training and SSL status updates
- **Interactive Visualization**: Charts, progress bars, and status indicators
- **SSL Dashboard**: Complete SSL head analysis and performance tracking
- **Model Analysis**: Deep architecture inspection and parameter breakdown

### Development Priorities
1. **SSL Performance Validation**: Measure and validate SSL learning effectiveness
2. **Enhanced Evaluation**: Multi-engine tournaments with SSL-aware metrics
3. **SSL Visualization**: Advanced SSL decision explanation tools
4. **Automated Testing**: Comprehensive SSL validation suites

---

**Matrix0 Enhanced WebUI v2.1** - Complete monitoring and analysis platform for SSL-integrated chess AI development.

*Current focus: SSL performance validation and multi-engine evaluation with SSL-aware metrics.* 🚀
