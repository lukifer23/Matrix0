# Web UI Guide

Matrix0 includes a lightweight FastAPI-based web interface for viewing evaluation games and interacting with trained models.

## Running the server

```bash
uvicorn webui.server:app --host 127.0.0.1 --port 8000
```

The interface is currently read-only and intended for evaluation and analysis.
