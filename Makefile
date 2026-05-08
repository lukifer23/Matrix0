PYTHON?=/opt/homebrew/bin/python3.12
VENV_DIR=.venv
PIP=$(VENV_DIR)/bin/pip
PYTHON_EXEC=$(VENV_DIR)/bin/python

.PHONY: help install setup venv doctor env-info local-loop train selfplay data-stats lint download-lichess cleanup-temp-npz

help:
	@echo "Matrix0 Makefile"
	@echo "----------------"
	@echo "setup          - Create virtual environment and install dependencies."
	@echo "venv           - Create .venv with Python 3.12 and install dependencies."
	@echo "doctor         - Verify imports, torch, pytest, and MPS availability."
	@echo "env-info       - Print environment and model information."
	@echo "local-loop     - Run tiny self-play -> train -> eval benchmark and write JSON."
	@echo "install        - Install/update Python dependencies."
	@echo "train          - Run the training module directly."
	@echo "orchestrator   - Run the complete training pipeline with orchestrator."
	@echo "selfplay       - Start self-play workers to generate training data."
	@echo "data-stats     - Display statistics about the training data."
	@echo "lint           - Run linter and code formatter (requires ruff)."
	@echo "download-lichess - Download a month of Lichess data (e.g., make download-lichess MONTH=2023-01)."
	@echo "cleanup-temp-npz - Remove stray temporary .npz files from the data directory."

# Environment and Installation
setup: $(VENV_DIR)/bin/activate

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Virtual environment created. Run 'source $(VENV_DIR)/bin/activate'."
	touch $@

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

doctor: $(VENV_DIR)/bin/activate
	$(PYTHON_EXEC) -c "import chess, pytest, torch; from azchess.config import select_device; print('python ok'); print(f'torch={torch.__version__}'); print(f'mps_built={torch.backends.mps.is_built()} mps_available={torch.backends.mps.is_available()}'); print(f'selected_device={select_device(\"auto\")}')"

env-info: $(VENV_DIR)/bin/activate
	$(PYTHON_EXEC) -m azchess.tools.model_info --config config.yaml

local-loop: $(VENV_DIR)/bin/activate
	$(PYTHON_EXEC) -m azchess.tools.bench_local_loop --config config.yaml

# Core Workflows
train:
	@echo "Starting training module..."
	$(PYTHON_EXEC) -m azchess.training.train --config config.yaml --steps 1000

orchestrator:
	@echo "Starting complete training pipeline with orchestrator..."
	$(PYTHON_EXEC) -m azchess.orchestrator --config config.yaml

quick-orchestrator:
	@echo "Starting quick orchestrator run (3 games, no eval)..."
	$(PYTHON_EXEC) -m azchess.orchestrator --config config.yaml --workers 3 --games 3 --sims 300 --eval-games 0 --promotion-threshold 0.55 --device auto --tui table

selfplay:
	@echo "Starting self-play data generation..."
	$(PYTHON_EXEC) -m azchess.selfplay --config config.yaml --games 16

# Data Management
data-stats:
	@echo "Fetching data statistics..."
	$(PYTHON_EXEC) -m azchess.data_manager --action stats

download-lichess:
	@echo "Downloading Lichess data for month: $(MONTH)"
	$(PYTHON_EXEC) azchess/tools/process_lichess.py download $(MONTH)

cleanup-temp-npz:
	@echo "Cleaning up temporary NPZ files..."
	$(PYTHON_EXEC) scripts/cleanup_temp_npz.py

# Code Quality
lint:
	@echo "Running linter and formatter..."
	$(PIP) install ruff black
	ruff check .
	black .
