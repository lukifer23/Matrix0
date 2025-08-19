PYTHON=python3
VENV_DIR=.venv
PIP=$(VENV_DIR)/bin/pip
PYTHON_EXEC=$(VENV_DIR)/bin/python

.PHONY: help install setup train selfplay data-stats lint download-lichess cleanup-temp-npz

help:
	@echo "Matrix0 Makefile"
	@echo "----------------"
	@echo "setup          - Create virtual environment and install dependencies."
	@echo "install        - Install/update Python dependencies."
	@echo "train          - Run the comprehensive training script."
	@echo "selfplay       - Start self-play workers to generate training data."
	@echo "data-stats     - Display statistics about the training data."
	@echo "lint           - Run linter and code formatter (requires ruff)."
	@echo "download-lichess - Download a month of Lichess data (e.g., make download-lichess MONTH=2023-01)."
	@echo "cleanup-temp-npz - Remove stray temporary .npz files from the data directory."

# Environment and Installation
setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Virtual environment created. Run 'source $(VENV_DIR)/bin/activate'."
	touch $@

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Core Workflows
train:
	@echo "Starting comprehensive training..."
	$(PYTHON_EXEC) train_comprehensive.py --config config.yaml

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

