PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYBIN=$(VENV)/bin/python

.PHONY: venv install run-selfplay train eval orchestrate

venv:
	$(PY) -m venv $(VENV)
	@echo "Run 'source $(VENV)/bin/activate' to activate."

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run-selfplay:
	$(PYBIN) -m azchess.selfplay --games 8

train:
	$(PYBIN) -m azchess.train

eval:
	$(PYBIN) -m azchess.eval --ckpt_a checkpoints/model.pt --ckpt_b checkpoints/best.pt

orchestrate:
	$(PYBIN) -m azchess.orchestrator --config config.yaml

