PYTHON ?= python
ARGS ?=

.PHONY: train smoke eval test lint format upstream
train:
	TORCH_DEVICE_BACKEND_AUTOLOAD=0 $(PYTHON) -m src.train $(ARGS)
smoke:
	TORCH_DEVICE_BACKEND_AUTOLOAD=0 $(PYTHON) -m src.train experiment=smoke $(ARGS)
eval:
	TORCH_DEVICE_BACKEND_AUTOLOAD=0 $(PYTHON) -m src.eval $(ARGS)
test:
	TORCH_DEVICE_BACKEND_AUTOLOAD=0 $(PYTHON) -m pytest $(ARGS)
lint:
	$(PYTHON) -m ruff check src tests
format:
	$(PYTHON) -m ruff format src tests
upstream:
	$(PYTHON) -m src.upstream.fetch
