.PHONY: help setup sync test data jupyter lab scratch clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Setup ---

setup: sync data ## Full setup: install deps + generate synthetic data

sync: ## Install all dependencies
	uv sync --extra dev

data: ## Generate synthetic data
	uv run python scripts/generate_synthetic_data.py

# --- Demos (copies to scratch/ so originals stay clean) ---
#
# Workflow:
#   1. make prep-01 prep-02   (copy demos you need to scratch/)
#   2. make scratch            (open one Jupyter server for all of scratch/)
#   3. Open any notebook from the file browser
#
# Or just:  make demo-01      (copy + open Jupyter in one step)

prep-%: ## Copy demo NN to scratch/ (e.g. make prep-01)
	@mkdir -p scratch
	@nb=$$(find demos -path "demos/$*_*/demo_*.ipynb" | head -1); \
	if [ -z "$$nb" ]; then \
		echo "No demo notebook found for week $*"; exit 1; \
	fi; \
	cp "$$nb" "scratch/$$(basename $$nb)"; \
	echo "Copied $$nb -> scratch/$$(basename $$nb)"

prep-all: ## Copy all demo notebooks to scratch/
	@mkdir -p scratch
	@for nb in demos/*/demo_*.ipynb; do \
		cp "$$nb" "scratch/$$(basename $$nb)"; \
		echo "Copied $$nb -> scratch/$$(basename $$nb)"; \
	done

scratch: ## Open Jupyter server for scratch/ workspace
	uv run jupyter notebook --notebook-dir=scratch

demo-%: prep-% ## Copy demo NN + open Jupyter (e.g. make demo-01)
	uv run jupyter notebook --notebook-dir=scratch

# --- General ---

jupyter: ## Open Jupyter in demos/ directory (read-only browsing)
	uv run jupyter notebook --notebook-dir=demos

lab: ## Open JupyterLab in demos/ directory
	uv run jupyter lab --notebook-dir=demos

test: ## Run all tests
	uv run pytest tests/ -v

clean: ## Remove generated/temp files and scratch notebooks
	rm -rf .pytest_cache scratch
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '.ipynb_checkpoints' -type d -exec rm -rf {} + 2>/dev/null || true
