.PHONY: help setup sync lint format check typecheck test test-fast test-cov \
        notebooks run lab clean reset ci dev

# `src` is imported as a flat package and the project is not installed into the venv
# (see [tool.uv] package = false in pyproject.toml), so every recipe needs the repo root
# on the import path. Jupyter, nbconvert and Streamlit all run with a cwd other than the
# repo root, so this cannot be left to the default sys.path.
export PYTHONPATH := $(CURDIR)

# Simulation size profile, read by src/config.py. `full` = the published numbers
# (10k replications); `ci` = the reduced-rep profile used by `make ci`.
HTA_PROFILE ?= full
export HTA_PROFILE

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	uv run python -m ipykernel install --user --name hypothesis-testing-arena
	@echo "\n✅ Ready. Run 'make test'."

sync: ## Sync deps
	uv sync --all-extras

lint: format check typecheck ## All linters
format: ## ruff format
	uv run ruff format src/ tests/ app/
check: ## ruff check
	uv run ruff check --fix src/ tests/ app/
typecheck: ## ty check
	uv run ty check src/

test: ## pytest
	uv run pytest
test-fast: ## pytest without the simulation-heavy cases
	uv run pytest -x -q -m "not slow"
test-cov: ## pytest with coverage
	uv run pytest --cov=src --cov-report=term-missing

notebooks: ## Execute all notebooks in place (figures -> outputs/figures, numbers -> outputs/results)
	@for nb in notebooks/0*.ipynb; do \
		echo "▶ Executing $$nb ..."; \
		uv run jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=900 "$$nb" || exit 1; \
	done
	@echo "\n✅ All notebooks executed."

# --inplace so the committed notebooks carry their rendered output, which is what a reader of
# the repository actually looks at; without it nbconvert leaves *.nbconvert.ipynb clutter.
# The 900s timeout (not 300s) is sized for notebook 04: the full 16-scenario arena at 10,000
# replications takes about nine and a half minutes on a cold cache, and seconds on a warm one.

run: ## Streamlit app
	uv run streamlit run app/streamlit_app.py --server.port 8501
lab: ## JupyterLab
	uv run jupyter lab --no-browser --port 8888

clean: ## Clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ty_cache htmlcov .coverage .pytest_cache .ruff_cache
	@echo "🧹 Cleaned."
reset: clean ## Full reset
	rm -rf .venv outputs/cache

ci: HTA_PROFILE := ci
ci: sync lint test ## CI (reduced-rep simulation profile)
dev: lint test ## Fast loop
