.PHONY: install data prepare train leakage label-delay store drift bench all serve mlflow-ui test lint check

# Keep the laptop usable: models and BLAS stay on 4 threads, heavy steps run at low priority.
export FRAUDOPS_THREADS ?= 4
export OMP_NUM_THREADS ?= 4
export MLFLOW_DISABLE_AGENT_HINT = 1
COMPETITION := ieee-fraud-detection

install:
	uv sync

data:               ## Kaggle download (accept the competition rules first; needs ~/.kaggle/kaggle.json)
	mkdir -p data/raw
	uvx kaggle competitions download $(COMPETITION) -f train_transaction.csv -p data/raw
	uvx kaggle competitions download $(COMPETITION) -f train_identity.csv -p data/raw

prepare:            ## CSVs -> data/transactions.parquet -> data/features.parquet
	nice -n 10 uv run fraudops prepare

train:              ## baseline, LightGBM, ablation, Optuna, calibration, registry
	nice -n 10 uv run fraudops train --trials 30

leakage:            ## random split / future-peeking aggregates vs the honest setup
	nice -n 10 uv run fraudops leakage

label-delay:        ## value of faster chargebacks: labels usable after 7/30/60/90 days
	nice -n 10 uv run fraudops label-delay

store:              ## online feature store snapshot at the end of validation
	uv run fraudops build-store

drift:              ## weekly PSI + PR-AUC on the val/test period
	nice -n 10 uv run fraudops drift

bench:              ## HTTP latency + online/offline parity on 5,000 test transactions
	uv run fraudops bench -n 5000
	uv run fraudops bench -n 5000 --no-explain

all: prepare train leakage label-delay store drift bench

serve:
	uv run fraudops serve --port 8000

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050

test:
	uv run pytest --cov --cov-report=term

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

check: lint test
