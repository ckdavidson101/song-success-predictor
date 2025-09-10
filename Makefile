PY=python3

.PHONY: setup dev backend frontend test lint fmt data-kaggle data-manifest

setup:
	$(PY) -m venv backend/.venv
	. backend/.venv/bin/activate && pip install -r backend/requirements.txt
	cd frontend && npm install

dev:
	# backend
	cd backend && . .venv/bin/activate && uvicorn app.main:app --reload & \
	# frontend
	cd frontend && npm run dev

backend:
	cd backend && . .venv/bin/activate && uvicorn app.main:app --reload

frontend:
	cd frontend && npm run dev

test:
	cd backend && . .venv/bin/activate && pytest -q

lint:
	cd backend && . .venv/bin/activate && ruff check . && mypy app && black --check .

fmt:
	cd backend && . .venv/bin/activate && black .

data-kaggle:
	cd backend && . .venv/bin/activate && $(PY) ../ml/utils/kaggle_ingest.py --raw-dir ../data/raw/kaggle/spotify_1m --out ../data/interim/spotify_features.parquet

data-manifest:
	. backend/.venv/bin/activate && $(PY) -m ml.utils.sampling --in data/interim/spotify_features.parquet --out data/interim/manifest_v1.parquet --counts data/interim/manifest_v1_counts.csv --n 50000 --seed 42
