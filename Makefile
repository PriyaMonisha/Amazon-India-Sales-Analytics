.PHONY: setup etl eda train serve monitor all test clean feast-apply feast-materialize

# --- Setup ---
setup:
	python -m venv venv
	venv/Scripts/python.exe -m pip install --upgrade pip
	venv/Scripts/python.exe -m pip install -r requirements/dev.txt
	cp .env.example .env
	@echo "Setup complete. Edit .env as needed, then run: make etl"

# --- Data pipeline (run in order) ---
etl:
	APP_ENV=local python notebooks/01_data_engineering.py

eda: etl
	APP_ENV=local python notebooks/02_eda.py

train:
	FAST_MODE=true APP_ENV=local python notebooks/03_model_training.py

# --- Serving (local dev) ---
serve:
	APP_ENV=local uvicorn api.main:app --reload --port 8000

app:
	APP_ENV=local streamlit run streamlit_app/app.py

# --- Feast ---
feast-apply:
	cd feast_repo && feast apply

feast-materialize:
	cd feast_repo && feast materialize-incremental $$(date -u +%Y-%m-%dT%H:%M:%S)

# --- Monitoring (Docker only) ---
monitor:
	docker compose up prometheus grafana -d

# --- Full stack ---
all:
	docker compose up --build

# --- Tests ---
test:
	python -m pytest tests/ -v --tb=short

# --- Cleanup ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cache cleaned."
