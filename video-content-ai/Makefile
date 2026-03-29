.PHONY: dev install lint test docker clean

# ── Development ─────────────────────────────────────────────────
dev:
	python -m app.main

install:
	pip install -e ".[dev]"

# ── Quality ─────────────────────────────────────────────────────
lint:
	ruff check app/

test:
	pytest tests/ -v

# ── Docker ──────────────────────────────────────────────────────
docker:
	docker compose up --build

docker-down:
	docker compose down

# ── CLI ─────────────────────────────────────────────────────────
serve:
	python -m app.cli serve

# ── Cleanup ─────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache dist build *.egg-info
