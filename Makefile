.PHONY: setup lint test app format

setup:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

app:
	streamlit run app/streamlit_app.py
