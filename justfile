lint:
    uv run ruff format
    uv run ruff check --fix

test:
    uv run pytest -v tests
