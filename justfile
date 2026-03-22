@_default:
    just --list

# CI: lint, format (ruff, ty)
qa:
    uv run ruff check --fix src/
    uv run ty check src/
    uv run ruff format src/

# Generate all graphics (or specific graph with number)
run: qa
    uv run src/main.py
