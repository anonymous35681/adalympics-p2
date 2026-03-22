@_default:
    just --list

# CI: lint, format (ruff, ty)
qa:
    uv run ruff check --fix src/ scripts/
    uv run ty check src/ scripts/
    uv run ruff format src/ scripts/

# Generate all graphics (or specific graph with number)
run arg='': qa
    uv run src/main.py {{arg}}
