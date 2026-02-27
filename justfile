# Conductor — task runner recipes

# Show available commands
[group('info')]
help:
    @just --list

# Run tests
[group('quality')]
test *args='':
    uv run pytest {{ args }}

# Run tests with coverage
[group('quality')]
test-cov:
    uv run pytest --cov --cov-report=term-missing

# Lint with ruff
[group('quality')]
lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

# Format code
[group('quality')]
fmt:
    uv run ruff check --fix src/ tests/
    uv run ruff format src/ tests/

# Run pre-commit hooks on all files
[group('quality')]
precommit:
    uv run pre-commit run --all-files

# Full local verification
[group('quality')]
verify:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "=== Lint ==="
    just lint
    echo ""
    echo "=== Tests with coverage ==="
    just test-cov
    echo ""
    echo "=== All checks passed ==="
