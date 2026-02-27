# Conductor

General-purpose agent orchestrator for managing autonomous coding agents
via a warm tmux pool, file-based I/O, Pydantic contracts, and SQLite state.

## Overview

Conductor orchestrates autonomous `agent chat` sessions running in tmux.
It breaks work into granular micro-steps, dispatches them to a warm agent
pool with model-tier awareness, and tracks progress in a local SQLite
database with GitHub as the durable projection.

**Key principles:**

- **Mechanical work in Python** -- context loading, file parsing, git
  operations, GitHub sync are handled directly by the conductor.
- **Creative work via agents** -- design, architecture, test writing,
  code writing, and debugging are dispatched to agent sessions.
- **Structured contracts** -- Pydantic models define the JSON protocol
  between steps. Validation failures trigger automatic retries.
- **Project-agnostic** -- all project-specific configuration lives in
  `.conductor/conductor.toml` in the consuming repository.

## Installation

```bash
pip install conductor
# or
uv pip install conductor
```

## Usage

```bash
# Initialize config in your project
conductor --init

# Run the orchestrator (coming soon)
conductor run
```

## Design

See the [design document](https://github.com/gerchowl/scitadel/issues/23)
for the full architecture specification.

## Development

```bash
git clone https://github.com/gerchowl/conductor.git
cd conductor
uv sync --all-extras
uv run pytest
```

## License

MIT
