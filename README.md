# claudegnostic

Diagnostics and statistics for Claude Code sessions.

## Install

Run without installing:

```bash
uvx claudegnostic --help
```

Install as a tool:

```bash
uv tool install claudegnostic
```

From a local checkout:

```bash
uv tool install .
```

## Usage

```bash
claudegnostic --help
claudegnostic hello --name you
```

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy
```

Optional: enable pre-commit.

```bash
uv run pre-commit install
```
