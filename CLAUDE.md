# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Repository Authority

This repository opts into **team-maintainer** behavior for the bd workflow:

- After finishing work on a bead (closing it), always `git add` the changed files and `git commit` them without asking.
- Use the bead id in the commit message (e.g. `feat: ... (claudegnostic-NNN)`).
- Do **not** `git push` or run Dolt remote sync unless the user explicitly asks.
- A current "do not commit" instruction from the user still wins.

## Build & Test

```bash
uv sync --dev              # install deps (incl. dev extras)
uv run pytest              # run tests
uv run ruff check .        # lint
uv run mypy                # type-check
uv run claudegnostic ...   # invoke the CLI from source
```

Optional pre-commit:

```bash
uv run pre-commit install
```

## Product Scope

`claudegnostic` is meant to be generally useful for **anyone** running Claude
Code. The target user is a developer who wants insight into their own session
data — usage patterns, token economics, cost, and workflow shape — without
sending anything off-machine.

Design implications:

- Zero-config defaults — auto-discover `~/.claude/projects`, write the DB to
  `XDG_DATA_HOME/claudegnostic/`.
- Degrade gracefully on sparse data — a user with three sessions should still
  get a usable report, not stack traces on empty percentiles.
- All analysis is local. The CLI never makes network calls. Reports are
  single-file artifacts the user can choose to share.
- Base install stays slim. Heavier surfaces (dashboard) are optional extras.

## Architecture Overview

Pipeline, in order:

1. **Parser** (`parser.py`) — reads Claude Code session JSONL files and
   normalises each assistant turn into a row dict.
2. **Ingest** (`ingest.py`) — discovers session files under a root, parses
   them, upserts into DuckDB, and rebuilds the `sessions` aggregation.
3. **Storage** (`storage.py`, `schema.py`) — owns the DuckDB connection and
   the `turns` / `sessions` schema. This is the contract everything else
   reads from.
4. **Analysis layer** (`analysis/`, planned) — pure functions returning
   DataFrames for each lens (cost, productivity, workflow archaeology). Both
   the report and the dashboard import from here. No duplication.
5. **Surfaces**:
   - **`report`** (planned) — Jinja2 + plotnine PNGs (base64-embedded) →
     single self-contained `.html` file. Default surface for most users.
   - **`dashboard`** (planned) — Streamlit app reading the same DuckDB.
     Optional install (`pip install 'claudegnostic[dashboard]'`).
   - **`ingest` / `stats`** (current) — the data-plumbing CLI commands.

The DuckDB file is the durable boundary. Analysis modules should depend only
on the schema, never on parser/ingest internals.

## Conventions & Patterns

**Plotting**

- Default: **plotnine** for static charts (per the global user preference).
- Reserve **Altair** for the dashboard's interactive charts only (hover,
  zoom, filtering). Don't mix interactive libraries elsewhere.
- Before any plotnine import in a non-notebook context, set
  `matplotlib.use("Agg")` — required for headless rendering in `report`.

**Report output**

- The HTML report is a **single self-contained file**. PNGs are
  base64-embedded; no sidecar `assets/` directory.
- Set chart DPI explicitly (100, not the matplotlib 300 default) to keep
  file size reasonable.
- Every section must handle the empty-data case explicitly. Prefer a clear
  "not enough data" note over a broken chart.

**Optional dependencies**

- Dashboard deps live behind an extra: `claudegnostic[dashboard]`.
- If `claudegnostic dashboard` is invoked without the extra installed,
  detect the `ImportError` and print the exact install command. Never let
  users hit a raw traceback.

**Privacy**

- Reports include `cwd`, `git_branch`, and byte/token counts. That's fine
  for personal use but surprising in a shared artifact. When adding new
  fields, ask: would the user be surprised this is in a file they might
  share?
- No telemetry, no network calls, ever.

**Schema changes**

- The `turns` and `sessions` schemas are the contract between ingest and
  analysis. Adding columns is safe; renaming or removing columns is a
  breaking change and needs a bead.
