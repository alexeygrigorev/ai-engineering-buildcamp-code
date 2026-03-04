# Documentation Agent

An AI-powered assistant for Evidently documentation, with a monitoring dashboard and fake-data generator for testing.

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env
```

## Running the app

```bash
# Documentation assistant (Streamlit)
make streamlit

# Monitoring dashboard (port 8502)
make dashboard
```

## Generating fake monitoring data

The fake data generator simulates realistic user sessions — questions, follow-up clicks, and feedback — and writes them directly to the SQLite database so you can see the dashboard in action without needing real traffic.

```bash
# Run continuously at 2 events/sec (Ctrl-C to stop)
make fake-data

# Custom rate and fixed number of sessions
uv run python -m logs.generate_fake_data --rate 5.0 --sessions 20

# All options
uv run python -m logs.generate_fake_data --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--rate` | `2.0` | Events per second |
| `--sessions` | `0` (∞) | Stop after N sessions |
| `--db` | `db/logs.db` | SQLite database path |
| `--quiet` | off | Suppress per-event output |

Each session generates:
- **Log records** — agent runs with token counts, execution times, and tool calls
- **`followup_clicked`** events — when a user clicks a suggested follow-up question
- **`user_feedback`** events — thumbs up/down feedback (`+1` / `-1`)

## Running tests

```bash
make tests

# Judge tests in parallel
make tests-judge
```
