# Local playable demo

The local demo is a six-max no-limit Hold'em table with a React/Vite client,
FastAPI service, SQLite recovery, and WebSocket event stream.

## Run

```bash
./scripts/run_local_demo.sh
```

Open `http://127.0.0.1:8790`. The first run installs the web dependencies and
builds the client. The
SQLite database is stored at `.local/poker-demo/poker_demo.sqlite3` and is not
tracked by Git.

The default provider is `Deterministic Mock`, which is offline and reproducible.
`Live · aliyun_99` is optional. It calls the already-authenticated remote
OpenCode CLI over the restricted SSH alias and fixes the model to
`opencode-go/deepseek-v4-flash`; it does not expose an HTTP LLM gateway.

## Frozen behavior

- Only Hero switches between `Human Player` and `LLM Agent` using the
  `closed_loop_shaper` strategy.
- Switching back to Human invalidates an in-flight LLM result. Provider errors,
  validation errors, the 15-second timeout, or the 200-call live budget also
  pause delegation and return control to Human without a fallback action.
- Human advice is read-only. It can observe public actions, but it never writes
  a strategy patch.
- Strategy changes are bounded, LLM-authored, post-hand patches. The player can
  inspect versions and diffs but cannot edit, freeze, or roll back them.
- Each hand begins at 100 BB. There is no hand limit; the table can only be ended
  after a hand completes.
- The anonymous owner is identified by an HttpOnly cookie. Anyone without that
  cookie receives a read-only view, and folded or non-showdown opponent cards
  remain hidden in replay state.

## Configuration

The backend serves the built client, `/api`, and the event WebSocket from
`127.0.0.1:8790`. Optional environment variables:

```bash
POKER_DEMO_DB=/absolute/path/demo.sqlite3
POKER_DEMO_LIVE_CALL_LIMIT=200
```

## Verify

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests
npm --prefix web run build
```
