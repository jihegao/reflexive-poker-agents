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
OpenCode CLI over the restricted SSH alias. Each of the six seats independently
selects a model from the server-discovered `opencode-go` catalog; it does not
expose an HTTP LLM gateway or accept arbitrary provider/model URLs.

## Frozen behavior

- Hero switches between `Human Player` and `LLM Agent`; each of the other five
  seats can independently switch between its frozen rule strategy and an LLM
  controller.
- Hero and every other seat persist an independent `opencode-go` model choice.
  Existing tables default to `deepseek-v4-flash`.
- The entry page configures all six seats at once: strategy, Human/Rule/LLM
  controller, and model. The service mode is inferred automatically; a table
  with an LLM seat uses the live provider, while an all-Rule table stays mock.
- Each LLM seat starts from its own persona (`closed_loop_shaper`, TAG, LAG,
  Rock, Calling Station, or Myopic), receives only its own decision history and
  reflection memory, and advances only its own bounded strategy-version chain
  after a hand in which it acted.
- Switching back to Human invalidates an in-flight LLM result. Provider errors,
  validation errors, the 60-second timeout, or the 200-call live budget pause
  delegation without silently changing the selected controller or taking a
  fallback action.
- Human advice is read-only. It can observe public actions, but it never writes
  a strategy patch.
- Strategy changes are bounded, per-seat LLM-authored, post-hand patches. The player can
  inspect versions and diffs but cannot edit, freeze, or roll back them.
- The left Agent list shows each LLM seat's model. Clicking an LLM seat opens
  its current-hand decision history and saved reflections; the right rail stays
  dedicated to Hero and shows the current LLM thought.
- The owner-only hand archive keeps every completed hand as a replay unit. Open
  `牌局档案` (or `查看本手复盘`) to inspect the public action timeline, the model's
  recorded situation summary/rationale/self-model/opponent-model/risk flags at
  each LLM action, and every participating LLM seat's post-hand review, belief
  updates, calibration, and next-hand strategy adjustment. These are
  model-authored audit explanations, not a transcript of hidden chain of thought.
- Completed-hand archives persist the public board, visible showdown cards,
  action amounts, bounded decision traces, and reflections. Owner-only reasoning
  data is omitted from spectator snapshots, and folded or non-showdown opponent
  cards remain hidden.
- Each hand begins at 100 BB. There is no hand limit; the table can only be ended
  after a hand completes, or ended immediately with the owner leave control.
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
