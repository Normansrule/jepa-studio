# JEV local API (`/api/jev/*`)

JEV's routes are mounted on the same local backend as the JEPA routes (`jepa-studio serve`,
see [../api.md](../api.md)): `127.0.0.1` only, the per-session `X-JEPA-Token` header on every
request, the `Host`/`Origin` checks, JSON bodies capped at 2 MB. Large outputs (images, videos)
are files in the workspace, fetched by URL, never inlined.

Long operations (a chat turn, a render) run in a background thread and are followed by
polling an event log, the same pattern as training runs: `GET .../events?since=<n>` returns
`{events: [...], next: <n>}`.

## Status and models

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/jev/status` | | `{version, ollama: {url, reachable, version, error?}, default_model, models: [...], tools: [{name, description, risk, category}], workspace_root, sandbox: {network_isolation, limits}, gpu}` |
| GET | `/api/jev/models` | | `{models: [{name, size_bytes, family, parameter_size, quantization, digest, modified}]}` (from Ollama `/api/tags`) |
| POST | `/api/jev/models/pull` | `{name}` | `{approval_id}` always; downloading a model is an approval-gated action. After approval, progress events go to `/api/jev/jobs/<job_id>/events` |
| POST | `/api/jev/models/import` | `{gguf_path, name}` | `{approval_id}`; imports a local GGUF file you own into Ollama with a Modelfile (`FROM <path>`), recording its SHA-256 |
| DELETE | `/api/jev/models/<name>` | | `{approval_id}` (deleting asks) |
| POST | `/api/jev/settings` | `{default_model?, limits?, network_allowed?}` | the new settings |

## Conversations and turns

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/api/jev/chat` | `{message, conversation_id?, model?, attachments?: [workspace paths]}` | `{conversation_id, turn_id}` |
| GET | `/api/jev/turns/<turn_id>/events` | `?since=<n>` | `{events, next, status}` |
| POST | `/api/jev/turns/<turn_id>/cancel` | | `{ok}` |
| GET | `/api/jev/conversations` | | `{conversations: [{id, title, created, updated, model}]}` |
| GET | `/api/jev/conversations/<id>` | | `{id, title, messages: [...], artifacts: [...]}` |
| DELETE | `/api/jev/conversations/<id>` | | `{approval_id}` (deletes its workspace too, so it asks) |

Turn event records (`event` field):

| event | fields | meaning |
|---|---|---|
| `start` | `model` | the turn began |
| `token` | `text` | streamed model text |
| `tool_call` | `call_id, name, args, risk, decision` | the model asked for a tool; `decision` is `run`, `ask` or `deny` |
| `approval_request` | `approval_id, call_id, name, args, reason` | waiting for you (`/api/jev/approvals/<id>`) |
| `progress` | `call_id, fraction, message` | from a running tool |
| `tool_result` | `call_id, ok, content, artifacts: [{id, path, kind, mime, title, url}]` | |
| `artifact` | `{id, path, kind, mime, title, url}` | also listed in tool_result |
| `done` | `text` | final answer |
| `error` | `message` | |
| `cancelled` | | |

## Approvals (the safety net)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/jev/approvals` | `?status=pending` | `{approvals: [{id, created, tool, args, reason, risk, conversation_id, turn_id, status}]}` |
| POST | `/api/jev/approvals/<id>` | `{decision: "approve" \| "deny", remember?: "session" \| null}` | `{ok, status}` |

Every decision (including automatic ones) is written to the audit log in the SQLite store.
`remember: "session"` allows the same tool with the same argument *shape* for the rest of the
backend session only; nothing is remembered across restarts.

## Artifacts

| Method | Path | Returns |
|---|---|---|
| GET | `/api/jev/artifacts/<id>` | the file bytes with its content type (`Content-Disposition: inline`), streamed; only files inside a workspace |
| GET | `/api/jev/artifacts` | `?conversation_id=` → `{artifacts: [...]}` |

`url` fields are `/api/jev/artifacts/<id>`; the front end fetches them with the token header and
shows them via `blob:` URLs (the CSP allows `img-src`/`media-src` `blob:`).

## Science (no model needed)

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/jev/science/blackhole` | `{spin, inclination_deg, distance_M, disk: {...}, width, height, frames?, seconds?}` | `{job_id}`; result artifact(s) via `/api/jev/jobs/<job_id>/events` |
| POST | `/api/jev/science/simulate` | `{kind: "nbody" \| "pendulum" \| "wave" \| ..., params}` | `{job_id}` |
| GET | `/api/jev/jobs/<job_id>/events` | `?since=` | `{events, next, status}` (same event shapes as turns) |
