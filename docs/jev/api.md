# JEV local API (`/api/jev/*`)

JEV's routes are mounted on the same local backend as the JEPA routes (`jepa-studio serve`,
see [../api.md](../api.md)): `127.0.0.1` only, the per-session `X-JEPA-Token` header on every
request, the `Host`/`Origin` checks, JSON bodies capped at 2 MB. Large outputs (images, videos)
are files in the workspace, fetched by URL, never inlined. Implementation: `jev/api.py`
(routes), `jepa_studio/server.py` (`_jev`, `_send_file`: the guard and streamed file responses).

JEV keeps its data in `<root>/jev-data/` (`settings.json`, `jev.db`, `workspaces/<conversation>/`,
`exports/`). It is not `<root>/jev/` because in a source checkout the root is the repository,
where `jev/` is the Python package.

Long operations (a chat turn, a render, a model download) run in a background thread and are
followed by polling an event log, the same pattern as training runs:
`GET .../events?since=<n>` returns `{events: [...], next: <n>, status}`. Every event has
`i` (its index), `t` (unix time) and `event`. Jobs live in memory: after a backend restart
their event logs are gone (conversations, messages, artifacts and the audit log are in SQLite).

Errors are `{error: "..."}` with: 400 bad input (including an approval whose `args_sha256` is
missing or does not match, and a workspace path that is a FIFO, socket or device), 401 token, 403 outside the workspace / granted
folders or refused by `approval_rules`, 404 unknown id, 409 the conversation is still answering,
413 file too large, 421 bad Host, 501 science module not installed, 502 Ollama returned an
error, 503 Ollama isn't running (the message says how to start it).

## Status, settings and models

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/jev/status` | | `{version, ollama: {url, reachable, version, error?}, default_model, models: [model], tools: [{name, description, risk, category}], workspace_root, data_dir, sandbox: {method, network_isolation, filesystem, rlimits, process_tree_kill, windows_job, probe, limits, fs_isolated, net_isolated, data_dir_protected, tmpfs_size_limit, cgroup_scope, weak, weak_reasons, isolation, isolation_notes, wsl: {detected, interop_enabled, windows_path_in_PATH}}, network_allowed, gpu: {cuda, mps, devices: [{name, memory_gb}]}, approval_rules: [{rule, default, action, covers, locked, rememberable}]}`. `sandbox.isolation` is `strong`, `weaker (WSL interop)` or `weak`; when `weak` (no file-system or no network isolation, `weak_reasons` says which) every tool that runs code or FFmpeg asks first (the locked `weak_sandbox` row); `isolation_notes` explains a `weaker` result (docs/jev/SAFETY.md) |
| GET | `/api/jev/models` | | `{models: [model], imported: [{name, path, sha256, bytes, created}]}`; `model` = `{name, size_bytes, family, parameter_size, quantization, digest, modified, remote}` (from Ollama `/api/tags`; `remote: true` marks Ollama cloud models, which JEV refuses unless `allow_cloud_models`) |
| POST | `/api/jev/models/pull` | `{name}` | `{approval_id, status: "pending", args, args_canonical, args_sha256}` (every approval-gated action replies with the exact arguments of the approval it created); downloading a model is an approval-gated action. Approving returns `job_id`; progress goes to `/api/jev/jobs/<job_id>/events`. If you set the `model_download` (and `network`) rows to `run` in `approval_rules`, the reply is `{approval_id, status: "auto_run", job_id}` |
| POST | `/api/jev/models/import` | `{gguf_path, name}` | `{approval_id, status}`; imports a local GGUF file you own (it must be inside a folder granted to JEPA Studio, or be a file granted by the desktop shell's `pick_gguf_file` dialog) into Ollama, recording its SHA-256 (in `models`, and in the job's `done` event) |
| DELETE | `/api/jev/models/<name>` | | `{approval_id, status}` (deleting asks). URL-encode the name (`llama3.1%3A8b`; `/` as `%2F`) |
| GET | `/api/jev/settings` | | the settings below plus `rules` (the approval table, as in `status.approval_rules`) and `paths: {data_dir, settings, db, workspaces, exports}` (where everything is on disk) |
| POST | `/api/jev/settings` | `{default_model?, limits?: {cpu_seconds, memory_mb, wall_seconds, max_output_mb, max_file_mb, max_workspace_mb, max_workspace_entries}, network_allowed?, ollama_url?}` | the new settings. `ollama_url` must be loopback unless `allow_remote_ollama` is set **in the file**; `approval_rules`, `allow_remote_ollama`, `allow_cloud_models` can only be changed by editing `jev-data/settings.json` (400 otherwise); limits outside sane bounds are refused. Every change is written to the audit log (`kind: "settings"`, `tool: "settings_update"`, `status: "applied"`, `args: {patch, changed: {<key>: {before, after}}}`) |

## All of your data

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/jev/export` | | a zip (`Content-Disposition: attachment`): `jev-data.json` (`format: "jev-export/1"`: every conversation with its turns, messages, tool calls, approvals and artifacts; the whole audit log; imported models; `settings: {file, effective, paths}`) and `workspaces/<folder>/...` (every workspace file and the science renders, without `.venv`, caches, symlinks or FIFOs). A copy (mode 0600, `*.app.zip`) is left in `jev-data/exports/`; the 3 newest of these are kept (exports made with `jev export` are never pruned) |
| POST | `/api/jev/data/delete` | `{keep_audit_log?: false}` | `{approval_id, status}` (the `delete` row: it asks). Approving stops running turns and science jobs, then deletes every conversation (rows and workspace folders), `_science/`, `exports/`, the imported-model records and, unless `keep_audit_log`, every audit row created before the request (the approval for this deletion stays as the first row of the new log; with `keep_audit_log` the tool arguments are redacted instead). Then `VACUUM`, so deleted rows do not linger in `jev.db`. `settings.json` (your approval rules) and the models in Ollama are kept; delete the `jev-data/` folder or use `ollama rm` for those. Only conversation workspaces inside `workspace_root` are removed, never the whole folder |

## Conversations and turns

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/api/jev/chat` | `{message, conversation_id?, model?, attachments?: [workspace paths]}` | `{conversation_id, turn_id}`; 409 while that conversation is still answering |
| GET | `/api/jev/turns/<turn_id>/events` | `?since=<n>` | `{events, next, status}`; `status` is `running`, `waiting_approval`, `done`, `error` or `cancelled` |
| POST | `/api/jev/turns/<turn_id>/cancel` | | `{ok}`; running code is killed, a pending approval is closed as `cancelled` |
| GET | `/api/jev/conversations` | | `{conversations: [{id, title, created, updated, model}]}` |
| POST | `/api/jev/conversations` | `{title?, model?}` | `{id, title, model, workspace, created, updated}` (an empty conversation, e.g. to upload files first) |
| GET | `/api/jev/conversations/<id>` | | `{id, title, model, workspace, created, updated, messages: [{id, role, content, tool_calls, tool_name, turn_id, created}], artifacts: [artifact]}` |
| DELETE | `/api/jev/conversations/<id>` | | `{approval_id, status}` (deletes its rows and workspace too, so it asks; the audit rows stay with their arguments redacted) |
| GET | `/api/jev/conversations/<id>/export` | | a zip (`Content-Disposition: attachment`): `conversation.json` (messages, turns, tool calls, approvals, artifacts) and `files/` (the workspace, without `.venv`). A copy (`<id>-*.app.zip`, mode 0600) stays in `jev-data/exports/`; the 3 newest per conversation are kept |
| POST | `/api/jev/conversations/<id>/files` | `{name, data_base64}` (or `content_base64`; <= ~1.5 MB so the body stays under 2 MB) or `{name, from_path}` (a file inside a folder granted to JEPA Studio) | `{path, name, bytes, artifact}`; `name` is a path inside the workspace |

Turn event records (`event` field):

| event | fields | meaning |
|---|---|---|
| `start` | `model, conversation_id` | the turn began |
| `token` | `text` | streamed model text (a tool call written as text is held back, not streamed) |
| `tool_call` | `call_id, name, args, risk, decision, reason` | the model asked for a tool; `decision` is `run`, `ask` or `deny` (strings in `args` are cut at 20 000 characters here, for display only) |
| `approval_request` | `approval_id, call_id, name, args, args_canonical, args_sha256, reason, risk, rules` | waiting for you (`/api/jev/approvals/<id>`); the turn's `status` is `waiting_approval`. `args` are the complete arguments, never cut; `args_canonical` is their canonical JSON (sorted keys, no spaces, ASCII only: every other character as a `\u` escape) and `args_sha256` its SHA-256. The app checks that `args_canonical` parses to exactly the `args` it shows, computes the SHA-256 itself and sends that. A call whose arguments are over 1 000 000 characters is refused instead of asked, because no card could show it |
| `approval_result` | `approval_id, call_id, status` | `approved`, `denied`, `expired` (no answer within `approval_timeout_s`, default 30 min) or `cancelled` |
| `progress` | `call_id, fraction, message` | from a running tool (`fraction` may be `null` when unknown) |
| `tool_result` | `call_id, ok, content, artifacts: [artifact], data?` | `content` is what the model reads (<= ~4000 chars) |
| `artifact` | `id, path, kind, mime, title, bytes, sha256, url` | also listed in tool_result |
| `done` | `text, rounds` | final answer (`rounds` = tool rounds used, at most `max_tool_rounds`, default 12) |
| `error` | `message` | |
| `cancelled` | | |

`artifact` = `{id, path, kind, mime, title, bytes, sha256, url}`; `kind` is one of `image`,
`video`, `audio`, `code`, `text`, `data`, `model`, `other`; `path` is relative to the workspace.

## Approvals (the safety net)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/jev/approvals` | `?status=pending`, `?limit=N` | `{approvals: [{id, created, kind, tool, args, args_sha256, args_canonical?, reason, risk, rules, conversation_id, turn_id, call_id, status, decided_by, decided, remember, job_id}]}` (newest first; `limit` 1 to 10 000, default 500; a pending approval carries its complete arguments and `args_canonical`) |
| POST | `/api/jev/approvals/<id>` | `{decision: "approve" \| "deny", remember?: "session" \| null, args_sha256}` | `{ok, status, job_id?, note?}`; `job_id` when approving started a job (model pull/import/delete, a long render). **Approving requires `args_sha256`**: the SHA-256 of the pending call's `args_canonical`, which the app computes itself from the canonical JSON it received (after checking that JSON is the arguments it displays). Missing, malformed or not matching the pending call's arguments is 400 and nothing is approved (denying needs no hash). Deciding twice returns `{ok: false, status}` |

Every decision (including automatic ones) is written to the audit log in the SQLite store:
`status` is `auto_run`, `auto_deny`, `remembered` (automatic), or `pending`, `approved`,
`denied`, `expired`, `cancelled`, `failed`, and `applied` for settings changes; `decided_by` is
`policy`, `session`, `settings`, `user`, `terminal`, `timeout` or `restart`. `kind` is `tool`,
`model_pull`, `model_import`, `model_delete`, `conversation_delete`, `data_delete`, `science` or
`settings`. Every row has `args_sha256`.
`remember: "session"` allows the same tool with **exactly the same arguments** (the same
`args_sha256`) **in the same conversation** for the rest of the backend session only; nothing is
remembered across restarts. Calls that hit a row whose `rememberable` is false (shell commands,
installs, network use, deletes, model downloads/imports/deletes, paths outside the workspace,
sending data off the machine) are never remembered (`note` says so) and are asked every time.

## Artifacts

| Method | Path | Returns |
|---|---|---|
| GET | `/api/jev/artifacts/<id>` | the file bytes with its content type (`Content-Disposition: inline`), streamed in 256 KB chunks with `Content-Length`; only files inside a workspace (the stored path is re-resolved, symlinks included, on every request: 403 otherwise) |
| GET | `/api/jev/artifacts` | `?conversation_id=` → `{artifacts: [artifact]}` |

`url` fields are `/api/jev/artifacts/<id>`; the front end fetches them with the token header and
shows them via `blob:` URLs (the CSP allows `img-src`/`media-src` `blob:`). Responses carry
`X-Content-Type-Options: nosniff` and `Content-Security-Policy: default-src 'none'; sandbox`.
Model-written HTML, JavaScript and XML are served as `text/plain; charset=utf-8` (the artifact
record keeps the original `mime`): never render them as documents or iframes in the app's origin.
SVG keeps `image/svg+xml`; show it only through `<img>`.

## Science (no model needed)

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/jev/science/blackhole` | `{spin, inclination_deg, distance_M, disk: {...}, width, height, frames?, seconds?}` | `{job_id, estimate_s}`; result artifact(s) via `/api/jev/jobs/<job_id>/events`. A render the science module estimates at over 10 minutes needs approval first: `{approval_id, status: "pending", reason, args, estimate_s}`, and approving it returns the `job_id` |
| POST | `/api/jev/science/simulate` | `{kind: "nbody" \| "pendulum" \| "wave" \| ..., params}` | `{job_id}` (or an approval as above) |
| GET | `/api/jev/jobs/<job_id>/events` | `?since=` | `{events, next, status}` (same event shapes as turns; science jobs end with `done {text, artifacts}`, model jobs with `done {text, sha256?}`) |
| POST | `/api/jev/jobs/<job_id>/cancel` | | `{ok}` |
| GET | `/api/jev/jobs` | | `{jobs: [{id, kind, status, created, finished, meta, events}]}` (last 100) |

Renders over 7680 × 4320 pixels or 20 000 frames are refused (400). One heavy job (render,
pull, import) runs at a time; others wait in `queued`.
