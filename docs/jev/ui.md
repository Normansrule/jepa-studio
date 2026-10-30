# JEV and Science in the app (front end)

The JEV tab and the Science tab are part of the one static front end in `site/`, shared by the
website and the desktop app. This page describes what they show, how they talk to the local
backend ([api.md](api.md)), and the accessibility and security rules they follow. For what the
agent may do on its own, see [SAFETY.md](SAFETY.md).

| Tab | Web (GitHub Pages) | Desktop app |
|---|---|---|
| **JEV** | A panel that says JEV needs the desktop app, links to the release downloads, lists what JEV can do and the three risk levels, and sends you to Science | Chat with streaming answers, tool cards, approvals and the audit log, files, status strip, Models, Settings and Doctor |
| **Science** | Live black hole and simulations (WebGL) | The same, plus "Render on this computer": an MP4 rendered by the backend, with progress |

The tab bar puts JEV and Science first, then a divider, then the JEPA workflow (Data, Train,
Inspect, Evaluate, World model, Explain, Export). With no `#tab` in the address and no remembered
tab, the desktop app opens on JEV and the website on Data (the JEPA demo that runs in a browser).
The JEPA progress stepper is hidden on JEV and Science.

## Files

| File | What it does |
|---|---|
| `site/js/tabs/jev.js` | the tab: tiers, status, models, conversations, composer, attachments, turns |
| `site/js/jev/client.js` | `/api/jev/*` calls through `DesktopBackend.req` (token header, 127.0.0.1 only), artifact `blob:` URLs, `follow()` for event logs |
| `site/js/jev/turn.js` | one turn in the log: streamed text and tool cards |
| `site/js/jev/approvals.js` | the approval dialog, the "N waiting" badges, the Approvals panel (pending + audit history) |
| `site/js/jev/gallery.js` | the per-conversation Files list and the image lightbox |
| `site/js/jev/doctor.js` | the status strip and the Doctor view (checks and per-OS commands) |
| `site/js/jev/models.js` | the Models panel: installed models, default, delete, download with examples sized by GPU memory, GGUF import |
| `site/js/jev/settings.js` | the Settings panel: settings form with validation, sandbox facts, data paths, export / delete all, approval rules |
| `site/js/jev/audit.js` | audit-log filters (decision, tool, dates) and the CSV export used by the Approvals panel |
| `site/js/jev/highlight.js` | the small in-house code highlighter and answer formatting (no HTML parsing) |
| `site/js/jev/icons.js` | inline SVG icons and the risk badge |
| `site/js/tabs/science.js` | the Science tab; loads `site/js/science/blackhole.js` and `sims.js` on demand |

## Talking to the backend

Every call goes through `DesktopBackend` (`site/js/backend.js`), which exists only when the shell
injected `window.__JEPA_DESKTOP__ = {port, token}`. The website never constructs it, so a visitor's
browser never tries to reach a local port (and the site's CSP, `connect-src 'self'`, would block
it anyway).

Routes the UI uses (all in [api.md](api.md)):

| Feature | Routes |
|---|---|
| Status strip, Doctor | `GET /status` every 15 s while the page is visible; `GET /models` |
| Chat | `POST /chat`, then `GET /turns/<id>/events?since=n` every 200 ms (900 ms while an approval is pending) until `done`, `error` or `cancelled`; `POST /turns/<id>/cancel` for Stop |
| Conversations | `GET /conversations`, `GET /conversations/<id>`, `POST /conversations` (an empty one, before the first upload), `DELETE /conversations/<id>` (answered with an approval), `GET /conversations/<id>/export` |
| Attachments | `POST /conversations/<id>/files` with `{name, mime, data_base64}`; files over 1.5 MB are refused in the page, because the base64 body must stay under the 2 MB cap. The returned `path` goes into `attachments` of the next `POST /chat` |
| Approvals | `GET /approvals?status=pending` every 3 s while visible, `GET /approvals?limit=5000` for the history (audit log), `POST /approvals/<id>` with `{decision, remember, args_sha256}` (the hash of the exact arguments the card showed; required to approve) |
| Models | `GET /models` (installed + imported SHA-256s), `POST /settings {default_model}`, `POST /models/pull`, `POST /models/import {gguf_path, name}` (the path comes from the shell's `pick_gguf_file`), `DELETE /models/<name>`; the last three answer with an approval, and after "Approve" the `job_id` in the decision's response is followed with `GET /jobs/<id>/events` |
| Settings | `GET /settings` (settings, `rules`, `paths`), `POST /settings` with only the changed keys, `GET /export` (a zip, saved with a normal download), `POST /data/delete {keep_audit_log}` (answered with an approval); sandbox facts come from `GET /status` |
| Files | `GET /artifacts/<id>` with the token header, shown through a `blob:` URL |
| Science renders | `POST /science/blackhole`, `POST /science/simulate`, then `GET /jobs/<id>/events`; a long render answers with an approval first |

The approval decision's response carries `job_id` when approving starts a job (model downloads,
imports, deletes, long renders), so the page can follow its progress. If the approval table is
set to `run` for that row, the request itself answers `{status: "auto_run", job_id}` and the page
follows the job directly.

## Streaming without flooding screen readers

The conversation is a `role="log"` region with `aria-live="polite"` and
`aria-relevant="additions"`. While a turn runs:

* the log has `aria-busy="true"`, and the text that streams in token by token is
  `aria-hidden="true"`;
* one short message per tool step ("Running render_black_hole.", "run_shell needs your approval.")
  goes to a separate, visually hidden `role="status"` line. Nothing is announced per token.

When the turn ends, the streamed text is replaced by the formatted answer (a new node), the log's
`aria-busy` turns false, and one sentence is appended for screen readers: "JEV answered after
2 tool calls: …" (the first 280 characters). The log keeps itself scrolled to the newest content
while you are at the bottom, and stays put when you scroll up to read.

## Tool cards

Each tool call is a card with:

* the tool name and a **risk badge**: an icon, a word and a colour, never colour alone
  (`Read only`: an eye, `Sandbox`: a box, `Asks first`: a shield). Screen readers hear
  "Sandbox risk level"; the tooltip and the dialog give the full explanation;
* its state, with an icon and a word: Running, Waiting for your approval, Done, Failed,
  Not allowed (refused by the policy), You denied this, Cancelled. The card's left rule repeats it
  (warning, good or bad status colour);
* **Arguments**, collapsed by default: a `code` argument is shown as highlighted code with line
  numbers, everything else as pretty-printed JSON;
* a progress bar (`role="progressbar"` with `aria-valuenow`) and the tool's message while it runs;
  a progress event without a fraction shows only its message;
* the result text (first 6 000 characters) and the files the tool made, also added to the
  conversation's Files list.

## The approval dialog

When a request arrives (from a turn's `approval_request` event or the pending list) and you are on
the JEV or Science tab, a modal `<dialog>` opens; on another tab, a notice with a Review button
appears and the JEV tab shows the count. The dialog shows:

* a question naming the action ("Allow JEV to run a command on your computer?", "… to download a
  model?", "… to delete something?", "… to use the internet?", …) and one sentence on what will
  happen;
* the exact command, path, address or model in a monospace block, so you approve what will
  actually run; all arguments, never cut, are one click away ("All arguments", scrollable), with
  the SHA-256 of those exact arguments. Approving sends that hash back
  (`POST /approvals/<id> {decision, remember, args_sha256}`) and the backend refuses an approval
  whose hash does not match the pending call;
* why it asks (the backend's `reason`) and the tool with its risk badge;
* **Deny**, **Approve for this session**, **Approve once**. "For this session" covers only these
  exact arguments of this tool in this conversation; shell commands, installs, network use,
  deletes, models and paths outside the workspace are asked every time. Focus starts on Deny. **Esc** is Deny
  (the dialog's `cancel` event). Tab moves between the buttons; focus returns to where it was when
  the dialog closes. With several requests waiting, it says "1 of 3 requests" and shows the next
  after each answer.

If the answer cannot be saved, the dialog stays open and says so. If the backend reports that the
request was already decided, or that the approved action failed to start, a notice says that
instead of "Approved".

"N waiting" appears as a count on the JEV tab (its accessible name becomes "JEV, 2 approvals
waiting") and on the Approvals button, until every request is answered. The **Approvals** panel
lists pending requests (each with Review) and the audit history: time, tool, the exact command or
path, and the decision with an icon and a word (Approved, Denied, Ran automatically, Expired, …).

## Models panel

**Models** (in the status strip, or **Manage models** next to the model menu) opens a sheet with:

* **Installed**: one row per Ollama model with its size, parameter count, quantization, family
  and the first 12 characters of Ollama's digest (the full digest is the tooltip); a GGUF you
  imported also shows the SHA-256 JEV recorded for the file. **Make default** saves
  `default_model` and selects it in the composer; **Delete…** asks (the dialog names the model).
* **Download a model**: a name field and **Request download** (asks, then a progress bar).
  Below it, *examples* that fit this computer: the tier for the largest GPU memory in
  `status.gpu.devices` (CPU only or under 6 GB, 6 to 11, 12 to 23, 24 GB and more; an Apple GPU
  counts as the 6 to 11 tier) and the next smaller tier, each with its approximate download
  size, whether it is already installed, and a note on models without native tool calling.
  Clicking one only fills the name; nothing downloads without the approval.
* **Import a GGUF file**: **Choose a .gguf file…** calls the shell's `pick_gguf_file`. The page
  never supplies a path: the native dialog returns one, and the shell has already granted the
  backend that single file after checking its name and `GGUF` header. The panel proposes a
  name from the file name (lowercase, other characters as `-`), validates it like the backend
  (`MODEL_NAME`), and **Request import** asks. When the job finishes the recorded SHA-256 is
  shown. Without the shell (no `window.__TAURI__`), the button is disabled and the reason gives
  the `jev models import` command.
* **Where the model files are**: Ollama's default folder per OS and how to move it with
  `OLLAMA_MODELS`.

## Settings panel

**Settings** opens a form for the keys the API may change, each validated in the page with
the backend's bounds before **Save settings** is enabled (errors are next to the field, linked
with `aria-describedby`, and the field gets `aria-invalid`); only changed keys are sent:

* default model (the installed models, plus the current value if it is not installed);
* Ollama address: must be loopback (`127.0.0.0/8`, `::1`, `localhost`). Another address needs
  **Allow an Ollama on another machine**, which shows a warning that prompts and files would
  leave the computer, and is still refused while `allow_remote_ollama` is off in the file (the
  message names the settings file): the page cannot turn that switch on;
* the five sandbox limits (`cpu_seconds`, `wall_seconds`, `memory_mb`, `max_output_mb`,
  `max_file_mb`), whole numbers within `jev/settings.py`'s bounds;
* `network_allowed`, explained: it lifts the sandbox's network block and skips that one
  question, while downloads, installs, shell commands, sending data and deleting still ask.

Below the form, read only: what the sandbox enforces here (network and file isolation, method,
resource limits, how runaway code is stopped, from `status.sandbox`); **Your data**, the paths of
the data folder, database, settings file, workspaces and exports with Copy buttons (the shell
has no command to open folders, on purpose), **Export all data (.zip)** and **Delete all data…**
(asks; "Keep the audit log" is a checkbox); the approval table (rule, what it covers, the
current action with an icon and a word, the default when you changed it, Locked rows); and the
settings only the file can change.

## Audit log

The Approvals panel's **History (audit log)** loads up to 5 000 decisions and filters them in
the page by decision (Approved by you, Denied by you, Ran automatically, Refused by policy,
Waiting, Expired / cancelled / failed), tool and a date range (local days, inclusive), with a
count ("2 of 5 decisions"). **Export CSV** saves the filtered rows with every field (ids, ISO
times, status, decided_by, rules, reason, arguments as JSON); a cell that would start with `=`,
`+`, `-` or `@` gets a leading `'` so a spreadsheet never runs it as a formula. Below 620 px
each row stacks: time and decision, tool, then what it did.

## Files (artifacts)

Images (with a lightbox: Esc closes it and focus returns), videos (`<video controls>`), audio
(`<audio controls>`), code and text (highlighted, line numbers drawn by CSS so copying copies no
numbers), other files as a download. Downloads are `<a download>` links on the same `blob:` URL.
File names and titles are always inserted as text. Object URLs are revoked when you switch
conversation.

Content Security Policy for this: the website and the desktop shell both allow
`img-src 'self' data: blob:` and `media-src 'self' blob:`; `connect-src` is unchanged
(`'self'` on the web; `'self'` plus the one backend port on the desktop, see
[../desktop.md](../desktop.md)).

## Status strip and Doctor

The strip shows, each with an icon and words: Ollama running (and its version) or not reachable;
the chosen model or "No model yet"; whether sandboxed code is offline (`sandbox.network_isolation`
is `"enforced"`) or could reach the network; the GPU from `gpu.devices` (or "CPU only").

Doctor turns the same facts into checks with fixes: install and start Ollama, download a model,
what missing network isolation means. Commands are shown for macOS, Windows or Linux (guessed from
the browser, switchable, remembered) with Copy buttons; links are given as text because the
desktop window does not open external pages. When `/api/jev/status` does not answer, Doctor says
whether the backend lacks the JEV routes (HTTP 404) or did not respond, and uses the shell's own
Ollama probe (`backend_status().ollama`) as a second opinion.

Send stays disabled, with the reason under the box, until the backend answers, Ollama is reachable
and a model is chosen.

## Science tab

The tab loads `site/js/science/blackhole.js` and `sims.js` with a dynamic `import()` only when it is
shown, and destroys the view when you leave the tab or hide the page (a fresh `<canvas>` each
time, since a destroyed WebGL context cannot be reused). If a module is missing or WebGL cannot
start, the stage shows a plain explanation and a Try again button instead.

The UI calls:

```js
const bh = mountBlackHole(canvas, { reducedMotion, spin, distance, inclination, disk, doppler, beaming, quality });
bh.setParams({ spin, distance, inclination, disk, doppler, beaming, quality });  // on every change
bh.snapshot();   // Blob | data URL | canvas; the PNG is saved with a normal download
bh.destroy();

const sim = mountSim(canvas, kind, { reducedMotion, ...params });   // kind: nbody | double_pendulum | wave
sim.setParams(params); sim.reset(); sim.destroy();
```

`distance` is in units of M, `inclination` in degrees from the spin axis, `quality` is `low`,
`medium` or `high`. Simulation parameters: `nbody` `{bodies, speed}`, `double_pendulum`
`{theta1, theta2, massRatio, speed}` (degrees), `wave` `{speed, damping, freq}`. If the module
says it cannot draw spin (`capabilities.spin === false` on the handle or the module), a note
under the slider says so and that rendering on the computer uses it.

"Render high-quality video on this computer" (desktop) posts the same settings in the backend's
units (`spin`, `inclination_deg`, `distance_M`, `disk: {enabled, doppler, beaming}`, `width`,
`height`, `seconds`, `frames`, `quality: "final"`; simulations as `{kind, params}` with
`jev/science/sims.py`'s names), follows the job, and lists the MP4 in the tab's own Files list.

## Design notes

New tokens (in `site/css/app.css`, light / dark, contrast against `--surface` and `--surface-2`):

| Token | Light | Dark | Contrast (light: surface / surface-2; dark: surface / surface-2) | Use |
|---|---|---|---|---|
| `--risk-read` | `#4a5468` | `#b4bccb` | 7.6 / 6.7; 8.1 / 7.1 | Read only badge |
| `--risk-sandbox` | `#1f6f8b` | `#62b6d6` | 5.7 / 5.0; 6.8 / 5.9 | Sandbox badge |
| `--risk-approve` | `#985c00` | `#f2b13a` | 5.4 / 4.8; 8.2 / 7.2 | Asks first badge, approval dialog rule, waiting counts |
| `--code-kw`, `--code-str`, `--code-num`, `--code-fn` | `#7a3eb1`, `#1d7a4f`, `#985c00`, `#1f5fbf` | `#c39af0`, `#7fd3a4`, `#f2b13a`, `#8ab8ff` | all ≥ 4.6 | syntax colours |
| `--space` | `#05070c` | same | | the black-hole stage and the lightbox background |

The JEPA data-flow colours are not reused for JEV. The transcript is a document, not chat
bubbles: role names in small bold type, your message on `--surface-2`, JEV's answer on the panel.
Hit targets are 44 px on phones; below 900 px the conversation list becomes a drawer (a modal
`<dialog>`), and below 1240 px the Files list moves under the chat.

## Tests

`tests/test_jev_ui.py` drives both tabs in Chromium with an in-test fake of the JEV routes: a
scripted turn (tokens, a sandbox tool with an image, an `approval_request` for `run_shell`,
approval by keyboard, an MP4), Esc to deny a second request, the history, Doctor with Ollama down,
the phone drawer, 44 px targets and no horizontal scroll at 375 px, attachments, a saved
conversation, the web tier panel, the Science tab live or in fallback, and a desktop render job.
The Models, Settings and audit-log tests cover listing, make-default, delete, pull and GGUF import
(with `__TAURI__.core.invoke('pick_gguf_file')` stubbed) through the approval dialog; settings
validation, the remote-address warning, saving only changed keys, export and delete-all; the
audit filters and CSV; and the three panels at 375 px in light and dark (no horizontal scroll,
44 px targets).
`tests/test_desktop_tier.py` also opens the JEV tab against the real backend.
