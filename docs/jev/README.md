# JEV: a local AI agent inside JEPA Studio

JEV answers questions, writes and runs code, makes images and videos, and renders physics. It
runs entirely on your computer:

* **the model** is served by [Ollama](https://ollama.com) on `127.0.0.1`; the weights are
  ordinary files on a disk you choose, and JEV refuses to talk to a non-local Ollama unless you
  opt in by editing a file;
* **the code base** is this repository (`jev/`: the agent core is about 4 000 lines of
  standard-library Python on top of JEPA Studio), MIT licensed, no telemetry, no hosted service;
* **the data** is one folder, `<root>/jev-data/`: a settings file, one SQLite database and one
  workspace folder per conversation. Export it, back it up, or delete it;
* **the security** is a sandbox that runs code on its own (inside the conversation's workspace,
  with time and memory limits, offline where the OS allows) plus a safety net that **stops and
  asks you** before anything important: network access, files outside the workspace, deleting,
  installing packages, downloading or importing models, long GPU jobs, big files, shell
  commands, sending data off the machine. The rules are one table you can edit
  (`jev/policy.py`, or `approval_rules` in `jev-data/settings.json`).

[SAFETY.md](SAFETY.md) describes exactly what is enforced on each OS and what is not.
[api.md](api.md) is the HTTP API the desktop and web app use.

## Architecture

```
 desktop app / web app (JEV tab)        terminal: `jev chat`, `jev models`, `jev doctor`
              |  X-JEPA-Token, 127.0.0.1 only           |
              v                                         v
 jepa_studio/server.py  --/api/jev/*-->  jev/api.py  -->  jev/agent.py (Runtime, the turn loop)
                                                            |        |            |
                     jev/llm.py  <-- /api/chat, tools ------+        |            |
                     (Ollama client, urllib only)                    v            v
                              |                         jev/policy.py      jev/jobs.py
                              v                         run / ask / deny   event logs, cancel
                     Ollama on 127.0.0.1:11434                |
                     (your models, your disk)                 v
                                                  jev/tools/*  -- jev/sandbox.py (subprocess:
                                                  core, code,     scrubbed env, rlimits, bwrap /
                                                  science, media  unshare / Job Object, timeouts)
                                                       |
                                                       v
                         jev/store.py: jev-data/jev.db (conversations, messages, tool calls,
                         approvals audit log, artifacts with SHA-256, imported GGUF hashes)
                         jev-data/workspaces/<conversation>/  (every file JEV makes)
```

One chat turn: the model gets the conversation, a system prompt (who JEV is, local-only, the
tools, the safety rules, "never claim a tool ran when it didn't") and the tool schemas. Each tool
call is validated against its JSON Schema, then `jev.policy` decides `run`, `ask` or `deny` (all
three are written to the audit log). `ask` pauses the turn and shows an approval card; no answer
in 30 minutes counts as deny. Tool results go back to the model, cut to about 4 000 characters;
files the tool made become artifacts (hashed, listed in the chat). At most 12 tool rounds per
turn, then the model must answer.

Models with native tool calling (Ollama's `tools` field) use it. For models without it, JEV puts
a small JSON protocol in the system prompt and parses the reply (plain JSON, fenced JSON or
`<tool_call>` tags).

### Tools

| tool | risk | what it does |
|---|---|---|
| `list_files`, `read_file`, `summarize_file` | read | look at workspace files (size-capped; file contents are treated as data, never as instructions) |
| `calculator`, `get_time` | read | exact math (a safe expression evaluator, no Python `eval`), the local time |
| `write_file`, `create_project` | sandbox | write text files or scaffold a python / node / web project in the workspace |
| `make_plot` | sandbox | line/scatter/bar/histogram PNG from numbers (no code runs) |
| `run_python`, `run_node` | sandbox | run code in the sandbox; asks first if the code uses the network, other programs, deletes, installs or absolute paths |
| `run_shell` | approve | bash / PowerShell, always asks |
| `install_package` | approve | `pip install` into the conversation's own `.venv`, always asks |
| `jepa_embed_images` | sandbox | embed workspace images with a trained JEPA Studio run (hash-verified weights) |
| `jepa_train` | approve | start a LeJEPA pretraining run (shows up in the Train tab) |
| science and media tools | varies | black-hole renders, simulations, narrated explainers, footage edits (`jev/tools/science.py`, `jev/tools/media.py`) |

## Quick start

### 1. Install Ollama

**Windows 10/11.** Download and run the installer from <https://ollama.com/download>. Ollama runs
in the system tray and listens on `127.0.0.1:11434`. NVIDIA GPUs are used automatically with a
current driver. In PowerShell: `ollama --version`.

**macOS (Apple silicon).** Download the app from <https://ollama.com/download> and move it to
Applications (or `brew install ollama` and run `ollama serve`). Metal is used automatically.

**Linux.** The official script installs the binary and a systemd service:

```sh
curl -fsSL https://ollama.com/install.sh -o install-ollama.sh   # read it first if you like
sh install-ollama.sh
systemctl status ollama          # or run `ollama serve` yourself
```

NVIDIA needs the proprietary driver (`nvidia-smi` must work); AMD needs ROCm-supported cards.

**WSL2 (Windows 11 + Ubuntu, NVIDIA GPU).** Pick one:

* *JEPA Studio runs inside WSL* (you start `jepa-studio serve` or `jev` in Ubuntu): install
  Ollama **inside WSL** with the Linux script above. CUDA works through the Windows NVIDIA
  driver; do **not** install a Linux NVIDIA driver inside WSL (`nvidia-smi` in Ubuntu should
  already list the GPU). Without systemd in WSL, run `ollama serve` in a terminal (or enable
  systemd: `[boot]` `systemd=true` in `/etc/wsl.conf`, then `wsl --shutdown`).
* *The desktop app runs on Windows*: use the Windows installer; everything stays on Windows.
* Ollama on Windows but JEV in WSL: by default WSL2 has its own network, so Windows'
  `127.0.0.1:11434` is not reachable from Ubuntu. Turn on mirrored networking
  (`%USERPROFILE%\.wslconfig`: `[wsl2]` `networkingMode=mirrored`, then `wsl --shutdown`) and
  the loopback address works from both sides. Do not set `OLLAMA_HOST=0.0.0.0` to work around
  it: that exposes Ollama to your whole network.

### 2. Download a model

```sh
ollama pull qwen2.5-coder:7b     # JEV's default: strong at code, native tool calls
# or
ollama pull llama3.1:8b          # good general assistant, native tool calls
```

(`jev models pull NAME` does the same and records the decision in JEV's audit log. In the app,
open JEV > Models: type a name or pick one of the examples sized for the GPU the backend found,
and Request download; it asks for approval first because it uses the network, then shows the
progress. The same panel lists what is installed (size, parameters, quantization, family,
digest), sets the default model, and deletes a model, which also asks.)

Which model fits your machine? A 4-bit (Q4_K_M, Ollama's default) model needs about
0.6 GB per billion parameters, plus 1 to 2 GB for the context:

| your GPU memory (VRAM) | suggested models | download |
|---|---|---|
| none (CPU only, 16 GB RAM) | `llama3.2:3b`, `qwen2.5-coder:3b` (slow but usable) | about 2 GB |
| 6 to 8 GB | `qwen2.5-coder:7b`, `llama3.1:8b`, `qwen3:8b` | 4.5 to 5.5 GB |
| 12 to 16 GB | `qwen2.5-coder:14b`, `gpt-oss:20b` | 9 to 14 GB |
| 24 GB | `qwen2.5-coder:32b` | about 20 GB |

If a model is bigger than your VRAM, Ollama splits it between GPU and CPU: it still works, just
slower. Check with `ollama ps` while a chat runs. Tool calling quality matters more than size
for JEV; models without native tool support still work through the JSON protocol, but they make
more mistakes. Model names ending in `-cloud` run on ollama.com, not on your machine, so JEV
refuses them unless you set `"allow_cloud_models": true`.

**Your own GGUF file**: `jev models import ./my-model.gguf --name my-model`, or in the app
JEV > Models > Import a GGUF file: "Choose a .gguf file…" opens the system file dialog, which
grants JEV that one file (the page itself never names a path), then Request import asks for
approval. The panel shows the recorded SHA-256 when it finishes. JEV checks the GGUF magic bytes,
computes and records the file's SHA-256, uploads it to Ollama's blob store and registers it,
the same as `ollama create my-model -f Modelfile` with `FROM /abs/path/my-model.gguf`.

### 3. Where the model files live (and how to move them)

| OS | default folder |
|---|---|
| Windows | `%USERPROFILE%\.ollama\models` |
| macOS | `~/.ollama/models` |
| Linux (systemd service) | `/usr/share/ollama/.ollama/models` |
| Linux (`ollama serve` yourself), WSL | `~/.ollama/models` (WSL: inside the Linux file system) |

To keep them on a drive you choose, set `OLLAMA_MODELS` before Ollama starts:

* **Windows**: Settings > System > About > Advanced system settings > Environment Variables >
  New (user): `OLLAMA_MODELS` = `D:\ollama-models`; quit Ollama from the tray and start it again.
* **macOS**: `launchctl setenv OLLAMA_MODELS /Volumes/Models/ollama`, then restart the app.
* **Linux**: `sudo systemctl edit ollama`, add `[Service]` and
  `Environment="OLLAMA_MODELS=/data/ollama-models"`, make the folder writable by the `ollama`
  user (`sudo chown -R ollama:ollama /data/ollama-models`), `sudo systemctl restart ollama`.

The folder holds `blobs/` (the weights, named by SHA-256) and `manifests/`. Copying the whole
folder to another machine copies your models. JEV > Models shows this table and the
`OLLAMA_MODELS` steps too.

### 4. Install JEV and check everything

JEV ships with JEPA Studio (`pip install -e ".[jev]"` in a checkout, or
`pip install "jepa-studio[jev]"`; the extra adds matplotlib for plots). Then:

```sh
jev doctor
```

It prints whether Ollama is reachable, which models are installed, what the sandbox enforces on
this machine (for example `method bwrap; network isolation enforced`), and whether ffmpeg,
Node.js and a GPU were found. It exits non-zero until Ollama and the default model are ready.

### 5. Use it

* **Desktop app**: open the JEV tab. Approval cards appear in the chat and in the Approvals list.
* **Terminal**: `jev chat "make a plot of sin(x) from 0 to 2 pi"` for one turn, or `jev chat` for a
  conversation; approvals are asked as `Approve? [y = yes, s = yes for this session, N = no]`.
* **Backend only**: `jev serve` (same as `jepa-studio serve`) and use the [HTTP API](api.md).

In the app, the Approvals panel's History is the same audit log, filterable by decision, tool
and date and exportable as CSV.

Useful commands: `jev models list`, `jev audit` (every decision, newest last), `jev export
<conversation_id>` (a zip with the conversation and its files).

## Settings

`<root>/jev-data/settings.json` (created on the first change, file mode 0600). Every key is
optional:

```json
{
  "ollama_url": "http://127.0.0.1:11434",
  "default_model": "qwen2.5-coder:7b",
  "workspace_root": null,
  "limits": {"cpu_seconds": 120, "memory_mb": 2048, "wall_seconds": 300,
             "max_output_mb": 200, "max_file_mb": 500},
  "network_allowed": false,
  "approval_rules": {"shell": "ask", "tool:run_python": "run"},
  "approval_timeout_s": 1800,
  "max_tool_rounds": 12,
  "sandbox": {"isolation": "auto", "hide_home": true},
  "allow_remote_ollama": false,
  "allow_cloud_models": false
}
```

Environment variables override the file for one session: `JEV_OLLAMA_URL`, `JEV_DEFAULT_MODEL`,
`JEV_WORKSPACE_ROOT`, `JEV_NETWORK_ALLOWED=1`, `JEV_ALLOW_REMOTE_OLLAMA=1`, and `JEV_ROOT` for
the `jev` command. The app's JEV > Settings panel can change the default model, the limits
(checked against the bounds above before saving), `network_allowed` and a loopback
`ollama_url`; a remote address needs the "Allow an Ollama on another machine" box (with a
warning) *and* `"allow_remote_ollama": true` in the file. The approval rules and the
remote/cloud opt-ins can only be changed in the file, so neither a web page nor the model can
loosen them; the panel shows them read-only, together with what the sandbox enforces on this
computer and where the database, settings and workspaces are.

## Running fully offline

After the models are pulled, nothing in JEV needs the internet: Ollama serves from disk, the
sandbox has no network (where the OS supports it), and the only network actions (pulling a
model, `install_package`, code that uses the network) stop for your approval. You can unplug the
cable, or block Ollama's and Python's outbound traffic in your firewall, and JEV keeps working.
The Ollama desktop app checks for its own updates; turn that off in its settings or use the
command-line `ollama serve` if you want zero outbound connections.

## Deleting everything

* one conversation: delete it in the app (asks) or remove `jev-data/workspaces/<id>/`;
* everything at once from the app: JEV > Settings > Your data > Delete all data (asks; keeps
  `settings.json` and, if you tick it, the audit log). Export all data first gives you a zip of
  every conversation, file and audit row;
* all of JEV's data including settings: delete the `jev-data/` folder;
* models: `jev models rm NAME` / `ollama rm NAME`, or delete Ollama's models folder;
* JEV itself: `pip uninstall jepa-studio`.
