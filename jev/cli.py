"""`jev` command line: doctor, models, chat, audit, export, serve.

    jev doctor                         is Ollama running? which models? what does the sandbox enforce?
    jev models list
    jev models pull llama3.1:8b        asks y/N (downloads use the network)
    jev models import ./my.gguf --name my-model     asks y/N; records the file's SHA-256
    jev models rm llama3.1:8b          asks y/N
    jev chat "plot sin(x) from 0 to 2 pi"           one turn, approvals asked in the terminal
    jev chat                           interactive; empty line or Ctrl-D quits, Ctrl-C cancels a turn
    jev audit                          the approvals audit log (every decision, automatic ones too)
    jev export <conversation_id>       zip with the conversation JSON and its workspace files
    jev serve                          same as `jepa-studio serve` (desktop/web backend)

`--root DIR` (or JEV_ROOT) picks the backend root; JEV keeps its data in DIR/jev-data/.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _rt(args):
    from .agent import Runtime
    return Runtime(args.root)


def _ask(prompt: str, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _audit(rt, tool: str, args: dict, reason: str, rules: list, approved: bool) -> None:
    rt.store.add_approval({"id": "a-" + secrets.token_hex(6), "kind": tool, "tool": tool, "args": args,
                           "reason": reason, "risk": "approve", "rules": rules,
                           "status": "approved" if approved else "denied", "decided_by": "terminal",
                           "decided": time.time()})


# ------------------------------------------------------------------------------------------ doctor
def _gpu_line() -> str:
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            if out:
                return "NVIDIA " + "; ".join(out.splitlines())
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple GPU (MPS)"
    except Exception:  # noqa: BLE001
        pass
    return "none found (models run on the CPU: slower, pick a 3B-8B model)"


def cmd_doctor(args) -> int:
    from .llm import Ollama, OllamaError
    from .sandbox import Sandbox
    from .settings import Settings, SettingsError

    ok = True
    print(f"JEV doctor   root {Path(args.root).resolve()}")
    try:
        st = Settings(args.root)
        s = st.resolved()
    except SettingsError as e:
        print(f"  settings    PROBLEM: {e}")
        return 2
    print(f"  settings    {st.path} ({'yours' if st.path.exists() else 'defaults; file not created yet'})")
    llm = Ollama.from_settings(s)
    models = []
    try:
        print(f"  ollama      {s['ollama_url']}  OK, version {llm.version()}")
        models = llm.tags()
    except OllamaError as e:
        ok = False
        print(f"  ollama      NOT REACHABLE at {s['ollama_url']}\n              {e}")
        print("              Install: https://ollama.com/download (Windows/macOS app) or, on Linux/WSL,\n"
              "              curl -fsSL https://ollama.com/install.sh | sh   then: ollama serve")
    if models:
        names = [m["name"] for m in models]
        print(f"  models      {len(names)}: " + ", ".join(names[:12]) + (" ..." if len(names) > 12 else ""))
        if s["default_model"] not in names and f"{s['default_model']}:latest" not in names:
            ok = False
            print(f"              default model {s['default_model']} is not installed: jev models pull "
                  f"{s['default_model']}")
    elif ok:
        print(f"  models      none yet: jev models pull {s['default_model']}")
        ok = False
    caps = Sandbox(s).capabilities()
    print(f"  sandbox     method {caps['method']}; network isolation {caps['network_isolation']}")
    print(f"              files: {caps['filesystem']}")
    print(f"              limits: {caps['rlimits']}; {json.dumps(caps['limits'])}")
    if caps.get("probe"):
        print(f"              probe: {caps['probe']}")
    for exe, why in (("ffmpeg", "video tools (narrated explainers, footage edits, black-hole videos)"),
                     ("node", "run_node"), ("bash" if os.name != "nt" else "powershell", "run_shell")):
        w = shutil.which(exe)
        print(f"  {exe:<11} {w or 'missing'}{'' if w else f'  (needed for {why})'}")
    try:
        import matplotlib
        print(f"  matplotlib  {matplotlib.__version__}")
    except ImportError:
        print("  matplotlib  missing (make_plot): pip install \"jepa-studio[jev]\"")
    print(f"  gpu         {_gpu_line()}")
    print("  result      " + ("ready" if ok else "not ready: fix the lines above"))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ models
def cmd_models(args) -> int:
    from .llm import OllamaError, sha256_file

    rt = _rt(args)
    llm = rt.llm()
    try:
        if args.action in (None, "list"):
            for m in llm.tags():
                size = (m["size_bytes"] or 0) / 2**30
                print(f"{m['name']:<40} {size:6.1f} GB  {m.get('parameter_size') or '':>6}  "
                      f"{m.get('quantization') or ''}{'  (cloud: runs on ollama.com)' if m['remote'] else ''}")
            for m in rt.store.list_models():
                print(f"imported: {m['name']}  sha256 {m['sha256']}  from {m['path']}")
            return 0
        if args.action == "pull":
            reason = f"Download {args.name} from the Ollama registry (network)"
            yes = _ask(reason + "?", args.yes)
            _audit(rt, "model_pull", {"name": args.name}, reason, ["model_download", "network"], yes)
            if not yes:
                print("not downloaded")
                return 1
            last = [""]

            def prog(d):
                line = d["status"] + (f" {d['fraction'] * 100:5.1f}%" if d.get("fraction") is not None else "")
                if line != last[0]:
                    print("\r" + line.ljust(70), end="", flush=True)
                    last[0] = line
            llm.pull(args.name, prog)
            print(f"\npulled {args.name}")
            return 0
        if args.action == "import":
            p = Path(args.name).expanduser().resolve()
            if not args.as_name:
                print("give the model a name: --name my-model")
                return 2
            if p.suffix.lower() != ".gguf" or not p.is_file():
                print(f"{p} is not a .gguf file")
                return 2
            reason = f"Import {p} ({p.stat().st_size / 2**30:.2f} GB) into Ollama as {args.as_name}"
            yes = _ask(reason + "?", args.yes)
            _audit(rt, "model_import", {"name": args.as_name, "gguf_path": str(p)}, reason, ["model_import"], yes)
            if not yes:
                return 1
            print("computing SHA-256 ...")
            digest = sha256_file(p)
            llm.create_from_gguf(args.as_name, p, lambda d: print("  " + d["status"]), sha256=digest)
            rt.store.record_model(args.as_name, str(p), digest, p.stat().st_size)
            print(f"imported {args.as_name}  sha256 {digest}")
            return 0
        if args.action == "rm":
            reason = f"Delete {args.name} from Ollama (removes its files)"
            yes = _ask(reason + "?", args.yes)
            _audit(rt, "model_delete", {"name": args.name}, reason, ["model_delete", "delete"], yes)
            if not yes:
                return 1
            llm.delete(args.name)
            print(f"deleted {args.name}")
            return 0
    except OllamaError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 2


# ------------------------------------------------------------------------------------------ chat
def run_chat_turn(rt, message: str, conversation_id: str | None, model: str | None, assume: str | None = None,
                  out=None) -> tuple[str, str]:
    """Runs one turn, printing tokens and asking approvals on the terminal. Returns (conversation_id, status)."""
    out = out or sys.stdout
    r = rt.start_turn(message, conversation_id, model)
    job = rt.jobs.get(r["turn_id"])
    n = 0
    try:
        while True:
            page = job.since(n)
            for e in page["events"]:
                ev = e["event"]
                if ev == "token":
                    out.write(e["text"])
                elif ev == "tool_call":
                    out.write(f"\n  [tool] {e['name']} -> {e['decision']}" + (f" ({e['reason']})" if e["decision"] !=
                                                                             "run" else "") + "\n")
                elif ev == "approval_request":
                    out.write(f"\n  APPROVAL NEEDED for {e['name']}: {e['reason']}\n")
                    shown = json.dumps(e["args"], indent=2)
                    shown = shown if len(shown) < 3000 else shown[:3000] + "\n..."
                    out.write("    " + shown.replace("\n", "\n    ") + "\n")
                    out.flush()
                    if assume:
                        ans = assume
                    else:
                        try:
                            ans = input("  Approve? [y = yes, s = yes for this session, N = no] ").strip().lower()
                        except EOFError:
                            ans = "n"
                    rt.broker.decide(e["approval_id"], "approve" if ans in ("y", "yes", "s") else "deny",
                                     "session" if ans == "s" else None, by="terminal")
                elif ev == "tool_result":
                    first = (e["content"] or "").strip().splitlines()[:6]
                    out.write("  " + ("ok" if e["ok"] else "failed") + ": " + "\n    ".join(first) + "\n")
                elif ev == "artifact":
                    out.write(f"  file: {e['path']} ({e['kind']}, {e['bytes']} bytes)\n")
                elif ev == "error":
                    out.write(f"\n  error: {e['message']}\n")
                elif ev == "cancelled":
                    out.write("\n  (cancelled)\n")
                elif ev == "done":
                    out.write("\n")
            out.flush()
            n = page["next"]
            if page["status"] in ("done", "error", "cancelled") and job.finished is not None and \
                    n >= len(job.events):
                return r["conversation_id"], page["status"]
            time.sleep(0.05)
    except KeyboardInterrupt:
        rt.cancel(r["turn_id"])
        rt.jobs.wait(r["turn_id"], 10)
        out.write("\n  (cancelled)\n")
        return r["conversation_id"], "cancelled"


def cmd_chat(args) -> int:
    rt = _rt(args)
    cid = args.conversation
    if args.message:
        cid, status = run_chat_turn(rt, " ".join(args.message), cid, args.model)
        print(f"(conversation {cid}; workspace {rt.store.workspace(cid)})")
        return 0 if status == "done" else 1
    print("JEV chat. Empty line or Ctrl-D quits; Ctrl-C cancels the current answer.")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except EOFError:
            break
        if not msg:
            break
        print("jev> ", end="", flush=True)
        cid, _ = run_chat_turn(rt, msg, cid, args.model)
    if cid:
        print(f"(conversation {cid}; workspace {rt.store.workspace(cid)})")
    return 0


# ------------------------------------------------------------------------------------------ misc
def cmd_audit(args) -> int:
    rt = _rt(args)
    for a in reversed(rt.store.list_approvals(args.status, args.limit)):
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a["created"]))
        print(f"{t}  {a['status']:<10} {a['tool'] or '':<20} {(a['decided_by'] or ''):<9} {a['reason']}")
    print(f"(audit log: {rt.store.path}, table approvals)")
    return 0


def cmd_export(args) -> int:
    rt = _rt(args)
    out = rt.store.export_conversation(args.conversation, Path(args.out or Path(rt.settings.dir) / "exports"))
    print(out)
    return 0


def cmd_serve(args, rest) -> int:
    from jepa_studio.cli import main as jmain
    jmain(["serve", "--root", args.root, *rest])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jev", description="JEV: a local AI agent inside JEPA Studio (Ollama).")
    ap.add_argument("--root", default=os.environ.get("JEV_ROOT", "."), help="backend root (data in ROOT/jev-data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check Ollama, models, sandbox, ffmpeg, GPU")
    p = sub.add_parser("models", help="list / pull / import / rm Ollama models")
    p.add_argument("action", nargs="?", choices=["list", "pull", "import", "rm"], default="list")
    p.add_argument("name", nargs="?", help="model name (pull/rm) or GGUF file (import)")
    p.add_argument("--name", dest="as_name", help="name for an imported GGUF")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    p = sub.add_parser("chat", help="chat in the terminal (approvals asked here)")
    p.add_argument("message", nargs="*")
    p.add_argument("--model")
    p.add_argument("--conversation", help="continue a conversation id")
    p = sub.add_parser("audit", help="show the approvals audit log")
    p.add_argument("--status")
    p.add_argument("--limit", type=int, default=50)
    p = sub.add_parser("export", help="export a conversation as a zip")
    p.add_argument("conversation")
    p.add_argument("--out")
    sub.add_parser("serve", help="start the local backend (same as jepa-studio serve)")
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "serve":
        return cmd_serve(args, rest)
    if rest:
        ap.error("unrecognized arguments: " + " ".join(rest))
    if args.cmd == "models" and args.action in ("pull", "import", "rm") and not args.name:
        ap.error(f"models {args.action} needs a name")
    fn = {"doctor": cmd_doctor, "models": cmd_models, "chat": cmd_chat, "audit": cmd_audit, "export": cmd_export}
    return fn[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
