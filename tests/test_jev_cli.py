"""`jev` command line: doctor, models, chat with terminal approvals, audit, export."""
import builtins
import subprocess
import sys

from jev.cli import main

from test_jev_support import dead_url, fake_ollama, jev_env  # noqa: F401 - fixtures


def test_doctor_without_ollama_is_helpful_and_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JEV_OLLAMA_URL", dead_url())
    code = main(["--root", str(tmp_path), "doctor"])
    out = capsys.readouterr().out
    assert code != 0
    assert "NOT REACHABLE" in out and "ollama serve" in out and "https://ollama.com/download" in out
    assert "sandbox" in out and "network isolation" in out


def test_doctor_as_a_subprocess_exits_nonzero(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "JEV_OLLAMA_URL": dead_url(), "PYTHONPATH": str(__import__("pathlib").Path(
        __file__).resolve().parent.parent)}
    r = subprocess.run([sys.executable, "-m", "jev.cli", "--root", str(tmp_path), "doctor"], capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 1 and "Ollama isn't running" in r.stdout


def test_doctor_with_ollama_ready(jev_env, capsys):  # noqa: F811
    root, _ = jev_env
    assert main(["--root", str(root), "doctor"]) == 0
    out = capsys.readouterr().out
    assert "OK, version 0.9.9-fake" in out and "result      ready" in out


def test_doctor_default_model_missing(jev_env, capsys):  # noqa: F811
    root, fake = jev_env
    fake.models.pop("qwen2.5-coder:7b")
    assert main(["--root", str(root), "doctor"]) == 1
    assert "jev models pull qwen2.5-coder:7b" in capsys.readouterr().out


def test_models_commands_ask_first(jev_env, capsys, monkeypatch, tmp_path):  # noqa: F811
    root, fake = jev_env
    assert main(["--root", str(root), "models"]) == 0
    assert "llama3.1:8b" in capsys.readouterr().out
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    assert main(["--root", str(root), "models", "pull", "phi3:mini"]) == 1
    assert "phi3:mini" not in fake.models
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    assert main(["--root", str(root), "models", "pull", "phi3:mini"]) == 0
    assert "phi3:mini" in fake.models
    g = tmp_path / "x.gguf"
    g.write_bytes(b"GGUF1234")
    assert main(["--root", str(root), "models", "import", str(g), "--name", "mine", "--yes"]) == 0
    assert "mine" in fake.models and "sha256" in capsys.readouterr().out
    assert main(["--root", str(root), "models", "rm", "mine", "-y"]) == 0
    assert "mine" not in fake.models
    assert main(["--root", str(root), "audit"]) == 0
    out = capsys.readouterr().out
    assert "denied" in out and "approved" in out and "model_pull" in out


def test_chat_one_shot_with_terminal_approval(jev_env, capsys, monkeypatch):  # noqa: F811
    root, fake = jev_env
    fake.script = [{"content": "", "tool_calls": [{"name": "run_shell", "arguments": {"command": "echo from-shell"}}]},
                   {"content": "Shell said hi."}]
    answers = iter(["y"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert main(["--root", str(root), "chat", "say", "hi", "via", "shell"]) == 0
    out = capsys.readouterr().out
    assert "APPROVAL NEEDED for run_shell" in out and "from-shell" in out and "Shell said hi." in out
    cid = out.strip().split("(conversation ")[-1].split(";")[0]
    assert main(["--root", str(root), "export", cid]) == 0
    assert capsys.readouterr().out.strip().endswith(".zip")


def test_chat_denies_on_eof(jev_env, capsys, monkeypatch):  # noqa: F811
    root, fake = jev_env
    fake.script = [{"content": "", "tool_calls": [{"name": "run_shell", "arguments": {"command": "echo nope"}}]},
                   {"content": "Okay, not running it."}]

    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr(builtins, "input", eof)
    assert main(["--root", str(root), "chat", "run it"]) == 0
    out = capsys.readouterr().out
    assert "failed: The user denied" in out and "Okay, not running it." in out


def test_models_pull_generative_asks_and_lists(jev_env, capsys, monkeypatch, tmp_path):  # noqa: F811
    from jev.media import generative as G
    root, fake = jev_env
    monkeypatch.setenv("JEV_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(G, "detect_gpu", lambda: {"backend": "cuda", "name": "Fake 16 GB", "vram_gb": 16.0})
    got = []

    def fake_download(name, settings=None, **kw):
        got.append(name)
        d = G.models_dir(settings) / name
        d.mkdir(parents=True)
        (d / "model_index.json").write_text("{}")
        return {"model": name, "folder": str(d), "files": 1, "bytes": 2, "class_name": "X"}
    monkeypatch.setattr(G, "download_model", fake_download)
    assert main(["--root", str(root), "models"]) == 0
    assert "available: sdxl-turbo" in capsys.readouterr().out
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    assert main(["--root", str(root), "models", "pull", "sdxl-turbo"]) == 1 and got == []
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    assert main(["--root", str(root), "models", "pull", "sdxl-turbo"]) == 0 and got == ["sdxl-turbo"]
    assert "Hugging Face" not in fake.models and "sdxl-turbo" not in fake.models   # not sent to Ollama
    assert main(["--root", str(root), "models", "pull", "flux-schnell", "-y"]) == 0
    out = capsys.readouterr().out
    assert "needs ~24 GB" in out and "refused" in out                                  # 16 GB card warned
    assert main(["--root", str(root), "models"]) == 0
    assert "installed: sdxl-turbo" in capsys.readouterr().out
    assert main(["--root", str(root), "audit"]) == 0
    assert "download_generative_model" in capsys.readouterr().out
