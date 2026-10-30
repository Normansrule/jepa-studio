"""JEV settings and the Ollama client, against the fake Ollama server."""
import json
import os
import stat

import pytest

from jev.llm import MockLLM, Ollama, OllamaError, OllamaUnavailable, ToolsUnsupported, is_cloud_model
from jev.settings import Settings, SettingsError, check_ollama_url

from test_jev_support import dead_url, fake_ollama  # noqa: F401 - fixture


# ----------------------------------------------------------------------------- settings
@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434",
                                 "http://127.0.0.2:1"])
def test_loopback_ollama_urls_allowed(url):
    assert check_ollama_url(url, allow_remote=False) == url


@pytest.mark.parametrize("url", ["http://192.168.1.10:11434", "http://ollama.example.com", "http://0.0.0.0:11434",
                                 "http://10.0.0.1:11434", "https://evil.example/api", "file:///etc/passwd",
                                 "http://127.0.0.1.evil.example:11434"])
def test_non_loopback_ollama_refused_unless_allowed(url, tmp_path):
    with pytest.raises(SettingsError):
        check_ollama_url(url, allow_remote=False)
    if url.startswith("http"):
        assert check_ollama_url(url, allow_remote=True)
    with pytest.raises(SettingsError):
        Ollama(url)
    # through the settings file / API too
    st = Settings(tmp_path, env={})
    with pytest.raises(SettingsError):
        st.update({"ollama_url": url})
    with pytest.raises(SettingsError, match="only be changed by editing"):
        st.update({"allow_remote_ollama": True})


def test_settings_file_private_env_override_and_absurd_limits(tmp_path):
    st = Settings(tmp_path, env={"JEV_DEFAULT_MODEL": "llama3.1:8b"})
    assert st.resolved()["default_model"] == "llama3.1:8b"
    assert st.resolved()["workspace_root"].endswith(os.path.join("jev-data", "workspaces"))
    st.update({"limits": {"wall_seconds": 60}, "network_allowed": False})
    assert json.loads(st.path.read_text())["limits"] == {"wall_seconds": 60}
    if os.name != "nt":
        assert stat.S_IMODE(st.path.stat().st_mode) == 0o600
    for bad in ({"limits": {"memory_mb": 10 ** 9}}, {"limits": {"cpu_seconds": 0}}, {"limits": {"bogus": 1}},
                {"network_allowed": "yes"}, {"approval_rules": {"shell": "run"}}):
        with pytest.raises(SettingsError):
            st.update(bad)
    assert st.resolved()["limits"]["wall_seconds"] == 60     # failed updates changed nothing
    # remote Ollama only with the explicit file/env opt-in
    with pytest.raises(SettingsError, match="not on this machine"):
        Settings(tmp_path / "b", env={"JEV_OLLAMA_URL": "http://192.168.1.5:11434"})
    st3 = Settings(tmp_path / "c", env={"JEV_OLLAMA_URL": "http://192.168.1.5:11434", "JEV_ALLOW_REMOTE_OLLAMA": "1"})
    assert st3.resolved()["ollama_url"] == "http://192.168.1.5:11434"


# ----------------------------------------------------------------------------- client
def test_version_tags_show(fake_ollama):  # noqa: F811
    o = Ollama(fake_ollama.url)
    assert o.version() == "0.9.9-fake"
    names = [m["name"] for m in o.tags()]
    assert "qwen2.5-coder:7b" in names
    m = o.tags()[0]
    assert set(m) >= {"name", "size_bytes", "family", "parameter_size", "quantization", "digest", "modified"}
    assert o.supports_tools("llama3.1:8b")
    fake_ollama.no_tools.add("llama3.1:8b")
    assert not Ollama(fake_ollama.url).supports_tools("llama3.1:8b")


def test_chat_streams_tokens_and_tool_calls(fake_ollama):  # noqa: F811
    fake_ollama.script = [{"content": "Let me write that.", "tool_calls": [
        {"name": "write_file", "arguments": {"path": "a.py", "content": "print(1)"}}]}]
    toks = []
    out = Ollama(fake_ollama.url).chat("qwen2.5-coder:7b", [{"role": "user", "content": "x"}],
                                       tools=[{"type": "function", "function": {"name": "write_file"}}],
                                       on_token=toks.append)
    assert "".join(toks) == "Let me write that." and len(toks) > 1
    assert out["tool_calls"][0]["name"] == "write_file"
    assert out["tool_calls"][0]["arguments"] == {"path": "a.py", "content": "print(1)"}
    assert fake_ollama.chats()[-1]["tools"][0]["function"]["name"] == "write_file"
    assert fake_ollama.chats()[-1]["stream"] is True


def test_tools_unsupported_and_missing_model(fake_ollama):  # noqa: F811
    fake_ollama.no_tools.add("llama3.1:8b")
    o = Ollama(fake_ollama.url)
    with pytest.raises(ToolsUnsupported):
        o.chat("llama3.1:8b", [{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    with pytest.raises(OllamaError, match="pull"):
        o.chat("nope:1b", [{"role": "user", "content": "x"}])


def test_pull_progress_and_failure(fake_ollama):  # noqa: F811
    o = Ollama(fake_ollama.url)
    seen = []
    o.pull("tinyllama", seen.append)
    fr = [s["fraction"] for s in seen if s["fraction"] is not None]
    assert fr == [0.0, 0.25, 0.5, 1.0] and seen[-1]["status"] == "success"
    assert "tinyllama" in [m["name"] for m in o.tags()]
    with pytest.raises(OllamaError, match="does not exist"):
        o.pull("does-not-exist")


def test_create_from_gguf_uploads_blob_and_hashes(fake_ollama, tmp_path):  # noqa: F811
    import hashlib
    g = tmp_path / "tiny.gguf"
    g.write_bytes(b"GGUF" + os.urandom(5000))
    digest = Ollama(fake_ollama.url).create_from_gguf("tiny-local", g)
    assert digest == hashlib.sha256(g.read_bytes()).hexdigest()
    assert fake_ollama.blobs[digest] == 5004
    assert "tiny-local" in fake_ollama.models
    bad = tmp_path / "not.gguf"
    bad.write_bytes(b"PK\x03\x04 zip")
    with pytest.raises(OllamaError, match="not a GGUF"):
        Ollama(fake_ollama.url).create_from_gguf("x", bad)


def test_delete(fake_ollama):  # noqa: F811
    o = Ollama(fake_ollama.url)
    o.delete("llama3.1:8b")
    assert "llama3.1:8b" not in fake_ollama.models
    with pytest.raises(OllamaError):
        o.delete("llama3.1:8b")


def test_not_running_message_and_no_proxy(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")    # must be ignored for the local Ollama
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        Ollama(dead_url()).version()


def test_cloud_models_detected():
    assert is_cloud_model("gpt-oss:120b-cloud") and is_cloud_model("qwen3-coder:480b-cloud")
    assert is_cloud_model("x", {"remote_host": "https://ollama.com:443"})
    assert not is_cloud_model("llama3.1:8b")


def test_mock_llm_scripts_replies():
    m = MockLLM([{"content": "a b", "tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]}])
    toks = []
    out = m.chat("mock", [], on_token=toks.append)
    assert "".join(toks) == "a b" and out["tool_calls"][0]["arguments"] == {"expression": "1+1"}
