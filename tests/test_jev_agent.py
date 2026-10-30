"""The agent loop end to end: tool calls, artifacts, approvals, the JSON fallback, limits, cancel."""
import json
import time

import pytest

from jev.agent import Runtime, parse_tool_calls
from jev.llm import MockLLM

from test_jev_support import fake_ollama, jev_env  # noqa: F401 - fixtures

SQ = "total = sum(i * i for i in range(10))\nprint('sum', total)\nopen('out.csv', 'w').write('n,sq\\n3,9\\n')\n"


def _wait_event(job, name, timeout=30):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        for e in job.events:
            if e["event"] == name:
                return e
        time.sleep(0.02)
    raise AssertionError(f"no {name} event; got {[e['event'] for e in job.events]}")


def _settings(root, **kw):
    d = root / "jev-data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(json.dumps(kw))


def _run(rt, msg, **kw):
    r = rt.start_turn(msg, **kw)
    job = rt.jobs.wait(r["turn_id"], 90)
    assert job.finished is not None, [e["event"] for e in job.events]
    return r, job


def test_end_to_end_with_fake_ollama(jev_env):  # noqa: F811
    root, fake = jev_env
    fake.script = [
        {"content": "I'll write the script first.",
         "tool_calls": [{"name": "write_file", "arguments": {"path": "sq.py", "content": SQ}}]},
        {"content": "", "tool_calls": [{"name": "run_python",
                                        "arguments": {"code": "import runpy\nrunpy.run_path('sq.py')"}}]},
        lambda body: {"content": "The sum of squares 0..9 is 285." if "sum 285" in body["messages"][-1]["content"]
                      else "I did not see the result."},
    ]
    rt = Runtime(root)
    r, job = _run(rt, "Sum the squares of 0..9 with Python and save a CSV.")
    ev = [e["event"] for e in job.events]
    assert ev[0] == "start" and ev[-1] == "done" and job.status == "done", job.events[-3:]
    calls = [e for e in job.events if e["event"] == "tool_call"]
    assert [(c["name"], c["decision"], c["risk"]) for c in calls] == [("write_file", "run", "sandbox"),
                                                                     ("run_python", "run", "sandbox")]
    results = [e for e in job.events if e["event"] == "tool_result"]
    assert all(x["ok"] for x in results), results
    arts = [e for e in job.events if e["event"] == "artifact"]
    assert {a["path"] for a in arts} == {"sq.py", "out.csv"}
    assert all(a["url"] == f"/api/jev/artifacts/{a['id']}" for a in arts)
    assert {a["kind"] for a in arts} == {"code", "data"}
    assert job.events[-1]["text"] == "The sum of squares 0..9 is 285."
    assert "".join(e["text"] for e in job.events if e["event"] == "token").startswith("I'll write the script first.")
    # what Ollama saw: native tools, the system prompt, tool results as role "tool"
    chats = fake.chats()
    assert chats[0]["tools"] and chats[0]["messages"][0]["role"] == "system"
    assert "Never claim that a tool ran" in chats[0]["messages"][0]["content"]
    assert chats[2]["messages"][-1]["role"] == "tool" and chats[2]["messages"][-1]["tool_name"] == "run_python"
    # store: messages, tool calls, an audit row for every decision
    msgs = rt.store.messages(r["conversation_id"])
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    audit = rt.store.list_approvals()
    assert sorted(a["tool"] for a in audit) == ["run_python", "write_file"]
    assert {a["status"] for a in audit} == {"auto_run"}
    ws = rt.store.workspace(r["conversation_id"])
    assert (ws / "out.csv").read_text() == "n,sq\n3,9\n"
    # a follow-up turn sees the history
    fake.script = [{"content": "Yes, 285."}]
    _run(rt, "Was it 285?", conversation_id=r["conversation_id"])
    hist = [m["content"] for m in fake.chats()[-1]["messages"]]
    assert any("Sum the squares" in h for h in hist)


def test_shell_approval_approve_deny_and_timeout(tmp_path):
    shell = {"content": "", "tool_calls": [{"name": "run_shell", "arguments": {"command": "echo approved-$((6*7))"}}]}
    rt = Runtime(tmp_path, llm=MockLLM([shell, {"content": "ran it"}, shell, {"content": "ok, skipped"}]))
    # approve
    r = rt.start_turn("run echo")
    job = rt.jobs.get(r["turn_id"])
    req = _wait_event(job, "approval_request")
    assert req["name"] == "run_shell" and "shell" in req["rules"] and job.status == "waiting_approval"
    pending = rt.broker.list("pending")
    assert [p["id"] for p in pending] == [req["approval_id"]]
    assert rt.broker.decide(req["approval_id"], "approve")["status"] == "approved"
    job = rt.jobs.wait(r["turn_id"], 30)
    res = [e for e in job.events if e["event"] == "tool_result"][0]
    assert res["ok"] and "approved-42" in res["content"]
    assert job.events[-1] == {**job.events[-1], "event": "done", "text": "ran it"}
    # deny
    r = rt.start_turn("run echo again", conversation_id=r["conversation_id"])
    job = rt.jobs.get(r["turn_id"])
    req = _wait_event(job, "approval_request")
    rt.broker.decide(req["approval_id"], "deny")
    job = rt.jobs.wait(r["turn_id"], 30)
    res = [e for e in job.events if e["event"] == "tool_result"][0]
    assert not res["ok"] and "denied" in res["content"]
    assert job.events[-1]["text"] == "ok, skipped"
    assert {a["status"] for a in rt.store.list_approvals()} == {"approved", "denied"}
    # timeout: unanswered approvals count as deny
    _settings(tmp_path, approval_timeout_s=1)
    rt2 = Runtime(tmp_path, llm=MockLLM([shell, {"content": "nobody answered"}]))
    r, job = _run(rt2, "run echo")
    res = [e for e in job.events if e["event"] == "tool_result"][0]
    assert not res["ok"] and "No answer" in res["content"]
    assert [e for e in job.events if e["event"] == "approval_result"][0]["status"] == "expired"
    assert any(a["status"] == "expired" and a["decided_by"] == "timeout" for a in rt2.store.list_approvals())


def test_policy_deny_and_invalid_args_reach_the_model(tmp_path):
    script = [{"content": "", "tool_calls": [{"name": "read_file", "arguments": {"path": "../../etc/passwd"}},
                                             {"name": "calculator", "arguments": {"expr": "1+1"}},
                                             {"name": "no_such_tool", "arguments": {}}]},
              {"content": "Those were refused."}]
    llm = MockLLM(script)
    rt = Runtime(tmp_path, llm=llm)
    _, job = _run(rt, "read the password file")
    decisions = [(e["name"], e["decision"]) for e in job.events if e["event"] == "tool_call"]
    assert decisions == [("read_file", "deny"), ("calculator", "deny"), ("no_such_tool", "deny")]
    tool_msgs = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert "Refused by JEV's safety policy" in tool_msgs[0] and "escapes the workspace" in tool_msgs[0]
    assert "invalid arguments" in tool_msgs[1] and "no tool named" in tool_msgs[2]
    statuses = [a["status"] for a in rt.store.list_approvals()]
    assert statuses.count("auto_deny") == 3


def test_json_fallback_for_models_without_tools(jev_env):  # noqa: F811
    root, fake = jev_env
    fake.no_tools.add("llama3.1:8b")
    fake.script = [{"content": "```json\n{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"6*7\"}}\n```"},
                   lambda body: {"content": "It is 42." if "= 42" in body["messages"][-1]["content"] else "??"}]
    rt = Runtime(root)
    _, job = _run(rt, "what is 6*7", model="llama3.1:8b")
    assert job.events[-1]["text"] == "It is 42."
    toks = "".join(e["text"] for e in job.events if e["event"] == "token")
    assert "calculator" not in toks                     # the JSON call is not streamed to the user
    chats = fake.chats()
    assert "tools" not in chats[0] and "TOOL PROTOCOL" in chats[0]["messages"][0]["content"]
    assert chats[1]["messages"][-1]["content"].startswith("TOOL RESULT (calculator, ok)")


def test_native_model_that_writes_calls_as_text(jev_env):  # noqa: F811
    root, fake = jev_env
    # some models (qwen2.5-coder) answer a native-tools request with the call written as text
    fake.script = [{"content": "<tool_call>{\"name\": \"get_time\", \"arguments\": {}}</tool_call>"},
                   {"content": "done"}]
    rt = Runtime(root)
    _, job = _run(rt, "time?")
    assert [e["name"] for e in job.events if e["event"] == "tool_call"] == ["get_time"]
    assert job.events[-1]["text"] == "done"


def test_switches_to_json_protocol_when_chat_rejects_tools(jev_env):  # noqa: F811
    root, fake = jev_env
    fake.reject_tools.add("qwen2.5-coder:7b")    # /api/show claims tools, /api/chat refuses them
    fake.script = [{"content": "{\"tool\": \"calculator\", \"arguments\": {\"expression\": \"2**10\"}}"},
                   {"content": "1024"}]
    rt = Runtime(root)
    _, job = _run(rt, "2^10?")
    assert job.events[-1]["text"] == "1024", job.events
    chats = fake.chats()
    assert "tools" not in chats[-2] and "TOOL PROTOCOL" in chats[-2]["messages"][0]["content"]


def test_parse_tool_calls_robust():
    names = {"calculator", "write_file"}
    assert parse_tool_calls('{"tool": "calculator", "arguments": {"expression": "1"}}', names)[0]["name"] == "calculator"
    assert parse_tool_calls('[{"name": "calculator", "arguments": "{\\"expression\\": \\"2\\"}"}]',
                            names)[0]["arguments"] == {"expression": "2"}
    assert parse_tool_calls('```json\n{"function": {"name": "write_file", "arguments": {"path": "a", '
                            '"content": "b"}}}\n```', names)[0]["name"] == "write_file"
    assert len(parse_tool_calls('<tool_call>{"name": "calculator", "arguments": {"expression": "1"}}</tool_call>\n'
                                '<tool_call>{"name": "calculator", "arguments": {"expression": "2"}}</tool_call>',
                                names)) == 2
    # a call is only a call when it is the whole reply (review M2): quoted JSON never runs
    assert not parse_tool_calls('Sure!\n```json\n{"function": {"name": "write_file", "arguments": {"path": "a", '
                                '"content": "b"}}}\n```', names)
    assert not parse_tool_calls('I will call {"tool": "calculator", "arguments": {"expression": "3"}} now', names)
    assert not parse_tool_calls('I will call {"tool": "calculator", "arguments": {}} now', names, strict=True)
    assert not parse_tool_calls('{"tool": "rm_rf", "arguments": {}}', names)       # unknown tools are ignored
    assert not parse_tool_calls('Here is JSON: {"a": 1}', names)
    assert not parse_tool_calls("", names)


def test_max_rounds_then_final_answer_without_tools(tmp_path):
    _settings(tmp_path, max_tool_rounds=2)
    calc = {"content": "", "tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]}
    llm = MockLLM([calc, calc, {"content": "Final: 2"}])
    rt = Runtime(tmp_path, llm=llm)
    _, job = _run(rt, "loop")
    assert job.events[-1]["text"] == "Final: 2" and job.events[-1]["rounds"] == 2
    assert llm.calls[-1]["tools"] is None
    assert "limit of 2 tool rounds" in llm.calls[-1]["messages"][-1]["content"]


def test_cancel_kills_running_code(tmp_path):
    slow = {"content": "", "tool_calls": [{"name": "run_python", "arguments": {"code": "import time\ntime.sleep(60)"}}]}
    rt = Runtime(tmp_path, llm=MockLLM([slow, {"content": "never"}]))
    r = rt.start_turn("sleep")
    job = rt.jobs.get(r["turn_id"])
    _wait_event(job, "tool_call")
    time.sleep(0.5)
    t0 = time.monotonic()
    assert rt.cancel(r["turn_id"])
    job = rt.jobs.wait(r["turn_id"], 20)
    assert job.status == "cancelled" and time.monotonic() - t0 < 10
    assert job.events[-1]["event"] == "cancelled"
    # the conversation is free again
    rt._llm_override = MockLLM([{"content": "hi"}])
    _run(rt, "hello", conversation_id=r["conversation_id"])


def test_busy_conversation_and_cloud_model_refusal(tmp_path):
    rt = Runtime(tmp_path, llm=MockLLM([{"content": "x"}]))
    _, job = _run(rt, "hi", model="gpt-oss:120b-cloud")
    assert job.status == "error" and "leave this computer" in job.events[-1]["message"]
    with pytest.raises(ValueError):
        rt.start_turn("")
    with pytest.raises(ValueError):
        rt.start_turn("x", model="bad model name; rm -rf")
    with pytest.raises(KeyError):
        rt.start_turn("x", conversation_id="c-nope")
    with pytest.raises(PermissionError):
        rt.start_turn("x", attachments=["../../etc/passwd"])


def test_plot_and_artifacts_from_code(tmp_path):
    pytest.importorskip("matplotlib")
    script = [{"content": "", "tool_calls": [{"name": "make_plot", "arguments": {
        "kind": "line", "series": [{"name": "sin", "y": [0, 1, 0, -1, 0]}], "title": "wave"}}]},
              {"content": "", "tool_calls": [{"name": "run_python", "arguments": {"code": (
                  "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
                  "plt.plot([1,2,3]); plt.savefig('code_plot.png'); print('saved')")}}]},
              {"content": "Two plots."}]
    rt = Runtime(tmp_path, llm=MockLLM(script))
    r, job = _run(rt, "plot")
    arts = [e for e in job.events if e["event"] == "artifact"]
    assert {a["path"] for a in arts} == {"plot.png", "code_plot.png"}
    assert all(a["mime"] == "image/png" and a["kind"] == "image" for a in arts)
    files = rt.store.list_artifacts(r["conversation_id"])
    assert len(files) == 2 and all(f["bytes"] > 1000 for f in files)
