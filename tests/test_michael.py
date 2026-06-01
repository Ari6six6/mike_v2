"""Tests for michael's internals: slugs, projects, paths, staging, trash, replay."""
from __future__ import annotations

import pathlib
import shutil

import pytest

import main as m
import michael.globals as michael_globals


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Patch all michael path globals to live under tmp_path/.michael."""
    state = tmp_path / ".michael"
    monkeypatch.setattr(michael_globals, "STATE_DIR", state)
    monkeypatch.setattr(michael_globals, "GLOBAL_CONFIG_PATH", state / "config.json")
    monkeypatch.setattr(michael_globals, "GLOBAL_EVENTS_PATH", state / "events.jsonl")
    monkeypatch.setattr(michael_globals, "STATE_FILE_PATH", state / "state.json")
    monkeypatch.setattr(michael_globals, "PROJECTS_DIR", state / "projects")
    monkeypatch.setattr(michael_globals, "REPL_HISTORY_PATH", state / "repl_history")
    monkeypatch.setattr(michael_globals, "GLOBAL_DATASET_DIR", state / "dataset")
    state.mkdir()
    return state


@pytest.fixture
def workspace(tmp_path):
    """Fresh project workspace with a couple of files plus dotted+skip dirs."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "foo.py").write_text("x = 1\n")
    (ws / "README.md").write_text("# hi\n")
    (ws / ".git").mkdir()
    (ws / ".git" / "HEAD").write_text("ref: x\n")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "junk.js").write_text("// no\n")
    return ws


# ---- slugify -------------------------------------------------------------

def test_slugify_basic():
    assert m.slugify("Hello World") == "hello-world"


def test_slugify_strips_specials():
    assert m.slugify("foo!@# bar/baz") == "foo-bar-baz"


def test_slugify_empty_raises():
    with pytest.raises(michael_globals.MichaelError):
        m.slugify("")
    with pytest.raises(michael_globals.MichaelError):
        m.slugify("///")


def test_slugify_truncates_to_64():
    s = m.slugify("a" * 200)
    assert len(s) <= 64


# ---- project model -------------------------------------------------------

def test_create_project_round_trip(home, workspace):
    p = m.create_project("my proj", workspace)
    assert p.slug == "my-proj"
    assert p.name == "my proj"
    assert pathlib.Path(p.path).resolve() == workspace.resolve()
    loaded = m.Project.load("my-proj")
    assert loaded == p


def test_create_project_collision_appends_suffix(home, workspace, tmp_path):
    m.create_project("foo", workspace)
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    p2 = m.create_project("foo", ws2)
    assert p2.slug == "foo-2"


def test_list_projects_sorted(home, tmp_path):
    for n in ("zeta", "alpha", "mike"):
        ws = tmp_path / n
        ws.mkdir()
        m.create_project(n, ws)
    slugs = [p.slug for p in m.list_projects()]
    assert slugs == sorted(slugs)


# ---- path-escape guard ---------------------------------------------------

def test_resolve_in_project_ok(home, workspace):
    p = m.create_project("x", workspace)
    r = m._resolve_in_project(p, "src/foo.py")
    assert r == (workspace / "src" / "foo.py").resolve()


def test_resolve_in_project_refuses_escape(home, workspace):
    p = m.create_project("x", workspace)
    with pytest.raises(m.MichaelError):
        m._resolve_in_project(p, "../escape.txt")


def test_resolve_in_project_refuses_absolute(home, workspace):
    p = m.create_project("x", workspace)
    with pytest.raises(m.MichaelError):
        m._resolve_in_project(p, "/etc/passwd")


# ---- file hashes & diff --------------------------------------------------

def test_file_hashes_skips_dotted_and_skipdirs(home, workspace):
    h = m._file_hashes(workspace)
    assert ".git/HEAD" not in h
    assert "node_modules/junk.js" not in h
    assert "src/foo.py" in h
    assert "README.md" in h


def test_diff_hashes_classifies_correctly():
    before = {"a": "1", "b": "2", "c": "3"}
    after = {"a": "1", "b": "9", "d": "4"}
    d = m._diff_hashes(before, after)
    assert d["added"] == ["d"]
    assert d["removed"] == ["c"]
    assert d["modified"] == ["b"]


# ---- check_expected ------------------------------------------------------

def test_check_expected_match():
    delta = {"added": ["a"], "modified": ["b"], "removed": []}
    assert m._check_expected(["a", "b"], delta) == ""


def test_check_expected_extra():
    delta = {"added": ["a", "c"], "modified": [], "removed": []}
    msg = m._check_expected(["a"], delta)
    assert "extra" in msg and "c" in msg


def test_check_expected_missing():
    delta = {"added": [], "modified": [], "removed": []}
    msg = m._check_expected(["a"], delta)
    assert "missing" in msg


# ---- staging -------------------------------------------------------------

def test_stage_project_skips_dotted_and_skipdirs(home, workspace):
    p = m.create_project("x", workspace)
    stage = m._stage_project(p)
    try:
        assert (stage / "src" / "foo.py").read_text() == "x = 1\n"
        assert not (stage / ".git").exists()
        assert not (stage / "node_modules").exists()
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)


def test_apply_in_staging_write_file_does_not_touch_real(home, workspace):
    p = m.create_project("x", workspace)
    stage = m._stage_project(p)
    real_root = workspace.resolve()
    ext_root = stage.parent / "_ext"
    ext_root.mkdir(exist_ok=True)
    try:
        m._apply_in_staging(
            "write_file",
            {"path": "src/bar.py", "content": "y = 2\n"},
            stage,
            real_root,
            ext_root,
        )
        assert (stage / "src" / "bar.py").read_text() == "y = 2\n"
        assert not (workspace / "src" / "bar.py").exists()
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)


def test_apply_in_staging_refuses_central_fs(home, workspace):
    """Writing to ~/.michael/ must be blocked regardless of staging."""
    p = m.create_project("x", workspace)
    stage = m._stage_project(p)
    real_root = workspace.resolve()
    ext_root = stage.parent / "_ext"
    ext_root.mkdir(exist_ok=True)
    central = str(michael_globals.STATE_DIR / "evil.txt")
    try:
        with pytest.raises(m.MichaelError, match="Central FS violation"):
            m._apply_in_staging(
                "write_file",
                {"path": central, "content": "x"},
                stage,
                real_root,
                ext_root,
            )
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)


# ---- trash + undo --------------------------------------------------------

def test_save_trash_and_undo_modified(home, workspace):
    p = m.create_project("x", workspace)
    real = workspace.resolve()
    delta = {"added": [], "modified": ["src/foo.py"], "removed": []}
    m._save_trash(p, "write_file",
                  {"path": "src/foo.py", "content": "x = 99\n"},
                  delta, real, verify_rc=None)
    (real / "src" / "foo.py").write_text("x = 99\n")
    m._undo_one(p)
    assert (real / "src" / "foo.py").read_text() == "x = 1\n"


def test_undo_added_deletes_file(home, workspace):
    p = m.create_project("x", workspace)
    real = workspace.resolve()
    delta = {"added": ["src/new.py"], "modified": [], "removed": []}
    m._save_trash(p, "write_file",
                  {"path": "src/new.py", "content": "z = 3\n"},
                  delta, real, verify_rc=None)
    (real / "src" / "new.py").write_text("z = 3\n")
    m._undo_one(p)
    assert not (real / "src" / "new.py").exists()


def test_undo_with_no_trash_errors(home, workspace):
    p = m.create_project("x", workspace)
    with pytest.raises(m.MichaelError):
        m._undo_one(p)


# ---- sync_to_real --------------------------------------------------------

def test_sync_to_real_applies_added_and_modified(home, workspace, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "src").mkdir()
    (stage / "src" / "foo.py").write_text("modified\n")
    (stage / "src" / "new.py").write_text("added\n")
    delta = {
        "added": ["src/new.py"],
        "modified": ["src/foo.py"],
        "removed": [],
    }
    m._sync_to_real(stage, workspace, delta)
    assert (workspace / "src" / "foo.py").read_text() == "modified\n"
    assert (workspace / "src" / "new.py").read_text() == "added\n"


def test_sync_to_real_handles_removed(home, workspace, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    delta = {"added": [], "modified": [], "removed": ["src/foo.py"]}
    m._sync_to_real(stage, workspace, delta)
    assert not (workspace / "src" / "foo.py").exists()


# ---- event log + replay --------------------------------------------------

def test_replay_instance_lifecycle(home):
    m.append_event("instance.start_requested", {"id": "1", "model": "coder"})
    m.append_event("instance.started", {"id": "1", "model": "coder"})
    state = m.replay_global()
    assert state["models"]["coder"]["instance_state"] == "running"


def test_iter_events_skips_garbage(home):
    log = home / "events.jsonl"
    log.write_text(
        '{"seq": 1, "ts": "x", "type": "test.ok", "payload": {}}\n'
        "this is not json\n"
        '{"seq": 2, "ts": "x", "type": "test.ok", "payload": {}}\n'
    )
    events = m.iter_events(log)
    assert len(events) == 2
    assert events[0]["seq"] == 1
    assert events[1]["seq"] == 2


# ---- filesystem_snapshot shape -------------------------------------------

def test_filesystem_snapshot_lists_files_and_skips_junk(home, workspace):
    snap = m.filesystem_snapshot(workspace)
    assert "src/foo.py" in snap
    assert "README.md" in snap
    assert ".git" not in snap
    assert "node_modules" not in snap


# ---- _config_is_unset ----------------------------------------------------

def test_config_is_unset_when_missing(home):
    assert m._config_is_unset() is True


def test_config_is_unset_when_blank(home):
    cfg = m.Config()
    cfg.save()
    assert m._config_is_unset() is True


def test_config_is_unset_false_when_keys_set(home):
    cfg = m.make_stub_config()
    cfg.vast_api_key = "x"
    cfg.models["god"].vast_instance_id = "12345"
    cfg.save()
    assert m._config_is_unset() is False


# ---- single-model helpers -----------------------------------------------

def test_full_toolset_always_available():
    full = {t["function"]["name"] for t in m.TOOLS}
    assert "write_file" in full
    assert "run_shell" in full
    assert "read_file" in full


def test_stub_config_has_single_god_profile():
    cfg = m.make_stub_config()
    assert "god" in cfg.models
    assert cfg.default_model == "god"
    assert len(cfg.models) == 1


def test_get_model_returns_god_by_default(home):
    cfg = m.make_stub_config()
    cfg.save()
    loaded = m.Config.load()
    name, profile = loaded.get_model()
    assert name == "god"


# ---- tool schema: expected_changes is required --------------------------

def _tool(name):
    return next(t for t in m.TOOLS if t["function"]["name"] == name)


def test_write_file_schema_requires_expected_changes():
    schema = _tool("write_file")
    required = schema["function"]["parameters"]["required"]
    assert "expected_changes" in required


def test_apply_patch_schema_requires_expected_changes():
    schema = _tool("apply_patch")
    required = schema["function"]["parameters"]["required"]
    assert "expected_changes" in required


# ---- predicted-delta gate (review reporter, not auto-reject) ------------

def test_execute_with_staging_missing_expected_returns_error_to_llm(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    result = m.execute_with_staging(
        "write_file",
        {"path": "src/bar.py", "content": "y = 2\n"},
        p, cfg, pending,
    )
    assert result.startswith("error: expected_changes is required")
    assert not (workspace / "src" / "bar.py").exists()
    assert pending.stage_root is None
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert "tool.delta_missing" in types


def test_execute_with_staging_mismatch_rolls_back_and_errors(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    # LLM predicts a different file than what the write actually changes.
    result = m.execute_with_staging(
        "write_file",
        {
            "path": "src/bar.py",
            "content": "y = 2\n",
            "expected_changes": ["src/wrong.py"],
        },
        p, cfg, pending,
    )
    # Mismatch is an error: rolled back, LLM told to re-propose.
    assert result.startswith("mismatch:")
    assert "predicted:" in result and "actual:" in result
    # Change was rolled back — not in change_log.
    assert len(pending.change_log) == 0
    # Real workspace untouched.
    assert not (workspace / "src" / "bar.py").exists()
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert "tool.delta_mismatch" in types
    assert "tool.staged" not in types


def test_execute_with_staging_review_returns_diff_without_prompt(home, workspace, monkeypatch):
    """A clean prediction match returns review data and stages the change.
    The user is NOT prompted; commit is deferred to the Ja gate."""
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    # If anything tries to prompt the user, fail loudly.
    monkeypatch.setattr(
        m.typer, "prompt",
        lambda *a, **k: pytest.fail("user must not be prompted in review mode"),
    )
    result = m.execute_with_staging(
        "write_file",
        {
            "path": "src/bar.py",
            "content": "y = 2\n",
            "expected_changes": ["src/bar.py"],
        },
        p, cfg, pending,
    )
    assert "predicted:" in result and "actual:" in result
    assert "match" in result.lower()
    # Stage holds the file; real workspace does not.
    assert (pending.stage_root / "src" / "bar.py").read_text() == "y = 2\n"
    assert not (workspace / "src" / "bar.py").exists()
    assert len(pending.change_log) == 1


def test_pending_changes_accumulates_across_calls(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    r1 = m.execute_with_staging(
        "write_file",
        {"path": "src/a.py", "content": "a = 1\n", "expected_changes": ["src/a.py"]},
        p, cfg, pending,
    )
    r2 = m.execute_with_staging(
        "write_file",
        {"path": "src/b.py", "content": "b = 2\n", "expected_changes": ["src/b.py"]},
        p, cfg, pending,
    )
    assert "predicted:" in r1 and "predicted:" in r2
    # One persistent stage_root reused; two entries in the change log.
    assert pending.stage_root is not None
    assert len(pending.change_log) == 2
    # Both files exist in stage; neither in real.
    assert (pending.stage_root / "src" / "a.py").is_file()
    assert (pending.stage_root / "src" / "b.py").is_file()
    assert not (workspace / "src" / "a.py").exists()
    assert not (workspace / "src" / "b.py").exists()


def test_commit_pending_syncs_all_entries_then_discards(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    m.execute_with_staging(
        "write_file",
        {"path": "src/a.py", "content": "a = 1\n", "expected_changes": ["src/a.py"]},
        p, cfg, pending,
    )
    m.execute_with_staging(
        "write_file",
        {"path": "src/b.py", "content": "b = 2\n", "expected_changes": ["src/b.py"]},
        p, cfg, pending,
    )
    summaries = m.commit_pending(p, pending)
    assert len(summaries) == 2
    assert (workspace / "src" / "a.py").read_text() == "a = 1\n"
    assert (workspace / "src" / "b.py").read_text() == "b = 2\n"
    # Stage is discarded after commit.
    assert pending.stage_root is None
    assert pending.change_log == []
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert types.count("tool.executed") == 2


# ---- Header 4 / build_protocol ------------------------------------------

def test_build_protocol_lists_four_headers():
    text = m.build_protocol()
    for h in ("H1", "H2", "H3", "H4"):
        assert h in text



def test_build_header_includes_protocol(home, workspace):
    p = m.create_project("x", workspace)
    pkg = m.build_header(p, "system stub")
    assert "H4: Protocol" in pkg
    assert "H1:" in pkg and "H2:" in pkg and "H3:" in pkg


# ---- REPL surface --------------------------------------------------------

def test_repl_commands_include_core_commands():
    assert "run" in m.REPL_COMMANDS
    assert "new" in m.REPL_COMMANDS
    assert "gpu" in m.REPL_COMMANDS


# ---- workbench -----------------------------------------------------------

import michael.workbench as wb


def test_workbench_project_root_outside_context_raises():
    with pytest.raises(RuntimeError, match="no active project"):
        wb.project_root()


def test_workbench_project_root_inside_context(home, workspace):
    p = m.create_project("wb-test", workspace)
    token = wb._set_context(p)
    try:
        assert wb.project_root() == pathlib.Path(p.path)
        assert wb.project_slug() == p.slug
    finally:
        wb._reset_context(token)


def test_workbench_context_cleared_after_reset(home, workspace):
    p = m.create_project("wb-reset", workspace)
    token = wb._set_context(p)
    wb._reset_context(token)
    with pytest.raises(RuntimeError, match="no active project"):
        wb.project_root()


def test_workbench_read_file_blocks_central_fs(home, workspace):
    p = m.create_project("wb-perm", workspace)
    token = wb._set_context(p)
    try:
        central_path = str(michael_globals.STATE_DIR / "secret.txt")
        with pytest.raises(m.MichaelError, match="Central FS violation"):
            wb.read_file(central_path)
    finally:
        wb._reset_context(token)


def test_workbench_run_shell_blocks_central_fs_reference(home, workspace):
    p = m.create_project("wb-shell", workspace)
    token = wb._set_context(p)
    try:
        with pytest.raises(m.MichaelError):
            wb.run_shell("cat ~/.michael/config.json")
    finally:
        wb._reset_context(token)


# ---- appmodel ------------------------------------------------------------

import michael.appmodel as am


def test_appmodel_save_and_load(home, workspace):
    p = m.create_project("am-test", workspace)
    model = am.make_model(
        "testapp", "v1",
        base_url="https://api.example.com",
        auth={"type": "bearer"},
        notes="test model",
    )
    am.save_model(p, model)
    loaded = am.load_model(p, "testapp", "v1")
    assert loaded.name == "testapp"
    assert loaded.version == "v1"
    assert loaded.base_url == "https://api.example.com"
    assert loaded.auth == {"type": "bearer"}
    assert loaded.notes == "test model"


def test_appmodel_list_returns_all(home, workspace):
    p = m.create_project("am-list", workspace)
    am.save_model(p, am.make_model("app-a", "1.0"))
    am.save_model(p, am.make_model("app-b", "2.0"))
    models = am.list_models(p)
    assert {mo.name for mo in models} == {"app-a", "app-b"}


def test_appmodel_missing_raises(home, workspace):
    p = m.create_project("am-miss", workspace)
    with pytest.raises(m.MichaelError, match="no model"):
        am.load_model(p, "ghost", "v0")


# ---- vLLM launch command (regression: "no PID returned") -----------------

def test_start_vllm_cmd_has_no_pkill():
    """The launch command must not chain a pkill -f against the server module
    path: its own shell argv contains that path, so pkill would kill the shell
    before `echo $!` runs (empty stdout -> "vLLM failed to launch")."""
    from michael.backends import _start_vllm_cmd
    from michael.config import GpuConfig
    cmd = _start_vllm_cmd(GpuConfig(model_repo="org/model"), ngpu=1)
    assert "pkill" not in cmd
    assert cmd.rstrip().endswith("echo $!")
    assert "org/model" in cmd
    # launches via the resolved interpreter ("$PY"), never a hardcoded `python`
    # (vast images often have no bare `python` -> nohup: No such file).
    assert 'nohup "$PY" -m vllm.entrypoints.openai.api_server' in cmd
    assert "nohup python -m" not in cmd
    # no overrides by default; explicit dtype/quantization render as flags
    assert "--dtype" not in cmd
    assert "--quantization" not in cmd
    cmd_x = _start_vllm_cmd(
        GpuConfig(model_repo="org/model"), ngpu=1, dtype="half", quantization="awq"
    )
    assert "--dtype half" in cmd_x
    assert "--quantization awq" in cmd_x
    # tool calling MUST be enabled — the agent always sends tools=…, and vLLM
    # 400s on any request containing tools unless the server enables it.
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser" in cmd


def test_start_vllm_cmd_caps_kv_cache_memory():
    """The launch command must bound the KV cache so the engine can start on a
    single GPU. Modern checkpoints advertise huge native contexts (e.g.
    Hermes-4.3-36B's 524288 tokens); without --max-model-len vLLM sizes the KV
    cache for that full length (~128 GiB) and aborts at startup with a "KV
    cache memory" ValueError. The default GpuConfig must emit both
    --max-model-len and --gpu-memory-utilization."""
    from michael.backends import _start_vllm_cmd
    from michael.config import GpuConfig

    cmd = _start_vllm_cmd(GpuConfig(model_repo="NousResearch/Hermes-4.3-36B"), ngpu=1)
    assert "--max-model-len 32768" in cmd
    assert "--gpu-memory-utilization 0.92" in cmd

    # A custom cap flows through verbatim.
    cmd2 = _start_vllm_cmd(
        GpuConfig(model_repo="org/model", max_model_len=8192, gpu_memory_utilization=0.95),
        ngpu=1,
    )
    assert "--max-model-len 8192" in cmd2
    assert "--gpu-memory-utilization 0.95" in cmd2

    # 0 is the opt-out: omit the flag entirely and let vLLM decide.
    cmd0 = _start_vllm_cmd(
        GpuConfig(model_repo="org/model", max_model_len=0, gpu_memory_utilization=0),
        ngpu=1,
    )
    assert "--max-model-len" not in cmd0
    assert "--gpu-memory-utilization" not in cmd0


def test_http_error_message_surfaces_body_and_hints():
    """A model-server 4xx must surface the response body (the real reason) and,
    for recognisable Ollama cases, a concrete next step — not the generic
    httpx status line that hid the cause behind an MDN link."""
    from michael.backends import _http_error_message

    class _Resp:
        def __init__(self, code, obj, text=""):
            self.status_code = code
            self._obj = obj
            self.text = text

        def json(self):
            if self._obj is None:
                raise ValueError("not json")
            return self._obj

    # model lacks a tool template (Ollama returns 400 with this string)
    msg = _http_error_message(
        _Resp(400, {"error": "llama2 does not support tools"}), "llama2"
    )
    assert "400" in msg
    assert "does not support tools" in msg
    assert "gpu.model_repo" in msg  # actionable hint present

    # model not pulled (nested {"error": {"message": ...}} shape)
    msg = _http_error_message(
        _Resp(404, {"error": {"message": 'model "foo:1b" not found'}}), "foo:1b"
    )
    assert "not found" in msg
    assert "michael gpu up" in msg

    # empty served_model_name
    msg = _http_error_message(_Resp(400, {"error": "model is required"}), "")
    assert "served_model_name is empty" in msg

    # non-JSON body still surfaces the raw text, never an empty message
    msg = _http_error_message(_Resp(400, None, text="Bad Request: bad field"), "qwen2.5:72b")
    assert "Bad Request: bad field" in msg


def test_vllm_tool_parser_per_model():
    from michael.backends import _vllm_tool_parser
    assert _vllm_tool_parser("Qwen/Qwen3-32B-AWQ") == "hermes"
    assert _vllm_tool_parser("Qwen/Qwen2.5-72B-Instruct-AWQ") == "hermes"
    assert _vllm_tool_parser("deepseek-ai/DeepSeek-V4-Flash") == "deepseek_v3"
    assert _vllm_tool_parser("meta-llama/Llama-3.1-70B") == "llama3_json"
    assert _vllm_tool_parser("mistralai/Mistral-7B") == "mistral"
    assert _vllm_tool_parser("org/unknown-model") == "hermes"


def test_prompt_backend_selection(monkeypatch):
    """Backend chooser maps numbers/names to vllm|ollama and keeps current on junk."""
    import michael.cli as cli
    cases = [("2", "vllm", "ollama"), ("1", "ollama", "vllm"),
             ("ollama", "vllm", "ollama"), ("zzz", "vllm", "vllm")]
    for answer, current, expected in cases:
        monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: answer)
        assert cli._prompt_backend_selection(current) == expected


def test_gpu_py_resolver_selects_available_interpreter():
    """_GPU_PY is valid POSIX sh and resolves "$PY" to a real interpreter."""
    import os
    import subprocess
    from michael.backends import _GPU_PY
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    cp = subprocess.run(
        ["bash", "-c", _GPU_PY + 'echo "$PY"'],
        capture_output=True, text=True, timeout=30,
    )
    assert cp.returncode == 0, cp.stderr
    py = cp.stdout.strip().split("\n")[-1]
    assert py, f"resolver produced no interpreter (stderr={cp.stderr!r})"
    assert os.path.exists(py) or shutil.which(py)


def test_stop_vllm_cmd_pattern_excludes_own_command_line():
    """The kill pattern must match a real vLLM process but NOT the stop
    command's own argv, otherwise pkill self-terminates its shell."""
    import re
    from michael.backends import _stop_vllm_cmd
    cmd = _stop_vllm_cmd()
    assert "pkill" in cmd
    found = re.search(r"pkill -f '([^']+)'", cmd)
    assert found, cmd
    pattern = found.group(1)
    # self-exclusion: the pattern does not match the command line it lives in
    assert re.search(pattern, cmd) is None
    # but it still matches a genuine running vLLM server
    assert re.search(pattern, "python -m vllm.entrypoints.openai.api_server --model x")


# ---- security: forge_tool requires confirmation -------------------------

import michael.agent as agent
import michael.tools as tools
from michael.config import Config


def test_forge_tool_is_not_auto_exec():
    # forge_tool imports LLM-supplied code into this process; it must pass
    # through the confirmation gate rather than auto-executing.
    assert "forge_tool" not in michael_globals.AUTO_EXEC_TOOLS


def test_forge_tool_rejected_writes_nothing(home, workspace, monkeypatch):
    p = m.create_project("forge-no", workspace)
    monkeypatch.setattr(tools, "confirm_tool_call", lambda n, a, pr: ("no", a))
    pending = tools.PendingChanges()
    args = {"name": "evil", "code": "import os\nos.system('echo pwned')\n"}
    result = tools.dispatch_tool_call("forge_tool", args, p, Config(), None, pending)
    assert result == "[user rejected this tool call]"
    assert not (pathlib.Path(p.path) / "tools" / "evil.py").exists()


def test_forge_tool_accepted_creates_file(home, workspace, monkeypatch):
    p = m.create_project("forge-yes", workspace)
    monkeypatch.setattr(tools, "confirm_tool_call", lambda n, a, pr: ("yes", a))
    pending = tools.PendingChanges()
    code = (
        "TOOL_SCHEMA = {'type': 'function', 'function': {'name': 'greet'}}\n"
        "def greet(**kwargs):\n    return 'hi'\n"
    )
    result = tools.dispatch_tool_call(
        "forge_tool", {"name": "greet", "code": code}, p, Config(), None, pending
    )
    assert "created" in result
    assert (pathlib.Path(p.path) / "tools" / "greet.py").is_file()


# ---- agent loop: context windowing --------------------------------------

def _grp(i):
    return [
        {"role": "assistant", "content": "a" * 1000,
         "tool_calls": [{"id": f"c{i}", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": "t" * 1000},
    ]


def _well_formed(msgs):
    # every 'tool' message must be preceded somewhere earlier by an assistant.
    seen_assistant = False
    for mm in msgs:
        if mm["role"] == "assistant":
            seen_assistant = True
        if mm["role"] == "tool" and not seen_assistant:
            return False
    return True


def test_window_messages_keeps_all_under_budget():
    msgs = [{"role": "system", "content": "H"}, {"role": "user", "content": "go"}]
    for i in range(3):
        msgs += _grp(i)
    out, dropped = agent._window_messages(msgs, 10_000_000)
    assert dropped == 0
    assert out == msgs


def test_window_messages_trims_oldest_groups():
    msgs = [{"role": "system", "content": "H"}, {"role": "user", "content": "go"}]
    for i in range(10):
        msgs += _grp(i)
    out, dropped = agent._window_messages(msgs, 5_000)
    assert dropped > 0
    # pinned header + prompt preserved
    assert out[0]["role"] == "system" and out[1]["role"] == "user"
    # window after the pinned pair starts on an assistant (group boundary)
    assert out[2]["role"] == "assistant"
    # never orphan a tool reply from its assistant call
    assert _well_formed(out)
    # newest group is always retained
    assert out[-1]["tool_call_id"] == "c9"


# ---- recon report: durable per-run persistence --------------------------

import json as _json
import types as _types


def _proj(tmp_path, slug="recone"):
    return _types.SimpleNamespace(path=str(tmp_path / slug), slug=slug)


def test_recon_report_writes_raw_and_report(tmp_path):
    proj = _proj(tmp_path)
    captured = [
        {"tool": "port_scan", "args": {"target": "example.com"},
         "result": "22/tcp open ssh\n80/tcp open http"},
        {"tool": "dir_enum", "args": {"base": "http://example.com"},
         "result": "/admin (403)"},
    ]
    agent._write_recon_report(proj, Config(), captured, reason="committed")

    recon = tmp_path / "recone" / "recon"
    raw = (recon / "raw.jsonl").read_text().strip().splitlines()
    assert len(raw) == 2
    rec0 = _json.loads(raw[0])
    assert rec0["tool"] == "port_scan"
    assert "22/tcp" in rec0["result"]
    assert rec0["reason"] == "committed"

    report = (recon / "report.md").read_text()
    assert "port_scan" in report
    assert "22/tcp open ssh" in report
    assert "tool results captured: 2" in report


def test_recon_report_flags_empty_run_loudly(tmp_path):
    proj = _proj(tmp_path)
    agent._write_recon_report(proj, Config(), [], reason="no-tool-exit")
    report = (tmp_path / "recone" / "recon" / "report.md").read_text()
    assert "NO DATA CAPTURED" in report
    assert "tool results captured: 0" in report


def test_recon_report_appends_across_runs(tmp_path):
    proj = _proj(tmp_path)
    agent._write_recon_report(
        proj, Config(), [{"tool": "a", "args": {}, "result": "r1"}], reason="committed")
    agent._write_recon_report(
        proj, Config(), [{"tool": "b", "args": {}, "result": "r2"}], reason="max-turns")
    raw = (tmp_path / "recone" / "recon" / "raw.jsonl").read_text().strip().splitlines()
    assert len(raw) == 2
    # report.md reflects the most recent run only
    report = (tmp_path / "recone" / "recon" / "report.md").read_text()
    assert "exit reason: max-turns" in report


# ---- analyst: telemetry, features, judgment, corpus + scorecard ----------

from michael.config import ModelProfile


def test_completion_response_carries_usage_and_finish_reason():
    # The per-turn telemetry depends on these fields surviving on the response
    # dataclasses — guard against a refactor silently dropping them.
    from michael.backends import _CompletionResponse, _Choice
    r = _CompletionResponse(
        choices=[_Choice(content="hi", tool_calls=None, finish_reason="stop")],
        usage={"total_tokens": 7},
    )
    assert r.usage["total_tokens"] == 7
    assert r.choices[0].finish_reason == "stop"


def test_compute_run_features_counts_and_recovery():
    import michael.analyst as analyst
    events = [
        {"type": "agent.started", "payload": {}},
        {"type": "turn.telemetry", "payload": {"turn": 1, "latency_ms": 100,
                                               "total_tokens": 40, "tool_calls": ["write_file"]}},
        {"type": "tool.verify_failed", "payload": {}},
        {"type": "turn.telemetry", "payload": {"turn": 2, "latency_ms": 200,
                                               "total_tokens": 60, "tool_calls": ["apply_patch"]}},
        {"type": "tool.executed", "payload": {}},
    ]
    captured = [
        {"tool": "x", "args": {}, "result": "error: boom"},
        {"tool": "y", "args": {}, "result": "ok"},
    ]
    f = analyst.compute_run_features(events, captured, reason="committed")
    assert f["committed"] is True
    assert f["turns_used"] == 2
    assert f["verify_failures"] == 1
    assert f["recovery_after_failure"] is True   # failure then later tool.executed
    assert f["dead_end_turns"] == 1              # the "error: boom" result
    assert f["total_tokens"] == 100
    assert f["avg_latency_ms"] == 150
    assert f["tool_histogram"] == {"write_file": 1, "apply_patch": 1}


def test_compute_run_features_windows_to_last_run():
    import michael.analyst as analyst
    events = [
        {"type": "agent.started", "payload": {}},
        {"type": "tool.verify_failed", "payload": {}},   # belongs to the OLD run
        {"type": "agent.started", "payload": {}},
        {"type": "turn.telemetry", "payload": {"turn": 1, "tool_calls": []}},
    ]
    f = analyst.compute_run_features(events, [], reason="no-tool-exit")
    assert f["verify_failures"] == 0   # old-run failure excluded by the window
    assert f["turns_used"] == 1


def test_extract_json_handles_fenced_and_prose():
    import michael.analyst as analyst
    assert analyst._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert analyst._extract_json('here you go: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert analyst._extract_json("no json here") is None


def test_normalize_records_enforces_enum_and_stamps_metadata():
    import michael.analyst as analyst
    proj = _types.SimpleNamespace(slug="s", mode="recon")
    recs, dropped = analyst._normalize_records(
        [{"class": "recovery", "turn": 1}, {"class": "bogus"}, {"turn": 2}],
        run_id="s:t", project=proj, features={"committed": True},
        analyst_model="analyst", ts="2026-06-01T00:00:00+00:00",
    )
    assert len(recs) == 1 and dropped == 2
    assert recs[0]["schema"] == "michael.dataset.v1"
    assert recs[0]["run_id"] == "s:t"
    assert recs[0]["project"] == "s"
    assert recs[0]["features"] == {"committed": True}


class _StubLLMClient:
    """Stand-in for backends.LLMClient: returns canned analyst JSON."""
    canned = ""

    def __init__(self, *a, **k):
        pass

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **k):
        return _types.SimpleNamespace(
            choices=[_types.SimpleNamespace(content=type(self).canned)],
            usage={},
        )


def test_run_analyst_writes_corpus_and_scorecard(home, workspace, monkeypatch):
    import michael.analyst as analyst
    import michael.backends as backends
    p = m.create_project("an-run", workspace)
    # Seed a finished run in the project event log.
    m.append_event("agent.started", {"model": "god"}, project=p)
    m.append_event("prompt.sent", {"prompt": "fix the bug"}, project=p)
    m.append_event("turn.telemetry", {"turn": 1, "latency_ms": 120,
                                       "finish_reason": "tool_calls", "total_tokens": 50,
                                       "tool_calls": ["write_file"], "n_tool_calls": 1,
                                       "content_chars": 10, "dropped_context_msgs": 0}, project=p)
    m.append_event("assistant.message", {"chars": 10, "turn": 1, "text": "patching"}, project=p)

    _StubLLMClient.canned = _json.dumps({
        "records": [
            {"turn": 1, "class": "productive_edit",
             "input": {"goal": "fix the bug"},
             "action": {"tool_calls": [{"name": "write_file", "args": {}}]},
             "outcome": {"verify_rc": 0, "delta_mismatch": False},
             "label_rationale": "clean patch", "quality": 0.9},
            {"turn": 2, "class": "NOT_A_REAL_CLASS"},  # must be dropped
        ],
        "scorecard": {"dominant_class": "successful_commit",
                      "scores": {"goal_completion": 0.9}, "overall": 0.85,
                      "narrative": "good run"},
    })
    monkeypatch.setattr(backends, "LLMClient", _StubLLMClient)

    cfg = Config(
        models={"analyst": ModelProfile(endpoint="http://x/v1", served_model_name="judge")},
        analyst_enabled=True,
    )
    analyst.run_analyst(p, cfg, reason="committed",
                        captured=[{"tool": "write_file", "args": {}, "result": "ok"}])

    # Per-project corpus
    recs = (pathlib.Path(p.path) / "dataset" / "records.jsonl").read_text().strip().splitlines()
    assert len(recs) == 1   # the bogus-class record was dropped
    rec0 = _json.loads(recs[0])
    assert rec0["class"] == "productive_edit"
    assert rec0["run_id"].startswith("an-run:")
    assert rec0["project"] == "an-run"
    # Cross-project corpus copy
    assert (michael_globals.GLOBAL_DATASET_DIR / "an-run.jsonl").is_file()
    # Scorecard
    scs = list((pathlib.Path(p.path) / "scorecards").glob("*.json"))
    assert len(scs) == 1
    sc = _json.loads(scs[0].read_text())
    assert sc["dominant_class"] == "successful_commit"
    assert sc["features"]["committed"] is True   # deterministic features attached
    assert sc["n_records"] == 1
    # Events
    events = m.iter_events(p.events_path)
    assert any(e["type"] == "run.scored" for e in events)
    assert any(e["type"] == "analyst.completed" for e in events)


def test_run_analyst_degrades_when_model_unreachable(home, workspace, monkeypatch):
    import michael.analyst as analyst
    p = m.create_project("an-degrade", workspace)
    m.append_event("agent.started", {"model": "god"}, project=p)
    # _call_analyst returns None (e.g. GPU down) -> features-only scorecard, no records.
    monkeypatch.setattr(analyst, "_call_analyst", lambda *a, **k: None)
    cfg = Config(
        models={"analyst": ModelProfile(endpoint="http://x/v1", served_model_name="judge")},
        analyst_enabled=True,
    )
    analyst.run_analyst(p, cfg, reason="committed", captured=[])
    assert not (pathlib.Path(p.path) / "dataset").exists()
    scs = list((pathlib.Path(p.path) / "scorecards").glob("*.json"))
    assert len(scs) == 1
    sc = _json.loads(scs[0].read_text())
    assert sc["scores"] is None
    assert sc["features"]["committed"] is True


def test_recon_report_runs_analyst_only_when_enabled(home, workspace):
    # Default Config has analyst_enabled=False -> recon files written, no analyst output.
    p = m.create_project("an-off", workspace)
    agent._write_recon_report(
        p, Config(), [{"tool": "x", "args": {}, "result": "r"}], reason="committed")
    assert (pathlib.Path(p.path) / "recon" / "raw.jsonl").is_file()
    assert not (pathlib.Path(p.path) / "dataset").exists()
    assert not (pathlib.Path(p.path) / "scorecards").exists()
    events = m.iter_events(p.events_path)
    assert not any(e["type"] == "run.scored" for e in events)
