"""Tests for michael.router and toolbox.route_task."""
from __future__ import annotations

import pytest

from michael.config import Config, ModelProfile
from michael.router import route, _find_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**profiles: dict) -> Config:
    """Build a minimal Config with the given model profiles."""
    models = {
        name: ModelProfile(**kw) for name, kw in profiles.items()
    }
    return Config(models=models, default_model=next(iter(models)))


def _profile(endpoint: str = "http://localhost:11434/v1",
             tool_uncapable: bool = False) -> dict:
    return {"endpoint": endpoint, "served_model_name": "test-model",
            "tool_uncapable": tool_uncapable}


# ---------------------------------------------------------------------------
# route() — heuristic dispatch
# ---------------------------------------------------------------------------

def test_route_exploit_keywords_pick_junior():
    cfg = _cfg(god=_profile(), junior=_profile())
    assert route("write an exploit for CVE-2024-1234", cfg) == "junior"


def test_route_payload_keyword_picks_junior():
    cfg = _cfg(god=_profile(), junior=_profile())
    assert route("generate a reverse shell payload for this target", cfg) == "junior"


def test_route_recon_keywords_pick_god():
    cfg = _cfg(god=_profile(), junior=_profile())
    assert route("enumerate subdomains and run a port scan", cfg) == "god"


def test_route_log_keywords_pick_analyst():
    cfg = _cfg(god=_profile(), analyst=_profile())
    assert route("triage this Wazuh alert from the SIEM", cfg) == "analyst"


def test_route_unmatched_task_falls_back_to_default():
    cfg = _cfg(god=_profile(), junior=_profile())
    assert route("refactor the parser module to handle unicode", cfg) == "god"


def test_route_case_insensitive():
    cfg = _cfg(god=_profile(), junior=_profile())
    assert route("EXPLOIT this vulnerability", cfg) == "junior"


def test_route_no_profiles_falls_back_gracefully():
    # No models at all — returns whatever default_model says ("god").
    cfg = Config(models={}, default_model="god")
    result = route("write an exploit", cfg)
    assert result == "god"


# ---------------------------------------------------------------------------
# tool_uncapable profiles are skipped
# ---------------------------------------------------------------------------

def test_route_skips_tool_uncapable_profile():
    # junior exists but is tool_uncapable — should not be selected for exploit tasks.
    cfg = _cfg(god=_profile(), junior=_profile(tool_uncapable=True))
    assert route("write an exploit for CVE-2024-1234", cfg) == "god"


def test_route_skips_profile_with_no_endpoint():
    # junior exists but has no endpoint — not selectable.
    cfg = _cfg(
        god=_profile(),
        junior={"endpoint": None, "served_model_name": "x", "tool_uncapable": False},
    )
    assert route("write an exploit for CVE-2024-1234", cfg) == "god"


# ---------------------------------------------------------------------------
# _find_profile helpers
# ---------------------------------------------------------------------------

def test_find_profile_exact_match():
    cfg = _cfg(god=_profile(), junior=_profile())
    assert _find_profile("junior", cfg) == "junior"


def test_find_profile_substring_match():
    cfg = _cfg(god=_profile(), junior_v2=_profile())
    assert _find_profile("junior", cfg) == "junior_v2"


def test_find_profile_no_match_returns_empty():
    cfg = _cfg(god=_profile())
    assert _find_profile("analyst", cfg) == ""


# ---------------------------------------------------------------------------
# router_enabled=False path via route_task tool
# ---------------------------------------------------------------------------

def test_route_task_disabled_returns_error(monkeypatch):
    import toolbox.route_task as rt
    from michael.config import Config

    cfg = Config(models={"god": ModelProfile(endpoint="http://x/v1",
                                              served_model_name="m")},
                 default_model="god",
                 router_enabled=False)
    monkeypatch.setattr(rt, "_load_config", lambda: cfg, raising=False)

    # Patch Config.load() so the tool uses our cfg.
    monkeypatch.setattr("michael.config.Config.load", lambda: cfg)

    result = rt.route_task(task="do something", prompt="do something")
    assert "router_enabled is false" in result


def test_route_task_enabled_calls_specialist(monkeypatch):
    import toolbox.route_task as rt
    from michael.config import Config

    cfg = Config(
        models={
            "god": ModelProfile(endpoint="http://localhost:11434/v1",
                                served_model_name="hermes"),
            "junior": ModelProfile(endpoint="http://localhost:11435/v1",
                                   served_model_name="deephat"),
        },
        default_model="god",
        router_enabled=True,
    )
    monkeypatch.setattr("michael.config.Config.load", lambda: cfg)

    # Stub _ensure_tunnel to a no-op (no real GPU).
    monkeypatch.setattr("michael.backends._ensure_tunnel", lambda *a, **k: None,
                        raising=False)

    # Stub LLMClient so no network call is made.
    class _FakeChoice:
        content = "exploit code here"

    class _FakeResp:
        choices = [_FakeChoice()]

    class _FakeCompletion:
        def create(self, **_kw):
            return _FakeResp()

    class _FakeChat:
        completions = _FakeCompletion()

    class _FakeClient:
        def __init__(self, _endpoint):
            self.chat = _FakeChat()

    monkeypatch.setattr("michael.backends.LLMClient", _FakeClient)

    result = rt.route_task(
        task="write an exploit for CVE-2024-1234",
        prompt="Write shellcode targeting the vuln in full detail.",
    )
    assert "[routed to: junior]" in result
    assert "exploit code here" in result
