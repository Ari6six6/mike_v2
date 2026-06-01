"""The analyst role — post-run judgment, training corpus, and scorecards.

After every run (when ``analyst_enabled`` is set), the analyst reads the just-
finished run from the project event log plus the captured tool results, computes
a set of *deterministic* heuristic features, and asks a dedicated ``analyst``
model to judge the run. The analyst LLM is the judge: it assigns each
instructive turn to a fixed data class and emits

  1. a classed training corpus  (``<project>/dataset/records.jsonl`` and a
     cross-project copy under ``~/.michael/dataset/<slug>.jsonl``) — input→
     action→outcome records shaped for SFT / preference data, and
  2. a per-run scorecard         (``<project>/scorecards/<run_id>.json`` plus a
     ``run.scored`` event).

The heuristic features are *evidence the analyst reads*, never the verdict.

This module never raises into the agent loop: ``run_analyst`` wraps everything
in a broad guard. The run it analyses has already produced its real artifacts
and committed before the analyst is invoked, so an analyst failure (including an
unreachable GPU on an air-gapped box) only costs the scorecard — the run stands.
When the model is unreachable, the analyst still writes a *features-only*
scorecard (``scores: null``) so scoring degrades rather than disappears.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Optional

import michael.globals as G
from michael.config import Config
from michael.project import Project, append_event, iter_events

# The fixed set of data classes the analyst must choose from. Enforced on every
# emitted training record; unknown classes are dropped loudly.
CLASSES: frozenset[str] = frozenset(
    {
        "successful_commit",
        "productive_edit",
        "recovery",
        "dead_end",
        "wasted_turn",
        "hallucinated_path",
        "rejected_by_user",
        "max_turns_exhausted",
        "verify_failure_loop",
        "clean_recon",
    }
)

DATASET_SCHEMA = "michael.dataset.v1"
SCORECARD_SCHEMA = "michael.scorecard.v1"

# Truncation budgets for the digest sent to the analyst model.
_MAX_ASSISTANT_CHARS = 1_500
_MAX_RESULT_CHARS = 1_500
_MAX_CAPTURED = 40


# ---------------------------------------------------------------------------
# Heuristic features (deterministic — evidence, not verdict)
# ---------------------------------------------------------------------------


def _run_window(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the slice of events belonging to the most recent run.

    The window starts at the last ``agent.started`` and runs to the end of the
    log. At analyst time ``agent.ended`` has not been written yet, and the
    analyst's own ``run.scored`` / ``analyst.completed`` events are emitted only
    *after* features are computed — so neither leaks into the window, and the
    analyst never re-analyses its own output.
    """
    start = 0
    for i, ev in enumerate(events):
        if ev.get("type") == "agent.started":
            start = i
    return events[start:]


def _is_error_result(result: str) -> bool:
    r = (result or "").strip().lower()
    if not r:
        return True
    return r.startswith("error") or r.startswith("[tool call error]") or "traceback (most recent call last)" in r


def compute_run_features(
    events: list[dict[str, Any]],
    captured: list[dict[str, Any]],
    *,
    reason: str,
) -> dict[str, Any]:
    """Fold the run window + captured results into deterministic features.

    Every feature maps to an event already emitted by the agent loop or to the
    captured tool results. ``reason`` is the exit reason passed by the agent
    loop (``committed`` / ``no-tool-exit`` / ``aborted`` / ``error`` /
    ``max-turns``) — authoritative since ``agent.ended`` isn't logged yet.
    """
    window = _run_window(events)

    def _count(t: str) -> int:
        return sum(1 for e in window if e.get("type") == t)

    telemetry = [e for e in window if e.get("type") == "turn.telemetry"]

    total_tokens = 0
    latencies: list[int] = []
    tool_histogram: dict[str, int] = {}
    for e in telemetry:
        p = e.get("payload", {})
        tt = p.get("total_tokens")
        if isinstance(tt, (int, float)):
            total_tokens += int(tt)
        lat = p.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(int(lat))
        for name in p.get("tool_calls", []) or []:
            tool_histogram[name] = tool_histogram.get(name, 0) + 1

    turns_used = max((e.get("payload", {}).get("turn", 0) for e in telemetry), default=0)

    verify_failures = _count("tool.verify_failed")
    delta_mismatches = _count("tool.delta_mismatch")
    rejected_tools = _count("tool.rejected")

    # recovery: a failure/rejection somewhere in the window, followed later by a
    # successful tool.executed (the loop dug itself back out).
    recovery = False
    failure_seen = False
    for e in window:
        t = e.get("type")
        if t in ("tool.verify_failed", "tool.delta_mismatch", "tool.rejected"):
            failure_seen = True
        elif t == "tool.executed" and failure_seen:
            recovery = True
            break

    dead_end_turns = sum(1 for rec in captured if _is_error_result(rec.get("result", "")))

    return {
        "exit_reason": reason,
        "committed": reason == "committed",
        "turns_used": turns_used,
        "max_turns_hit": reason == "max-turns",
        "verify_failures": verify_failures,
        "delta_mismatches": delta_mismatches,
        "rejected_tools": rejected_tools,
        "recovery_after_failure": recovery,
        "dead_end_turns": dead_end_turns,
        "context_trims": _count("context.trimmed"),
        "total_tokens": total_tokens,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "tool_histogram": tool_histogram,
        "n_captured": len(captured),
    }


# ---------------------------------------------------------------------------
# Digest + prompt construction
# ---------------------------------------------------------------------------


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n… [truncated {len(s) - n} chars]"


def _build_digest(
    window: list[dict[str, Any]], captured: list[dict[str, Any]], features: dict[str, Any]
) -> str:
    """A compact, turn-indexed run summary for the analyst model."""
    prompts = [
        e.get("payload", {}).get("prompt", "")
        for e in window
        if e.get("type") == "prompt.sent"
    ]
    # turn -> assistant text
    assistants = {
        e.get("payload", {}).get("turn"): e.get("payload", {}).get("text", "")
        for e in window
        if e.get("type") == "assistant.message"
    }
    telemetry = [e.get("payload", {}) for e in window if e.get("type") == "turn.telemetry"]

    lines: list[str] = ["## User prompt(s)"]
    for p in prompts:
        lines.append(f"- {_truncate(p, 500)}")

    lines.append("\n## Turns")
    for p in telemetry:
        turn = p.get("turn")
        lines.append(f"\n### turn {turn}")
        lines.append(
            f"finish_reason={p.get('finish_reason')} "
            f"latency_ms={p.get('latency_ms')} "
            f"total_tokens={p.get('total_tokens')} "
            f"tool_calls={p.get('tool_calls')}"
        )
        atext = assistants.get(turn)
        if atext:
            lines.append("assistant:")
            lines.append(_truncate(atext, _MAX_ASSISTANT_CHARS))

    lines.append("\n## Tool results (captured, truncated)")
    for rec in captured[:_MAX_CAPTURED]:
        lines.append(f"\n- tool={rec.get('tool')} args={json.dumps(rec.get('args', {}))[:300]}")
        lines.append(_truncate(rec.get("result", ""), _MAX_RESULT_CHARS))
    if len(captured) > _MAX_CAPTURED:
        lines.append(f"\n… [{len(captured) - _MAX_CAPTURED} more results omitted]")

    lines.append("\n## Deterministic features (evidence — your verdict must be consistent with these)")
    lines.append(json.dumps(features, indent=2, sort_keys=True))

    return "\n".join(lines)


def _load_contract(cfg: Config) -> str:
    """Read the analyst scripture directly (a tight, single-file contract).

    Read directly rather than via load_scripture so the analyst gets only its
    own contract, not the recon/model/build scripture meant for the agent loop.
    """
    p = pathlib.Path(cfg.scripture_dir).expanduser() / "analyst.md"
    try:
        return p.read_text(errors="replace")
    except OSError:
        return (
            "You are the analyst. Judge the run. Reply with a single JSON object "
            '{"records": [...], "scorecard": {...}}.'
        )


# ---------------------------------------------------------------------------
# Output parsing / validation
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Pull the first balanced top-level JSON object out of model text.

    Models often wrap JSON in prose or code fences. Try a direct parse first,
    then fall back to scanning for the outermost balanced ``{...}``.
    """
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_records(
    raw: Any,
    *,
    run_id: str,
    project: Project,
    features: dict[str, Any],
    analyst_model: str,
    ts: str,
) -> tuple[list[dict[str, Any]], int]:
    """Validate analyst records against the class enum; stamp run metadata.

    Returns (valid_records, n_dropped).
    """
    if not isinstance(raw, list):
        return [], 0
    out: list[dict[str, Any]] = []
    dropped = 0
    for rec in raw:
        if not isinstance(rec, dict):
            dropped += 1
            continue
        cls = rec.get("class")
        if cls not in CLASSES:
            G.err.print(f"[yellow]analyst: dropping record with invalid class {cls!r}[/]")
            dropped += 1
            continue
        rec.update(
            {
                "schema": DATASET_SCHEMA,
                "run_id": run_id,
                "project": project.slug,
                "mode": project.mode,
                "analyst_model": analyst_model,
                "ts": ts,
            }
        )
        rec.setdefault("features", features)
        out.append(rec)
    return out, dropped


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _append_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_run_id(run_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in run_id)


def _write_scorecard(project: Project, scorecard: dict[str, Any]) -> pathlib.Path:
    sc_dir = pathlib.Path(project.path) / "scorecards"
    sc_dir.mkdir(parents=True, exist_ok=True)
    path = sc_dir / f"{_safe_run_id(scorecard['run_id'])}.json"
    path.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_analyst(
    project: Project,
    cfg: Config,
    *,
    reason: str,
    captured: list[dict[str, Any]],
) -> None:
    """Judge the just-finished run; write a training corpus + scorecard.

    Never raises into the caller. On model-unreachable, degrades to a
    features-only scorecard so scoring survives offline.
    """
    try:
        profile = cfg.models.get("analyst")
        if profile is None or not profile.endpoint or not profile.served_model_name:
            append_event(
                "analyst.failed",
                {"reason": "models.analyst profile incomplete (need endpoint + served_model_name)"},
                project=project,
            )
            return

        events = iter_events(project.events_path)
        window = _run_window(events)
        features = compute_run_features(events, captured, reason=reason)

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        started = next(
            (e.get("ts") for e in reversed(window) if e.get("type") == "agent.started"),
            ts,
        )
        run_id = f"{project.slug}:{started}"

        digest = _build_digest(window, captured, features)
        contract = _load_contract(cfg)

        text = _call_analyst(cfg, profile, contract, digest)
        if text is None:
            _write_degraded_scorecard(project, run_id, features, reason, "analyst")
            return

        parsed = _extract_json(text)
        if parsed is None:
            append_event(
                "analyst.failed",
                {"reason": "analyst response was not parseable JSON", "run_id": run_id},
                project=project,
            )
            _write_degraded_scorecard(project, run_id, features, reason, "analyst")
            return

        records, dropped = _normalize_records(
            parsed.get("records"),
            run_id=run_id,
            project=project,
            features=features,
            analyst_model="analyst",
            ts=ts,
        )
        if records:
            _append_jsonl(pathlib.Path(project.path) / "dataset" / "records.jsonl", records)
            _append_jsonl(G.GLOBAL_DATASET_DIR / f"{project.slug}.jsonl", records)

        sc_in = parsed.get("scorecard") if isinstance(parsed.get("scorecard"), dict) else {}
        scorecard = {
            "schema": SCORECARD_SCHEMA,
            "run_id": run_id,
            "project": project.slug,
            "mode": project.mode,
            "exit_reason": reason,
            "dominant_class": sc_in.get("dominant_class"),
            "scores": sc_in.get("scores"),
            "overall": sc_in.get("overall"),
            "narrative": sc_in.get("narrative", ""),
            "features": features,  # deterministic, authoritative — overrides any LLM copy
            "n_records": len(records),
            "analyst_model": "analyst",
            "ts": ts,
        }
        sc_path = _write_scorecard(project, scorecard)

        append_event(
            "run.scored",
            {
                "run_id": run_id,
                "dominant_class": scorecard["dominant_class"],
                "overall": scorecard["overall"],
                "n_records": len(records),
                "dropped_records": dropped,
                "scorecard": str(sc_path),
            },
            project=project,
        )
        append_event(
            "analyst.completed",
            {"run_id": run_id, "n_records": len(records), "dropped_records": dropped},
            project=project,
        )
        G.console.print(
            f"[dim]analyst: {len(records)} record(s), scorecard → {sc_path}[/]"
        )
    except Exception as exc:  # never break the run
        G.err.print(f"[yellow]analyst failed (run is unaffected):[/] {exc}")
        try:
            append_event("analyst.failed", {"reason": str(exc)}, project=project)
        except Exception:
            pass


def _call_analyst(cfg: Config, profile: Any, contract: str, digest: str) -> Optional[str]:
    """Call the analyst model as a pure text oracle. Returns text or None on failure.

    Mirrors spawn_specialist: lazy import so tests can monkeypatch
    michael.backends.LLMClient, ensure the tunnel for the analyst's GPU.
    """
    from michael.backends import LLMClient

    gpu_key = profile.gpu_name or "analyst"
    gpu = cfg.get_gpu(gpu_key)
    if gpu and gpu.ssh_host:
        from michael.backends import _ensure_tunnel

        try:
            _ensure_tunnel(gpu_key, gpu)
        except Exception as exc:
            G.err.print(f"[yellow]analyst: tunnel for '{gpu_key}' failed: {exc}[/]")
            return None

    try:
        client = LLMClient(profile.endpoint)
        resp = client.chat.completions.create(
            model=profile.served_model_name,
            messages=[
                {"role": "system", "content": contract},
                {"role": "user", "content": digest},
            ],
            timeout=float(profile.request_timeout_s or 120),
        )
        return (resp.choices[0].content or "").strip()
    except Exception as exc:
        G.err.print(f"[yellow]analyst: model call failed: {exc}[/]")
        return None


def _write_degraded_scorecard(
    project: Project,
    run_id: str,
    features: dict[str, Any],
    reason: str,
    analyst_model: str,
) -> None:
    """Features-only scorecard when the analyst model can't be reached/parsed."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    scorecard = {
        "schema": SCORECARD_SCHEMA,
        "run_id": run_id,
        "project": project.slug,
        "mode": project.mode,
        "exit_reason": reason,
        "dominant_class": None,
        "scores": None,
        "overall": None,
        "narrative": "analyst model unavailable — deterministic features only",
        "features": features,
        "n_records": 0,
        "analyst_model": analyst_model,
        "ts": ts,
    }
    try:
        sc_path = _write_scorecard(project, scorecard)
        append_event(
            "run.scored",
            {
                "run_id": run_id,
                "dominant_class": None,
                "overall": None,
                "n_records": 0,
                "degraded": True,
                "scorecard": str(sc_path),
            },
            project=project,
        )
    except Exception as exc:
        G.err.print(f"[yellow]analyst: degraded scorecard write failed: {exc}[/]")
