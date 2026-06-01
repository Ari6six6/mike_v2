"""Router — heuristic dispatch from task description to model profile.

route(task, cfg) -> str

Returns the name of the model profile best suited for the task. The decision
is purely rule-based: keyword matching against the task text, filtered by
profile capability (tool_uncapable profiles are oracle-only). Falls back to
the default/senior profile when no rule matches or when router_enabled=False.

TODO: add learned-bias path once analyst scorecards accumulate.
  - Read <project>/scorecards/*.json + dataset/records.jsonl
  - Weight keyword scores by historical success rate per (task_class, profile)
  - Gate behind cfg.router_enabled (already exists) — default off
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from michael.config import Config


# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------
# Each rule: (compiled pattern, profile_hint).
# profile_hint is a substring that must appear in a profile name to qualify
# (e.g. "junior" matches a profile literally named "junior").
# Rules are evaluated in order; first match wins.

_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(exploit|payload|shellcode|cve|rce|lfi|sqli|xss|injection|bypass|"
            r"privilege.escal|privesc|reverse.shell|bind.shell|obfuscat|evasion|"
            r"malware|dropper|implant)\b",
            re.IGNORECASE,
        ),
        "junior",
    ),
    (
        re.compile(
            r"\b(recon|enumerate|scan|fingerprint|dns|nmap|nessus|shodan|censys|"
            r"osint|passive|subdomain|port.scan|service.detect)\b",
            re.IGNORECASE,
        ),
        "god",
    ),
    (
        re.compile(
            r"\b(log|alert|siem|soc|triage|wazuh|splunk|elk|incident|ioc|"
            r"threat.intel|sigma|yara|detection.rule)\b",
            re.IGNORECASE,
        ),
        "analyst",
    ),
]


def route(task: str, cfg: "Config") -> str:
    """Return the profile name that should handle *task*.

    Selection logic:
    1. Walk _RULES in order; find the first pattern that matches.
    2. If the hinted profile exists in cfg.models and has an endpoint, use it.
    3. Otherwise fall through to the default profile.

    tool_uncapable profiles are never chosen for tasks that imply tool use
    (any task that isn't pure text generation). For now we treat all router-
    selected tasks as potentially tool-using, so tool_uncapable profiles are
    skipped unless they are the only available profile or explicitly hinted
    by name AND have no tool-requiring signals in the task.
    """
    default_name = cfg.default_model or (next(iter(cfg.models)) if cfg.models else "god")

    for pattern, hint in _RULES:
        if not pattern.search(task):
            continue
        # Find the best matching profile for this hint.
        candidate = _find_profile(hint, cfg)
        if candidate:
            return candidate
        # Hint matched but no suitable profile — fall through to next rule.

    return default_name


def _find_profile(hint: str, cfg: "Config") -> str:
    """Return a profile name whose key contains *hint* and has a set endpoint.

    Prefers an exact match, then a substring match. Returns "" if nothing
    qualifies or if the matched profile is tool_uncapable (oracle-only).
    """
    # Exact match first.
    if hint in cfg.models:
        p = cfg.models[hint]
        if p.endpoint and not p.tool_uncapable:
            return hint

    # Substring match.
    for name, p in cfg.models.items():
        if hint in name and p.endpoint and not p.tool_uncapable:
            return name

    return ""
