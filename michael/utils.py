"""Filesystem snapshot and context-package builder (headers H1–H4)."""
from __future__ import annotations

import os
import pathlib
import re
from typing import TYPE_CHECKING, Any

import michael.globals as G
from michael.project import iter_events

if TYPE_CHECKING:
    from michael.project import Project


# ---------------------------------------------------------------------------
# Filesystem snapshot
# ---------------------------------------------------------------------------


def _is_text(path: pathlib.Path, sniff: int = 8192) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sniff)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def filesystem_snapshot(root: pathlib.Path) -> str:
    """Listing of the project tree + inlined contents for small text files."""
    root = root.resolve()
    listing_lines: list[str] = []
    text_files: list[tuple[pathlib.Path, int]] = []

    if not root.is_dir():
        return f"(project root does not exist: {root})"

    for dp, dirs, files in os.walk(root):
        dp_path = pathlib.Path(dp)
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in G.SKIP_DIRS]
        rel_dp = dp_path.relative_to(root)
        for fname in sorted(files):
            f = dp_path / fname
            try:
                size = f.stat().st_size
            except OSError:
                continue
            rel = (rel_dp / fname).as_posix() if str(rel_dp) != "." else fname
            listing_lines.append(f"{rel} ({size}b)")
            if size <= G.MAX_FILE_BYTES_INLINE and _is_text(f):
                text_files.append((f, size))

    parts: list[str] = []
    parts.append("Directory listing (relative to project root):")
    parts.append("\n".join(listing_lines) if listing_lines else "(empty)")
    parts.append("")
    parts.append(
        f"File contents (text only; per-file cap {G.MAX_FILE_BYTES_INLINE}b, "
        f"total cap {G.MAX_TOTAL_BYTES_INLINE}b):"
    )

    text_files.sort(key=lambda x: x[1])
    bodies: list[str] = []
    total = 0
    for f, size in text_files:
        rel = f.relative_to(root).as_posix()
        if total + size > G.MAX_TOTAL_BYTES_INLINE:
            bodies.append(f"==== {rel} (skipped: total cap reached) ====")
            continue
        try:
            content = f.read_text(errors="replace")
        except OSError:
            continue
        bodies.append(f"==== {rel} ({size}b) ====\n{content}")
        total += size

    parts.append("\n\n".join(bodies) if bodies else "(no text files inlined)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# H1 / H3 history builders
# ---------------------------------------------------------------------------


def _prompt_history_lines(project: "Project") -> list[str]:
    out: list[str] = []
    n = 0
    for ev in iter_events(project.events_path):
        if ev.get("type") == "prompt.sent":
            n += 1
            prompt = (ev.get("payload") or {}).get("prompt", "")
            out.append(f"[{n}] {prompt}")
    return out


def _action_log_lines(project: "Project") -> list[str]:
    out: list[str] = []
    n = 0
    for ev in iter_events(project.events_path):
        t = ev.get("type", "")
        p = ev.get("payload", {}) or {}
        if t == "tool.executed":
            n += 1
            line = f"[{n}] {p.get('summary', t)}"
            brief = p.get("brief_result", "")
            if brief:
                first_lines = "\n    ".join(brief.splitlines()[:4])
                line += f"\n    {first_lines}"
            out.append(line)
        elif t == "tool.rejected":
            n += 1
            out.append(f"[{n}] {p.get('summary', t)}  [REJECTED BY USER]")
        elif t == "tool.verify_failed":
            n += 1
            rc = p.get("verify_rc", "?")
            out.append(
                f"[{n}] {p.get('summary', t)}  [VERIFY FAILED rc={rc}, user not prompted]"
            )
        elif t == "tool.undone":
            n += 1
            out.append(
                f"[{n}] undone: {p.get('tool', '?')} ({p.get('trash_id', '?')})"
            )
    return out


# ---------------------------------------------------------------------------
# H4: Protocol Bible
# ---------------------------------------------------------------------------


_MODE_CONTEXT = {
    "recon": (
        "You are operating in RECON MODE.\n\n"
        "INTENTION: Produce the most comprehensive possible dataset on the target.\n"
        "Thoroughness is the metric — cover every available angle before stopping.\n"
        "A finding you skipped is a finding lost forever to this pipeline.\n\n"
        "TARGET: Defined in your MISSION or the user's prompt. Use target_define to\n"
        "formalize it. The target is the organizing principle for this entire run.\n\n"
        "YOUR OUTPUT FEEDS MODEL MODE. That stage can only work with what you give it.\n"
        "Gaps in your recon = gaps in the model = gaps in what gets built.\n\n"
        "OUTPUT DESTINATIONS:\n"
        "  targets/<domain>.md — structured findings (DNS, TLS, HTTP stack,\n"
        "    subdomains, IP intel, endpoints, secrets, version disclosures)\n"
        "  recon/raw.jsonl — auto-saved by the system, no action needed\n\n"
        "RULES:\n"
        "  - Use every applicable recon tool before stopping\n"
        "  - If a tool fails, note the failure in targets/<domain>.md —\n"
        "    absence of data is data\n"
        "  - Commit before exiting — staged writes not committed are discarded\n\n"
        "PIPELINE HANDOFF:\n"
        "  Before calling commit_changes, write a copy of targets/<domain>.md to\n"
        "  the Results path shown in project metadata, named <slug>-recon.md\n"
        "  (e.g. my-target-recon.md). This is how the model-mode project picks\n"
        "  up your work. The user will reference it by name."
    ),
    "model": (
        "You are operating in MODEL MODE.\n\n"
        "INTENTION: Produce the most precise possible model of the system described\n"
        "in your input data. Precision — not comprehensiveness — is the metric.\n"
        "One well-sourced fact is worth more than ten inferences.\n\n"
        "YOUR INPUT: One or more data files the user provides. Read them with\n"
        "read_file using the path the user specifies. Absolute paths work.\n"
        "Do not reach out to the network. Do not scan. Your raw material is on disk.\n\n"
        "YOUR OUTPUT FEEDS BUILD MODE. What you write here is the contract that build\n"
        "mode acts on. A false assumption there is worse than an acknowledged gap.\n\n"
        "OUTPUT: models/<name>-<version>.json via AppModel format.\n\n"
        "PRECISION RULES — non-negotiable:\n"
        "  1. If you have no evidence for a field, leave it empty or mark it unknown.\n"
        "     Never fill gaps with plausible-sounding guesses.\n"
        "  2. If a version is uncertain, say so: 'nginx 1.x — minor unconfirmed'.\n"
        "     Never write a specific value you cannot source.\n"
        "  3. Document what you DON'T know. Gaps are valuable information. Use the\n"
        "     notes field: 'auth flow not observed in input data — structure unknown'.\n"
        "  4. Source every key fact: cite which file and which section the value\n"
        "     came from. If you cannot cite it, do not include it.\n"
        "  5. The model is complete when all input data has been consumed and every\n"
        "     known gap is documented. Stop there.\n\n"
        "PIPELINE HANDOFF:\n"
        "  Before calling commit_changes, write the primary AppModel JSON to\n"
        "  the Results path shown in project metadata, named <slug>-model.json.\n"
        "  The build-mode project will read it from there."
    ),
    "build": (
        "You are operating in BUILD MODE.\n\n"
        "INTENTION: Build exactly what the user asks. Nothing more.\n\n"
        "YOUR PRIMARY INPUT: models/*.json — read the relevant AppModel first.\n"
        "It defines the target environment, stack, auth patterns, and endpoint\n"
        "signatures. Do not assume what you haven't read.\n\n"
        "YOUR TOOLS: Core tools only — write_file, apply_patch, read_file,\n"
        "run_shell, run_in_sandbox, commit_changes, forge_tool.\n"
        "No recon tools exist in this mode. If you need a reusable capability,\n"
        "forge it.\n\n"
        "RULES:\n"
        "  - Read the relevant AppModel before writing any code\n"
        "  - Test in run_in_sandbox before committing\n"
        "  - Commit when done and tested — not before\n"
        "  - Do not speculate about the target environment beyond what the model says"
    ),
}


def build_protocol(mode: str = "recon") -> str:
    """Header 4 — the protocol."""
    mode_context = _MODE_CONTEXT.get(mode, _MODE_CONTEXT["recon"])
    return "\n".join([
        "You are connected to the user's machine through Project Michael.",
        "Michael is event-sourced: every user prompt and every tool call you",
        "execute is logged. You are amnesiac across user prompts; the package",
        "below is your entire memory of this project.",
        "",
        "PACKAGE STRUCTURE (rendered in this order every run):",
        "  System Prompt + H4 (this protocol) — injected first, forms your",
        "       operating context and rules.",
        "  Toolbox — dynamic tools available to you (bundled, global, project).",
        "  Tool Body — tools you have previously built and delivered.",
        "  H1 — User's prompts in this project, verbatim and in order. The",
        "       user's formal/technical language is the source of truth; do",
        "       not re-derive intent from your own past output.",
        "  H3 — Every tool call you have executed in this project, with",
        "       outcomes. This is your causal chain.",
        "  H2 — Filesystem snapshot of the project workspace as of NOW.",
        "       Placed last so it is freshest in your context window.",
        "",
        "HEADER DISAMBIGUATION:",
        "In this project 'header' has two meanings — never confuse them.",
        "",
        "  Context headers (H1/H2/H3/H4) — the sections of this package: user",
        "  prompt history, filesystem snapshot, tool-call log, and this protocol.",
        "  When the user says 'read the headers', 'check the headers', 'look at",
        "  the headers', or 'what do the headers say' WITHOUT a URL or domain in",
        "  the same message, they mean these context sections. Read H1 and H3.",
        "",
        "  HTTP headers — key-value pairs returned by a web server in a response.",
        "  Only interpret 'headers' this way when a URL or domain is explicitly",
        "  present and a web request is clearly implied.",
        "",
        "Default: no URL → context headers. Always.",
        "",
        "OPERATING MODE:",
        mode_context,
        "",
        "FILESYSTEM ZONES:",
        "Two zones exist on this machine.",
        "",
        "  Central FS (~/.michael/) — READ-ONLY to you. This is Michael's",
        "  internal state: event logs, config, endpoint cache, project",
        "  metadata. You may read_file inside it to diagnose issues, but",
        "  write_file, apply_patch, and any run_shell command referencing",
        "  this path are blocked at the tool layer.",
        "",
        "  Work FS (everything else) — Unrestricted. write_file and",
        "  apply_patch accept any absolute path or project-relative path",
        "  outside ~/.michael/. run_shell has full system access except for",
        "  commands referencing ~/.michael/ which are blocked.",
        "",
        "STAGING:",
        "write_file and apply_patch write to a staging copy — nothing touches",
        "the real workspace until you call commit_changes(). You MUST include",
        "`expected_changes` on every write: your prediction of which paths will",
        "be added, modified, or removed. Michael computes the actual delta and",
        "feeds prediction vs reality back to you. Mismatch is information, not",
        "failure — read it and decide what to do next.",
        "",
        "COMMITTING:",
        "When your work is complete and you are satisfied, call",
        "commit_changes(summary='...') to apply all staged changes to disk.",
        "Do NOT call it until the goal is fully met. If you finish without",
        "staging any changes (e.g. an informational task), just respond — the",
        "loop exits naturally with nothing committed.",
        "",
        "MANDATORY OUTPUT RULE:",
        "Staging without committing burns data — the staging layer is discarded on",
        "exit. A run that calls write_file but not commit_changes is broken.",
        *(
            [
                "In recon/model mode: if any tool other than read_file, list_dir,",
                "search_memory, search_tools, fetch_url, load_model, or forge_tool",
                "returned results this run, you MUST write_file to persist findings",
                "AND call commit_changes before exiting. No exceptions beyond a pure",
                "informational exchange.",
            ] if mode in ("recon", "model") else [
                "In build mode: call commit_changes when your work is complete and",
                "tested. Do not commit partial or untested work.",
            ]
        ),
        "",
        "SANDBOX:",
        "Use run_in_sandbox to test code in an isolated podman container before",
        "writing it. run_shell runs in the project workspace without sandboxing.",
        "Both require user confirmation.",
        "INFRASTRUCTURE: the sandbox is rootless podman on a remote VPS — NOT",
        "Docker, not localhost. Never propose docker commands or docker-compose.",
        "Never assume a local Docker daemon exists. All container execution goes",
        "through run_in_sandbox or run_shell (which SSHes to the VPS).",
        "",
        "LONG-TERM MEMORY:",
        "Call search_memory(query) to retrieve context from previous sessions —",
        "what you explored, what the sandbox returned, what failed. Use it early",
        "before re-discovering what you already know.",
        "",
        "TOOLBOX STEWARDSHIP:",
        "tools/ (project-local) and ~/.michael/toolbox/ (global) are your growing",
        "capability set. Every run is an opportunity to leave them better than you",
        "found them. This is not optional scaffolding — it is how Michael compounds",
        "capability across sessions.",
        "",
        "The rule: if you reached for something that didn't exist and had to inline",
        "the logic, that logic belongs in a tool. Write it before calling",
        "commit_changes(). Export TOOL_SCHEMA (OpenAI function schema dict),",
        f"TOOL_TAGS = ['{mode}'] to scope it to this mode, and a callable with the",
        "same name. It auto-loads immediately — no restart needed.",
        "",
        "General-purpose tools go in ~/.michael/toolbox/ — available to any project",
        "with the same mode. Project-specific tools go in tools/ — local only,",
        "always loaded regardless of mode.",
        "A tool is worth writing if you can imagine calling it again on a different",
        "prompt. If it's truly one-off, inline is fine. Use judgment.",
        "",
        *(
            [
                "TARGET MODELING:",
                "Recon tool output (explore_service, web_dns_recon, web_http_probe, etc.)",
                "is rich but transient — only a 600-char excerpt survives in H3. Michael",
                "auto-saves every raw result to recon/raw.jsonl immediately on execution,",
                "as a safety net. If you see recon/raw.jsonl in H2 and the corresponding",
                "targets/<domain>.md is missing or stale, your first act MUST be to",
                "read_file('recon/raw.jsonl'), synthesize it, and write the target model",
                "before doing anything else. This recovers data from prior sessions where",
                "the LLM exited without writing.",
                "For live recon: write structured findings to targets/<domain>.md in the",
                "project root. Read the existing file first; update incrementally rather",
                "than overwriting. The filesystem snapshot (H2) ensures this model",
                "persists and grows across sessions. A target model is the primary",
                "working artifact for any recon run.",
                "",
                "SOURCE MAPPING:",
                "When version numbers are confirmed (server banners, generator tags, JS",
                "bundles), call source_map(package, version) to fetch the canonical",
                "directory structure from public registries (GitHub, npm, PyPI). Compare",
                "expected paths against what the target actually serves: paths that exist",
                "in source but return 403/404 indicate hardening; paths that exist in source",
                "AND return 200 are normal; paths that return 200 but are sensitive (install",
                "scripts, config templates, version files) are findings. Write this to the",
                "target model under 'Expected vs Observed Filesystem'.",
                "Before commit_changes() on any recon session: list detected versions and",
                "confirm source_map was called for each, or explain why not.",
                "",
            ] if mode == "recon" else []
        ),
        *(
            [
                "APP MODELS:",
                "load_model(name, version) returns the AppModel JSON for this project:",
                "base_url, auth, endpoints, stack, notes — the structured contract for build.",
                "Saves you the exploration turn when the ground truth is already there.",
                "To read a handoff file from another project: use read_file with the",
                "absolute path shown under Results in the project metadata",
                "(e.g. read_file('/absolute/path/to/results/my-project-model.json')).",
                "read_file accepts absolute paths — use them for cross-project input.",
                "",
            ] if mode in ("model", "build") else []
        ),
        "Tools (full schemas in the API call):",
        "  write_file(path, content, expected_changes)        stages a file write",
        "  apply_patch(path, unified_diff, expected_changes)  stages a patch",
        "  commit_changes(summary)                            applies all staged changes — call when done",
        "  read_file(path)                                    auto-executes",
        "  list_dir(path='.')                                 auto-executes",
        "  search_memory(query)                               auto-executes",
        "  search_tools(query)                                auto-executes; searches delivered tool catalog",
        "  forge_tool(name, code)                             auto-executes; creates a tool in tools/ immediately",
        "  fetch_url(url, method, headers, body)              auto-executes; HTTP fetch",
        "  load_model(name, version)                          auto-executes; returns AppModel JSON",
        "  run_in_sandbox(python_code)                        isolated podman, requires confirmation",
        "  run_shell(cmd, timeout_s=60)                       project workspace, requires confirmation",
        "",
        "Paths may be project-relative or absolute (including ~/). Absolute paths work",
        "for read_file when reading cross-project input. For writes, the path must not",
        "target ~/.michael/ (the Central FS). Do not use '..' to traverse the filesystem.",
    ])


def _tool_body_section() -> str:
    """Summarize the global tool catalog for injection into the context header."""
    from michael.project import load_catalog
    catalog = load_catalog()
    if not catalog:
        return "Tool Body: (empty — no tools delivered yet)\nUse search_tools(query) to search when entries exist."
    lines = ["Tool Body (tools you have built and delivered — consult before rebuilding):"]
    for slug, entry in sorted(catalog.items()):
        desc = entry.get("description", "(no description)")
        installed = entry.get("installed_as")
        run_cmd = entry.get("run_cmd", "—")
        display_cmd = installed if installed else run_cmd
        lines.append(f"  {slug}: {desc}")
        lines.append(f"    run: {display_cmd}")
    lines.append("")
    lines.append("Use search_tools(query) to find a specific tool by keyword.")
    return "\n".join(lines)


def _load_mission(project: "Project", max_chars: int = 8000) -> str:
    p = pathlib.Path(project.path) / "MISSION.md"
    if not p.is_file():
        return ""
    try:
        full = p.read_text(errors="replace").strip()
        if len(full) <= max_chars:
            return full
        # Keep the most recent sections (tail) when the file grows large
        tail = full[-max_chars:]
        cut = tail.find("\n## ")
        if cut != -1:
            tail = tail[cut + 1:]
        return f"[... earlier entries truncated — {len(full) - len(tail)} chars omitted ...]\n\n{tail}"
    except OSError:
        return ""


def _load_news(project: "Project") -> str:
    p = pathlib.Path(project.path) / "NEWS.md"
    if not p.is_file():
        return ""
    try:
        return p.read_text(errors="replace").strip()
    except OSError:
        return ""


def load_scripture(scripture_dir: str, mode: str = "") -> str:
    """Read scripture files, filtered by mode.

    A file whose stem matches a known mode name (recon, model, build) is only
    loaded when that mode is active. All other files load unconditionally.
    """
    p = pathlib.Path(scripture_dir).expanduser()
    if not p.is_dir():
        return ""
    known_modes = {"recon", "model", "build", "analyst", "router"}
    parts: list[str] = []
    for f in sorted(p.iterdir()):
        if not (f.is_file() and _is_text(f)):
            continue
        if f.stem in known_modes and f.stem != mode:
            continue  # mode-specific file, wrong mode
        try:
            parts.append(f"--- {f.name} ---\n{f.read_text(errors='replace')}")
        except OSError:
            continue
    return "\n\n".join(parts)


_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_TOOL_TAGS_RE = re.compile(r'TOOL_TAGS\s*=\s*\[([^\]]+)\]')


def _toolbox_listing(project_path: str, mode: str = "recon") -> str:
    """Summarise available dynamic tools, filtered to the current mode."""
    def _scan(d: pathlib.Path, apply_filter: bool) -> list[str]:
        if not d.is_dir():
            return []
        names: list[str] = []
        for f in sorted(d.glob("*.py")):
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            if "TOOL_SCHEMA" not in text:
                continue
            if apply_filter:
                tm = _TOOL_TAGS_RE.search(text)
                if tm:
                    raw_tags = [t.strip().strip("'\"") for t in tm.group(1).split(",")]
                    if mode not in raw_tags:
                        continue
            m = _TOOL_NAME_RE.search(text)
            names.append(m.group(1) if m else f.stem)
        return names

    bundled = pathlib.Path(__file__).parent.parent / "toolbox"
    global_box = G.GLOBAL_TOOLS_DIR
    project_box = pathlib.Path(project_path) / "tools"

    lines = [f"Toolbox (dynamic tools available in {mode} mode):"]
    for label, path, apply_filter in [
        ("bundled toolbox/", bundled, True),
        ("global ~/.michael/toolbox/", global_box, True),
        ("project tools/", project_box, False),
    ]:
        names = _scan(path, apply_filter)
        entry = ", ".join(names) if names else "(empty)"
        lines.append(f"  {label}: {entry}")
    lines.append(
        "  Write a .py file to project tools/ or ~/.michael/toolbox/ "
        "exporting TOOL_SCHEMA + TOOL_TAGS + a callable to add a new tool."
    )
    return "\n".join(lines)


def build_slim_header(
    project: "Project",
    system_prompt: str,
    tool_schemas: "Optional[list]" = None,
) -> str:
    """Minimal system prompt for small/weak models: no filesystem dump, no protocol wall.

    Sends only a short instruction block + compact tool list. Total size stays
    well under 1 K tokens so 1-2B parameter instruct models can actually follow it.
    """
    parts = [
        system_prompt.strip() if system_prompt.strip() else
        "You are a capable AI assistant operating inside the Michael agent framework.",
        "",
        "To call a tool, output EXACTLY this format on its own line:",
        '  <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>',
        "",
        "Rules:",
        "  - Use <tool_call> to read files, run code, write files, etc.",
        "  - You may emit multiple blocks per response; all will execute.",
        "  - Omit all <tool_call> blocks to reply without calling a tool.",
        "  - Use commit_changes when you have finished and want to save work.",
        "",
    ]
    if tool_schemas:
        parts.append("Available tools:")
        parts.append("")
        for schema in tool_schemas:
            fn = schema.get("function", {})
            name = fn.get("name", "")
            if not name:
                continue
            desc = fn.get("description", "").split("\n")[0][:150]
            props = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            parts.append(f"  {name}")
            if desc:
                parts.append(f"    {desc}")
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                req = " [required]" if pname in required else ""
                parts.append(f"    - {pname} ({ptype}){req}")
            parts.append("")
    parts += [
        f"Project: {project.name}",
        f"Root: {project.path}",
    ]
    return "\n".join(parts)


def _build_text_tool_protocol(tool_schemas: list) -> str:
    """Render a text-format tool calling instruction block + compact tool catalog."""
    lines = [
        "This model does not use the native function-calling API.",
        "To call a tool, output EXACTLY this format — one block per call, on its own line:",
        "",
        '  <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>',
        "",
        "Rules:",
        "  - The content inside the tags must be valid JSON with keys 'name' and 'arguments'.",
        "  - You may emit multiple <tool_call> blocks in one response; all will execute.",
        "  - To finish without calling a tool, simply omit any <tool_call> blocks.",
        "  - To commit and exit, use commit_changes via a <tool_call> block.",
        "  - Do NOT wrap the tag in markdown fences or embed it inside prose.",
        "",
        "Available tools:",
        "",
    ]
    for schema in tool_schemas:
        fn = schema.get("function", {})
        name = fn.get("name", "")
        if not name:
            continue
        desc = fn.get("description", "").split("\n")[0][:200]
        props = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        lines.append(f"  {name}")
        if desc:
            lines.append(f"    {desc}")
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            pdesc = pdef.get("description", "")[:120]
            req = " [required]" if pname in required else ""
            lines.append(f"    - {pname} ({ptype}){req}: {pdesc}")
        lines.append("")
    return "\n".join(lines)


def build_header(
    project: "Project",
    system_prompt: str,
    scripture: str = "",
    *,
    tool_schemas: "Optional[list]" = None,
    tool_uncapable: bool = False,
) -> str:
    """Pack the four-header context package sent to a fresh LLM instance."""
    prompts = _prompt_history_lines(project)
    actions = _action_log_lines(project)
    snap = filesystem_snapshot(pathlib.Path(project.path))
    protocol = build_protocol(mode=project.mode)
    toolbox = _toolbox_listing(project.path, mode=project.mode)

    tool_body = _tool_body_section()

    mission = _load_mission(project)
    parts = [
        system_prompt,
        "",
    ]
    if mission:
        parts += ["=== Mission ===", mission, ""]
    news = _load_news(project)
    if news:
        parts += ["=== Latest Session Notes ===", news, ""]
    parts += [
        "=== H4: Protocol ===",
        protocol,
        "",
    ]
    if tool_uncapable and tool_schemas:
        parts += [
            "=== Tool Calling Protocol (TEXT FORMAT — REQUIRED) ===",
            _build_text_tool_protocol(tool_schemas),
            "",
        ]
    parts += [
        "=== Toolbox ===",
        toolbox,
        "",
        "=== Tool Body ===",
        tool_body,
        "",
    ]
    if scripture:
        parts += [
            "=== Scripture ===",
            scripture,
            "",
        ]
    parts += [
        "=== Project ===",
        f"Name: {project.name}",
        f"Slug: {project.slug}",
        f"Mode: {project.mode}",
        f"Root: {project.path}",
        f"Results: {G.RESULTS_DIR}",
        "",
        "=== H1: User's prompts in this project (verbatim, in order) ===",
        "\n".join(prompts) if prompts else "(this is the user's first prompt)",
        "",
        "=== H3: Tool calls executed in this project (in order) ===",
        "\n".join(actions) if actions else "(none yet)",
        "",
        "=== H2: Filesystem snapshot ===",
        snap,
    ]
    return "\n".join(parts)
