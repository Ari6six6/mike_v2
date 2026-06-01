# Handoff: the Router role (next session)

> A note from a past session to a future one. Read this top-to-bottom before
> touching code. It assumes the **analyst** role has already landed (branch
> `claude/gpu-data-collection-c5tgc`): per-turn `turn.telemetry`, deterministic
> run features, a classed training corpus, and per-run scorecards.

## Context — where this fits

Project Michael is becoming a "firm" of LLM roles. So far:

- **Doers** — the senior model running the flat tool loop (`michael/agent.py`
  `_run_agent_loop`). One model is chosen per run (`--model` / `default_model`).
- **Oracles** — specialist models called as pure text oracles via
  `spawn_specialist(model_name, prompt)` (`toolbox/spawn_specialist.py`). The
  senior must invoke this **manually** and decide which specialist to use.
- **Analyst** — post-run judge (`michael/analyst.py`): writes `dataset/` records
  + `scorecards/` and `run.scored` events.

The **router** is the missing seat: instead of the senior hand-picking a
specialist every time, the router *automatically* decides — given a task or
subtask — which model profile should handle it (or that the senior should keep
it). It's the dispatch/orchestration layer the analyst's data eventually feeds.

This was deliberately left out of the analyst PR to keep that slice clean.

## What already exists (verify before building — cite file:line)

- **Model profiles**: `cfg.models.<name>` (`michael/config.py:14-26` ModelProfile;
  loaded `config.py:101-109`). `cfg.get_model(name)` (`config.py:151-161`) returns
  `(name, profile)`. Profiles carry `tool_uncapable`, `enable_thinking`,
  `gpu_name`, `slim_context`.
- **Named GPUs + tunnels**: `cfg.gpus.<name>` (`config.py:51-79`), `cfg.get_gpu()`
  (`config.py:133-142`), `_ensure_tunnel(name, gpu)` (`michael/backends.py:482-534`),
  unique `gpu_port` per GPU.
- **Specialist call pattern** (reuse verbatim): `spawn_specialist.py:64-104` —
  resolve profile, ensure tunnel, `LLMClient(profile.endpoint).chat.completions
  .create(...)` with no tools, return text.
- **Run loop seam**: tool dispatch happens at `agent.py` ~`_run_agent_loop`
  (the `for tc in tool_calls` block). The senior already decides to call
  `spawn_specialist` as one tool among many.
- **Feedback substrate** (from the analyst): `<project>/scorecards/<run_id>.json`
  (`scores`, `dominant_class`, `features`) and `<project>/dataset/records.jsonl`
  (`class`, `quality` per turn). A learned router can read these.

## Goal

Add a routing layer that, given a unit of work, selects the best model profile
to execute it — automatically — and (later) learns the mapping from the analyst's
scorecards. Ship it as a clean vertical slice, same as the analyst:

1. **Routing decision** — a `michael/router.py` with a pure function
   `route(task, cfg, *, features=None) -> str` returning a profile name. Start
   **rule/heuristic-based** (the analyst precedent: heuristics first, the
   model-as-judge layered on later — confirm this ordering with the user, since
   for the analyst they were unsure about heuristics-first but rejected pure
   offline). Candidate signals: task keywords (codegen→junior, recon→senior),
   declared task type, profile capabilities (`tool_uncapable` ⇒ oracle-only),
   project `mode`.
2. **A `route_task` tool** (or auto-dispatch) so the senior can delegate a
   subtask and have the router pick + run the specialist, returning text — i.e.
   `spawn_specialist` with the `model_name` chosen *for* the senior instead of
   *by* it. Decide with the user: a new tool vs. wrapping `spawn_specialist`.
3. **Learned routing (later, gated)** — once scorecards accumulate, bias routing
   toward the profile with the best historical `scores` for that task class.
   Read `~/.michael/dataset/*.jsonl` + `scorecards/`. Keep behind a config flag,
   default off, exactly like `analyst_enabled`.

## Files likely touched

- `michael/router.py` *(new)* — `route()`, capability checks, optional learned
  bias from analyst data.
- `toolbox/route_task.py` *(new)* OR extend `toolbox/spawn_specialist.py` — the
  tool surface the senior calls.
- `michael/config.py` — a `router_enabled` flag + `CONFIG_HELP`; maybe a
  `models.<name>.skills`/tags field so routing can match task→profile.
- `michael/agent.py` — only if doing auto-dispatch inside the loop (vs. a tool).
- `scripture/` — if the router becomes a model-judged role, a `router.md`
  contract (remember: `load_scripture` treats `recon/model/build/analyst` as
  mode-specific stems — add `"router"` there too so it doesn't leak into every
  run; see `michael/utils.py` `load_scripture`).

## Reuse, don't reinvent

- Tunnel + `LLMClient` call → copy `spawn_specialist.py:85-104`.
- Profile resolution → `cfg.get_model` / `cfg.models`.
- Feature inputs → `michael.analyst.compute_run_features` (already computes the
  signals a learned router would want).
- Config flag + CONFIG_HELP pattern → `analyst_enabled` in `config.py`.
- Event logging → `append_event("route.decided", {...}, project=project)`; add the
  new event type to the docs block in `michael/project.py:17-48`.

## Verification (offline, no GPU)

- Unit-test `route()` purely: feed tasks + a fake `cfg.models` (capabilities) and
  assert the chosen profile; assert `tool_uncapable` profiles are never handed
  tool-requiring work.
- Test the learned-bias path by seeding fake scorecards/dataset and asserting the
  router prefers the higher-scoring profile for a class.
- Test the tool with a monkeypatched `michael.backends.LLMClient` stub (same idiom
  as `tests/test_michael.py::test_run_analyst_writes_corpus_and_scorecard`).
- Disabled path: `router_enabled=False` ⇒ behavior identical to today (senior
  calls `spawn_specialist` manually). Prove zero behavior change.

## Out of scope (name it, don't drift)

- Multi-agent parallel execution / a real scheduler. The router *chooses*; it does
  not run several specialists concurrently (yet).
- Sandbox target replicas (the recon "engine" half deliberately skipped).
- Actually fine-tuning specialists — the dataset feeds that downstream.

## First moves for the next session

1. Re-read this file and the analyst code (`michael/analyst.py`) for the
   heuristics-first precedent.
2. Ask the user: (a) heuristics-first or model-judged router? (b) new
   `route_task` tool vs. auto-dispatch in the loop? (c) should routing learn from
   scorecards now or later?
3. Plan → confirm → implement the slice → tests → push to a fresh
   `claude/router-*` branch (do **not** reuse the analyst branch).
