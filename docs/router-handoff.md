# Router Handoff — Note from Session `claude/router-handoff-docs-9gTsg`

Written 2026-06-01. This doc is the only context you need to start the router
build. Read it fully before writing a line of code. Then ask the four questions
at the bottom before you start.

---

## What the router is

Michael has two kinds of model profiles: **doers** (the senior Hermes loop that
reads files, runs tools, and commits) and **oracles** (specialists called via
`spawn_specialist`, which are stateless text generators with no tools). Right
now the senior decides manually which oracle to call, or the user hard-codes
`--model` on the CLI.

The router sits between the task intake and model dispatch. Its job is:
given a task description (the user's prompt), pick the right model profile
to run it on. That's the whole thing.

Why it matters: once you have 3+ profiles — a general senior, a code
specialist, an exploit writer, a log analyst — ad-hoc selection breaks down.
The router makes dispatch automatic and, over time, learnable from feedback.

---

## What already exists to reuse

**`cfg.models` and `cfg.get_model()`** — `michael/config.py:151`. All model
profiles live in `cfg.models` (a `dict[str, ModelProfile]`). `get_model(name)`
resolves a profile by name. The router reads this dict to know what profiles
are available; it doesn't maintain its own registry.

**`_ensure_tunnel`** — imported at `michael/tools.py:14`, used at lines 318 and
410. Handles SSH control-master bringup for remote GPUs. `spawn_specialist` in
`toolbox/spawn_specialist.py:88-92` shows the exact call pattern: look up the
profile's `gpu_name`, get the `GpuConfig`, call `_ensure_tunnel(gpu_key, gpu)`
before making the HTTP request. Any routing path that dispatches to a remote
profile needs this same setup.

**`spawn_specialist` call pattern** — `toolbox/spawn_specialist.py:64-104`.
This is the complete reference implementation for calling a model profile
as a text oracle: load config, validate profile has endpoint and
served_model_name, optionally bring up tunnel, call `LLMClient`. The router's
dispatch path for oracle profiles mirrors this exactly.

**Analyst scorecards / dataset** — the analyst role (not yet built, but planned
as a mode-judged project role) will produce structured scorecards per run:
model profile used, task type, outcome, latency, user rating. This is the
natural learning substrate for part 3 below. The router's bias table reads from
these scorecards once they exist; until then it operates on heuristics alone.

---

## The build — three parts

### Part 1: Heuristic `route()`

A pure function: `route(prompt: str, profiles: dict[str, ModelProfile]) -> str`.

Returns a profile name. Initial logic: keyword and tag matching.

```
exploit / CVE / shellcode / payload  → "junior" (exploit specialist)
log / alert / SIEM / triage          → "analyst"  (log model)
code / refactor / fix / test         →  default doer
<anything else>                      →  default doer (god / hermes)
```

Tags on `ModelProfile` are a natural extension point — add a `tags: list[str]`
field and let the heuristic score against them. But don't add tags on day one;
start with keyword matching and see if it's enough.

Location: new file `michael/router.py`. No CLI command, no tool schema yet —
just a function you can test in isolation.

### Part 2: `route_task` tool

Expose `route()` to the LLM as a tool so the senior can explicitly re-route
mid-run if it figures out the task is better handled by a specialist.

Schema: `route_task(prompt: str) -> str` — returns the selected profile name
and a brief reason. The LLM can then call `load_model(profile)` (already
exists) to switch.

Location: `toolbox/route_task.py`. Follows the standard `TOOL_SCHEMA` +
function pattern. Auto-executes (no confirmation needed — it's read-only).

Wire it into `_load_dynamic_tools` at `michael/agent.py:107` — it should load
in all modes, same as `spawn_specialist`.

### Part 3: Learned bias from scorecards (optional, do last)

Once the analyst is building scorecards, the router can weight its keyword
scores by observed success rates per (task-type, profile) pair. Keep it simple:
a JSON bias table in `~/.michael/router_bias.json`, updated by a
`michael router update-bias` CLI subcommand that reads recent scorecards.

Don't build this until scorecards exist. Mention it in a `# TODO` in
`michael/router.py` and leave it there.

---

## Files to touch

| File | What changes |
|------|--------------|
| `michael/router.py` | New — the `route()` function and bias loader |
| `toolbox/route_task.py` | New — tool schema + thin wrapper around `route()` |
| `michael/agent.py` | Wire `route()` into dispatch before the first LLM call (optional: auto-route if no `--model` flag given) |
| `michael/project.py` | Only if router becomes a project mode — add `"router"` to `VALID_MODES` (line 55) |
| `michael/utils.py` | See the `load_scripture` gotcha below |
| `michael/config.py` | Only if you add `tags: list[str]` to `ModelProfile` — defer this |

---

## `load_scripture` gotcha — read this before you add a router mode

`michael/utils.py:439` — `load_scripture`:

```python
known_modes = {"recon", "model", "build"}
...
if f.stem in known_modes and f.stem != mode:
    continue  # mode-specific file, wrong mode
```

If you create a `scripture/router.txt` file (so the router mode gets its own
system-prompt fragment), it will silently load in ALL modes because `"router"`
is not in `known_modes` — there's nothing to filter it out. The fix is a
one-liner: add `"router"` to the set at line 448. The same gap was just fixed
for the analyst mode on this branch (`"analyst"` was missing and had to be
added). Don't forget it.

If the router is never a project mode and never gets a scripture file, you
can skip this. But if it does — fix line 448 first.

---

## Offline verification

```bash
# Unit test route() directly — no GPU needed
python - <<'EOF'
from michael.router import route
from michael.config import ModelProfile
profiles = {
    "god":    ModelProfile(endpoint="x", served_model_name="hermes"),
    "junior": ModelProfile(endpoint="y", served_model_name="deephat"),
}
assert route("write an exploit for CVE-2024-1234", profiles) == "junior"
assert route("refactor the parser module", profiles) == "god"
print("ok")
EOF

# Load the tool schema and confirm it appears in tool listing
python -c "
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('rt', 'toolbox/route_task.py')
m = importlib.util.load_from_spec(spec); spec.loader.exec_module(m)
print(m.TOOL_SCHEMA['function']['name'])
"
```

No integration test against a live model is needed to ship parts 1 and 2.
Part 3 requires scorecards on disk; mock them with a fixture JSON file.

---

## Out of scope

- Multi-label routing (a task belonging to two profiles) — not needed, pick one.
- Routing based on cost or latency — premature; add to bias table in part 3 if wanted.
- A `michael route <prompt>` CLI command for dry-run inspection — nice to have,
  but not blocking. Add it after parts 1 and 2 are solid.
- Changing how `michael run --model` works — that flag stays authoritative and
  skips the router entirely.

---

## First moves — ask these before writing code

1. **Should the router auto-run on every `michael run` call (no `--model` flag),
   or only when the LLM explicitly calls `route_task`?**
   Auto-run is lower friction; explicit tool call gives the LLM more control.
   Recommendation: auto-run as the default, tool call as the override.

2. **Which profile names are in the config right now, and do any already have
   implied task domains?** Run `michael config` or read `~/.michael/config.json`
   before hard-coding keyword → profile mappings. The heuristics need to
   map to real profile names.

3. **Is the analyst mode (and its scorecards) already built, or still planned?**
   If it doesn't exist yet, skip part 3 entirely and leave the TODO comment.

4. **Should `ModelProfile` gain a `tags` field, or should keyword matching stay
   purely in `router.py`?** Tags make the heuristic extensible but add schema
   churn. Defer unless the user asks for it.

---

## Branch reminder

**Do not reuse `claude/router-handoff-docs-9gTsg`** — that branch only contains
this doc. Start fresh:

```bash
git checkout main
git pull origin main
git checkout -b claude/router-<short-description>
```

Then build on that branch and push there.
