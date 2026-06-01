# Project Michael — Architecture & Deployment

Project Michael is an event-sourced, air-gapped AI control loop CLI. The phone (Termux or
Linux) is the control plane; a hardened VPS handles sandboxed code execution; a Vast.ai GPU
cluster runs the LLM inference. Every prompt rebuilds a four-header context package from the
project event log — the LLM is stateless, the log is its memory.

---

## Architecture

```
Phone (Termux / Linux)          VPS (Ubuntu 24.04, rootless podman)
  michael CLI ──────SSH──────▶  sandbox execution
       │                              ▲
       │ HTTP (OpenAI protocol)       │ staged code detonated here
       ▼                              │
  Vast.ai GPU cluster ─────────────▶ results back to CLI
  Ollama (OpenAI-compat)
```

**Named model profiles:**
Multiple model profiles can coexist in `config.json` under `models.<name>`. The default
profile is `god` (auto-created by `michael init`). Additional profiles (e.g. `hermes`) can
be added manually and selected per-run with `michael run --model hermes <prompt>` or made
the default by setting `default_model`. `michael gpu up` fills in the `endpoint` and
`served_model_name` for whichever profile is active at setup time.

**Four-header context package** (sent on every fresh LLM instance):
- H1 — user's prompts verbatim, in order
- H2 — live filesystem snapshot of the project
- H3 — every tool call executed in this project (causal chain)
- H4 — protocol Bible (the contract the LLM operates under)

**The commit_changes gate:** The LLM iterates with Michael — reading files, running the sandbox,
patching code — and calls `commit_changes(summary=…)` when it has something worth keeping.
On that call, staged writes are flushed to the project tree, pre-change snapshots are saved
for `michael undo`, any detectable deliverable is installed under `~/.michael/bin/`, and the
run exits. Confirmable tools (`write_file`, `apply_patch`, `run_in_sandbox`, `run_shell`,
`commit_changes`) prompt y/n by default; auto-exec tools (read/list/search/forge/fetch) run
without prompting. The loop also exits naturally if the LLM replies with no tool calls — in
that case nothing is committed and staged changes are discarded.

**Dual filesystem zones:**
| Zone | Path | LLM tool access |
|------|------|-----------------|
| Central FS | `~/.michael/` | Read-only. Writes blocked at Python layer before any I/O. |
| Work FS | Everything else | Unrestricted — `write_file`/`apply_patch` accept absolute paths; `run_shell` has full system access. |

The Central FS holds all headers source data (events, config, state). Michael's application code
writes there freely; LLM tool calls are categorically blocked from doing so. Enforcement lives in
`michael/permissions.py` and is applied inside every write path in `michael/tools.py`.

---

## Deploy Checklist

### 1. Termux / Linux (control plane)

```bash
# Clone and install
git clone <repo> project_michael && cd project_michael
bash bootstrap_termux.sh          # Termux
# or: pip install -r requirements.txt && echo 'alias michael="python main.py"' >> ~/.bashrc

# Initialise config
michael init
michael config                    # fill in vast_api_key, optionally gpu.model_repo
```

### 2. VPS (sandbox, run once as root)

```bash
git clone <repo> && bash bootstrap.sh
```

The script creates a `michael` user, hardens SSH, installs rootless podman, builds the
sandbox image, and creates `~/workspace`.

After bootstrap, copy your SSH public key to the VPS:
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub michael@<vps-ip>
michael ssh-test                  # verify roundtrip
```

### 3. Vast.ai GPU

1. Rent an instance — any GPU with enough VRAM for your chosen model (default
   `qwen2.5:72b` needs ~45 GB). Any PyTorch / CUDA template works; no on-start
   command needed.
2. Optionally set `gpu.model_repo` in `~/.michael/config.json` if you want a
   model other than the default `qwen2.5:72b`. Any Ollama tag works
   (`llama3.1:70b`, `qwen2.5-coder:32b`, etc.).
3. Start inference: `michael gpu up`. On first run it prompts for the Vast SSH
   command, installs Ollama (one curl line), pulls the model, and caches the
   endpoint. Subsequent runs reuse the cached config.

### 4. First run

```bash
michael new myproject             # creates project, sets it active
michael run fix the auth bug in login.py
```

The LLM reads your code, iterates, calls `commit_changes` when done. Done.

---

## Config Keys

| Key | Description |
|-----|-------------|
| `vast_api_key` | Vast.ai console API key |
| `default_model` | Profile to use (default: `god`). Override per-run with `--model <name>` |
| `gpu.model_repo` | vLLM: HuggingFace ID e.g. `NousResearch/Hermes-4.3-36B`; Ollama: tag e.g. `qwen2.5:72b` |
| `gpu.gpu_port` | OpenAI-compat port on the GPU (ollama default `11434`, vLLM default `8000`) |
| `gpu.max_model_len` | vLLM only: max context length (`--max-model-len`). Caps KV cache to fit VRAM (default `32768`). Lower it if startup fails with a "KV cache memory" error; `0` lets vLLM use the model's full native max (often too large for one GPU) |
| `gpu.gpu_memory_utilization` | vLLM only: fraction of GPU VRAM the engine may use (`--gpu-memory-utilization`, default `0.92`). Raise toward `0.95` for more KV cache, lower on load-time OOM |
| `models.<name>.request_timeout_s` | LLM request timeout in seconds |
| `models.<name>.served_model_name` | Auto-filled by `gpu up` from `gpu.model_repo` |
| `models.<name>.gpu_name` | Named GPU that serves this model (empty = primary `gpu`). Set by `michael gpu up <name>`. |
| `models.<name>.enable_thinking` | Enable `<think>` reasoning traces — set `true` for Hermes 4.3 on the senior model. |
| `models.<name>.tool_uncapable` | Set `true` for base-model fine-tunes with no native function-calling (e.g. the junior specialist). |
| `gpus.<name>.ssh_host` | SSH host for a named GPU instance. Set by `michael gpu up <name>`. |
| `gpus.<name>.gpu_port` | Local tunnel port for the named GPU — must be unique (e.g. god=11434, junior=11435). |
| `gpus.<name>.model_repo` | HuggingFace ID (vLLM) or Ollama tag for the named GPU. |
| `gpus.<name>.inference_backend` | `vllm` or `ollama` — auto-detected on `gpu up`. |
| `vps.host` | VPS public IP/hostname (empty = no remote sandbox) |
| `vps.user` | SSH user (default: `michael`) |
| `vps.ssh_key_path` | Path to private key (default: `~/.ssh/id_ed25519`) |
| `vps.workspace_dir` | Workspace dir on the VPS |
| `sandbox.image` | Tag of the sandbox image built by `bootstrap.sh` |
| `sandbox.memory_mb` | Sandbox memory cap in MB |
| `sandbox.cpus` | Sandbox CPU cap |
| `sandbox.pids` | Sandbox PID cap |
| `sandbox.timeout_s` | Default sandbox timeout in seconds |
| `system_prompt` | Default system prompt for the agent loop |
| `system_prompt_file` | If set, reads system prompt from this file |
| `log_responses` | If true (default), stores full LLM responses in events.jsonl |
| `boot_poll_s` | Poll interval while waiting for the inference server |

---

## Command Reference

| Command | Description |
|---------|-------------|
| `michael init` | Write stub config if missing |
| `michael show` | List all projects |
| `michael new [name]` | Create a new project |
| `michael use <slug>` | Switch active project |
| `michael current` | Print active project |
| `michael config` | Open `config.json` in `$EDITOR` |
| `michael gpu up [name]` | Provision GPU (default: `god`). `michael gpu up junior` provisions a second GPU on its own SSH tunnel and port. |
| `michael gpu new` | Swap to a new GPU — clear cached SSH/instance state, re-prompt, then `gpu up` |
| `michael gpu down [name]` | Pause the GPU instance (default: `god`) |
| `michael status` | Derived state from event log |
| `michael run <prompt…>` | **Run the agent.** Everything after `run` is the prompt |
| `michael log [--tail N]` | Show event log (last 20 by default) |
| `michael sandbox <file.py>` | Run Python file in isolated sandbox |
| `michael undo [--list] [<id>]` | Restore the most recent (or named) change |
| `michael ssh-test` | Verify VPS reachability, report handshake time |

---

## How a Run Works

```
michael run refactor the parser to handle unicode edge cases
```

1. Michael packages H1–H4 (your prompts, filesystem, tool history, protocol) and sends it with
   your prompt to the model endpoint (ollama, OpenAI-compatible).
2. The LLM iterates: reads files, patches code, runs the sandbox — up to `MAX_AGENT_TURNS`
   (60) turns. You see dim status lines and a y/n prompt before each confirmable tool runs.
3. When the LLM calls `commit_changes(summary=…)`, Michael flushes every staged write,
   snapshots originals to the trash dir, and (if a deliverable is detected) installs a
   wrapper under `~/.michael/bin/<slug>`.
4. The terminal shows what was applied. The command exits.

If the LLM responds with no tool calls, the run exits naturally and staged changes are
discarded. Ctrl-C also discards staged changes.

---

## Tools Available to the LLM

| Tool | Behaviour |
|------|-----------|
| `write_file(path, content, expected_changes)` | Staged; written on `commit_changes` |
| `apply_patch(path, unified_diff, expected_changes)` | Same staging flow as `write_file` |
| `read_file(path)` | Auto-executes, no confirmation |
| `list_dir(path='.')` | Auto-executes, no confirmation |
| `search_memory(query)` | Auto-executes; searches stored LLM responses in this project |
| `search_tools(query)` | Auto-executes; looks up tool schemas by name/keyword |
| `fetch_url(url)` | Auto-executes; HTTP GET of arbitrary content |
| `forge_tool(name, schema, code)` | Auto-executes; writes a new tool to `<project>/tools/<name>.py`, available **next run** |
| `spawn_specialist(model_name, prompt)` | Auto-executes; calls a specialist model as a pure text oracle — no tools, no loop. Returns raw generated text. The senior (Hermes) validates and iterates using its own tools. |
| `load_model(profile)` | Auto-executes; switch to a different model profile mid-run |
| `run_in_sandbox(python_code)` | Confirms; isolated podman (local or remote via SSH) |
| `run_shell(cmd, timeout_s=60)` | Confirms; runs in project workspace |
| `commit_changes(summary)` | Confirms; flushes staged writes, installs deliverable if any, exits the loop |

In addition, all `*.py` files under `<repo>/toolbox/`, `~/.michael/toolbox/`, and
`<project>/tools/` that export a `TOOL_SCHEMA` are loaded as tools at the start of each
run (later directories override on name collision). The bundled `toolbox/` ships ~25 recon
tools (DNS, TLS, HTTP fingerprint, directory enumeration, etc.).

---

## Deliverables

When the LLM commits, `detect_deliverable` (michael/project.py) looks for a top-level
script in the project that responds to `--help`. If found, Michael:

- runs `<deliverable> --help` to probe it,
- on success, registers it and writes a wrapper at `~/.michael/bin/<slug>` so the project
  is callable system-wide once `~/.michael/bin` is on `PATH`,
- on probe failure, logs the failure and suggests `michael run` to repair.

---

## State Layout

```
~/.michael/
  config.json                    # global config (chmod 600)
  events.jsonl                   # global event log (instance lifecycle)
  state.json                     # derived state (endpoint cache, etc.)
  projects/
    <slug>/
      config.json                # per-project stub
      events.jsonl               # per-project log (prompts, tool calls, LLM responses)
      trash/                     # pre-change snapshots for undo
  ssh-*.sock                     # SSH control-master sockets
  repl_history                   # REPL command history
```
