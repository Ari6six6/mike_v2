"""CLI commands, Typer bindings, and the interactive REPL."""
import datetime
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

import michael.globals as G

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
from michael.agent import _run_agent_loop
from michael.backends import (
    VastClient,
    _ensure_tunnel,
    _gpu_ssh_run,
    _ping_endpoint,
    _require_endpoint,
    _restart_vllm_on_gpu,
    _ssh_argv,
    _ssh_preflight,
    _GPU_PY,
    _gpu_compute_cap,
    _gpu_vllm_overrides,
    _vllm_crash_report,
    _start_ollama_cmd,
    _start_vllm_cmd,
    _stop_vllm_cmd,
    gpu_port_forward_cmd,
    llm_client,
    make_backend,
    parse_vast_ssh_cmd,
)
from michael.config import Config, CONFIG_HELP, GpuConfig, make_stub_config
from michael.project import (
    Project,
    VALID_MODES,
    append_event,
    create_project,
    detect_deliverable,
    get_active_project,
    get_active_slug,
    iter_events,
    list_projects,
    load_catalog,
    register_deliverable,
    replay_global,
    require_active_project,
    set_active_slug,
    slugify,
)
from michael.agent import _load_dynamic_tools
from michael.tools import TOOLS, _list_trash, _undo_one, _dispatch_dynamic_tool_from_path
from michael.utils import (
    build_header,
    load_scripture,
    _prompt_history_lines,
    _action_log_lines,
)

app = typer.Typer(
    no_args_is_help=False,
    rich_markup_mode="rich",
    help="michael — air-gapped AI control loop",
)

gpu_app = typer.Typer(help="GPU instance management (vLLM or Ollama).", invoke_without_command=True)
app.add_typer(gpu_app, name="gpu")

SUPPORTED_MODELS: list[str] = []  # Ollama menu (opt-in path); no baked-in tags
_MODEL_MIN_DISK_GB: dict[str, int] = {}

VLLM_SUPPORTED_MODELS = [
    "NousResearch/Hermes-4.3-36B",
    "deepseek-ai/DeepSeek-V4-Flash",
]
_VLLM_MODEL_LABELS: dict[str, str] = {
    "NousResearch/Hermes-4.3-36B":     "36B (Seed-OSS base), hybrid <think>, native tool-calling — full precision bf16 ~72 GB, one 80 GB+ card",
    "deepseek-ai/DeepSeek-V4-Flash":   "MoE, V4 Flash",
}
_VLLM_MODEL_MIN_DISK_GB: dict[str, int] = {
    "NousResearch/Hermes-4.3-36B":     75,
    "deepseek-ai/DeepSeek-V4-Flash":   30,
}

tools_app = typer.Typer(help="Inspect and run dynamic tools.")
app.add_typer(tools_app, name="tools")


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


_SHELL_MARKER = "# michael shell integration"
_SHELL_LINES = (
    "\n{marker}\n"
    "export PATH=\"{bin}:$PATH\"\n"
    "mcd() {{ cd \"$(michael path)\"; }}\n"
)


def _shell_profile() -> Optional[pathlib.Path]:
    shell = os.environ.get("SHELL", "")
    home = pathlib.Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    if "bash" in shell:
        for name in (".bashrc", ".bash_profile"):
            p = home / name
            if p.is_file():
                return p
        return home / ".bashrc"
    return None


def _inject_shell_integration() -> str:
    profile = _shell_profile()
    if profile is None:
        return "[yellow]unknown shell — add manually:[/]\n  export PATH=\"{bin}:$PATH\"\n  mcd() {{ cd \"$(michael path)\"; }}".format(bin=G.MICHAEL_BIN_DIR)
    text = profile.read_text() if profile.is_file() else ""
    if _SHELL_MARKER in text:
        return f"[dim]shell integration already in {profile}[/]"
    profile.parent.mkdir(parents=True, exist_ok=True)
    with profile.open("a") as f:
        f.write(_SHELL_LINES.format(marker=_SHELL_MARKER, bin=G.MICHAEL_BIN_DIR))
    return f"[green]wrote shell integration → {profile}[/]\n[dim]run: source {profile}[/]"


def cmd_init() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    G.MICHAEL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    if not G.GLOBAL_CONFIG_PATH.is_file():
        make_stub_config().save()
        G.console.print(f"[green]wrote stub[/] {G.GLOBAL_CONFIG_PATH}")
    else:
        G.console.print(f"[dim]config exists[/] {G.GLOBAL_CONFIG_PATH}")
    append_event("config.loaded", {"path": str(G.GLOBAL_CONFIG_PATH)})
    shell_msg = _inject_shell_integration()
    G.console.print(shell_msg)
    G.console.print(
        Panel(
            "Edit ~/.michael/config.json — fill in:\n\n"
            "  [bold]vast_api_key[/]              your Vast.ai console API key\n"
            "  [bold]gpu.model_repo[/]            HF model id (default 'NousResearch/Hermes-4.3-36B')\n\n"
            "[dim]Optional, for remote sandbox on the VPS:[/]\n"
            "  [bold]vps.host[/]                  VPS public IP/hostname\n"
            "  [bold]vps.user[/]                  ssh user (default: michael)\n"
            "  [bold]vps.ssh_key_path[/]          path to private key\n"
            "  [bold]vps.workspace_dir[/]         /home/michael/workspace\n\n"
            "[dim]Leave vps.host empty to run without sandbox.[/]",
            title="checklist",
            border_style="green",
        )
    )


def cmd_show() -> None:
    projects = list_projects()
    if not projects:
        G.console.print("0")
        return
    active = get_active_slug()
    table = Table(title=f"projects ({len(projects)})", border_style="cyan")
    table.add_column("active", justify="center")
    table.add_column("slug", style="bold")
    table.add_column("mode")
    table.add_column("name")
    table.add_column("path")
    table.add_column("created")
    for p in projects:
        mark = "*" if p.slug == active else ""
        table.add_row(mark, p.slug, p.mode, p.name, p.path, p.created_at)
    G.console.print(table)


def cmd_new(name: Optional[str]) -> None:
    if not name:
        name = (typer.prompt("name") or "").strip()
    if not name:
        G.err.print("name is required")
        return
    try:
        slug_preview = slugify(name)
    except G.MichaelError as e:
        G.err.print(str(e))
        return
    default_path = G.WORKBENCH_DIR / "codebases" / slug_preview
    path_str = typer.prompt("path", default=str(default_path))
    path = pathlib.Path(path_str).expanduser().resolve()
    mission = typer.prompt("mission  (objective — leave blank to skip)", default="").strip()
    mode_str = typer.prompt("mode     [recon/model/build]", default="recon").strip().lower()
    if mode_str not in VALID_MODES:
        G.err.print(f"invalid mode '{mode_str}' — must be one of: {', '.join(VALID_MODES)}")
        return
    proj = create_project(name, path, mode=mode_str)
    if mission:
        date_str = datetime.date.today().isoformat()
        (path / "MISSION.md").write_text(f"## {date_str}\n\n{mission}\n")
        G.console.print(f"[dim]mission saved to MISSION.md[/]")
    set_active_slug(proj.slug)
    append_event("project.activated", {"slug": proj.slug})
    G.console.print(f"[green]created[/] {proj.slug} at {proj.path}")
    G.console.print(f"[dim]workspace is empty — add your code there, then run: michael run <prompt>[/]")


def cmd_use(slug: str) -> None:
    proj = Project.load(slug)
    set_active_slug(proj.slug)
    append_event("project.activated", {"slug": proj.slug})
    G.console.print(f"[green]active[/] {proj.slug}")


def cmd_current() -> None:
    p = get_active_project()
    if not p:
        G.console.print("(no active project)")
        return
    G.console.print(f"{p.slug} — {p.name} — {p.mode} — {p.path}")


def cmd_config() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    if not G.GLOBAL_CONFIG_PATH.is_file():
        make_stub_config().save()
    help_lines = [f"[bold]{k}[/] — {v}" for k, v in CONFIG_HELP.items()]
    G.console.print(
        Panel(
            "\n".join(help_lines),
            title=f"config: {G.GLOBAL_CONFIG_PATH}",
            border_style="green",
        )
    )
    current_text = G.GLOBAL_CONFIG_PATH.read_text()
    edited = typer.edit(current_text, extension=".json")
    if edited is None or edited == current_text:
        G.console.print("[dim]no changes[/]")
        return
    try:
        json.loads(edited)
    except json.JSONDecodeError as e:
        G.err.print(f"invalid JSON, not saved: {e}")
        return
    G.GLOBAL_CONFIG_PATH.write_text(edited)
    os.chmod(G.GLOBAL_CONFIG_PATH, 0o600)
    G.console.print("[green]config saved[/]")


def _prompt_model_selection(
    current: str,
    backend: str = "ollama",
    custom_models: "list[str] | None" = None,
) -> str:
    """Interactive numbered menu for model selection. Returns the chosen tag or HF ID.

    custom_models is a mutable list; any newly entered model is appended in-place
    so the caller can persist it to config.
    """
    if custom_models is None:
        custom_models = []

    if backend == "vllm":
        builtin = list(VLLM_SUPPORTED_MODELS)
        labels = _VLLM_MODEL_LABELS
        hint = "HuggingFace model ID, e.g. mistralai/Mistral-7B-Instruct-v0.3"
    else:
        builtin = list(SUPPORTED_MODELS)
        labels: dict[str, str] = {}
        hint = "Ollama tag, e.g. hermes3:8b-q8_0"

    # merge built-ins + saved custom (dedup, preserve order)
    all_models: list[str] = list(builtin)
    for m in custom_models:
        if m not in all_models:
            all_models.append(m)

    G.console.print(f"\n[bold]Available models ({backend}):[/]")
    for i, tag in enumerate(all_models, 1):
        marker = " [green]← current[/]" if tag == current else ""
        is_custom = tag not in builtin
        custom_tag = " [magenta][custom][/]" if is_custom else ""
        G.console.print(f"  [cyan]{i}.[/] {tag}{custom_tag}  [dim]({labels.get(tag, 'custom')})[/]{marker}")
    G.console.print(f"  [dim]or type any {hint} to add it[/]")

    default_prompt = str(all_models.index(current) + 1) if current in all_models else str(1)
    raw = typer.prompt("Model", default=default_prompt).strip()

    # numeric selection
    try:
        idx = int(raw)
        if 1 <= idx <= len(all_models):
            return all_models[idx - 1]
    except ValueError:
        pass

    # exact match in combined list
    if raw in all_models:
        return raw

    # treat as a new custom model
    if raw:
        if raw not in custom_models:
            custom_models.append(raw)
            G.console.print(f"[green]Added[/] {raw!r} to your saved model list.")
        return raw

    G.console.print(f"[yellow]invalid choice, keeping {current or all_models[0]}[/]")
    return current or all_models[0]


def _prompt_backend_selection(current: str) -> str:
    """Interactive chooser for the inference backend. Returns 'vllm' or 'ollama'."""
    options = [
        ("vllm", "best on modern GPUs (Ampere/Ada+); MoE & AWQ models"),
        ("ollama", "bundles its own CUDA runtime; works on older GPUs/drivers"),
    ]
    G.console.print("\n[bold]Inference backend:[/]")
    for i, (name, desc) in enumerate(options, 1):
        marker = " [green]← current[/]" if name == current else ""
        G.console.print(f"  [cyan]{i}.[/] {name}  [dim]({desc})[/]{marker}")
    default_idx = next((i for i, (n, _) in enumerate(options, 1) if n == current), 1)
    raw = typer.prompt("Backend", default=str(default_idx)).strip()
    try:
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
    except ValueError:
        if raw in ("vllm", "ollama"):
            return raw
    G.console.print(f"[yellow]invalid choice, keeping {current}[/]")
    return current


def _select_vast_instance(cfg: "Config", gpu: "GpuConfig") -> bool:
    """List Vast.ai instances and let user pick one. Populates gpu in-place.

    Returns True on success, False if user chose manual entry or API unavailable.
    """
    if not cfg.vast_api_key:
        return False
    try:
        vast = VastClient(cfg.vast_api_key)
        instances = vast.list()
        vast.close()
    except G.MichaelError as e:
        G.console.print(f"[dim]Vast.ai API unavailable ({e}) — falling back to manual SSH entry[/]")
        return False

    if not instances:
        G.console.print(
            "[dim]No instances found on Vast.ai — rent one from the console, then re-run `michael gpu`.[/]"
        )
        return False

    table = Table(title="Vast.ai Instances", border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("ID", style="bold")
    table.add_column("GPU")
    table.add_column("Status")
    table.add_column("IP / Host")
    for i, inst in enumerate(instances, 1):
        gpu_label = f"{inst.get('num_gpus', 1)}× {inst.get('gpu_name') or '?'}"
        ip = inst.get("public_ipaddr") or inst.get("ssh_host") or "?"
        status = inst.get("actual_status") or inst.get("status") or "?"
        table.add_row(str(i), str(inst.get("id", "?")), gpu_label, status, ip)
    G.console.print(table)

    raw = typer.prompt(f"Select instance [1-{len(instances)}] or 0 for manual SSH entry", default="1").strip()
    try:
        choice = int(raw)
    except ValueError:
        choice = 0
    if choice < 1 or choice > len(instances):
        return False

    inst = instances[choice - 1]
    gpu.vast_instance_id = str(inst["id"])
    gpu.gpu_name = inst.get("gpu_name") or ""
    G.console.print(f"[green]Instance {gpu.vast_instance_id} selected.[/]")

    # Vast.ai API ssh_port is unreliable — always get exact details from the console command
    G.console.print(
        "[bold cyan]Paste the SSH command from the instance page[/] "
        "[dim](either the direct or relay one — e.g. ssh -p 13182 root@1.2.3.4)[/]"
    )
    ssh_str = typer.prompt("SSH command").strip()
    user, host, port = parse_vast_ssh_cmd(ssh_str)
    gpu.ssh_user = user
    gpu.ssh_host = host
    gpu.ssh_port = port
    G.console.print(f"[dim]connecting as {gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}[/]")
    return True


def _manual_ssh_setup(cfg: "Config", gpu: "GpuConfig") -> None:
    """Prompt user for SSH command and auto-detect instance ID from API."""
    G.console.print(
        "[bold cyan]Paste the SSH command from your Vast.ai console[/] "
        "[dim](e.g. ssh root@1.2.3.4 -p 10022)[/]"
    )
    ssh_str = typer.prompt("SSH command").strip()
    user, host, port = parse_vast_ssh_cmd(ssh_str)
    gpu.ssh_user = user
    gpu.ssh_host = host
    gpu.ssh_port = port

    if cfg.vast_api_key and not gpu.vast_instance_id:
        try:
            vast = VastClient(cfg.vast_api_key)
            for inst in vast.list():
                if inst.get("ssh_host") == host or inst.get("public_ipaddr") == host:
                    gpu.vast_instance_id = str(inst["id"])
                    G.console.print(f"[dim]auto-detected instance id: {gpu.vast_instance_id}[/]")
                    break
            vast.close()
        except G.MichaelError:
            pass

    G.console.print(f"[green]GPU config saved[/] ({gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port})")


def _boot_poll(gpu: "GpuConfig", max_boot: int = 300, poll_s: int = 10) -> None:
    """Poll SSH until the instance responds or timeout. Raises MichaelError on failure."""
    G.console.print("[dim]start requested — waiting for SSH to come up…[/]")
    elapsed = 0
    while elapsed < max_boot:
        time.sleep(poll_s)
        elapsed += poll_s
        try:
            cp = _gpu_ssh_run(gpu, "echo ok", timeout=60)
            if cp.returncode == 0:
                return
            reason = (cp.stderr or "").strip()[:120] or f"rc={cp.returncode}"
        except G.MichaelError as ssh_exc:
            reason = str(ssh_exc)[:120]
        G.console.print(f"[dim]· {elapsed}s — waiting for SSH ({reason})[/]")
    raise G.MichaelError(f"instance did not respond to SSH within {max_boot}s")


def _resume_known_instance(cfg: "Config", gpu: "GpuConfig") -> None:
    """Start a known Vast.ai instance (by vast_instance_id) and wait for SSH."""
    if not cfg.vast_api_key:
        G.console.print(
            f"[dim]no vast_api_key — assuming instance is already running, "
            f"connecting to {gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}…[/]"
        )
        cp = _gpu_ssh_run(gpu, "echo ok", timeout=60)
        if cp.returncode != 0:
            raise G.MichaelError(
                f"GPU unreachable: {cp.stderr.strip()[:200]}\n"
                "Check ssh_key_path in config or try ssh manually."
            )
        return

    G.console.print(f"[dim]checking instance {gpu.vast_instance_id} via Vast.ai API…[/]")
    try:
        vast = VastClient(cfg.vast_api_key)
        info = vast.get(gpu.vast_instance_id)
        vast.close()
    except G.MichaelError as e:
        if "404" in str(e) or "no_such_instance" in str(e):
            G.console.print("[yellow]Instance not found — it was probably destroyed. Clearing stale config…[/]")
            gpu.vast_instance_id = ""
            gpu.ssh_host = ""
            cfg.gpu = gpu
            cfg.save()
            if not _select_vast_instance(cfg, gpu):
                _manual_ssh_setup(cfg, gpu)
            cfg.gpu = gpu
            cfg.save()
            return
        raise G.MichaelError(f"Vast.ai API error: {e}") from e

    # Empty info = instance no longer exists (API returned 200 with no data)
    if not info:
        G.console.print("[yellow]Instance not found — it was probably destroyed. Clearing stale config…[/]")
        gpu.vast_instance_id = ""
        gpu.ssh_host = ""
        cfg.gpu = gpu
        cfg.save()
        if not _select_vast_instance(cfg, gpu):
            _manual_ssh_setup(cfg, gpu)
        cfg.gpu = gpu
        cfg.save()
        return

    status = info.get("actual_status") or info.get("status") or ""
    if status == "running":
        G.console.print(f"[dim]instance already running — reconnecting…[/]")
        try:
            cp = _gpu_ssh_run(gpu, "echo ok", timeout=60)
            ssh_ok = cp.returncode == 0
        except G.MichaelError:
            ssh_ok = False
        if not ssh_ok:
            G.console.print(
                f"[yellow]SSH failed on saved port {gpu.ssh_port} — port may have changed.[/]\n"
                f"Paste the current SSH command from the Vast.ai instance page:"
            )
            ssh_str = typer.prompt("SSH command").strip()
            user, host, port = parse_vast_ssh_cmd(ssh_str)
            gpu.ssh_user = user
            gpu.ssh_host = host
            gpu.ssh_port = port
            cfg.gpu = gpu
            cfg.save()
        return

    G.console.print(f"[dim]instance status: {status!r} — starting via Vast.ai API…[/]")
    try:
        vast = VastClient(cfg.vast_api_key)
        vast.start(gpu.vast_instance_id)
        vast.close()
    except G.MichaelError as e:
        if "404" in str(e) or "no_such_instance" in str(e):
            G.console.print("[yellow]Instance not found — it was probably destroyed. Clearing stale config…[/]")
            gpu.vast_instance_id = ""
            gpu.ssh_host = ""
            cfg.gpu = gpu
            cfg.save()
            if not _select_vast_instance(cfg, gpu):
                _manual_ssh_setup(cfg, gpu)
            cfg.gpu = gpu
            cfg.save()
            return
        raise G.MichaelError(f"failed to start instance: {e}") from e

    _boot_poll(gpu)


def _reconnect_ssh_only(cfg: "Config", gpu: "GpuConfig") -> None:
    """Reconnect when ssh_host is known but vast_instance_id is not."""
    G.console.print(f"[dim]connecting to {gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}…[/]")
    try:
        cp = _gpu_ssh_run(gpu, "echo ok", timeout=30)
        ok = cp.returncode == 0
    except G.MichaelError:
        ok = False

    if ok:
        if cfg.vast_api_key:
            try:
                vast = VastClient(cfg.vast_api_key)
                for inst in vast.list():
                    if inst.get("ssh_host") == gpu.ssh_host or inst.get("public_ipaddr") == gpu.ssh_host:
                        gpu.vast_instance_id = str(inst["id"])
                        G.console.print(f"[dim]re-detected instance id: {gpu.vast_instance_id}[/]")
                        break
                vast.close()
            except G.MichaelError:
                pass
        return

    G.console.print("[yellow]SSH unreachable — clearing stale config and re-selecting instance…[/]")
    gpu.ssh_host = ""
    gpu.ssh_port = 22
    gpu.ssh_user = "root"
    gpu.vast_instance_id = ""
    cfg.gpu = gpu
    cfg.save()
    _clear_gpu_known_hosts()
    if not _select_vast_instance(cfg, gpu):
        _manual_ssh_setup(cfg, gpu)
    cfg.gpu = gpu
    cfg.save()


def _ollama_models_to_load(gpu: "GpuConfig") -> "list[str]":
    """The known-good tags to pull/warm, in load order: senior first, then oracle.

    Both come from config (no prompting). The oracle is optional — an empty
    ``ollama_oracle_repo`` (or one equal to the senior) collapses to a single
    co-resident model. ``model_repo`` is honoured as a legacy fallback for the
    senior so pre-existing single-model Ollama configs keep working.
    """
    senior = getattr(gpu, "ollama_senior_repo", "") or gpu.model_repo
    oracle = getattr(gpu, "ollama_oracle_repo", "")
    tags = [senior]
    if oracle and oracle != senior:
        tags.append(oracle)
    return tags


def _ollama_max_card_vram_gb(gpu: "GpuConfig") -> int:
    """Largest single-card VRAM in GB (0 if nvidia-smi is unavailable).

    Co-residency needs ONE card big enough for both models, so we take the max
    per-card total rather than the sum across cards.
    """
    cp = _gpu_ssh_run(
        gpu,
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null",
        timeout=30,
    )
    best = 0
    for line in cp.stdout.splitlines():
        v = line.strip()
        if v.isdigit():
            best = max(best, int(v) // 1024)  # MiB → GB
    return best


def _ollama_ensure_model(cfg: "Config", gpu: "GpuConfig", tag: str) -> None:
    """Pull one Ollama tag if not already present, reporting progress as events.

    Idempotent: a tag already on the card is skipped. Disk is preflighted
    against ``_MODEL_MIN_DISK_GB`` so an out-of-space card fails with an
    actionable message instead of a half-written blob.
    """
    cp = _gpu_ssh_run(
        gpu,
        f"ollama list 2>/dev/null | awk 'NR>1 {{print $1}}' | grep -Fxq {tag!r} "
        f"&& echo present || echo missing",
        timeout=60,
    )
    if "present" in cp.stdout:
        G.console.print(f"[dim]model {tag} already present[/]")
        return

    disk_kb = _gpu_ssh_run(gpu, "df / | awk 'NR==2{print $4}'", timeout=60).stdout.strip()
    min_gb = _MODEL_MIN_DISK_GB.get(tag, 30)
    if disk_kb.isdigit() and int(disk_kb) < min_gb * 1_000_000:
        avail_gb = int(disk_kb) // 1_000_000
        raise G.MichaelError(
            f"Not enough disk space to pull {tag} "
            f"(only ~{avail_gb} GB free, need ~{min_gb} GB). Free space and retry:\n"
            f"  ssh -p {gpu.ssh_port} {gpu.ssh_user}@{gpu.ssh_host} "
            f"'rm -rf /root/.ollama/models/ && df -h /'"
        )
    G.console.print(f"[cyan]Pulling model {tag} (this can take a while)…[/]")
    _gpu_ssh_run(
        gpu,
        "rm -f /tmp/ollama_pull.exit && "
        "( nohup bash -c "
        f"'ollama pull {tag} > /tmp/ollama_pull.log 2>&1; "
        "echo $? > /tmp/ollama_pull.exit' "
        "> /dev/null 2>&1 < /dev/null & ) && echo started",
        timeout=60,
    )
    _max_pull_s = 3600
    _poll_s = 15
    _elapsed = 0
    while _elapsed < _max_pull_s:
        time.sleep(_poll_s)
        _elapsed += _poll_s
        cp = _gpu_ssh_run(
            gpu, "cat /tmp/ollama_pull.exit 2>/dev/null || echo running", timeout=180
        )
        done = cp.stdout.strip()
        if done and done != "running":
            rc = int(done) if done.lstrip("-").isdigit() else 1
            if rc != 0:
                tail = _gpu_ssh_run(
                    gpu, "tail -30 /tmp/ollama_pull.log 2>/dev/null", timeout=60
                ).stdout
                raise G.MichaelError(f"ollama pull of {tag} failed (rc={rc}):\n{tail.strip()}")
            G.console.print(f"[green]model {tag} pulled[/]")
            return
        tail = _ANSI.sub("", _gpu_ssh_run(
            gpu, "tail -1 /tmp/ollama_pull.log 2>/dev/null", timeout=180
        ).stdout.strip().replace("\r", " "))
        G.console.print(
            f"[dim]· {tag} {_elapsed}s — {(tail[:110] + '…') if len(tail) > 110 else (tail or 'starting pull…')}[/]"
        )
        append_event("gpu.poll", {"elapsed_s": _elapsed, "phase": "pull", "model": tag})
    raise G.MichaelError(
        f"ollama pull of {tag} did not finish within {_max_pull_s}s. "
        "SSH in and tail /tmp/ollama_pull.log for the real status."
    )


def _ollama_warm_model(gpu: "GpuConfig", tag: str) -> None:
    """Preload a tag into VRAM with an indefinite keep-alive so it stays HOT.

    Best-effort: a model that fails to warm here will still load on the first
    real request, so we report the failure but do not abort the whole bring-up.
    """
    cp = _gpu_ssh_run(
        gpu,
        f"curl -sf http://localhost:{gpu.gpu_port}/api/generate "
        f"-d '{{\"model\":\"{tag}\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":-1}}' "
        f">/dev/null 2>&1 && echo warmed || echo warm_failed",
        timeout=600,
    )
    if "warmed" in cp.stdout:
        G.console.print(f"[green]model {tag} loaded into VRAM (kept hot)[/]")
    else:
        G.console.print(
            f"[yellow]could not preload {tag} (will load on first request): "
            f"{cp.stderr.strip()[:120] or cp.stdout.strip()[:120]}[/]"
        )


def _point_all_profiles_at(cfg: "Config", *, endpoint: str, served_model_name: str) -> str:
    """Point every model profile at one endpoint serving one model.

    Used by the single-model path (vLLM full precision): god, oracle, analyst —
    every profile resolves to the same endpoint and the same served_model_name.
    Profiles still differ by their own knobs (enable_thinking, slim_context).
    Missing 'god'/'oracle' profiles are created. Returns the endpoint.
    """
    from michael.config import ModelProfile

    senior_name = cfg.default_model or "god"
    if senior_name not in cfg.models:
        cfg.models[senior_name] = ModelProfile(enable_thinking=True)
    cfg.default_model = senior_name
    if "oracle" not in cfg.models:
        cfg.models["oracle"] = ModelProfile(enable_thinking=False)

    for prof in cfg.models.values():
        prof.endpoint = endpoint
        prof.served_model_name = served_model_name
        prof.gpu_name = ""  # single shared GPU
    cfg.save()
    return endpoint


def _assign_ollama_profiles(
    cfg: "Config", gpu: "GpuConfig", senior_tag: str, oracle_tag: str
) -> str:
    """Point every model profile at the one shared Ollama endpoint.

    The senior ('god' / default_model) and the oracle (the tool_uncapable
    profile, created as 'oracle' if absent) differ ONLY by served_model_name —
    both share this endpoint, this tunnel, this port. The analyst profile, if
    present, is repointed at the same endpoint so it rides the senior's warm GPU
    with no second instance. Returns the endpoint.
    """
    from michael.config import ModelProfile

    endpoint = f"http://localhost:{gpu.gpu_port}/v1"

    senior_name = cfg.default_model or "god"
    senior = cfg.models.setdefault(senior_name, ModelProfile(enable_thinking=True))
    senior.endpoint = endpoint
    senior.served_model_name = senior_tag
    senior.gpu_name = ""  # single shared GPU — no named-GPU tunnel selection
    if not cfg.default_model:
        cfg.default_model = senior_name

    if oracle_tag:
        # Reuse an existing tool_uncapable profile if there is one; else 'oracle'.
        oracle_name = next(
            (n for n, p in cfg.models.items() if p.tool_uncapable and n != senior_name),
            "oracle",
        )
        oracle = cfg.models.setdefault(oracle_name, ModelProfile(tool_uncapable=True))
        oracle.endpoint = endpoint
        oracle.served_model_name = oracle_tag
        oracle.tool_uncapable = True
        oracle.gpu_name = ""

    analyst = cfg.models.get("analyst")
    if analyst is not None:
        analyst.endpoint = endpoint
        analyst.gpu_name = ""

    cfg.save()
    return endpoint


def _run_ollama_setup(cfg: "Config", gpu: "GpuConfig", profile_name: str = "") -> None:
    """One-shot, non-interactive Ollama bring-up.

    Everything here runs without a single prompt — the two model tags are read
    from config (``gpu.ollama_senior_repo`` / ``gpu.ollama_oracle_repo``). The
    only human step is the SSH handshake, which has already happened upstream in
    ``_run_gpu_setup_protocol``. Steps (each reported to the console + event
    log): install ollama → start the daemon with both-models-hot env → pull both
    tags → warm both into VRAM → point both profiles at the one shared endpoint.
    """
    tags = _ollama_models_to_load(gpu)
    senior_tag = tags[0]
    oracle_tag = tags[1] if len(tags) > 1 else ""
    G.console.print(
        f"[bold]ollama one-shot[/] — senior [cyan]{senior_tag}[/]"
        + (f" + oracle [cyan]{oracle_tag}[/]" if oracle_tag else " (senior only)")
        + " · co-resident on one card, one endpoint"
    )

    # ── Install ollama if missing ──
    cp = _gpu_ssh_run(gpu, "command -v ollama >/dev/null && echo installed || echo missing")
    if "missing" in cp.stdout:
        G.console.print("[cyan]Installing ollama on the GPU (may take a minute on slow instances)…[/]")
        _gpu_ssh_run(
            gpu,
            "rm -f /tmp/ollama_install.exit && "
            "( nohup bash -c "
            "'curl -fsSL https://ollama.com/install.sh | sh > /tmp/ollama_install.log 2>&1; "
            "echo $? > /tmp/ollama_install.exit' "
            "> /dev/null 2>&1 < /dev/null & ) && echo started",
            timeout=30,
        )
        _max_install_s = 600
        _poll_s = 5
        _elapsed = 0
        while _elapsed < _max_install_s:
            time.sleep(_poll_s)
            _elapsed += _poll_s
            cp = _gpu_ssh_run(
                gpu, "cat /tmp/ollama_install.exit 2>/dev/null || echo running", timeout=30
            )
            done = cp.stdout.strip()
            if done and done != "running":
                rc = int(done) if done.lstrip("-").isdigit() else 1
                if rc != 0:
                    tail = _gpu_ssh_run(
                        gpu, "tail -30 /tmp/ollama_install.log 2>/dev/null", timeout=30
                    ).stdout
                    raise G.MichaelError(f"ollama install failed (exit {rc}):\n{tail}")
                break
        else:
            tail = _gpu_ssh_run(
                gpu, "tail -20 /tmp/ollama_install.log 2>/dev/null", timeout=30
            ).stdout
            raise G.MichaelError(f"ollama install timed out after {_max_install_s}s:\n{tail}")
        G.console.print("[green]ollama installed[/]")

    # ── Ensure ollama daemon is running ──
    cp = _gpu_ssh_run(gpu, _start_ollama_cmd(gpu), timeout=60)
    pid = cp.stdout.strip().split("\n")[-1]
    if not pid.isdigit():
        raise G.MichaelError(
            f"ollama failed to launch (no PID returned)\n"
            f"stdout: {cp.stdout.strip()!r}\nstderr: {cp.stderr.strip()!r}"
        )
    G.console.print(f"[dim]ollama daemon started: pid={pid}[/]")

    time.sleep(2)
    cp = _gpu_ssh_run(
        gpu,
        f"kill -0 {pid} 2>/dev/null && echo alive || "
        "(echo dead; echo '--- /tmp/ollama.log ---'; cat /tmp/ollama.log 2>/dev/null | head -30)",
        timeout=60,
    )
    if "alive" not in cp.stdout:
        raise G.MichaelError(f"ollama died shortly after launch (pid={pid}):\n{cp.stdout.strip()}")

    # ── Wait for the endpoint to answer ──
    _max_wait_s = 60
    _elapsed = 0
    daemon_ready = False
    while _elapsed < _max_wait_s:
        time.sleep(2)
        _elapsed += 2
        cp = _gpu_ssh_run(
            gpu,
            f"curl -sf http://localhost:{gpu.gpu_port}/v1/models > /dev/null 2>&1 "
            f"&& echo ready || echo down",
            timeout=60,
        )
        if "ready" in cp.stdout:
            daemon_ready = True
            break
    if not daemon_ready:
        disk = _gpu_ssh_run(gpu, "df / | awk 'NR==2{print $5}'", timeout=60).stdout.strip()
        if disk.rstrip("%").isdigit() and int(disk.rstrip("%")) >= 95:
            raise G.MichaelError(
                f"GPU disk is full ({disk} used). Free space and retry:\n"
                f"  ssh -p {gpu.ssh_port} {gpu.ssh_user}@{gpu.ssh_host} "
                f"'rm -rf /root/.ollama/models/ && df -h /'"
            )
        diag = _gpu_ssh_run(
            gpu,
            "echo '--- /tmp/ollama.log ---'; cat /tmp/ollama.log 2>&1; "
            "echo '--- ollama processes ---'; ps -ef | grep -i ollama | grep -v grep; "
            "echo '--- port ---'; ss -tlnp 2>/dev/null | grep "
            f"{gpu.gpu_port} || netstat -tlnp 2>/dev/null | grep {gpu.gpu_port} "
            "|| echo '(nothing listening)'",
            timeout=60,
        ).stdout
        raise G.MichaelError(
            f"ollama daemon did not become ready within {_max_wait_s}s\n{diag.strip()}"
        )

    # ── VRAM floor (warn-only): co-residency needs one card big enough for both ──
    floor = getattr(gpu, "ollama_min_vram_gb", 0) or 0
    if floor:
        card_gb = _ollama_max_card_vram_gb(gpu)
        if card_gb and card_gb < floor:
            G.console.print(
                f"[yellow]warning: largest GPU is ~{card_gb} GB, below the validated "
                f"{floor} GB floor for {senior_tag}"
                + (f" + {oracle_tag}" if oracle_tag else "")
                + " at Q8_0.[/]\n[yellow]Proceeding anyway — if a model fails to load "
                "(OOM), drop gpu.ollama_senior_repo/ollama_oracle_repo to smaller tags "
                "and re-run `michael gpu up`.[/]"
            )
        elif card_gb:
            G.console.print(f"[dim]largest GPU ~{card_gb} GB ≥ {floor} GB floor[/]")

    # ── Pull both tags (idempotent), then warm both into VRAM ──
    for tag in tags:
        _ollama_ensure_model(cfg, gpu, tag)
    for tag in tags:
        _ollama_warm_model(gpu, tag)

    # Report co-residency from the server's own view.
    ps = _gpu_ssh_run(gpu, "ollama ps 2>/dev/null", timeout=60).stdout.strip()
    if ps:
        G.console.print(f"[dim]ollama ps:\n{ps}[/]")

    # ── Point every profile at the one shared endpoint ──
    endpoint = _assign_ollama_profiles(cfg, gpu, senior_tag, oracle_tag)
    append_event(
        "gpu.ready",
        {"host": gpu.ssh_host, "models": tags, "endpoint": endpoint, "backend": "ollama"},
    )

    pf_cmd = gpu_port_forward_cmd(gpu)
    model_lines = f"  senior : {senior_tag}\n" + (f"  oracle : {oracle_tag}\n" if oracle_tag else "")
    G.console.print(
        Panel(
            f"[bold green]ollama is ready[/] — both models co-resident on one endpoint\n\n"
            f"{model_lines}\n"
            f"[bold]Open a new terminal and run:[/]\n\n"
            f"  {pf_cmd}\n\n"
            f"[dim]Keep that terminal open. Then use:[/]\n"
            f"  michael run <your prompt>",
            title="port forward",
            border_style="green",
        )
    )


def _run_vllm_setup(cfg: "Config", gpu: "GpuConfig", profile_name: str = "") -> None:
    """Install vLLM if missing, start server (downloads model on first run), save endpoint."""
    # ── Detect GPU count ──
    ngpu_cp = _gpu_ssh_run(
        gpu,
        "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l",
        timeout=30,
    )
    ngpu_str = ngpu_cp.stdout.strip()
    ngpu = int(ngpu_str) if ngpu_str.isdigit() and int(ngpu_str) > 0 else 1
    G.console.print(f"[dim]detected {ngpu} GPU(s)[/]")

    # ── Resolve the GPU's Python interpreter ──
    # Vast images often lack a bare `python`, and the CUDA torch stack lives in
    # an env that non-interactive SSH does not put on PATH. _GPU_PY picks the
    # right interpreter; we install into and launch from that same one.
    py_cp = _gpu_ssh_run(gpu, _GPU_PY + 'echo "$PY"', timeout=30)
    gpu_py = py_cp.stdout.strip().split("\n")[-1]
    if not gpu_py:
        raise G.MichaelError(
            "no Python interpreter found on the GPU (tried python3, python, "
            "conda and venv paths).\nUse a template that ships Python — the "
            "Vast.ai PyTorch template works."
        )
    G.console.print(f"[dim]using GPU interpreter: {gpu_py}[/]")

    # ── Install vLLM if missing ──
    cp = _gpu_ssh_run(
        gpu,
        _GPU_PY + '"$PY" -c "import vllm" 2>/dev/null && echo installed || echo missing',
        timeout=30,
    )
    if "missing" in cp.stdout:
        G.console.print("[cyan]Installing vLLM on the GPU (pip install, may take ~2 min)…[/]")
        cp = _gpu_ssh_run(gpu, _GPU_PY + '"$PY" -m pip install vllm --quiet', timeout=600)
        if cp.returncode != 0:
            raise G.MichaelError(f"vLLM install failed:\n{(cp.stderr or cp.stdout)[:500]}")
        G.console.print("[green]vLLM installed[/]")

    # ── Drop flashinfer if curand.h is not accessible to nvcc ──────────────
    # flashinfer's sampling kernels JIT-compile on first use and require
    # curand.h.  On Vast.ai images where curand-dev is absent the JIT build
    # crashes and vLLM aborts at startup.  Test with nvcc itself (the same
    # compiler flashinfer uses) — test -f was unreliable.  If the compile
    # fails, uninstall flashinfer so vLLM uses its built-in flash-attn
    # attention + PyTorch sampling fallback (confirmed "FLASH_ATTN backend"
    # in the startup log, so no attention regression).
    cp = _gpu_ssh_run(
        gpu,
        'printf \'#include <curand.h>\\nint main(){return 0;}\\n\' > /tmp/_curand_test.cu && '
        '/usr/local/cuda/bin/nvcc /tmp/_curand_test.cu -o /tmp/_curand_test 2>/dev/null && '
        'echo CURAND_OK || echo CURAND_MISSING',
        timeout=30,
    )
    if "CURAND_MISSING" in cp.stdout:
        G.console.print("[yellow]curand.h not found by nvcc — uninstalling flashinfer (vLLM will use flash-attn + PyTorch sampler)[/]")
        _gpu_ssh_run(
            gpu,
            _GPU_PY + '"$PY" -m pip uninstall flashinfer -y 2>/dev/null; true',
            timeout=60,
        )
        G.console.print("[green]flashinfer removed — vLLM fallback active[/]")

    # ── Preflight: torch must be able to talk to this GPU's driver ──
    # pip's torch is built for a recent CUDA; on older cards (e.g. Titan RTX)
    # the driver can be too old, and vLLM dies deep in engine init with a
    # cryptic stack. Catch it here with an actionable message instead.
    cp = _gpu_ssh_run(
        gpu,
        _GPU_PY + '"$PY" -c "import torch; torch.zeros(1).cuda()" 2>&1 && echo CUDA_OK',
        timeout=120,
    )
    if "CUDA_OK" not in cp.stdout:
        out = (cp.stdout + cp.stderr).strip()
        hint = ""
        if "too old" in out.lower() or ("driver" in out.lower() and "cuda" in out.lower()):
            hint = (
                "\n\nThe installed torch needs a newer CUDA than this GPU's driver "
                "supports — common on older cards. Either rerun `michael gpu up` and "
                "choose the 'ollama' backend (it bundles its own CUDA runtime), or "
                "use a newer GPU (Ampere/Ada, e.g. A100/L40/RTX 4090)."
            )
        raise G.MichaelError(f"torch cannot initialize CUDA on this GPU:\n{out[-800:]}{hint}")

    # ── Start vLLM server ──
    # Stop any prior server in its own SSH session first — never chain the kill
    # with the launch (see _stop_vllm_cmd: pkill -f would match the launching
    # shell's own argv and kill it before `echo $!`, the "no PID returned" bug).
    # Pre-Ampere GPUs (Titan RTX et al., compute cap < 8.0) need launch
    # overrides vLLM won't pick on its own: --dtype half (no bfloat16) and, for
    # AWQ checkpoints, --quantization awq (the auto-selected awq_marlin kernel
    # needs sm80+). Without these the engine dies at init.
    # Pascal and older (compute < 7.0, e.g. Tesla P40 = 6.1) are not supported
    # by vLLM v1 at all — its CUDA kernels require Volta (sm70+).
    _cap = _gpu_compute_cap(gpu)
    if 0 < _cap < 7.0:
        raise G.MichaelError(
            f"GPU compute capability {_cap:.1f} (Pascal or older) is not supported by vLLM v1,\n"
            f"which requires Volta or newer (sm70 / compute 7.0+).\n\n"
            f"Rerun `michael gpu up` and select backend 2 (ollama) — it bundles its own\n"
            f"CUDA runtime and works on older GPUs like the Tesla P40."
        )
    dtype, quant = _gpu_vllm_overrides(gpu, _cap)
    if dtype or quant:
        extras = " ".join(
            f"--{k} {v}" for k, v in (("dtype", dtype), ("quantization", quant)) if v
        )
        G.console.print(f"[dim]pre-Ampere GPU — launching with {extras}[/]")
    if getattr(gpu, "max_model_len", 0):
        G.console.print(
            f"[dim]context capped at --max-model-len {gpu.max_model_len} "
            f"(raise gpu.max_model_len for longer context on bigger GPUs)[/]"
        )
    _gpu_ssh_run(gpu, _stop_vllm_cmd(), timeout=30)
    cp = _gpu_ssh_run(gpu, _start_vllm_cmd(gpu, ngpu, dtype, quant), timeout=60)
    pid = cp.stdout.strip().split("\n")[-1]
    if not pid.isdigit():
        raise G.MichaelError(
            f"vLLM failed to launch (no PID returned)\n"
            f"stdout: {cp.stdout.strip()!r}\nstderr: {cp.stderr.strip()!r}"
        )
    G.console.print(f"[dim]vLLM server started: pid={pid}, tensor-parallel-size={ngpu}[/]")

    time.sleep(2)
    cp = _gpu_ssh_run(
        gpu, f"kill -0 {pid} 2>/dev/null && echo alive || echo dead", timeout=60
    )
    if "alive" not in cp.stdout:
        raise G.MichaelError(
            f"vLLM died shortly after launch (pid={pid}):\n{_vllm_crash_report(gpu)}"
        )

    # ── Poll /v1/models until ready ──
    # First start downloads model weights from HuggingFace before the endpoint becomes healthy.
    _max_wait_s = 2400  # 40 min ceiling; breaks out early once ready
    _poll_s = 15
    _elapsed = 0
    server_ready = False
    G.console.print(
        f"[dim]waiting for vLLM to load {gpu.model_repo} "
        f"(first run downloads from HuggingFace, ~15–30 min)…[/]"
    )
    while _elapsed < _max_wait_s:
        time.sleep(_poll_s)
        _elapsed += _poll_s
        cp = _gpu_ssh_run(
            gpu,
            f"curl -sf http://localhost:{gpu.gpu_port}/v1/models > /dev/null 2>&1 "
            f"&& echo ready || echo down",
            timeout=60,
        )
        if "ready" in cp.stdout:
            server_ready = True
            break
        # Fail fast if the engine crashed instead of waiting out the 40-min ceiling.
        live = _gpu_ssh_run(
            gpu, f"kill -0 {pid} 2>/dev/null && echo alive || echo dead", timeout=30
        )
        if "alive" not in live.stdout:
            raise G.MichaelError(
                f"vLLM engine exited during startup (pid={pid}):\n"
                f"{_vllm_crash_report(gpu)}"
            )
        tail_cp = _gpu_ssh_run(gpu, "tail -2 /tmp/vllm.log 2>/dev/null", timeout=60)
        tail_line = _ANSI.sub("", tail_cp.stdout.strip().replace("\r", " "))
        G.console.print(
            f"[dim]· {_elapsed}s — {(tail_line[:120] + '…') if len(tail_line) > 120 else (tail_line or 'loading model…')}[/]"
        )
        append_event("gpu.poll", {"elapsed_s": _elapsed, "phase": "vllm_warmup"})

    if not server_ready:
        disk = _gpu_ssh_run(gpu, "df / | awk 'NR==2{print $5}'", timeout=60).stdout.strip()
        if disk.rstrip("%").isdigit() and int(disk.rstrip("%")) >= 95:
            raise G.MichaelError(
                f"GPU disk is full ({disk} used). Free space and retry:\n"
                f"  ssh -p {gpu.ssh_port} {gpu.ssh_user}@{gpu.ssh_host} "
                f"'rm -rf ~/.cache/huggingface/hub/ && df -h /'"
            )
        diag = _gpu_ssh_run(
            gpu,
            "echo '--- /tmp/vllm.log (last 40 lines) ---'; tail -40 /tmp/vllm.log 2>&1; "
            "echo '--- vllm processes ---'; ps -ef | grep -i vllm | grep -v grep; "
            "echo '--- port ---'; ss -tlnp 2>/dev/null | grep "
            f"{gpu.gpu_port} || netstat -tlnp 2>/dev/null | grep {gpu.gpu_port} "
            "|| echo '(nothing listening)'",
            timeout=60,
        ).stdout
        raise G.MichaelError(
            f"vLLM server did not become ready within {_max_wait_s}s\n{diag.strip()}"
        )

    # ── Save endpoint — one model, every profile points at it ──
    endpoint = _point_all_profiles_at(cfg, endpoint=f"http://localhost:{gpu.gpu_port}/v1",
                                      served_model_name=gpu.model_repo)
    append_event("gpu.ready", {"host": gpu.ssh_host, "model": gpu.model_repo, "endpoint": endpoint, "backend": "vllm"})

    pf_cmd = gpu_port_forward_cmd(gpu)
    gpu_label = f" ({gpu.gpu_name})" if gpu.gpu_name else ""
    G.console.print(
        Panel(
            f"[bold green]vLLM is ready[/]{gpu_label} — {gpu.model_repo}\n\n"
            f"[bold]Open a new terminal and run:[/]\n\n"
            f"  {pf_cmd}\n\n"
            f"[dim]Keep that terminal open. Then use:[/]\n"
            f"  michael run <your prompt>\n\n"
            f"[dim]Tail the server log:[/]\n"
            f"  ssh -p {gpu.ssh_port} {gpu.ssh_user}@{gpu.ssh_host} 'tail -f /tmp/vllm.log'",
            title="port forward",
            border_style="green",
        )
    )


def _run_gpu_setup_protocol(cfg: "Config", gpu: "GpuConfig", profile_name: str = "") -> None:
    """Verify SSH, auto-detect installed backend, then dispatch to setup."""
    # ── Check local SSH key exists before attempting connection ──
    key_path = pathlib.Path(gpu.ssh_key_path).expanduser()
    if not key_path.exists():
        raise G.MichaelError(
            f"SSH private key not found: {key_path}\n\n"
            f"Generate one with:  ssh-keygen -t ed25519\n"
            f"Then add the public key (~/.ssh/id_ed25519.pub) to your Vast.ai account:\n"
            f"  console.vast.ai → Account → SSH Keys"
        )

    # ── Wait for SSH — instance may still be booting ──
    G.console.print(f"[dim]waiting for SSH on {gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}…[/]")
    _ssh_retries = 6
    _ssh_wait = 15
    ssh_ok = False
    last_err = ""
    for attempt in range(1, _ssh_retries + 1):
        try:
            cp = _gpu_ssh_run(gpu, "echo ok", timeout=30)
            if cp.returncode == 0:
                ssh_ok = True
                break
            last_err = (cp.stderr or "").strip()[:200]
        except G.MichaelError as e:
            last_err = str(e)[:200]
        G.console.print(f"[dim]· attempt {attempt}/{_ssh_retries} — {last_err[:80] or 'no response'} (retry in {_ssh_wait}s)[/]")
        time.sleep(_ssh_wait)
    if not ssh_ok:
        raise G.MichaelError(
            f"SSH unreachable after {_ssh_retries} attempts ({gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}).\n"
            f"Last error: {last_err}\n\n"
            f"Things to check:\n"
            f"  1. Your SSH public key is saved in Vast.ai → Account → SSH Keys\n"
            f"     (paste contents of {key_path}.pub)\n"
            f"  2. Try manually:  ssh -p {gpu.ssh_port} {gpu.ssh_user}@{gpu.ssh_host}\n"
            f"  3. If that works, run `michael gpu` again."
        )

    # ── One model, one GPU, no prompts. The model is pinned in config
    #    (gpu.model_repo, default NousResearch/Hermes-4.3-36B) and served at full
    #    precision (bf16) via vLLM — quantization stays whatever config says
    #    (default ""=bf16). The interactive backend/model menus only appear if no
    #    model is pinned (a deliberately blanked config), so the normal path and
    #    every re-run run end-to-end with zero questions after the SSH handshake. ──
    if not gpu.model_repo:
        cp_ollama = _gpu_ssh_run(gpu, "command -v ollama >/dev/null 2>&1 && echo yes || echo no", timeout=30)
        cp_vllm = _gpu_ssh_run(
            gpu, _GPU_PY + '"$PY" -c "import vllm" 2>/dev/null && echo yes || echo no', timeout=30
        )
        has_ollama = "yes" in cp_ollama.stdout
        has_vllm = "yes" in cp_vllm.stdout
        default_backend = gpu.inference_backend or ("ollama" if (has_ollama and not has_vllm) else "vllm")
        gpu.inference_backend = _prompt_backend_selection(default_backend)
        if gpu.inference_backend == "vllm":
            custom = gpu.custom_vllm_models
            gpu.model_repo = _prompt_model_selection(gpu.model_repo, backend="vllm", custom_models=custom)
            gpu.custom_vllm_models = custom

    if gpu.inference_backend == "vllm":
        prec = f"quantized ({gpu.quantization})" if (getattr(gpu, "quantization", "") or "") else "full precision (bf16)"
        G.console.print(f"[dim]vLLM · {gpu.model_repo} · {prec} · one GPU — no prompts[/]")
    else:
        G.console.print(f"[dim]ollama · {gpu.ollama_senior_repo or gpu.model_repo} — no prompts[/]")

    # Single shared GPU — always cfg.gpu (named-GPU machinery removed).
    cfg.gpu = gpu
    cfg.save()

    # ── Dispatch to backend-specific setup ──
    if gpu.inference_backend == "vllm":
        _run_vllm_setup(cfg, gpu, profile_name)
    else:
        _run_ollama_setup(cfg, gpu, profile_name)


def cmd_gpu() -> None:
    """Select which GPU to use from your Vast.ai instances. Saves the selection; run `gpu up` to start it."""
    cfg = Config.load()
    gpu = cfg.gpu

    if not _select_vast_instance(cfg, gpu):
        _manual_ssh_setup(cfg, gpu)

    cfg.gpu = gpu
    cfg.save()

    label = f" ({gpu.gpu_name})" if gpu.gpu_name else ""
    G.console.print(
        f"[green]GPU selected{label}[/] — instance {gpu.vast_instance_id or gpu.ssh_host}\n"
        f"[dim]Run [bold]michael gpu up[/bold] to start it.[/]"
    )


def cmd_gpu_up(gpu_name: str = "god") -> None:
    """Start the one shared GPU: resume instance, install backend if needed, serve.

    Michael runs a single GPU serving every model behind one endpoint, so the
    legacy ``gpu_name`` argument is accepted only for CLI compatibility — any
    value resolves to the primary ``cfg.gpu``.
    """
    cfg = Config.load()
    if gpu_name not in ("", "god"):
        G.console.print(
            f"[dim]named GPUs were collapsed into one shared GPU — '{gpu_name}' "
            f"maps to the primary GPU.[/]"
        )

    gpu = cfg.gpu
    if not gpu.ssh_host and not gpu.vast_instance_id:
        cmd_gpu()
        cfg = Config.load()
        gpu = cfg.gpu
    if gpu.vast_instance_id:
        _resume_known_instance(cfg, gpu)
    else:
        G.console.print(f"[dim]connecting to {gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}…[/]")
        cp = _gpu_ssh_run(gpu, "echo ok", timeout=60)
        if cp.returncode != 0:
            raise G.MichaelError(
                f"GPU unreachable: {cp.stderr.strip()[:200]}\n"
                "Check ssh_key_path in config or try ssh manually."
            )
    _run_gpu_setup_protocol(cfg, gpu)


def _clear_gpu_known_hosts() -> None:
    """Drop the Michael-managed GPU known_hosts file.

    Called whenever GPU SSH state is reset, so that a freshly rented Vast.ai
    instance reusing a previous IP does not hard-fail on a host-key mismatch
    under StrictHostKeyChecking=accept-new.
    """
    try:
        G.GPU_KNOWN_HOSTS_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def cmd_gpu_new() -> None:
    """Forget the current GPU selection and pick a new one, then run gpu up."""
    cfg = Config.load()
    if cfg.gpu.ssh_host or cfg.gpu.vast_instance_id:
        label = f" ({cfg.gpu.gpu_name})" if cfg.gpu.gpu_name else ""
        G.console.print(
            f"[dim]clearing GPU{label}: {cfg.gpu.ssh_user}@{cfg.gpu.ssh_host}"
            f":{cfg.gpu.ssh_port} (instance {cfg.gpu.vast_instance_id or '—'})[/]"
        )
    cfg.gpu.ssh_host = ""
    cfg.gpu.ssh_port = 22
    cfg.gpu.ssh_user = "root"
    cfg.gpu.vast_instance_id = ""
    cfg.gpu.gpu_name = ""
    for profile in cfg.models.values():
        profile.endpoint = None
        profile.served_model_name = ""
    cfg.save()
    _clear_gpu_known_hosts()
    G.console.print("[green]gpu cleared[/]")
    cmd_gpu_up()


def cmd_gpu_down(gpu_name: str = "god") -> None:
    cfg = Config.load()
    gpu = cfg.gpu  # single shared GPU — gpu_name kept only for CLI compatibility
    if not gpu.ssh_host:
        raise G.MichaelError("no GPU configured — run `michael gpu up` first")

    # Stop inference server via SSH (best-effort — instance may already be off)
    if gpu.inference_backend == "vllm":
        stop_cmd = "pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true"
        server_name = "vLLM"
    else:
        stop_cmd = "systemctl stop ollama 2>/dev/null || pkill -x ollama 2>/dev/null || true"
        server_name = "ollama"
    cp = _gpu_ssh_run(gpu, stop_cmd, timeout=60)
    if cp.returncode == 0:
        G.console.print(f"[yellow]{server_name} stopped[/]")
    else:
        G.console.print(f"[dim]SSH unreachable — skipping {server_name} stop (instance likely already off)[/]")

    if gpu.vast_instance_id and cfg.vast_api_key:
        vast = VastClient(cfg.vast_api_key)
        try:
            vast.stop(gpu.vast_instance_id)
            G.console.print(f"[yellow]instance {gpu.vast_instance_id} stopped[/]")
            append_event("gpu.stopped", {"host": gpu.ssh_host, "instance_id": gpu.vast_instance_id})
        finally:
            vast.close()
    else:
        G.console.print("[dim]no vast_instance_id or vast_api_key — skipping API stop[/]")
        append_event("gpu.stopped", {"host": gpu.ssh_host})

    # One GPU serves every profile — clear them all.
    for prof in cfg.models.values():
        prof.endpoint = None
    cfg.save()


def cmd_gpu_logs(lines: int = 120, gpu_name: str = "god") -> None:
    """Tail the inference server log on the GPU so server-side crashes are visible.

    The CLI only sees 'Server disconnected' / 'Bad Request' from its side of the
    tunnel; the real traceback lives in the server log on the GPU. This surfaces
    it without a manual SSH.
    """
    cfg = Config.load()
    gpu = cfg.get_gpu(gpu_name)
    if not gpu.ssh_host:
        raise G.MichaelError(f"no GPU {gpu_name!r} configured — run `michael gpu up {gpu_name}` first")

    # Is the server even alive / listening?
    health = _gpu_ssh_run(
        gpu,
        f"curl -sf http://localhost:{gpu.gpu_port}/v1/models > /dev/null 2>&1 "
        f"&& echo ready || echo down",
        timeout=60,
    )
    if "ready" in health.stdout:
        G.console.print(f"[green]server is up[/] on port {gpu.gpu_port}")
    else:
        G.console.print(
            f"[yellow]server not responding[/] on port {gpu.gpu_port} — "
            "it likely crashed; the log below should say why"
        )

    if gpu.inference_backend == "vllm":
        # _vllm_crash_report greps for the root cause then tails the log.
        G.console.print(_vllm_crash_report(gpu))
    else:
        out = _gpu_ssh_run(
            gpu,
            "journalctl -u ollama --no-pager -n "
            f"{lines} 2>/dev/null || tail -{lines} /tmp/ollama*.log 2>&1",
            timeout=60,
        ).stdout
        G.console.print(out.strip() or "(no ollama log found)")


def cmd_status() -> None:
    cfg = Config.load()
    state = replay_global()
    active = get_active_project()
    table = Table(title="michael status", border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("active project", active.slug if active else "(none)")
    if cfg.vps_active():
        table.add_row("vps", f"{cfg.vps.user}@{cfg.vps.host}:{cfg.vps.port}")
        table.add_row("vps.workspace", cfg.vps.workspace_dir)
    else:
        table.add_row("vps", "[dim]not configured (no sandbox)[/]")

    table.add_row("default model", cfg.default_model or "[dim](first available)[/]")
    if not cfg.models:
        table.add_row("models", "[dim](none — edit config.json)[/]")
    for mname, profile in cfg.models.items():
        st = state.get("models", {}).get(mname, {})
        table.add_row(
            f"  {mname}",
            f"state={st.get('instance_state', 'unknown')}  "
            f"endpoint={st.get('endpoint') or profile.endpoint or '—'}",
        )

    table.add_row("errors (global)", str(state["errors"]))
    G.console.print(table)


def cmd_run(prompt: str, model_override: Optional[str] = None) -> None:
    project = require_active_project()
    cfg = Config.load()
    name, profile = cfg.get_model(model_override)
    _run_agent_loop(project, cfg, name, profile, prompt, verb_label="run")


def cmd_ask(prompt: str, system: str = "", file: Optional[str] = None) -> None:
    """Bare chat: no agent loop, no context package, no tools. Just a raw model reply."""
    cfg = Config.load()
    name, profile = cfg.get_model()
    endpoint = _require_endpoint(profile, name)
    _ssh_preflight(cfg)
    _ask_gpu = cfg.get_gpu(profile.gpu_name)
    if _ask_gpu.ssh_host:
        _ensure_tunnel(profile.gpu_name or "god", _ask_gpu)
    user_content = prompt
    if file:
        p = pathlib.Path(file).expanduser()
        if not p.exists():
            G.err.print(f"file not found: {p}")
            raise typer.Exit(1)
        file_text = p.read_text(errors="replace")
        user_content = f"{prompt}\n\n--- {p.name} ---\n{file_text}" if prompt else file_text
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    client = llm_client(endpoint)
    G.console.print(f"[dim]michael ask · {name} ({profile.served_model_name})[/]")
    try:
        chunks = client.chat.completions.create(
            model=profile.served_model_name,
            messages=messages,
            stream=True,
            timeout=float(profile.request_timeout_s),
        )
        for chunk in chunks:
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()
    finally:
        client.close()


def cmd_log(tail: int) -> None:
    project = get_active_project()
    if project:
        events = iter_events(project.events_path)
        title = f"events (project: {project.slug})"
    else:
        events = iter_events(G.GLOBAL_EVENTS_PATH)
        title = "events (global)"
    if not events:
        G.console.print("[dim](no events)[/]")
        return
    last = events[-tail:] if tail > 0 else events
    table = Table(
        title=f"{title} — last {len(last)} of {len(events)}",
        border_style="cyan",
    )
    table.add_column("seq", style="bold", justify="right")
    table.add_column("ts")
    table.add_column("type")
    table.add_column("payload")
    for ev in last:
        payload = json.dumps(ev.get("payload", {}), ensure_ascii=False, sort_keys=True)
        if len(payload) > 80:
            payload = payload[:77] + "..."
        table.add_row(
            str(ev.get("seq", "?")),
            str(ev.get("ts", "?")),
            str(ev.get("type", "?")),
            payload,
        )
    G.console.print(table)


def cmd_inspect() -> None:
    project = require_active_project()
    cfg = Config.load()
    scripture = load_scripture(cfg.scripture_dir)
    header = build_header(project, cfg.resolved_system_prompt(), scripture)
    prompts = _prompt_history_lines(project)
    actions = _action_log_lines(project)
    G.console.print(f"\n[bold cyan]Project:[/] {project.name}  [dim]({project.slug})[/]")
    G.console.print(
        f"[dim]H1 prompts: {len(prompts)} · H3 tool calls: {len(actions)} · "
        f"context size: {len(header):,} chars[/]\n"
    )
    G.console.print(header)


def cmd_undo(list_only: bool = False, trash_id: Optional[str] = None) -> None:
    project = require_active_project()
    if list_only:
        entries = _list_trash(project)
        if not entries:
            G.console.print("(no trash)")
            return
        table = Table(
            title=f"trash for {project.slug} (newest last)",
            border_style="cyan",
        )
        table.add_column("trash_id", style="bold")
        table.add_column("ts")
        table.add_column("tool")
        table.add_column("delta")
        table.add_column("verify")
        for m in entries:
            d = m.get("delta", {}) or {}
            delta_summary = (
                f"+{len(d.get('added', []))} "
                f"~{len(d.get('modified', []))} "
                f"-{len(d.get('removed', []))}"
            )
            v = m.get("verify_rc")
            v_str = "—" if v is None else f"rc={v}"
            table.add_row(
                str(m.get("trash_id", "?")),
                str(m.get("ts", "?")),
                str(m.get("tool", "?")),
                delta_summary,
                v_str,
            )
        G.console.print(table)
        return
    metadata = _undo_one(project, trash_id)
    append_event(
        "tool.undone",
        {
            "trash_id": metadata.get("trash_id"),
            "tool": metadata.get("tool"),
            "summary": metadata.get("summary", ""),
        },
        project=project,
    )
    G.console.print(
        f"[green]undone[/] {metadata.get('tool')} ({metadata.get('trash_id')})"
    )


def cmd_sandbox(file: pathlib.Path, net: bool = False, timeout: int = 30) -> None:
    cfg = Config.load()
    _ssh_preflight(cfg)
    backend = make_backend(cfg)
    project = get_active_project()
    code = pathlib.Path(file).read_text()
    cp = backend.run(code, network=net, timeout_s=timeout, project=project)
    stdout_tail = "\n".join((cp.stdout or "").splitlines()[-80:])
    stderr_tail = "\n".join((cp.stderr or "").splitlines()[-40:])
    G.console.print(
        Panel(
            stdout_tail or "(empty)",
            title=f"stdout (rc={cp.returncode})",
            border_style="green" if cp.returncode == 0 else "red",
        )
    )
    if stderr_tail:
        G.console.print(Panel(stderr_tail, title="stderr", border_style="red"))


def cmd_ssh_test() -> None:
    cfg = Config.load()
    if not cfg.vps_active():
        raise G.MichaelError("vps.host is not configured")
    t0 = time.monotonic()
    cp = subprocess.run(
        _ssh_argv(cfg.vps) + ["echo ok && podman --version 2>/dev/null || true"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    dt = round(time.monotonic() - t0, 3)
    if cp.returncode != 0:
        append_event("ssh.health", {"host": cfg.vps.host, "ok": False, "stderr": cp.stderr[:200]})
        raise G.MichaelError(f"ssh failed in {dt}s: {cp.stderr.strip()[:200]}")
    append_event("ssh.health", {"host": cfg.vps.host, "ok": True, "duration_s": dt})
    G.console.print(
        Panel(
            cp.stdout.strip() or "(no output)",
            title=f"ssh ok in {dt}s — {cfg.vps.user}@{cfg.vps.host}",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Catalog / install / deliver commands
# ---------------------------------------------------------------------------


def cmd_catalog() -> None:
    catalog = load_catalog()
    if not catalog:
        G.console.print("[dim]catalog is empty — deliver a tool first[/]")
        return
    table = Table(title=f"tool catalog ({len(catalog)} tools)", border_style="cyan")
    table.add_column("slug", style="bold")
    table.add_column("description")
    table.add_column("installed", style="green")
    table.add_column("built_at", style="dim")
    for slug, entry in sorted(catalog.items()):
        table.add_row(
            slug,
            str(entry.get("description", "—"))[:60],
            str(entry.get("installed_as") or "—"),
            str(entry.get("built_at", "—"))[:19],
        )
    G.console.print(table)


def cmd_install(slug: Optional[str]) -> None:
    catalog = load_catalog()
    if not catalog:
        raise G.MichaelError("catalog is empty — no tools to install")
    if slug is None:
        proj = get_active_project()
        if not proj:
            raise G.MichaelError("no active project and no slug given")
        slug = proj.slug
    entry = catalog.get(slug)
    if not entry:
        raise G.MichaelError(f"tool {slug!r} not found in catalog")
    deliverable = entry.get("deliverable", "")
    if not deliverable:
        raise G.MichaelError(f"no deliverable path recorded for {slug!r}")
    src = pathlib.Path(deliverable).expanduser()
    if not src.is_file():
        raise G.MichaelError(f"deliverable not found: {src}")
    G.MICHAEL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    link = G.MICHAEL_BIN_DIR / slug
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(src)
    if not src.stat().st_mode & 0o111:
        src.chmod(src.stat().st_mode | 0o755)
    run_cmd = str(link)
    from michael.project import save_catalog
    catalog[slug]["installed_as"] = str(link)
    catalog[slug]["run_cmd"] = run_cmd
    save_catalog(catalog)
    G.console.print(
        Panel(
            f"[bold green]installed[/] {slug}\n"
            f"  symlink: {link} → {src}\n\n"
            f"Add to PATH:\n  export PATH=\"{G.MICHAEL_BIN_DIR}:$PATH\"",
            title="michael install",
            border_style="green",
        )
    )


def cmd_path() -> None:
    p = get_active_project()
    if not p:
        raise G.MichaelError("no active project")
    G.console.print(p.path)


def cmd_deliver() -> None:
    project = require_active_project()
    det = detect_deliverable(project)
    if not det:
        raise G.MichaelError("no deliverable detected in this project (look for main.py, app.py, *.sh, etc.)")
    deliverable, run_cmd = det
    register_deliverable(project, deliverable, run_cmd)
    G.console.print(
        Panel(
            f"[bold green]delivered[/] {deliverable}\n"
            f"installed: [cyan]{G.MICHAEL_BIN_DIR / project.slug}[/]\n\n"
            f"[dim]Add to PATH: export PATH=\"{G.MICHAEL_BIN_DIR}:$PATH\"[/]",
            title="michael deliver",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Tools workspace commands
# ---------------------------------------------------------------------------

_TOOL_DIR_LABELS = [
    ("bundled", pathlib.Path(__file__).parent.parent / "toolbox"),
    ("global",  pathlib.Path(G.GLOBAL_TOOLS_DIR)),
]


def _tool_search_dirs(project_path: str | None) -> list[tuple[str, pathlib.Path]]:
    dirs = list(_TOOL_DIR_LABELS)
    if project_path:
        dirs.append(("project", pathlib.Path(project_path) / "tools"))
    return dirs


def _parse_kv_args(tokens: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise typer.BadParameter(f"expected key=value, got {token!r}")
        k, _, v = token.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _find_tool_file(name: str, project_path: str | None) -> pathlib.Path | None:
    # Project-local takes priority, then global, then bundled.
    search = list(reversed(_tool_search_dirs(project_path)))
    for _label, d in search:
        candidate = d / f"{name}.py"
        if candidate.exists():
            return candidate
    return None


def cmd_tools_list() -> None:
    project_path: str | None = None
    try:
        project_path = require_active_project().path
    except G.MichaelError:
        pass

    import importlib.util as _ilu

    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    # Reverse priority so highest-priority entry wins display
    for label, d in reversed(_tool_search_dirs(project_path)):
        if not d.is_dir():
            continue
        for py_file in sorted(d.glob("*.py")):
            try:
                spec = _ilu.spec_from_file_location(py_file.stem, py_file)
                mod = _ilu.module_from_spec(spec)       # type: ignore[arg-type]
                spec.loader.exec_module(mod)             # type: ignore[union-attr]
                if not hasattr(mod, "TOOL_SCHEMA"):
                    continue
                fn_name = mod.TOOL_SCHEMA.get("function", {}).get("name", py_file.stem)
                if fn_name in seen:
                    continue
                seen.add(fn_name)
                desc = mod.TOOL_SCHEMA.get("function", {}).get("description", "")
                desc = desc.strip().splitlines()[0][:72] if desc else ""
                rows.append((fn_name, desc, label))
            except Exception as exc:
                G.err.print(f"[dim]skipped {py_file.name}: {exc}[/]")

    if not rows:
        G.console.print("[dim]no dynamic tools found[/]")
        return

    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False, min_width=60)
    t.add_column("Name", style="cyan", no_wrap=True)
    t.add_column("Description", no_wrap=False)
    t.add_column("Source", style="dim", no_wrap=True)
    for name, desc, label in sorted(rows, key=lambda r: r[0]):
        t.add_row(name, desc, label)
    G.console.print(t)


def cmd_tools_run(name: str, kv_tokens: list[str]) -> None:
    project_path: str | None = None
    try:
        project_path = require_active_project().path
    except G.MichaelError:
        pass

    py_file = _find_tool_file(name, project_path)
    if py_file is None:
        raise G.MichaelError(f"tool {name!r} not found in any toolbox directory")

    try:
        args = _parse_kv_args(kv_tokens)
    except typer.BadParameter as e:
        raise G.MichaelError(str(e)) from e

    result = _dispatch_dynamic_tool_from_path(name, args, py_file)
    G.console.print(result)


def cmd_tools_show(name: str) -> None:
    project_path: str | None = None
    try:
        project_path = require_active_project().path
    except G.MichaelError:
        pass

    py_file = _find_tool_file(name, project_path)
    if py_file is None:
        raise G.MichaelError(f"tool {name!r} not found in any toolbox directory")

    from rich.syntax import Syntax
    G.console.print(f"[dim]{py_file}[/]")
    G.console.print(Syntax(py_file.read_text(), "python", line_numbers=True))


# ---------------------------------------------------------------------------
# Typer command bindings
# ---------------------------------------------------------------------------


@app.command(name="init")
def init_cmd() -> None:
    """Write stub config, create workbench dirs, inject shell integration. Idempotent."""
    cmd_init()


@app.command(name="show")
def show_cmd() -> None:
    """List projects."""
    cmd_show()


@app.command(name="new")
def new_cmd(
    name: Optional[str] = typer.Argument(None, help="Project name."),
) -> None:
    """Create a new project."""
    cmd_new(name)


@app.command(name="use")
def use_cmd(slug: str = typer.Argument(...)) -> None:
    """Set the active project."""
    cmd_use(slug)


@app.command(name="current")
def current_cmd() -> None:
    """Print the active project."""
    cmd_current()


@app.command(name="mission")
def mission_cmd(text: Optional[str] = typer.Argument(None, help="New mission text. Omit to view.")) -> None:
    """View or set the mission objective for the active project."""
    proj = require_active_project()
    p = pathlib.Path(proj.path) / "MISSION.md"
    if text is None:
        if p.is_file():
            G.console.print(p.read_text())
        else:
            G.console.print("[dim](no mission set — run: michael mission 'your objective')[/]")
    else:
        stripped = text.strip()
        if not stripped:
            G.console.print("[yellow]mission text is empty — nothing saved[/]")
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        date_str = datetime.date.today().isoformat()
        entry = f"## {date_str}\n\n{stripped}\n"
        existing = p.read_text() if p.is_file() else ""
        if existing.strip():
            p.write_text(existing.rstrip("\n") + "\n\n" + entry)
        else:
            p.write_text(entry)
        G.console.print("[green]mission updated[/]")


@app.command(name="config")
def config_cmd() -> None:
    """Open the global config file in $EDITOR (with help panel)."""
    cmd_config()


@gpu_app.callback()
def gpu_callback(ctx: typer.Context) -> None:
    """Pick which Vast.ai GPU to use. Shows your instances with hardware names; saves the selection."""
    if ctx.invoked_subcommand is None:
        cmd_gpu()


@gpu_app.command("up")
def gpu_up_cmd(
    name: str = typer.Argument("god", help="Accepted for compatibility; one shared GPU serves every model."),
) -> None:
    """One-shot bring-up of the shared GPU: after the SSH handshake, install the
    backend and load both Ollama models (senior + oracle) hot on one endpoint —
    no further prompts. Set gpu.inference_backend='vllm' for the interactive,
    single-model vLLM path."""
    cmd_gpu_up(name)


@gpu_app.command("new")
def gpu_new_cmd() -> None:
    """Forget the current primary GPU and pick + start a fresh one."""
    cmd_gpu_new()


@gpu_app.command("down")
def gpu_down_cmd(
    name: str = typer.Argument("god", help="Accepted for compatibility; one shared GPU."),
) -> None:
    """Stop the inference server and pause the Vast.ai instance via API."""
    cmd_gpu_down(name)


@gpu_app.command("logs")
def gpu_logs_cmd(
    name: str = typer.Argument("god", help="Accepted for compatibility; one shared GPU."),
) -> None:
    """Show the inference server log on the GPU (surfaces server-side crashes)."""
    cmd_gpu_logs(gpu_name=name)


@app.command(name="status")
def status_cmd() -> None:
    """Show derived state from the event log."""
    cmd_status()


@app.command(name="run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_cmd(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model profile to use (e.g. 'junior'). Defaults to config's default_model."),
    prompt: list[str] = typer.Argument(None, help="Prompt — every word after 'run' is the prompt."),
) -> None:
    """Run the agent on a prompt. Everything after 'run' is the prompt.

    Example: michael run fix the auth bug in login.py
    Example: michael run --model junior draft the exploit
    """
    text = " ".join(prompt or []).strip()
    if not text:
        G.err.print("michael run requires a prompt. Example: michael run fix the login bug")
        raise typer.Exit(1)
    cmd_run(text, model_override=model)


@app.command(name="ask", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ask_cmd(
    prompt: list[str] = typer.Argument(None, help="Prompt words."),
    system: str = typer.Option("", "--system", "-s", help="Optional system message."),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Inject file contents into the message."),
) -> None:
    """Bare chat with the model — no agent loop, no context, no tools.

    Examples:
      michael ask are you there?
      michael ask --file /sdcard/notes.txt summarize this
    """
    text = " ".join(prompt or []).strip()
    if not text and not file:
        G.err.print("michael ask requires a prompt or --file. Example: michael ask are you there?")
        raise typer.Exit(1)
    cmd_ask(text, system, file)


@app.command(name="log")
def log_cmd(
    tail: int = typer.Option(20, "--tail", "-n", help="How many events to show."),
) -> None:
    """Show the project event log (or global if no project active)."""
    cmd_log(tail)


@app.command(name="inspect")
def inspect_cmd() -> None:
    """Print the full H1–H4 context package the model will receive on the next run."""
    cmd_inspect()


@app.command(name="sandbox")
def sandbox_cmd(
    file: pathlib.Path = typer.Argument(..., exists=True, readable=True),
    net: bool = typer.Option(False, "--net", help="Allow bridge networking."),
    timeout: int = typer.Option(30, help="Wall-clock timeout in seconds."),
) -> None:
    """Run a Python file in the sandbox (local or VPS depending on config)."""
    cmd_sandbox(file, net, timeout)


@app.command(name="undo")
def undo_cmd(
    list_only: bool = typer.Option(False, "--list", "-l", help="List trash entries."),
    trash_id: Optional[str] = typer.Argument(None, help="Specific trash id to undo."),
) -> None:
    """Restore the most recent (or named) staged change."""
    cmd_undo(list_only=list_only, trash_id=trash_id)


@app.command(name="ssh-test")
def ssh_test_cmd() -> None:
    """Verify the VPS is reachable and report the SSH handshake time."""
    cmd_ssh_test()


@app.command(name="catalog")
def catalog_cmd() -> None:
    """List all delivered tools in the global catalog."""
    cmd_catalog()


@app.command(name="path")
def path_cmd() -> None:
    """Print the active project's workspace path (useful for cd $(michael path))."""
    cmd_path()


@app.command(name="deliver")
def deliver_cmd() -> None:
    """Detect, register, and install the active project's deliverable."""
    cmd_deliver()


@app.command(name="install", hidden=True)
def install_cmd(
    slug: Optional[str] = typer.Argument(None, help="Tool slug to reinstall."),
) -> None:
    """Reinstall a delivered tool's wrapper script (repair command)."""
    cmd_install(slug)


@tools_app.command(name="list")
def tools_list_cmd() -> None:
    """List all dynamic tools across bundled, global, and project toolboxes."""
    cmd_tools_list()


@tools_app.command(
    name="run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def tools_run_cmd(
    name: str = typer.Argument(..., help="Tool name to invoke."),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Run a dynamic tool by name. Pass arguments as key=value pairs."""
    cmd_tools_run(name, ctx.args if ctx else [])


@tools_app.command(name="show")
def tools_show_cmd(
    name: str = typer.Argument(..., help="Tool name to inspect."),
) -> None:
    """Print the source code of a dynamic tool."""
    cmd_tools_show(name)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

REPL_COMMANDS = {
    "project", "new", "run", "gpu", "config", "init",
    "tools", "quit", "exit", "help",
}


def _config_is_unset() -> bool:
    if not G.GLOBAL_CONFIG_PATH.is_file():
        return True
    try:
        cfg = Config.load()
    except G.MichaelError:
        return True
    if not cfg.vast_api_key:
        return True
    return not any(p.vast_instance_id for p in cfg.models.values())


class MichaelCompleter(Completer):
    """Tab-completion for the REPL."""

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        words = text.split()
        at_boundary = text.endswith(" ") or not text

        if not words or (len(words) == 1 and not at_boundary):
            prefix = words[0] if words else ""
            for cmd in sorted(REPL_COMMANDS):
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix))
            return

        head = words[0]
        if head == "project":
            prefix = words[1] if len(words) > 1 and not at_boundary else ""
            for p in list_projects():
                if p.slug.startswith(prefix):
                    yield Completion(p.slug, start_position=-len(prefix))
            return


def repl() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(G.REPL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=MichaelCompleter(),
        complete_while_typing=False,
    )
    G.console.print("[bold cyan]michael[/] [dim]— event-sourced LLM loop[/]")
    if _config_is_unset():
        G.console.print(
            "[yellow]setup required[/] [dim]type: config[/]"
        )
    while True:
        try:
            line = session.prompt("michael> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            continue
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        try:
            dispatch_repl(line)
        except G.MichaelError as e:
            G.err.print(f"michael: {escape(str(e))}")
        except typer.Abort:
            G.err.print("aborted")
        except KeyboardInterrupt:
            G.err.print("interrupted")


def _opt_value(rest: list[str], *flags: str) -> Optional[str]:
    for f in flags:
        if f in rest:
            i = rest.index(f)
            if i + 1 < len(rest):
                return rest[i + 1]
    return None


def dispatch_repl(line: str) -> None:
    try:
        parts = shlex.split(line)
    except ValueError as e:
        G.err.print(f"parse error: {e}")
        return
    if not parts:
        return
    cmd, rest = parts[0], parts[1:]

    if cmd == "help":
        G.console.print(
            "commands:\n"
            "  run <prompt>                      run the agent on a prompt\n"
            "  project [slug]                    select/list projects\n"
            "  new [name]                        create new project\n"
            "  up / down                         start/stop GPU (legacy — needs config.json)\n"
            "  gpu up / gpu down                 start instance, install ollama, pull model\n"
            "  gpu logs                          show the GPU inference server log\n"
            "  tools list                        list all dynamic tools\n"
            "  tools run <name> [key=value ...]  run a dynamic tool directly\n"
            "  tools show <name>                 print tool source\n"
            "  catalog                           list all delivered tools\n"
            "  path                              print active project workspace path\n"
            "  deliver                           detect + install active project's deliverable\n"
            "  config                            edit config\n"
            "  init                              initialize config + shell integration\n"
            "  upgrade                           git pull + re-apply shell integration\n"
            "  exit / quit                       exit michael"
        )
        return

    if cmd == "init":
        cmd_init()
    elif cmd == "config":
        cmd_config()
    elif cmd == "project":
        if rest:
            cmd_use(rest[0])
        else:
            cmd_show()
    elif cmd == "new":
        name = " ".join(rest) if rest else None
        cmd_new(name)
    elif cmd == "run":
        if not rest:
            G.err.print("run requires a prompt. Example: run fix the auth bug")
            return
        cmd_run(" ".join(rest))
    elif cmd == "gpu":
        sub = rest[0] if rest else ""
        gpu_name_arg = rest[1] if len(rest) > 1 else "god"
        if sub == "up":
            cmd_gpu_up(gpu_name_arg)
        elif sub == "new":
            cmd_gpu_new()
        elif sub == "down":
            cmd_gpu_down(gpu_name_arg)
        elif sub == "logs":
            cmd_gpu_logs(gpu_name=gpu_name_arg)
        else:
            cmd_gpu()
    elif cmd == "tools":
        sub = rest[0] if rest else "list"
        if sub == "list":
            cmd_tools_list()
        elif sub == "run":
            if len(rest) < 2:
                G.err.print("usage: tools run <name> [key=value ...]")
            else:
                cmd_tools_run(rest[1], rest[2:])
        elif sub == "show":
            if len(rest) < 2:
                G.err.print("usage: tools show <name>")
            else:
                cmd_tools_show(rest[1])
        else:
            G.err.print("usage: tools list | tools run <name> [key=value ...] | tools show <name>")
    elif cmd == "catalog":
        cmd_catalog()
    elif cmd == "path":
        cmd_path()
    elif cmd == "deliver":
        cmd_deliver()
    else:
        G.err.print(f"unknown command: {cmd!r}. try 'help'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    try:
        if len(sys.argv) == 1:
            repl()
        else:
            app()
    except G.MichaelError as e:
        G.err.print(f"michael: {escape(str(e))}")
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        G.err.print(f"command failed (exit {e.returncode})")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        G.err.print("interrupted")
        sys.exit(130)
