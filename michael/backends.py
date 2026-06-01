"""SSH helpers, Vast.ai client, LLM client, and sandbox backends."""
from __future__ import annotations

import atexit
import dataclasses
import json as _json
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import httpx

import michael.globals as G
from michael.config import Config, GpuConfig, ModelProfile, SandboxConfig, VpsConfig

if TYPE_CHECKING:
    from michael.project import Project


# ---------------------------------------------------------------------------
# Event helpers (lazy import to avoid circular dependency)
# ---------------------------------------------------------------------------


def append_event(event_type: str, payload: dict, *, project=None) -> None:
    from michael.project import append_event as _ae
    _ae(event_type, payload, project=project)


# ---------------------------------------------------------------------------
# VPS SSH helpers
# ---------------------------------------------------------------------------


def _ssh_argv(vps: VpsConfig) -> list[str]:
    args = [
        "ssh",
        "-o", f"ControlMaster=auto",
        "-o", f"ControlPath={G.STATE_DIR}/ssh-%r@%h:%p.sock",
        "-o", f"ControlPersist={vps.control_persist}",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-i", os.path.expanduser(vps.ssh_key_path),
        "-p", str(vps.port),
        f"{vps.user}@{vps.host}",
    ]
    return args


def _ssh_preflight(cfg: Config) -> None:
    if not cfg.vps_active():
        return
    cp = subprocess.run(
        _ssh_argv(cfg.vps) + ["podman --version"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if cp.returncode != 0:
        raise G.MichaelError(
            f"podman not available on VPS {cfg.vps.host}. "
            "Run `bash bootstrap.sh` on the VPS to install it."
        )


# ---------------------------------------------------------------------------
# GPU SSH helpers
# ---------------------------------------------------------------------------


def parse_vast_ssh_cmd(ssh_str: str) -> tuple[str, str, int]:
    """Parse a Vast.ai SSH string and return (user, host, port)."""
    try:
        tokens = shlex.split(ssh_str)
    except ValueError as e:
        raise G.MichaelError(f"could not parse SSH command: {e}") from e

    tokens = [t for t in tokens if t != "ssh"]
    port = 22
    user_host: Optional[str] = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-p" and i + 1 < len(tokens):
            try:
                port = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if tok.startswith("-p") and len(tok) > 2:
            try:
                port = int(tok[2:])
            except ValueError:
                pass
            i += 1
            continue
        if tok.startswith("-") and len(tok) == 2:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if "@" in tok:
            user_host = tok
        i += 1

    if not user_host:
        raise G.MichaelError(
            f"could not find user@host in SSH command: {ssh_str!r}"
        )
    parts = user_host.split("@", 1)
    return parts[0], parts[1], port


def _gpu_ssh_argv(gpu: GpuConfig) -> list[str]:
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={G.GPU_KNOWN_HOSTS_PATH}",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-i", os.path.expanduser(gpu.ssh_key_path),
        "-p", str(gpu.ssh_port),
        f"{gpu.ssh_user}@{gpu.ssh_host}",
    ]


def _gpu_ssh_run(
    gpu: GpuConfig, cmd: str, *, timeout: int = 60
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            _gpu_ssh_argv(gpu) + [cmd],
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise G.MichaelError(f"GPU command timed out after {timeout}s")


def gpu_port_forward_cmd(gpu: GpuConfig) -> str:
    key = os.path.expanduser(gpu.ssh_key_path)
    return (
        f"ssh -p {gpu.ssh_port} {gpu.ssh_user}@{gpu.ssh_host} "
        f"-L {gpu.gpu_port}:localhost:{gpu.gpu_port} "
        f"-N -o StrictHostKeyChecking=accept-new "
        f"-o UserKnownHostsFile={G.GPU_KNOWN_HOSTS_PATH} -i {key}"
    )


def _start_ollama_cmd(gpu: GpuConfig) -> str:
    """Background ollama serve detached from the SSH session, print its PID.

    Three statements, semicolon-separated, no chaining cleverness:
      1. pkill any existing ollama serve (ignored if none)
      2. touch the log so we can prove the redirect ran even if ollama crashes
      3. nohup the daemon with full std-stream redirection, echo the PID
    The caller is responsible for verifying the PID is still alive after a
    short sleep — that's done in a separate SSH call so SSH session timing
    can't affect the verification.
    """
    return (
        "pkill -x ollama 2>/dev/null; "
        "touch /tmp/ollama.log; "
        f"OLLAMA_HOST=0.0.0.0:{gpu.gpu_port} "
        "nohup ollama serve >/tmp/ollama.log 2>&1 </dev/null & "
        "echo $!"
    )


# Resolve a Python interpreter on the GPU. Vast.ai images commonly ship only
# `python3` (no bare `python`), so hardcoding `python` makes the launch die
# with "nohup: failed to run command 'python'". On top of that, the Vast.ai
# PyTorch template installs the CUDA torch stack into a specific env (conda or
# a venv) that a non-interactive SSH shell does not put on PATH — so we prefer
# the interpreter that can already `import torch` (vLLM needs it, and reusing
# the prebuilt CUDA torch avoids pip pulling a mismatched build), falling back
# to the first usable python. Prepend this to any remote command that needs the
# interpreter and reference it as "$PY"; installing AND launching with the same
# "$PY" guarantees vLLM is importable where we start it. POSIX-sh compatible.
_GPU_PY = (
    'PY=""; '
    'for _c in python3 python /opt/conda/bin/python /venv/main/bin/python '
    '/usr/local/bin/python3 /usr/bin/python3; do '
    '_p="$(command -v "$_c" 2>/dev/null)"; '
    '[ -z "$_p" ] && [ -x "$_c" ] && _p="$_c"; '
    '[ -z "$_p" ] && continue; '
    '[ -z "$PY" ] && PY="$_p"; '
    'if "$_p" -c "import torch" >/dev/null 2>&1; then PY="$_p"; break; fi; '
    'done; '
)


def _stop_vllm_cmd() -> str:
    """Kill any running vLLM server.

    This MUST run in its own SSH session — never chained ahead of the launch
    command. `pkill -f` matches against the *full command line* of every
    process, and the launch command's own argv contains the server module
    path, so chaining `pkill -f` with the launch makes pkill kill the very
    shell that is about to `echo $!` — the bug behind "vLLM failed to launch
    (no PID returned)". The `[v]llm` bracket is the standard self-exclusion
    trick: the regex matches a real `vllm.…` process but not this command's
    own argv (which contains the literal text `[v]llm.…`). The trailing
    `true` keeps the exit status 0 when no server was running.
    """
    return "pkill -f '[v]llm.entrypoints.openai.api_server' 2>/dev/null; true"


def _gpu_compute_cap(gpu: GpuConfig) -> float:
    """Return the GPU's CUDA compute capability (e.g. 7.5), or 0.0 if unknown."""
    cp = _gpu_ssh_run(
        gpu,
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1",
        timeout=30,
    )
    raw = cp.stdout.strip().split("\n")[-1].strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _gpu_vllm_overrides(gpu: GpuConfig) -> tuple[Optional[str], Optional[str]]:
    """Return (dtype, quantization) launch overrides for this GPU + model.

    Pre-Ampere cards (compute capability < 8.0, e.g. Titan RTX / Turing) need
    two things vLLM does not pick for them automatically:

    - `--dtype half`: they can't run bfloat16, which most checkpoints default to.
      Without this the engine dies at init with "Engine core initialization
      failed" within seconds.
    - `--quantization awq` for AWQ checkpoints: vLLM auto-selects the
      `awq_marlin` kernel, but Marlin requires sm80+, so on Turing it fails
      fast. The plain `awq` kernel works on sm75.

    On Ampere+ we return (None, None) and let vLLM decide.
    """
    cap = _gpu_compute_cap(gpu)
    if not (0 < cap < 8.0):
        return None, None
    quant = "awq" if "awq" in gpu.model_repo.lower() else None
    return "half", quant


def _vllm_tool_parser(model_repo: str) -> str:
    """Pick vLLM's tool-call parser for this model.

    Michael's agent loop sends `tools=…` + `tool_choice="auto"` on every
    request. vLLM's OpenAI server rejects any request containing `tools` with
    a 400 unless launched with `--enable-auto-tool-choice` and a matching
    `--tool-call-parser`. The parser must match the model's chat template:

      - Qwen2.5 / Qwen3        → hermes
      - DeepSeek V3 / V4        → deepseek_v3
      - Llama 3.x               → llama3_json
      - Mistral                 → mistral

    Defaults to hermes, the most broadly compatible parser.
    """
    m = model_repo.lower()
    if "deepseek" in m:
        return "deepseek_v3"
    if "llama" in m:
        return "llama3_json"
    if "mistral" in m:
        return "mistral"
    return "hermes"  # qwen and everything else


def _vllm_crash_report(gpu: GpuConfig) -> str:
    """Pull the real root cause out of /tmp/vllm.log after an engine crash.

    vLLM's V1 engine runs EngineCore in a child process; the actual error is
    logged *there*, above the API server's generic "Engine core initialization
    failed" wrapper. A small tail misses it, so we surface both a grep of the
    likely root-cause lines and a generous tail of the log.
    """
    pattern = (
        "error|exception|traceback|runtimeerror|valueerror|assert|"
        "importerror|modulenotfound|out of memory|no kernel image|"
        "not supported|unsupported|compute capability"
    )
    cmd = (
        "echo '--- likely root cause ---'; "
        f"grep -niE '{pattern}' /tmp/vllm.log 2>/dev/null | tail -30; "
        "echo '--- /tmp/vllm.log (last 120 lines) ---'; "
        "tail -120 /tmp/vllm.log 2>&1"
    )
    return _gpu_ssh_run(gpu, cmd, timeout=60).stdout.strip()


def _start_vllm_cmd(
    gpu: GpuConfig,
    ngpu: int = 1,
    dtype: Optional[str] = None,
    quantization: Optional[str] = None,
) -> str:
    """Background vLLM api_server detached from the SSH session, print its PID.

    Two statements, semicolon-separated:
      1. touch the log so we can prove the redirect ran even if vllm crashes
      2. nohup the server with full std-stream redirection, echo the PID

    `dtype` and `quantization`, when given, are passed as `--dtype` /
    `--quantization` (e.g. "half" + "awq" for pre-Ampere GPUs — see
    `_gpu_vllm_overrides`).

    `--max-model-len` and `--gpu-memory-utilization` come from `gpu` config.
    The former is critical: many modern checkpoints advertise a huge native
    context (e.g. Hermes-4.3-36B's 524288 tokens), and vLLM sizes the KV cache
    to serve one request at that full length. On a single GPU that demands far
    more VRAM than exists (128 GiB for 512K tokens), so the engine aborts at
    startup with "KV cache memory" before ever serving a request. Capping
    `--max-model-len` bounds the KV cache to what the card can hold. A value
    of 0 omits the flag and lets vLLM use the model's full max (rarely viable
    on one GPU). See `_check_enough_kv_cache_memory` in vLLM for the check.

    Killing a prior server is intentionally NOT done here — call
    `_stop_vllm_cmd` in a separate SSH session first. See `_stop_vllm_cmd`
    for why the two must never be chained in one shell.
    """
    dtype_flag = f"--dtype {dtype} " if dtype else ""
    quant_flag = f"--quantization {quantization} " if quantization else ""
    max_len = getattr(gpu, "max_model_len", 0) or 0
    max_len_flag = f"--max-model-len {max_len} " if max_len > 0 else ""
    mem_util = getattr(gpu, "gpu_memory_utilization", 0) or 0
    mem_util_flag = f"--gpu-memory-utilization {mem_util} " if mem_util > 0 else ""
    tool_parser = _vllm_tool_parser(gpu.model_repo)
    return (
        "touch /tmp/vllm.log; "
        + _GPU_PY +
        f'nohup "$PY" -m vllm.entrypoints.openai.api_server '
        f"--model {gpu.model_repo} "
        f"--port {gpu.gpu_port} "
        f"--host 0.0.0.0 "
        f"--tensor-parallel-size {ngpu} "
        f"--enable-auto-tool-choice "
        f"--tool-call-parser {tool_parser} "
        f"{max_len_flag}"
        f"{mem_util_flag}"
        f"{dtype_flag}"
        f"{quant_flag}"
        f">/tmp/vllm.log 2>&1 </dev/null & "
        "echo $!"
    )


def _restart_vllm_on_gpu(gpu: GpuConfig, *, poll_timeout_s: int = 1800) -> None:
    """Restart the vLLM server on the GPU and poll until ready.

    Default timeout is 30 minutes because the first start downloads the model
    from HuggingFace before the endpoint becomes healthy.
    """
    G.console.print("[yellow]restarting vLLM on GPU...[/]")
    ngpu_cp = _gpu_ssh_run(
        gpu,
        "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l",
        timeout=30,
    )
    ngpu_str = ngpu_cp.stdout.strip()
    ngpu = int(ngpu_str) if ngpu_str.isdigit() and int(ngpu_str) > 0 else 1

    dtype, quant = _gpu_vllm_overrides(gpu)
    _gpu_ssh_run(gpu, _stop_vllm_cmd(), timeout=30)
    cp = _gpu_ssh_run(gpu, _start_vllm_cmd(gpu, ngpu, dtype, quant), timeout=60)
    pid = cp.stdout.strip().split("\n")[-1]
    if not pid.isdigit():
        raise G.MichaelError(
            f"vLLM failed to launch (no PID returned)\n"
            f"stdout: {cp.stdout.strip()!r}\nstderr: {cp.stderr.strip()!r}"
        )
    elapsed = 0
    while elapsed < poll_timeout_s:
        time.sleep(15)
        elapsed += 15
        cp = _gpu_ssh_run(
            gpu,
            f"curl -sf http://localhost:{gpu.gpu_port}/v1/models > /dev/null 2>&1 "
            f"&& echo ready || echo down",
            timeout=60,
        )
        if "ready" in cp.stdout:
            G.console.print("[green]vLLM is ready[/]")
            return
        # Fail fast if the engine died instead of waiting out the full timeout.
        live = _gpu_ssh_run(
            gpu, f"kill -0 {pid} 2>/dev/null && echo alive || echo dead", timeout=30
        )
        if "alive" not in live.stdout:
            raise G.MichaelError(
                f"vLLM engine exited during startup (pid={pid}):\n"
                f"{_vllm_crash_report(gpu)}"
            )
        tail_cp = _gpu_ssh_run(gpu, "tail -2 /tmp/vllm.log 2>/dev/null", timeout=30)
        tail_line = tail_cp.stdout.strip().replace("\r", " ")
        G.console.print(f"[dim]· {elapsed}s — {tail_line[:120] or 'waiting for vLLM…'}[/]")
    raise G.MichaelError(
        f"vLLM did not come up within {poll_timeout_s}s — "
        "check /tmp/vllm.log on the GPU"
    )


def _restart_ollama_on_gpu(gpu: GpuConfig, *, poll_timeout_s: int = 300) -> None:
    """Restart the ollama daemon on the GPU and poll until ready."""
    G.console.print("[yellow]restarting ollama on GPU...[/]")
    cp = _gpu_ssh_run(gpu, _start_ollama_cmd(gpu), timeout=60)
    pid = cp.stdout.strip().split("\n")[-1]
    if not pid.isdigit():
        raise G.MichaelError(
            f"ollama failed to launch (no PID returned)\n"
            f"stdout: {cp.stdout.strip()!r}\nstderr: {cp.stderr.strip()!r}"
        )
    elapsed = 0
    while elapsed < poll_timeout_s:
        time.sleep(5)
        elapsed += 5
        cp = _gpu_ssh_run(
            gpu,
            f"curl -sf http://localhost:{gpu.gpu_port}/v1/models > /dev/null 2>&1 "
            f"&& echo ready || echo down",
            timeout=60,
        )
        if "ready" in cp.stdout:
            G.console.print("[green]ollama is ready[/]")
            return
    raise G.MichaelError(
        f"ollama did not come up within {poll_timeout_s}s — "
        "check /tmp/ollama.log or `journalctl -u ollama` on the GPU"
    )


_tunnel_procs: dict[str, "subprocess.Popen[bytes]"] = {}


def _close_tunnel(name: Optional[str] = None) -> None:
    """Terminate one named tunnel or all tunnels (if name is None)."""
    targets = [name] if name else list(_tunnel_procs.keys())
    for n in targets:
        proc = _tunnel_procs.pop(n, None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass


def _ensure_tunnel(name: str, gpu: "GpuConfig") -> None:
    """Auto-spawn SSH port-forward for a named GPU if endpoint is not reachable.

    Safe to call on every run — returns immediately if already up. Keyed by
    name so god and junior (or any other GPU) each maintain their own tunnel.
    """
    if not gpu.ssh_host:
        return
    endpoint = f"http://localhost:{gpu.gpu_port}/v1"
    if _ping_endpoint(endpoint):
        return  # already reachable — nothing to do
    G.console.print(f"[yellow]tunnel {name!r} not detected — starting SSH port-forward...[/]")
    key = os.path.expanduser(gpu.ssh_key_path)
    _tunnel_procs[name] = subprocess.Popen(
        [
            "ssh", "-p", str(gpu.ssh_port), f"{gpu.ssh_user}@{gpu.ssh_host}",
            "-L", f"{gpu.gpu_port}:localhost:{gpu.gpu_port}",
            "-N",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={G.GPU_KNOWN_HOSTS_PATH}",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-i", key,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(_close_tunnel)
    for _ in range(15):
        time.sleep(2)
        if _ping_endpoint(endpoint):
            G.console.print(f"[green]tunnel {name!r} up[/]")
            return
    _close_tunnel(name)
    raise G.MichaelError(
        f"SSH tunnel {name!r} failed to come up after 30s — "
        "check ssh_host / ssh_key_path in config"
    )


# ---------------------------------------------------------------------------
# Vast.ai API client
# ---------------------------------------------------------------------------


class VastClient:
    base = "https://console.vast.ai/api/v0"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise G.MichaelError("vast_api_key is not set")
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

    def close(self) -> None:
        self._client.close()

    def _wrap(self, fn_name: str, request) -> Any:
        try:
            r = request()
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:200]
            msg = f"vast {fn_name}: HTTP {e.response.status_code} — {body}"
            append_event("error", {"where": fn_name, "msg": msg})
            raise G.MichaelError(msg) from e
        except httpx.HTTPError as e:
            msg = f"vast {fn_name}: {e}"
            append_event("error", {"where": fn_name, "msg": msg})
            raise G.MichaelError(msg) from e

    def get(self, inst_id: str | int) -> dict[str, Any]:
        data = self._wrap(
            "get",
            lambda: self._client.get(f"{self.base}/instances/{inst_id}/"),
        )
        return data.get("instances", {}) or {}

    def start(self, inst_id: str | int) -> dict[str, Any]:
        return self._wrap(
            "start",
            lambda: self._client.put(
                f"{self.base}/instances/{inst_id}/", json={"state": "running"}
            ),
        )

    def stop(self, inst_id: str | int) -> dict[str, Any]:
        return self._wrap(
            "stop",
            lambda: self._client.put(
                f"{self.base}/instances/{inst_id}/", json={"state": "stopped"}
            ),
        )

    def list(self) -> list[dict[str, Any]]:
        data = self._wrap("list", lambda: self._client.get(f"{self.base}/instances/"))
        return data.get("instances", []) or []

    def endpoint_for(self, inst_id: str | int, port: int) -> Optional[str]:
        info = self.get(inst_id)
        if not info:
            return None
        ip = info.get("public_ipaddr") or info.get("ssh_host")
        if not ip:
            return None
        return f"http://{ip}:{port}/v1"


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ToolCall:
    id: str
    name: str
    arguments: str


@dataclasses.dataclass
class _Choice:
    content: Optional[str]
    tool_calls: Optional[list[_ToolCall]]
    finish_reason: Optional[str]


@dataclasses.dataclass
class _CompletionResponse:
    choices: list[_Choice]
    usage: Optional[dict]


def _http_error_message(r: httpx.Response, model: str) -> str:
    """Build an actionable error from a non-2xx model-server response.

    The OpenAI/Ollama endpoint puts the real reason in the response body
    (e.g. "model X does not support tools", "model not found"); httpx's
    default raise_for_status() discards it, leaving only a generic status
    line. Surface the body and, where recognisable, a concrete next step.
    """
    body = (r.text or "").strip()
    detail = body
    # Ollama / OpenAI errors are JSON: {"error": "..."} or {"error": {"message": "..."}}.
    try:
        parsed = r.json().get("error", "")
        if isinstance(parsed, dict):
            detail = parsed.get("message", "") or detail
        elif isinstance(parsed, str) and parsed:
            detail = parsed
    except Exception:
        pass
    detail = detail[:500] if detail else "(empty response body)"

    msg = f"model server rejected request ({r.status_code}): {detail}"
    low = detail.lower()
    if "does not support tools" in low or "support tools" in low:
        msg += (
            f"\n\nThe model '{model}' has no tool-calling template, but Michael always "
            "sends tools. Set `gpu.model_repo` to a tool-capable model "
            "(e.g. qwen2.5:72b, llama3.1:70b) and re-run `michael gpu up`."
        )
    elif "not found" in low and model and model in detail:
        msg += (
            f"\n\nThe model '{model}' is not pulled on the server. Run `michael gpu up` "
            "to pull it, or fix `models.god.served_model_name` / `gpu.model_repo`."
        )
    elif not model:
        msg += (
            "\n\nNo model name was sent (served_model_name is empty). "
            "Run `michael gpu up` to populate it from `gpu.model_repo`."
        )
    return msg


_TEXT_TOOL_CALL_RE = __import__("re").compile(r"<tool_call>(.*?)</tool_call>", __import__("re").DOTALL)


def _parse_text_tool_calls(content: str) -> list:
    """Parse <tool_call>JSON</tool_call> blocks from a model's text response."""
    results = []
    for i, m in enumerate(_TEXT_TOOL_CALL_RE.finditer(content)):
        try:
            parsed = _json.loads(m.group(1).strip())
        except _json.JSONDecodeError as exc:
            G.err.print(f"[yellow]text tool call {i}: malformed JSON — skipped: {exc}[/]")
            continue
        name = parsed.get("name", "")
        if not name:
            G.err.print(f"[yellow]text tool call {i}: missing 'name' — skipped[/]")
            continue
        arguments = parsed.get("arguments", {})
        results.append(
            _ToolCall(
                id=f"call_txt_{i}",
                name=name,
                arguments=_json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
            )
        )
    return results


class _Completions:
    def __init__(
        self,
        endpoint: str,
        http: httpx.Client,
        headers: dict,
        enable_thinking: bool = False,
        tool_uncapable: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._http = http
        self._headers = headers
        self._enable_thinking = enable_thinking
        self._tool_uncapable = tool_uncapable

    def create(
        self,
        *,
        model: str,
        messages: list,
        tools: Optional[list] = None,
        tool_choice: Optional[Any] = None,
        stream: bool = False,
        timeout: float = 60.0,
        stream_options: Optional[dict] = None,
        **_kw: Any,
    ) -> Any:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if tools and not self._tool_uncapable:
            body["tools"] = tools
        if self._enable_thinking and not any(m.get("role") == "tool" for m in messages):
            body["chat_template_kwargs"] = {"enable_thinking": True}
        if tool_choice is not None and not self._tool_uncapable:
            body["tool_choice"] = tool_choice
        if stream and stream_options:
            body["stream_options"] = stream_options
        if stream:
            return self._stream_iter(body, timeout)
        r = self._http.post(
            f"{self._endpoint}/chat/completions",
            json=body,
            timeout=timeout,
        )
        if r.status_code >= 400:
            raise G.MichaelError(_http_error_message(r, model))
        try:
            data = r.json()
        except Exception as exc:
            raise G.MichaelError(
                f"model server returned non-JSON response: {exc} — body: {r.text[:200]!r}"
            ) from exc
        return self._parse_response(data)

    def _stream_iter(self, body: dict, timeout: float):
        client = httpx.Client(
            headers=self._headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        try:
            with client.stream(
                "POST", f"{self._endpoint}/chat/completions", json=body
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    raise G.MichaelError(_http_error_message(r, body.get("model", "")))
                for line in r.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            yield self._parse_chunk(_json.loads(line[6:]))
                        except (_json.JSONDecodeError, KeyError):
                            continue
        finally:
            client.close()

    def _parse_response(self, data: dict) -> _CompletionResponse:
        choices = []
        for c in data.get("choices", []):
            m = c.get("message", {})
            if self._tool_uncapable:
                text_tcs = _parse_text_tool_calls(m.get("content") or "")
                tool_calls: Optional[list] = text_tcs or None
            else:
                tcs = m.get("tool_calls") or []
                tool_calls = [
                    _ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", ""),
                    )
                    for tc in tcs
                ] or None
            choices.append(
                _Choice(
                    content=m.get("content"),
                    tool_calls=tool_calls,
                    finish_reason=c.get("finish_reason"),
                )
            )
        return _CompletionResponse(choices=choices, usage=data.get("usage"))

    def _parse_chunk(self, data: dict) -> _Choice:
        c = data.get("choices", [{}])[0]
        delta = c.get("delta", {})
        tcs = delta.get("tool_calls") or []
        tool_calls: Optional[list] = [
            _ToolCall(
                id=tc.get("id", ""),
                name=(tc.get("function") or {}).get("name", ""),
                arguments=(tc.get("function") or {}).get("arguments", ""),
            )
            for tc in tcs
        ] or None
        return _Choice(
            content=delta.get("content"),
            tool_calls=tool_calls,
            finish_reason=c.get("finish_reason"),
        )


class _ChatCompletions:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class LLMClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        enable_thinking: bool = False,
        tool_uncapable: bool = False,
    ) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(headers=headers, timeout=120)
        _completions = _Completions(endpoint, self._http, headers, enable_thinking, tool_uncapable)
        self.chat = _ChatCompletions(_completions)

    def close(self) -> None:
        self._http.close()


def llm_client(
    endpoint: str,
    api_key: str = "",
    enable_thinking: bool = False,
    tool_uncapable: bool = False,
) -> LLMClient:
    return LLMClient(endpoint, api_key, enable_thinking, tool_uncapable)


def _require_endpoint(profile: ModelProfile, name: str) -> str:
    if not profile.endpoint:
        raise G.MichaelError(
            f"model '{name}' has no endpoint — run `michael up` or `michael gpu up` first"
        )
    return profile.endpoint


def _ping_endpoint(endpoint: str, *, timeout_s: float = 5.0) -> bool:
    try:
        r = httpx.get(f"{endpoint}/models", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sandbox backends
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int


class SandboxBackend(ABC):
    @abstractmethod
    def run(
        self,
        code: str,
        *,
        network: bool = False,
        timeout_s: int = 30,
        project=None,
    ) -> SandboxResult:
        ...


class LocalPodmanBackend(SandboxBackend):
    def __init__(self, cfg: SandboxConfig) -> None:
        self._cfg = cfg

    def run(
        self,
        code: str,
        *,
        network: bool = False,
        timeout_s: int = 30,
        project=None,
    ) -> SandboxResult:
        cfg = self._cfg
        if not shutil.which("podman"):
            raise G.MichaelError(
                "podman not found on PATH. "
                "Install podman locally, or configure a remote VPS sandbox:\n"
                "  michael config  →  set vps.host to your VPS IP"
            )
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            cmd = [
                "podman", "run", "--rm",
                "--memory", f"{cfg.memory_mb}m",
                "--cpus", str(cfg.cpus),
                "--pids-limit", str(cfg.pids),
                "--network", "bridge" if network else "none",
                "-v", f"{tmp}:/sandbox/script.py:ro",
                cfg.image,
                "python3", "/sandbox/script.py",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
                return SandboxResult(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                )
            except subprocess.TimeoutExpired:
                raise G.MichaelError(f"sandbox timed out after {timeout_s}s") from None
            except FileNotFoundError:
                raise G.MichaelError("podman not found — cannot run sandbox locally.") from None
        finally:
            pathlib.Path(tmp).unlink(missing_ok=True)


class SubprocessBackend(SandboxBackend):
    """No-isolation fallback: runs code directly with the host python3."""

    def __init__(self, timeout_s: int) -> None:
        self._timeout_s = timeout_s

    def run(
        self,
        code: str,
        *,
        network: bool = False,
        timeout_s: int = 30,
        project=None,
    ) -> SandboxResult:
        effective = timeout_s or self._timeout_s
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            try:
                result = subprocess.run(
                    ["python3", tmp],
                    capture_output=True,
                    text=True,
                    timeout=effective,
                    check=False,
                )
                return SandboxResult(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                )
            except subprocess.TimeoutExpired:
                raise G.MichaelError(f"sandbox timed out after {effective}s") from None
        finally:
            pathlib.Path(tmp).unlink(missing_ok=True)


class RemotePodmanBackend(SandboxBackend):
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def _stage(self, code: str, remote_path: str) -> None:
        cfg = self._cfg
        vps = cfg.vps
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            scp_cmd = [
                "scp",
                "-o", f"ControlMaster=auto",
                "-o", f"ControlPath={G.STATE_DIR}/ssh-%r@%h:%p.sock",
                "-o", f"ControlPersist={vps.control_persist}",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-i", os.path.expanduser(vps.ssh_key_path),
                "-P", str(vps.port),
                tmp,
                f"{vps.user}@{vps.host}:{remote_path}",
            ]
            cp_stage = subprocess.run(
                scp_cmd, capture_output=True, text=True, timeout=30, check=False
            )
            if cp_stage.returncode != 0:
                raise G.MichaelError(f"remote stage failed: {cp_stage.stderr[:200]}")
        finally:
            pathlib.Path(tmp).unlink(missing_ok=True)

    def run(
        self,
        code: str,
        *,
        network: bool = False,
        timeout_s: int = 30,
        project=None,
    ) -> SandboxResult:
        cfg = self._cfg
        _ssh_preflight(cfg)
        vps = cfg.vps
        sb = cfg.sandbox
        run_id = uuid.uuid4().hex[:8]
        remote_script = f"{vps.workspace_dir}/run_{run_id}.py"
        self._stage(code, remote_script)
        net_flag = "bridge" if network else "none"
        podman_cmd = (
            f"podman run --rm "
            f"--memory {sb.memory_mb}m "
            f"--cpus {sb.cpus} "
            f"--pids-limit {sb.pids} "
            f"--network {net_flag} "
            f"-v {remote_script}:/sandbox/script.py:ro "
            f"{sb.image} python3 /sandbox/script.py"
        )
        try:
            cp = subprocess.run(
                _ssh_argv(vps) + [podman_cmd],
                capture_output=True,
                text=True,
                timeout=timeout_s + 15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise G.MichaelError(f"sandbox timed out after {timeout_s}s") from None
        cleanup_cmd = f"rm -f {remote_script}"
        subprocess.run(
            _ssh_argv(vps) + [cleanup_cmd],
            capture_output=True, timeout=10, check=False,
        )
        return SandboxResult(
            stdout=cp.stdout,
            stderr=cp.stderr,
            returncode=cp.returncode,
        )


def make_backend(cfg: Config) -> SandboxBackend:
    if cfg.sandbox.passthrough:
        return SubprocessBackend(cfg.sandbox.timeout_s)
    if cfg.vps_active():
        return RemotePodmanBackend(cfg)
    return LocalPodmanBackend(cfg.sandbox)
