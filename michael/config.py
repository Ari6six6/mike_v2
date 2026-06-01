"""Config dataclasses: ModelProfile, VpsConfig, SandboxConfig, Config."""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import michael.globals as G


@dataclass
class ModelProfile:
    """One model served behind an OpenAI-compatible endpoint (Ollama)."""

    vast_instance_id: str = ""
    served_model_name: str = ""  # the tag to send in API requests, e.g. "qwen2.5:72b"
    request_timeout_s: int = 120
    endpoint: Optional[str] = None
    enable_thinking: bool = False
    tool_uncapable: bool = False
    slim_context: bool = False  # strip H1-H4 package; send only minimal prompt + tool list
    gpu_name: str = ""  # which GpuConfig entry serves this model (empty = primary "god" GPU)


@dataclass
class VpsConfig:
    """Remote VPS that runs rootless Podman for sandbox execution."""

    host: str = ""
    port: int = 22
    user: str = "michael"
    ssh_key_path: str = "~/.ssh/id_ed25519"
    workspace_dir: str = "/home/michael/workspace"
    control_persist: str = "10m"


@dataclass
class SandboxConfig:
    image: str = "michael-sandbox:alpine"
    memory_mb: int = 384
    cpus: float = 1.5
    pids: int = 128
    timeout_s: int = 30
    passthrough: bool = False


@dataclass
class GpuConfig:
    """Direct SSH + GPU inference config for a rented GPU."""

    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = "root"
    ssh_key_path: str = "~/.ssh/id_ed25519"
    vast_instance_id: str = ""
    gpu_name: str = ""  # hardware name from Vast.ai, e.g. "RTX 4090"
    model_repo: str = "qwen2.5:72b"  # vLLM: HuggingFace ID. Ollama path uses ollama_senior_repo/ollama_oracle_repo instead.
    gpu_port: int = 11434
    inference_backend: str = "ollama"  # "ollama" (default, one-shot multi-model) or "vllm" (interactive, single model)
    # ── Ollama one-shot: two known-good Q8_0 tags, co-resident on ONE card behind ONE endpoint ──
    # Senior is tool-capable (drives the agent loop); oracle is a small tool_uncapable text model.
    # Both are pulled, warmed, and kept HOT (OLLAMA_MAX_LOADED_MODELS=2, OLLAMA_KEEP_ALIVE=-1).
    ollama_senior_repo: str = "qwen2.5:32b-instruct-q8_0"   # tool-capable senior, ~35 GB at Q8_0
    ollama_oracle_repo: str = "qwen2.5-coder:7b-instruct-q8_0"  # small oracle, ~8 GB at Q8_0 ("" = senior only)
    ollama_min_vram_gb: int = 80  # validated co-resident floor (one A100-80G / H100-80G); warn-only if a card is smaller
    max_model_len: int = 32768  # vLLM --max-model-len; caps KV cache so it fits VRAM (0 = let vLLM decide)
    gpu_memory_utilization: float = 0.92  # vLLM --gpu-memory-utilization (fraction of VRAM for the engine)
    quantization: str = ""  # vLLM --quantization override (e.g. "bitsandbytes", "fp8", "gptq"); "" = auto
    nccl_p2p_disable: bool = False  # set True on Vast.ai multi-GPU if NCCL fails to init (adds NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1)
    custom_vllm_models: list = field(default_factory=list)   # user-added HuggingFace model IDs
    custom_ollama_models: list = field(default_factory=list)  # user-added Ollama tags


@dataclass
class Config:
    vast_api_key: str = ""
    models: dict[str, ModelProfile] = field(default_factory=dict)
    default_model: str = ""
    vps: VpsConfig = field(default_factory=VpsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    gpus: dict[str, GpuConfig] = field(default_factory=dict)  # named extra GPU instances
    system_prompt: str = G.DEFAULT_SYSTEM_PROMPT
    system_prompt_file: str = ""
    log_responses: bool = True
    analyst_enabled: bool = False  # run the post-run analyst (needs a models.analyst profile)
    boot_poll_s: int = 10
    scripture_dir: str = "scripture"
    workbench_root: str = "~/workbench"
    router_enabled: bool = False

    @classmethod
    def load(cls) -> "Config":
        data: dict[str, Any] = {}
        if G.GLOBAL_CONFIG_PATH.is_file():
            try:
                data = json.loads(G.GLOBAL_CONFIG_PATH.read_text())
            except json.JSONDecodeError as e:
                raise G.MichaelError(f"config.json is not valid JSON: {e}") from e

        if v := os.environ.get("VAST_API_KEY"):
            data["vast_api_key"] = v
        if v := os.environ.get("MICHAEL_DEFAULT_MODEL"):
            data["default_model"] = v

        models_raw = data.pop("models", {}) or {}
        models: dict[str, ModelProfile] = {}
        valid_mp = set(ModelProfile.__dataclass_fields__)
        for name, prof in models_raw.items():
            if isinstance(prof, dict):
                # god profiles get enable_thinking=True unless explicitly set to False
                if name == "god" and "enable_thinking" not in prof:
                    prof = {**prof, "enable_thinking": True}
                models[name] = ModelProfile(**{k: v for k, v in prof.items() if k in valid_mp})

        vps_raw = data.pop("vps", None) or {}
        valid_vps = set(VpsConfig.__dataclass_fields__)
        vps = VpsConfig(**{k: v for k, v in vps_raw.items() if k in valid_vps})

        sb_raw = data.pop("sandbox", None) or {}
        valid_sb = set(SandboxConfig.__dataclass_fields__)
        sandbox = SandboxConfig(**{k: v for k, v in sb_raw.items() if k in valid_sb})

        gpu_raw = data.pop("gpu", None) or {}
        valid_gpu = set(GpuConfig.__dataclass_fields__)
        gpu = GpuConfig(**{k: v for k, v in gpu_raw.items() if k in valid_gpu})

        gpus_raw = data.pop("gpus", {}) or {}
        gpus: dict[str, GpuConfig] = {}
        for gname, gcfg in gpus_raw.items():
            if isinstance(gcfg, dict):
                gpus[gname] = GpuConfig(**{k: v for k, v in gcfg.items() if k in valid_gpu})

        valid = set(cls.__dataclass_fields__) - {"models", "vps", "sandbox", "gpu", "gpus"}
        clean = {k: v for k, v in data.items() if k in valid}
        return cls(models=models, vps=vps, sandbox=sandbox, gpu=gpu, gpus=gpus, **clean)

    def get_gpu(self, name: str = "") -> GpuConfig:
        """Return the single shared GPU.

        Michael now runs one GPU serving every model behind one endpoint (the
        senior and the oracle are co-resident in Ollama). The ``name`` argument
        is accepted for call-site compatibility but ignored — every profile
        resolves to ``cfg.gpu``. The legacy ``cfg.gpus`` dict still loads from
        old config files (see ``Config.load``) so nothing crashes on upgrade,
        but it no longer drives tunnel selection.
        """
        return self.gpu

    def save(self) -> None:
        G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
        G.GLOBAL_CONFIG_PATH.write_text(
            json.dumps(_diff_from_default(self, Config()), indent=2, sort_keys=True)
        )
        os.chmod(G.GLOBAL_CONFIG_PATH, 0o600)

    def get_model(self, name: Optional[str] = None) -> tuple[str, ModelProfile]:
        if not self.models:
            raise G.MichaelError(
                "no model profiles configured — run `config` and add a 'models' entry"
            )
        chosen = name or self.default_model or next(iter(self.models))
        if chosen not in self.models:
            raise G.MichaelError(
                f"unknown model profile: {chosen!r}. Available: {sorted(self.models)}"
            )
        return chosen, self.models[chosen]

    def resolved_system_prompt(self) -> str:
        if self.system_prompt_file:
            p = pathlib.Path(self.system_prompt_file).expanduser()
            if p.is_file():
                return p.read_text()
        return self.system_prompt

    def vps_active(self) -> bool:
        return bool(self.vps and self.vps.host)


def _diff_from_default(obj: Any, default: Any) -> Any:
    """Recursively prune a dataclass against a default instance of the same type.

    Returns a dict containing only fields whose values differ from the default.
    Nested dataclasses are pruned recursively; if a nested object equals its
    default entirely, it is omitted. Dicts whose values are dataclasses (e.g.
    Config.models) keep their keys but each value is pruned individually.
    """
    if not dataclasses.is_dataclass(obj):
        return obj
    out: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        cur = getattr(obj, f.name)
        dflt = getattr(default, f.name)
        if dataclasses.is_dataclass(cur) and dataclasses.is_dataclass(dflt):
            pruned = _diff_from_default(cur, dflt)
            if pruned:
                out[f.name] = pruned
        elif isinstance(cur, dict):
            pruned_dict: dict[str, Any] = {}
            for k, v in cur.items():
                if dataclasses.is_dataclass(v):
                    pruned_dict[k] = _diff_from_default(v, type(v)())
                else:
                    pruned_dict[k] = v
            if pruned_dict != dflt:
                out[f.name] = pruned_dict
        elif cur != dflt:
            out[f.name] = cur
    return out


def make_stub_config() -> Config:
    """Minimal starting point for the one-shot Ollama flow: the two known-good
    model profiles the agent loop dispatches to.

      - ``god``    — tool-capable senior; drives the agent loop.
      - ``oracle`` — small ``tool_uncapable`` text model (spawn_specialist target).

    Both are filled with their shared endpoint + served_model_name by
    ``michael gpu up`` (which pulls the two ``gpu.ollama_*_repo`` tags onto one
    card behind one port). Every other field uses its dataclass default;
    save-time pruning keeps the on-disk file to just what the user (or
    `michael gpu up`) has actually written.
    """
    return Config(
        models={
            "god": ModelProfile(enable_thinking=True),
            "oracle": ModelProfile(tool_uncapable=True),
        },
        default_model="god",
        sandbox=SandboxConfig(passthrough=True),
    )


CONFIG_HELP: dict[str, str] = {
    "vast_api_key": "Vast.ai console API key.",
    "default_model": "Profile name to use (default: 'god'). Override per-run with `michael run --model <name>`.",
    "models.god.vast_instance_id": "Numeric ID of the rented GPU instance.",
    "models.god.served_model_name": "Model name sent in API requests. Auto-filled by `michael gpu up`. For vllm: HF ID (e.g. 'NousResearch/Hermes-4.3-36B'); for ollama: tag.",
    "models.god.request_timeout_s": "LLM request timeout (seconds).",
    "models.god.enable_thinking": "Enable <think> reasoning traces (Hermes 4.3 / QwQ). Recommended for the senior model.",
    "models.god.tool_uncapable": "If true, skip tools/tool_choice params and use text-format tool calling instead (for models without a function-calling template).",
    "models.god.gpu_name": "Which named GPU serves this model (empty = primary gpu). Set to 'junior' for the specialist model.",
    "models.oracle.served_model_name": "Oracle model tag — set by `michael gpu up` from gpu.ollama_oracle_repo. Shares the senior's endpoint.",
    "models.oracle.tool_uncapable": "True — the oracle is a small text model with no native function-calling; spawn_specialist calls it as a pure oracle.",
    "gpu.inference_backend": "Inference backend: 'ollama' (default — one-shot, both models co-resident on one card) or 'vllm' (interactive, single model, true-FP8). Ollama runs end-to-end with no prompts after the SSH handshake.",
    "gpu.model_repo": "vLLM only: HuggingFace ID e.g. 'NousResearch/Hermes-4.3-36B'. The Ollama path ignores this and uses gpu.ollama_senior_repo / gpu.ollama_oracle_repo.",
    "gpu.ollama_senior_repo": "Ollama tool-capable senior tag (Q8_0), pulled onto the card and pointed at by the 'god' profile. Default 'qwen2.5:32b-instruct-q8_0' (~35 GB).",
    "gpu.ollama_oracle_repo": "Ollama small tool_uncapable oracle tag (Q8_0), co-resident with the senior and pointed at by the 'oracle' profile. Default 'qwen2.5-coder:7b-instruct-q8_0' (~8 GB). Empty = load the senior only.",
    "gpu.ollama_min_vram_gb": "Validated co-resident VRAM floor (default 80 — one A100-80G / H100-80G holds both Q8_0 models with KV headroom). Warn-only: a smaller card still proceeds.",
    "gpu.gpu_port": "OpenAI-compat port on the GPU (ollama default: 11434, vllm default: 8000 — configurable).",
    "gpu.max_model_len": "vLLM only: max context length (--max-model-len). Caps KV cache to fit VRAM. Default 32768; lower it if the engine reports 'KV cache memory' errors at startup, raise it for longer context on bigger GPUs. 0 = let vLLM use the model's full max (often too large for a single GPU).",
    "gpu.gpu_memory_utilization": "vLLM only: fraction of GPU VRAM the engine may use (--gpu-memory-utilization), 0.0–1.0. Default 0.92. Raise toward 0.95 to squeeze in more KV cache, lower if you hit OOM during load.",
    "gpu.nccl_p2p_disable": "vLLM multi-GPU: set true if vLLM crashes immediately at startup with 'WorkerProc initialization failed' on a Vast.ai (or other cloud) multi-GPU instance. Adds NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1, forcing socket-based GPU communication instead of NVLink/InfiniBand — required when GPUs are on separate PCIe buses.",
    "vps.host": "VPS public IP/hostname (empty = no remote sandbox).",
    "vps.user": "SSH user (default: michael).",
    "vps.ssh_key_path": "Path to private key (default: ~/.ssh/id_ed25519).",
    "vps.workspace_dir": "Default workspace dir on the VPS.",
    "sandbox.image": "Tag of the sandbox image built by bootstrap.sh.",
    "sandbox.memory_mb": "Sandbox memory cap in MB.",
    "sandbox.cpus": "Sandbox CPU cap.",
    "sandbox.pids": "Sandbox PID cap.",
    "sandbox.timeout_s": "Default sandbox timeout (seconds).",
    "sandbox.passthrough": "If true, run sandbox code directly via python3 (no container isolation).",
    "system_prompt": "Default system prompt for the agent loop.",
    "system_prompt_file": "If set, read system prompt from this file.",
    "log_responses": "If true, log full LLM responses to events.jsonl. Strongly recommended when analyst_enabled is on — training records lose the assistant's reasoning text without it.",
    "analyst_enabled": "If true, run the analyst after every run: it judges the run into data classes, writes a classed training corpus and a per-run scorecard. Off by default (one extra GPU call per run). Requires a 'models.analyst' profile.",
    "models.analyst.endpoint": "Analyst model endpoint. Reuse the senior GPU by setting models.analyst.gpu_name to 'god' and pointing endpoint at the same server.",
    "models.analyst.served_model_name": "Analyst model tag/ID sent in API requests.",
    "models.analyst.gpu_name": "Which named GPU serves the analyst (set 'god' to reuse the senior's warm tunnel — no second GPU needed).",
    "models.analyst.request_timeout_s": "Analyst request timeout (seconds).",
    "boot_poll_s": "Poll interval while waiting for the inference server to come up.",
    "scripture_dir": "Path to scripture files (relative to repo root, default 'scripture').",
    "router_enabled": (
        "If true, the route_task tool is active and will automatically select a specialist "
        "model profile based on the task description. Default false — senior calls "
        "spawn_specialist manually with an explicit model_name."
    ),
}
