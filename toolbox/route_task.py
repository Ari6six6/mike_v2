"""route_task — route a subtask to the best specialist and run it.

The senior calls this instead of choosing a model_name manually for
spawn_specialist. The router picks the profile; this tool runs it.

Auto-executes — no y/n confirmation needed (read-only dispatch + oracle call).
"""
from __future__ import annotations

from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "route_task",
        "description": (
            "Route a subtask to the best available specialist model and run it as a "
            "pure text oracle. The router selects the model profile automatically "
            "based on the task description — you do not need to know which specialist "
            "to use. Returns the specialist's raw text output. "
            "You are responsible for validating the result and looping with a revised "
            "prompt if it falls short. "
            "Use this instead of spawn_specialist when you want automatic model selection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "One-sentence description of the subtask. "
                        "Used to select the specialist — be specific about the domain "
                        "(e.g. 'write an exploit for CVE-2024-1234', "
                        "'triage this Wazuh alert', 'generate a Python parser for X'). "
                        "This text is NOT sent to the specialist; include full context in prompt."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The complete prompt for the specialist. "
                        "Include ALL context — desired output format, relevant code or data, "
                        "constraints, and exactly what you want returned. "
                        "The specialist has no project context beyond what you provide here."
                    ),
                },
            },
            "required": ["task", "prompt"],
        },
    },
}


def route_task(task: str, prompt: str, **_: Any) -> str:
    from michael.backends import LLMClient
    from michael.config import Config
    from michael.router import route

    cfg = Config.load()

    if not cfg.router_enabled:
        return (
            "error: router_enabled is false in config — "
            "set router_enabled=true or call spawn_specialist directly with an explicit model_name."
        )

    profile_name = route(task, cfg)

    if profile_name not in cfg.models:
        return (
            f"error: router selected profile '{profile_name}' but it is not in cfg.models — "
            f"check config or add the profile."
        )

    profile = cfg.models[profile_name]

    if not profile.endpoint:
        return (
            f"error: routed to profile '{profile_name}' but endpoint is not set — "
            f"run `michael gpu up {profile_name}` first."
        )
    if not profile.served_model_name:
        return f"error: profile '{profile_name}'.served_model_name is not set."

    gpu_key = profile.gpu_name or profile_name
    gpu = cfg.get_gpu(gpu_key)
    if gpu and gpu.ssh_host:
        from michael.backends import _ensure_tunnel
        try:
            _ensure_tunnel(gpu_key, gpu)
        except Exception as exc:
            return f"error: tunnel for '{gpu_key}' failed to come up: {exc}"

    client = LLMClient(profile.endpoint)
    try:
        resp = client.chat.completions.create(
            model=profile.served_model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=float(profile.request_timeout_s or 120),
        )
    except Exception as exc:
        return f"error calling specialist '{profile_name}': {exc}"

    result = (resp.choices[0].content or "").strip()
    return f"[routed to: {profile_name}]\n\n{result}"
