"""spawn_specialist — call a specialist model as a pure text oracle.

The specialist (e.g. a fine-tuned 7B base model) receives the prompt and
returns generated text. It has no tools, no project context, and no memory
beyond what the prompt contains.

Hermes (the senior) is responsible for everything else:
  - writing the output to a file via write_file
  - validating / testing via run_in_sandbox or run_shell
  - looping with revised prompts if the result is unsatisfactory
  - calling commit_changes when done

Config required in ~/.michael/config.json:
  models.junior.endpoint          e.g. "http://localhost:11435/v1"
  models.junior.served_model_name e.g. "deephat-v1-7b"
  models.junior.gpu_name          "junior"  (set by michael gpu up junior)
"""
from __future__ import annotations

from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "spawn_specialist",
        "description": (
            "Call a specialist model as a pure text oracle. "
            "Send a prompt, receive generated text. "
            "The specialist has no tools, no file access, and no awareness of the project — "
            "it only knows what you put in the prompt. "
            "You are responsible for validating the output: write it to a file, run it, "
            "test it with your own tools, and loop with a revised prompt if it falls short. "
            "Use this to leverage domain-specific expertise for code generation, exploit "
            "writing, config synthesis, script drafting, or any task where a specialist "
            "model outperforms a general approach."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": (
                        "Name of the specialist model profile in config "
                        "(e.g. 'junior'). Must have endpoint and served_model_name set."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The complete prompt for the specialist. "
                        "Include ALL context it needs — desired output format, "
                        "any relevant code or data to work from, constraints, "
                        "and exactly what you want returned. "
                        "The specialist has no other context."
                    ),
                },
            },
            "required": ["model_name", "prompt"],
        },
    },
}


def spawn_specialist(model_name: str, prompt: str, **_: Any) -> str:
    from michael.backends import LLMClient
    from michael.config import Config

    cfg = Config.load()

    if model_name not in cfg.models:
        return (
            f"error: no model profile '{model_name}' in config — "
            f"add models.{model_name} with endpoint and served_model_name, "
            f"or run `michael gpu up {model_name}` to provision it."
        )
    profile = cfg.models[model_name]
    if not profile.endpoint:
        return (
            f"error: models.{model_name}.endpoint is not set — "
            f"run `michael gpu up {model_name}` first."
        )
    if not profile.served_model_name:
        return f"error: models.{model_name}.served_model_name is not set."

    client = LLMClient(profile.endpoint)
    try:
        resp = client.chat.completions.create(
            model=profile.served_model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=float(profile.request_timeout_s or 120),
        )
    except Exception as exc:
        return f"error calling specialist '{model_name}': {exc}"

    return (resp.choices[0].content or "").strip()
