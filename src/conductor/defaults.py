"""Compiled-in default configuration values for conductor.

This module is the single source of truth for all default settings.
Other modules should import from here rather than duplicating values.
"""

from __future__ import annotations

from typing import Final

POOL_DEFAULTS: Final[dict[str, int | str]] = {
    "max_sessions": 3,
    "idle_ttl_seconds": 60,
    "default_model": "sonnet-4.5",
}

MODEL_DEFAULTS: Final[dict[str, str]] = {
    "autonomous": "opus-4.6",
    "standard": "sonnet-4.5",
    "lightweight": "composer-1.5",
}

TIMEOUT_DEFAULTS: Final[dict[str, int]] = {
    "design": 300,
    "plan": 180,
    "architect": 300,
    "test": 120,
    "implement": 180,
    "verify": 60,
    "pr": 120,
}

HEALTH_DEFAULTS: Final[dict[str, int]] = {
    "poll_interval_seconds": 5,
    "idle_threshold_seconds": 30,
    "max_nudges": 2,
    "max_retries": 1,
}

STEP_DEFAULTS: Final[dict[str, str]] = {
    "1.1": "python",
    "1.2": "autonomous",
    "1.3": "python",
    "2.1": "python",
    "2.2": "autonomous",
    "2.3": "python",
    "3.1": "python",
    "3.2": "autonomous",
    "3.3": "autonomous",
    "3.4": "python",
    "4.1": "python",
    "4.2.*": "standard",
    "4.3": "python",
    "5.1": "python",
    "5.2.*": "standard",
    "5.3": "python",
    "5.4": "standard",
    "6.1": "python",
    "6.2": "standard",
    "6.3": "python",
    "7.1": "python",
    "7.2": "standard",
    "7.3": "python",
    "7.4": "python",
}


_BARE_KEY_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


def _quote_key(key: str) -> str:
    if key and all(c in _BARE_KEY_CHARS for c in key):
        return key
    return f'"{key}"'


def _format_toml_value(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    msg = f"Unsupported type: {type(value)}"
    raise TypeError(msg)


def _section_to_toml(name: str, data: dict[str, object]) -> str:
    lines = [f"[{name}]"]
    for key, value in data.items():
        lines.append(f"{_quote_key(key)} = {_format_toml_value(value)}")
    return "\n".join(lines)


def generate_toml() -> str:
    """Generate a TOML configuration string from compiled-in defaults."""
    sections = [
        _section_to_toml("pool", POOL_DEFAULTS),
        _section_to_toml("model", MODEL_DEFAULTS),
        _section_to_toml("timeout", TIMEOUT_DEFAULTS),
        _section_to_toml("health", HEALTH_DEFAULTS),
        _section_to_toml("step", STEP_DEFAULTS),
    ]
    return "\n\n".join(sections) + "\n"


def resolve_step_model(step_id: str) -> str:
    """Resolve a step ID to its model tier.

    Lookup order:
    1. Exact match on *step_id*
    2. Prefix wildcard — e.g. key ``"4.2.*"`` matches ``"4.2.1"``, ``"4.2.2"``
    3. Falls back to ``"standard"``
    """
    if step_id in STEP_DEFAULTS:
        return STEP_DEFAULTS[step_id]

    prefix = step_id.rsplit(".", maxsplit=1)[0]
    wildcard_key = f"{prefix}.*"
    if wildcard_key in STEP_DEFAULTS:
        return STEP_DEFAULTS[wildcard_key]

    return "standard"
