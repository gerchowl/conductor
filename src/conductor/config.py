from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from conductor.defaults import (
    HEALTH_DEFAULTS,
    MODEL_DEFAULTS,
    POOL_DEFAULTS,
    STEP_DEFAULTS,
    TIMEOUT_DEFAULTS,
    generate_toml,
)

CONFIG_FILENAME = "conductor.toml"
CONFIG_DIR = ".conductor"


@dataclass
class PoolConfig:
    max_sessions: int
    idle_ttl_seconds: int
    default_model: str


@dataclass
class HealthConfig:
    poll_interval_seconds: int
    idle_threshold_seconds: int
    max_nudges: int
    max_retries: int


@dataclass
class ConductorConfig:
    pool: PoolConfig
    models: dict[str, str] = field(default_factory=dict)
    timeouts: dict[str, int] = field(default_factory=dict)
    health: HealthConfig = field(
        default_factory=lambda: HealthConfig(**HEALTH_DEFAULTS),
    )
    steps: dict[str, str] = field(default_factory=dict)


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_defaults() -> dict:
    return {
        "pool": dict(POOL_DEFAULTS),
        "model": dict(MODEL_DEFAULTS),
        "timeout": dict(TIMEOUT_DEFAULTS),
        "health": dict(HEALTH_DEFAULTS),
        "step": dict(STEP_DEFAULTS),
    }


def _config_from_dict(data: dict) -> ConductorConfig:
    pool_data = data.get("pool", POOL_DEFAULTS)
    return ConductorConfig(
        pool=PoolConfig(
            max_sessions=pool_data["max_sessions"],
            idle_ttl_seconds=pool_data["idle_ttl_seconds"],
            default_model=pool_data["default_model"],
        ),
        models=data.get("model", dict(MODEL_DEFAULTS)),
        timeouts=data.get("timeout", dict(TIMEOUT_DEFAULTS)),
        health=HealthConfig(**data.get("health", HEALTH_DEFAULTS)),
        steps=data.get("step", dict(STEP_DEFAULTS)),
    )


def load_config(project_root: Path) -> ConductorConfig:
    """Load config: source defaults merged with .conductor/conductor.toml overrides."""
    defaults = _build_defaults()
    toml_path = project_root / CONFIG_DIR / CONFIG_FILENAME

    if not toml_path.is_file():
        return _config_from_dict(defaults)

    try:
        raw = toml_path.read_bytes()
        overrides = tomllib.loads(raw.decode())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        print(f"Warning: failed to parse {toml_path}: {exc}", file=sys.stderr)
        return _config_from_dict(defaults)

    merged = _deep_merge(defaults, overrides)
    return _config_from_dict(merged)


def init_config(project_root: Path) -> Path:
    """Write .conductor/conductor.toml from source defaults. Backup existing."""
    config_dir = project_root / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / CONFIG_FILENAME
    if config_path.exists():
        backup_path = config_path.with_suffix(".toml.bak")
        backup_path.write_text(config_path.read_text())

    config_path.write_text(generate_toml())
    return config_path


def resolve_step_model(config: ConductorConfig, step_id: str) -> str:
    """Resolve step ID to actual model ID via config.

    Exact match, then wildcard, then fallback to default_model.
    """
    tier = _resolve_step_tier(config.steps, step_id)
    return config.models.get(tier, config.pool.default_model)


def _resolve_step_tier(steps: dict[str, str], step_id: str) -> str:
    if step_id in steps:
        return steps[step_id]

    prefix = step_id.rsplit(".", maxsplit=1)[0]
    wildcard_key = f"{prefix}.*"
    if wildcard_key in steps:
        return steps[wildcard_key]

    return "standard"
