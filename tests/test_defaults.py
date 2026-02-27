from __future__ import annotations

import tomllib
from typing import ClassVar

import pytest

from conductor.defaults import (
    HEALTH_DEFAULTS,
    MODEL_DEFAULTS,
    POOL_DEFAULTS,
    STEP_DEFAULTS,
    TIMEOUT_DEFAULTS,
    generate_toml,
    resolve_step_model,
)

# ---------------------------------------------------------------------------
# Structure & type checks
# ---------------------------------------------------------------------------


class TestPoolDefaults:
    expected_keys: ClassVar[set[str]] = {
        "max_sessions",
        "idle_ttl_seconds",
        "default_model",
    }

    def test_keys(self):
        assert set(POOL_DEFAULTS) == self.expected_keys

    @pytest.mark.parametrize("key", ["max_sessions", "idle_ttl_seconds"])
    def test_int_values(self, key: str):
        assert isinstance(POOL_DEFAULTS[key], int)

    def test_default_model_is_str(self):
        assert isinstance(POOL_DEFAULTS["default_model"], str)


class TestModelDefaults:
    expected_keys: ClassVar[set[str]] = {"autonomous", "standard", "lightweight"}

    def test_keys(self):
        assert set(MODEL_DEFAULTS) == self.expected_keys

    def test_all_values_are_str(self):
        for v in MODEL_DEFAULTS.values():
            assert isinstance(v, str)


class TestTimeoutDefaults:
    expected_keys: ClassVar[set[str]] = {
        "design",
        "plan",
        "architect",
        "test",
        "implement",
        "verify",
        "pr",
    }

    def test_keys(self):
        assert set(TIMEOUT_DEFAULTS) == self.expected_keys

    def test_all_values_are_positive_ints(self):
        for v in TIMEOUT_DEFAULTS.values():
            assert isinstance(v, int)
            assert v > 0


class TestHealthDefaults:
    expected_keys: ClassVar[set[str]] = {
        "poll_interval_seconds",
        "idle_threshold_seconds",
        "max_nudges",
        "max_retries",
    }

    def test_keys(self):
        assert set(HEALTH_DEFAULTS) == self.expected_keys

    def test_all_values_are_non_negative_ints(self):
        for v in HEALTH_DEFAULTS.values():
            assert isinstance(v, int)
            assert v >= 0


class TestStepDefaults:
    def test_all_keys_are_dotted_strings(self):
        for key in STEP_DEFAULTS:
            assert isinstance(key, str)
            assert "." in key

    def test_all_values_are_valid_tiers(self):
        valid = {"python", "autonomous", "standard", "lightweight"}
        for v in STEP_DEFAULTS.values():
            assert v in valid, f"unexpected tier: {v}"


# ---------------------------------------------------------------------------
# generate_toml()
# ---------------------------------------------------------------------------


class TestGenerateToml:
    def test_returns_str(self):
        result = generate_toml()
        assert isinstance(result, str)

    def test_parseable(self):
        toml_str = generate_toml()
        parsed = tomllib.loads(toml_str)
        assert isinstance(parsed, dict)

    def test_contains_all_sections(self):
        parsed = tomllib.loads(generate_toml())
        for section in ("pool", "model", "timeout", "health", "step"):
            assert section in parsed

    def test_roundtrip_pool(self):
        parsed = tomllib.loads(generate_toml())
        for key, value in POOL_DEFAULTS.items():
            assert parsed["pool"][key] == value

    def test_roundtrip_model(self):
        parsed = tomllib.loads(generate_toml())
        for key, value in MODEL_DEFAULTS.items():
            assert parsed["model"][key] == value

    def test_roundtrip_timeout(self):
        parsed = tomllib.loads(generate_toml())
        for key, value in TIMEOUT_DEFAULTS.items():
            assert parsed["timeout"][key] == value

    def test_roundtrip_health(self):
        parsed = tomllib.loads(generate_toml())
        for key, value in HEALTH_DEFAULTS.items():
            assert parsed["health"][key] == value

    def test_roundtrip_step(self):
        parsed = tomllib.loads(generate_toml())
        for key, value in STEP_DEFAULTS.items():
            assert parsed["step"][key] == value


# ---------------------------------------------------------------------------
# resolve_step_model()
# ---------------------------------------------------------------------------


class TestResolveStepModel:
    def test_exact_match(self):
        assert resolve_step_model("1.1") == "python"
        assert resolve_step_model("1.2") == "autonomous"
        assert resolve_step_model("6.2") == "standard"

    def test_wildcard_match(self):
        assert resolve_step_model("4.2.1") == "standard"
        assert resolve_step_model("4.2.2") == "standard"
        assert resolve_step_model("5.2.99") == "standard"

    def test_wildcard_key_itself_is_exact(self):
        assert resolve_step_model("4.2.*") == "standard"

    def test_fallback(self):
        assert resolve_step_model("99.99") == "standard"
        assert resolve_step_model("unknown") == "standard"

    def test_exact_beats_wildcard(self):
        assert resolve_step_model("5.4") == "standard"
