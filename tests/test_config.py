from __future__ import annotations

from pathlib import Path

from conductor.config import (
    ConductorConfig,
    init_config,
    load_config,
    resolve_step_model,
)
from conductor.defaults import (
    HEALTH_DEFAULTS,
    MODEL_DEFAULTS,
    POOL_DEFAULTS,
    STEP_DEFAULTS,
    TIMEOUT_DEFAULTS,
    generate_toml,
)


class TestLoadConfigDefaults:
    def test_returns_pool_defaults_when_no_toml(self, tmp_path: Path):
        cfg = load_config(tmp_path)
        assert cfg.pool.max_sessions == POOL_DEFAULTS["max_sessions"]
        assert cfg.pool.idle_ttl_seconds == POOL_DEFAULTS["idle_ttl_seconds"]
        assert cfg.pool.default_model == POOL_DEFAULTS["default_model"]

    def test_returns_model_defaults_when_no_toml(self, tmp_path: Path):
        cfg = load_config(tmp_path)
        assert cfg.models == MODEL_DEFAULTS

    def test_returns_timeout_defaults_when_no_toml(self, tmp_path: Path):
        cfg = load_config(tmp_path)
        assert cfg.timeouts == TIMEOUT_DEFAULTS

    def test_returns_health_defaults_when_no_toml(self, tmp_path: Path):
        cfg = load_config(tmp_path)
        expected = HEALTH_DEFAULTS["poll_interval_seconds"]
        assert cfg.health.poll_interval_seconds == expected
        assert cfg.health.max_retries == HEALTH_DEFAULTS["max_retries"]

    def test_returns_step_defaults_when_no_toml(self, tmp_path: Path):
        cfg = load_config(tmp_path)
        assert cfg.steps == STEP_DEFAULTS


class TestLoadConfigPartialOverrides:
    def _write_toml(self, tmp_path: Path, content: str) -> None:
        d = tmp_path / ".conductor"
        d.mkdir()
        (d / "conductor.toml").write_text(content)

    def test_overrides_pool_partially(self, tmp_path: Path):
        self._write_toml(tmp_path, "[pool]\nmax_sessions = 10\n")
        cfg = load_config(tmp_path)
        assert cfg.pool.max_sessions == 10
        assert cfg.pool.idle_ttl_seconds == POOL_DEFAULTS["idle_ttl_seconds"]

    def test_overrides_single_model(self, tmp_path: Path):
        self._write_toml(tmp_path, '[model]\nautonomous = "gpt-5"\n')
        cfg = load_config(tmp_path)
        assert cfg.models["autonomous"] == "gpt-5"
        assert cfg.models["standard"] == MODEL_DEFAULTS["standard"]

    def test_overrides_single_timeout(self, tmp_path: Path):
        self._write_toml(tmp_path, "[timeout]\ndesign = 999\n")
        cfg = load_config(tmp_path)
        assert cfg.timeouts["design"] == 999
        assert cfg.timeouts["plan"] == TIMEOUT_DEFAULTS["plan"]

    def test_overrides_health_partially(self, tmp_path: Path):
        self._write_toml(tmp_path, "[health]\nmax_nudges = 5\n")
        cfg = load_config(tmp_path)
        assert cfg.health.max_nudges == 5
        expected = HEALTH_DEFAULTS["poll_interval_seconds"]
        assert cfg.health.poll_interval_seconds == expected

    def test_sections_not_overridden_keep_defaults(self, tmp_path: Path):
        self._write_toml(tmp_path, "[pool]\nmax_sessions = 7\n")
        cfg = load_config(tmp_path)
        assert cfg.timeouts == TIMEOUT_DEFAULTS
        assert cfg.steps == STEP_DEFAULTS


class TestLoadConfigCorruptToml:
    def test_corrupt_toml_returns_defaults(self, tmp_path: Path, capsys):
        d = tmp_path / ".conductor"
        d.mkdir()
        (d / "conductor.toml").write_text("{{{{not valid toml!!!!")
        cfg = load_config(tmp_path)
        assert cfg.pool.max_sessions == POOL_DEFAULTS["max_sessions"]
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_binary_garbage_returns_defaults(self, tmp_path: Path, capsys):
        d = tmp_path / ".conductor"
        d.mkdir()
        (d / "conductor.toml").write_bytes(b"\x80\x81\x82\x83")
        cfg = load_config(tmp_path)
        assert cfg.models == MODEL_DEFAULTS
        captured = capsys.readouterr()
        assert "Warning" in captured.err


class TestInitConfig:
    def test_creates_config_file(self, tmp_path: Path):
        path = init_config(tmp_path)
        assert path.exists()
        assert path.name == "conductor.toml"
        assert path.read_text() == generate_toml()

    def test_creates_conductor_dir(self, tmp_path: Path):
        init_config(tmp_path)
        assert (tmp_path / ".conductor").is_dir()

    def test_backs_up_existing_file(self, tmp_path: Path):
        d = tmp_path / ".conductor"
        d.mkdir()
        original_content = "# old config\n"
        (d / "conductor.toml").write_text(original_content)

        init_config(tmp_path)

        backup = d / "conductor.toml.bak"
        assert backup.exists()
        assert backup.read_text() == original_content
        assert (d / "conductor.toml").read_text() == generate_toml()

    def test_returns_path_to_config(self, tmp_path: Path):
        path = init_config(tmp_path)
        assert path == tmp_path / ".conductor" / "conductor.toml"


class TestResolveStepModel:
    def _default_config(self) -> ConductorConfig:
        return load_config(Path("/nonexistent"))

    def test_exact_match(self):
        cfg = self._default_config()
        result = resolve_step_model(cfg, "1.2")
        assert result == MODEL_DEFAULTS[STEP_DEFAULTS["1.2"]]

    def test_wildcard_match(self):
        cfg = self._default_config()
        result = resolve_step_model(cfg, "4.2.1")
        expected_tier = STEP_DEFAULTS["4.2.*"]
        assert result == MODEL_DEFAULTS[expected_tier]

    def test_fallback_to_standard(self):
        cfg = self._default_config()
        result = resolve_step_model(cfg, "99.99")
        assert result == MODEL_DEFAULTS["standard"]

    def test_python_tier_falls_back_to_default_model(self):
        """'python' is not in MODEL_DEFAULTS, so it falls back to pool.default_model."""
        cfg = self._default_config()
        result = resolve_step_model(cfg, "1.1")
        assert STEP_DEFAULTS["1.1"] == "python"
        assert result == cfg.pool.default_model

    def test_custom_step_mapping(self):
        cfg = self._default_config()
        cfg.steps["custom.1"] = "autonomous"
        result = resolve_step_model(cfg, "custom.1")
        assert result == MODEL_DEFAULTS["autonomous"]

    def test_full_chain_with_overrides(self, tmp_path: Path):
        d = tmp_path / ".conductor"
        d.mkdir()
        (d / "conductor.toml").write_text(
            '[model]\nautonomous = "custom-opus"\n\n[step]\n"1.2" = "autonomous"\n'
        )
        cfg = load_config(tmp_path)
        result = resolve_step_model(cfg, "1.2")
        assert result == "custom-opus"
