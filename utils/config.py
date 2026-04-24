from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{([^:}]+)(?::([^}]*))?\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} and ${VAR:-default} placeholders with environment variable values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default or "")

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _resolve_config_recursively(raw: Any) -> Any:
    """Walk config dict/list and resolve env var placeholders in all string values."""
    if isinstance(raw, str):
        return _resolve_env_vars(raw)
    if isinstance(raw, dict):
        return {key: _resolve_config_recursively(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return [_resolve_config_recursively(item) for item in raw]
    return raw


@dataclass(slots=True)
class AppConfig:
    raw: dict[str, Any]
    path: Path

    def section(self, key: str) -> dict[str, Any]:
        return self.raw.get(key, {})

    @property
    def project(self) -> dict[str, Any]:
        return self.section("project")

    @property
    def database(self) -> dict[str, Any]:
        return self.section("database")

    @property
    def storage(self) -> dict[str, Any]:
        return self.section("storage")

    @property
    def providers(self) -> dict[str, Any]:
        return self.section("providers")

    @property
    def jobs(self) -> dict[str, Any]:
        return self.section("jobs")

    @property
    def labels(self) -> dict[str, Any]:
        return self.section("labels")

    @property
    def events(self) -> dict[str, Any]:
        return self.section("events")

    @property
    def company_encoding(self) -> dict[str, Any]:
        return self.section("company_encoding")

    @property
    def runtime(self) -> dict[str, Any]:
        return self.section("runtime")

    @property
    def llm(self) -> dict[str, Any]:
        return self.section("llm")


def load_config(config_path: str | Path = "configs/config.yaml") -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _resolve_config_recursively(raw)
    return AppConfig(raw=raw, path=path)
