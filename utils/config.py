from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    return AppConfig(raw=raw, path=path)
