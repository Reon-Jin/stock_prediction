from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from utils.config import AppConfig
from utils.io import dump_json, ensure_dir
from utils.logger import get_logger


class ProviderError(RuntimeError):
    pass


class BaseProvider:
    provider_name = "base"
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger(self.provider_name)
        self.storage_conf = config.storage
        self.provider_conf = config.providers
        self.raw_cache_dir = ensure_dir(self.storage_conf.get("raw_cache_dir", "data/raw_cache"))
        self._proxy_env_backup: dict[str, str] = {}

    def build_snapshot_path(self, domain: str, payload: dict[str, Any]) -> Path:
        dt = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        digest = hashlib.md5(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:12]
        return self.raw_cache_dir / domain / f"{dt}_{digest}.json"

    def snapshot(self, domain: str, payload: dict[str, Any]) -> Path:
        path = self.build_snapshot_path(domain, payload)
        dump_json(path, payload)
        return path

    def wrap_fetch(self, domain: str, params: dict[str, Any], func: Callable[[], Any]) -> Any:
        @retry(
            stop=stop_after_attempt(int(self.provider_conf.get("max_retries", 3))),
            wait=wait_fixed(int(self.provider_conf.get("retry_wait_seconds", 2))),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _runner() -> Any:
            self._prepare_network_env()
            try:
                return func()
            finally:
                self._restore_network_env()

        result = _runner()
        if self.config.jobs.get("snapshot_raw_response", True):
            self.snapshot(domain, {"provider": self.provider_name, "params": params, "result": result})
        return result

    def _prepare_network_env(self) -> None:
        if not self.provider_conf.get("disable_env_proxy", True):
            return
        proxy_keys = [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]
        self._proxy_env_backup = {}
        for key in proxy_keys:
            if key in os.environ:
                self._proxy_env_backup[key] = os.environ[key]
                os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

    def _restore_network_env(self) -> None:
        if not self.provider_conf.get("disable_env_proxy", True):
            return
        for key in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]:
            os.environ.pop(key, None)
        for key, value in self._proxy_env_backup.items():
            os.environ[key] = value
        self._proxy_env_backup = {}

    def require_akshare(self) -> Any:
        try:
            import akshare as ak  # type: ignore

            return ak
        except ImportError as exc:
            raise ProviderError("akshare is required for the default providers") from exc

    def build_http_session(self) -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        session.headers.update(self.default_headers)
        return session

    def http_get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        session = self.build_http_session()
        if headers:
            session.headers.update(headers)
        response = session.get(
            url,
            params=params,
            timeout=timeout or float(self.provider_conf.get("request_timeout", 20)),
        )
        response.raise_for_status()
        return response.json()
