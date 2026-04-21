from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any


PASSWORD_ITERATIONS = 120_000
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\u4e00-\u9fff]{3,24}$")


def validate_username(username: str) -> bool:
    return bool(USERNAME_PATTERN.fullmatch(username.strip()))


def validate_email(email: str) -> bool:
    value = email.strip()
    return "@" in value and "." in value.split("@")[-1]


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_ITERATIONS,
    )
    return f"{salt}${base64.urlsafe_b64encode(digest).decode('ascii')}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, encoded_hash = stored_hash.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_ITERATIONS,
    )
    candidate = base64.urlsafe_b64encode(digest).decode("ascii")
    return hmac.compare_digest(candidate, encoded_hash)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def create_access_token(
    payload: dict[str, Any],
    secret_key: str,
    expires_in_hours: int = 24,
) -> str:
    data = dict(payload)
    data["exp"] = int((datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)).timestamp())
    body = _b64url_encode(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret_key.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    return f"{body}.{_b64url_encode(signature)}"


def decode_access_token(token: str, secret_key: str) -> dict[str, Any] | None:
    try:
        body, signature = token.split(".", 1)
    except ValueError:
        return None
    expected = _b64url_encode(hmac.new(secret_key.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int) or exp < int(datetime.now(timezone.utc).timestamp()):
        return None
    return payload
