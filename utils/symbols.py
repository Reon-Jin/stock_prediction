from __future__ import annotations

import re


SH_PREFIXES = ("600", "601", "603", "605", "688", "689", "510", "511", "512", "513", "515")
BJ_PREFIXES = ("430", "830", "831", "832", "833", "835", "836", "837", "838", "839", "870", "871", "872", "873", "920")


def normalize_symbol(symbol: str) -> str:
    raw = re.sub(r"[^0-9A-Za-z.]", "", str(symbol or "").strip().upper())
    if not raw:
        raise ValueError("empty symbol")
    if "." in raw:
        code, suffix = raw.split(".", 1)
        suffix = suffix.upper()
        if suffix in {"SH", "SZ", "BJ"} and len(code) == 6:
            return f"{code}.{suffix}"
    code = re.sub(r"[^0-9]", "", raw)
    if len(code) != 6:
        raise ValueError(f"invalid symbol: {symbol}")
    if code.startswith(BJ_PREFIXES):
        exchange = "BJ"
    elif code.startswith(SH_PREFIXES):
        exchange = "SH"
    else:
        exchange = "SZ"
    return f"{code}.{exchange}"


def infer_exchange(symbol: str) -> str:
    return normalize_symbol(symbol).split(".")[1]


def symbol_to_akshare(symbol: str) -> str:
    return normalize_symbol(symbol).split(".")[0]

