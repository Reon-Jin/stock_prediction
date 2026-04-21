from __future__ import annotations

import argparse


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--start", required=False, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", required=False, help="End date, YYYY-MM-DD")
    parser.add_argument("--incremental", action="store_true", help="Run incremental mode")
    parser.add_argument("--limit", type=int, default=None, help="Optional row or symbol limit")
    return parser

