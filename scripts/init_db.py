from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from warehouse.schema_init import init_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize database schema")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config yaml")
    args = parser.parse_args()
    init_schema(args.config)


if __name__ == "__main__":
    main()
