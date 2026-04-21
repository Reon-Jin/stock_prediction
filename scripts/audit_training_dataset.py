from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def audit_split(path: Path) -> None:
    df = pd.read_parquet(path)
    print(f"[{path.name}] rows={len(df):,} cols={len(df.columns)}")
    if df.empty:
        print("  empty parquet")
        return

    if {"symbol", "trade_date"}.issubset(df.columns):
        duplicate_count = int(df.duplicated(subset=["symbol", "trade_date"]).sum())
        print(f"  duplicate symbol-date rows={duplicate_count}")

    numeric = df.select_dtypes(include=["number", "boolean"])
    if not numeric.empty:
        missing = (numeric.isna().mean().sort_values(ascending=False).head(15) * 100).round(2)
        constants = [column for column in numeric.columns if numeric[column].nunique(dropna=True) <= 1][:15]
        print("  top missing %:")
        for column, ratio in missing.items():
            print(f"    {column}: {ratio:.2f}%")
        print(f"  constant-like columns={constants}")

    label_columns = [column for column in df.columns if column.startswith("label_")]
    if label_columns:
        print("  label means:")
        for column in label_columns[:15]:
            series = pd.to_numeric(df[column], errors="coerce")
            print(f"    {column}: mean={series.mean():.6f} std={series.std():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exported training parquet quality.")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=["data/exports/train.parquet", "data/exports/valid.parquet", "data/exports/test.parquet"],
        help="Parquet paths to inspect",
    )
    args = parser.parse_args()

    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        audit_split(path)


if __name__ == "__main__":
    main()
