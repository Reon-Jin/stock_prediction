from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.splitter import build_time_purged_split_config, validate_no_temporal_leakage
from utils.config import load_config


def _print_split_summary(summary: dict) -> None:
    print("Split ranges:")
    for split_name, info in summary["date_ranges"].items():
        print(
            f"  {split_name}: rows={info['rows']:,} symbols={info['unique_symbols']:,} "
            f"dates={info['unique_dates']:,} range={info['min_trade_date']}..{info['max_trade_date']}"
        )
    print("Boundary checks:")
    for name, info in summary["boundary_checks"].items():
        print(
            f"  {name}: left_end={info['left_end']} right_start={info['right_start']} "
            f"gap_days={info['actual_gap_days']} "
            f"label_reach={info['left_label_reach']} input_window_start={info['right_input_window_start']}"
        )
    print(f"Leakage check passed: {summary.get('passed', False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported purged time-series train/valid/test splits.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--export-dir", default="data/exports", help="Directory containing exported parquet splits")
    args = parser.parse_args()

    config = load_config(args.config)
    split_config = build_time_purged_split_config(config)
    export_dir = (PROJECT_ROOT / args.export_dir).resolve()
    summary = validate_no_temporal_leakage(
        export_dir / "train.parquet",
        export_dir / "valid.parquet",
        export_dir / "test.parquet",
        seq_length=split_config.seq_length,
        max_horizon=split_config.max_horizon,
        gap_days=split_config.gap_days,
        split_windows=split_config.windows,
    )
    _print_split_summary(summary)


if __name__ == "__main__":
    main()
