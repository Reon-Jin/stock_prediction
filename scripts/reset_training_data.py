from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from datasets.splitter import build_time_purged_split_config
from warehouse.repository import WarehouseRepository


def _remove_path(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return 1


def _resolve_cleanup_window(config) -> tuple[str, str]:
    try:
        split_config = build_time_purged_split_config(config)
    except Exception:
        start = str(config.project.get("default_start"))
        end = str(config.project.get("default_end"))
        return start, end
    return (
        split_config.train.start.strftime("%Y-%m-%d"),
        split_config.test.end.strftime("%Y-%m-%d"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove old training parquet/cache data and clear stale DB training rows.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--include-derived-features", action="store_true", help="Also delete event/company derived tables")
    parser.add_argument("--include-artifacts", action="store_true", help="Also delete train/artifacts outputs")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be deleted")
    args = parser.parse_args()

    config = load_config(args.config)
    repo = WarehouseRepository.from_config_path(args.config)

    export_dir = (PROJECT_ROOT / config.storage.get("export_dir", "data/exports")).resolve()
    cache_dir = (PROJECT_ROOT / "train" / "cache").resolve()
    artifacts_dir = (PROJECT_ROOT / "train" / "artifacts").resolve()

    removable_paths = [
        export_dir / "train.parquet",
        export_dir / "valid.parquet",
        export_dir / "test.parquet",
        export_dir / "train_events.parquet",
        export_dir / "valid_events.parquet",
        export_dir / "test_events.parquet",
        export_dir / "split_manifest.json",
        export_dir / "split_summary.json",
        cache_dir,
    ]
    if args.include_artifacts:
        removable_paths.append(artifacts_dir)

    print("Local cleanup targets:")
    for path in removable_paths:
        print(f"  - {path}")

    if args.dry_run:
        start, end = _resolve_cleanup_window(config)
        print("DB cleanup targets:")
        print(f"  - training_samples within rebuild window {start}..{end}")
        if args.include_derived_features:
            print(f"  - event_features_daily within rebuild window {start}..{end}")
            print(f"  - company_profiles within rebuild window {start}..{end} and current profile version")
            print("  - company_similarity for current similarity version")
        return

    removed_paths = sum(_remove_path(path) for path in removable_paths)
    start, end = _resolve_cleanup_window(config)
    deleted_training_samples = repo.delete_training_samples(start, end)

    print(f"Deleted local path entries: {removed_paths}")
    print(f"Deleted training_samples rows: {deleted_training_samples}")

    if args.include_derived_features:
        deleted_event_features = repo.delete_event_features(start, end)
        deleted_company_profiles = repo.delete_company_profiles(
            start=start,
            end=end,
            profile_version=str(config.company_encoding.get("profile_version", "cp1")),
        )
        deleted_company_similarity = repo.delete_company_similarity(
            str(config.company_encoding.get("similarity_version", "cs1"))
        )
        print(f"Deleted event_features_daily rows: {deleted_event_features}")
        print(f"Deleted company_profiles rows: {deleted_company_profiles}")
        print(f"Deleted company_similarity rows: {deleted_company_similarity}")


if __name__ == "__main__":
    main()
