from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.company_similarity_builder import build_company_similarity
from utils.cli import build_common_parser
from warehouse.models import CompanySimilarity

from jobs.common import bootstrap, resolve_start_end


def run(config_path: str, start: str | None = None, end: str | None = None) -> None:
    ctx = bootstrap("build_company_similarity", config_path)
    if not bool(ctx.config.company_encoding.get("enabled", True)):
        ctx.logger.info("company encoding disabled, skipping company similarity build")
        return
    start, end = resolve_start_end(ctx.config, start, end)
    topk = int(ctx.config.company_encoding.get("similarity_topk", 10))
    similarity_version = str(ctx.config.company_encoding.get("similarity_version", "cs1"))
    job_id = ctx.repo.record_job_start(
        "build_company_similarity",
        {"start": start, "end": end, "topk": topk, "similarity_version": similarity_version},
    )
    try:
        profiles = ctx.repo.fetch_table("company_profiles", end=end)
        similarity_df = build_company_similarity(
            company_profiles=profiles,
            topk=topk,
            similarity_version=similarity_version,
        )
        ctx.repo.delete_company_similarity(similarity_version)
        affected = ctx.repo.upsert(CompanySimilarity, similarity_df.to_dict("records"))
        ctx.logger.info("built company similarity rows=%s", affected)
        ctx.repo.record_job_end(job_id, "success", affected, "build completed")
    except Exception as exc:
        ctx.logger.exception("build company similarity failed")
        ctx.repo.record_job_end(job_id, "failed", message=str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Build company similarity graph")
    args = parser.parse_args()
    run(args.config, args.start, args.end)


if __name__ == "__main__":
    main()
