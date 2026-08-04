"""Import labeled personal-information benchmark CSV files into database A."""

import argparse

from app.core.config import load_settings
from app.repositories.benchmark_source import BenchmarkSourceRepository
from app.services.benchmark_import_service import parse_benchmark_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personal", required=True)
    parser.add_argument("--non-personal", required=True)
    parser.add_argument("--batch", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    if not settings.source_database_url:
        raise SystemExit("SOURCE_DATABASE_URL is not configured")
    personal = parse_benchmark_csv(args.personal, source_dataset="personal")
    non_personal = parse_benchmark_csv(
        args.non_personal,
        source_dataset="non_personal",
    )
    repository = BenchmarkSourceRepository(settings.source_database_url)
    total = repository.import_batch(args.batch, [*personal, *non_personal])
    print(
        f"Imported benchmark batch {args.batch}: "
        f"personal={len(personal)}, non_personal={len(non_personal)}, total={total}"
    )


if __name__ == "__main__":
    main()
