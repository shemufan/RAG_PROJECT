"""Read-only HTTP views for benchmark runs and predictions."""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.config import get_settings
from app.repositories.benchmark_target import BenchmarkTargetRepository
from app.schemas.benchmark import BenchmarkPredictionRow, BenchmarkRunSummary

router = APIRouter(prefix="/benchmark", tags=["benchmark"])
Outcome = Literal["TP", "FP", "TN", "FN", "FAILED", "UNLABELED"]


def get_benchmark_repository(request: Request) -> BenchmarkTargetRepository:
    """Lazily connect to B only when a benchmark query is requested."""
    cached = getattr(request.app.state, "benchmark_repository", None)
    if cached is not None:
        return cached
    settings = get_settings()
    if not settings.target_database_url:
        raise HTTPException(status_code=503, detail="TARGET_DATABASE_URL is not configured")
    repository = BenchmarkTargetRepository(settings.target_database_url)
    request.app.state.benchmark_repository = repository
    return repository


@router.get("/runs/{run_id}", response_model=BenchmarkRunSummary)
def get_benchmark_run(
    run_id: UUID,
    repository: BenchmarkTargetRepository = Depends(get_benchmark_repository),
) -> BenchmarkRunSummary:
    summary = repository.get_run(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="benchmark run not found")
    return summary


@router.get("/results", response_model=list[BenchmarkPredictionRow])
def query_benchmark_results(
    run_id: UUID,
    outcome: Outcome | None = None,
    predicted_personal: bool | None = None,
    need_review: bool | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    repository: BenchmarkTargetRepository = Depends(get_benchmark_repository),
) -> list[BenchmarkPredictionRow]:
    return repository.query_predictions(
        run_id=run_id,
        outcome=outcome,
        predicted_personal=predicted_personal,
        need_review=need_review,
        limit=limit,
        offset=offset,
    )
