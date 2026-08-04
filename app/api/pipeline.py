"""HTTP adapters for database classification runs and persisted result queries."""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from app.core.config import get_settings
from app.repositories.source_mysql import SourceMySQLRepository
from app.repositories.target_mysql import TargetMySQLRepository
from app.schemas.pipeline import (
    ClassificationEvidenceRow,
    ClassificationResultRow,
    PipelineRequest,
    PipelineSummary,
    RunDetail,
)
from app.services.database_pipeline import DatabaseClassificationPipeline

router = APIRouter(tags=["database-pipeline"])


def get_target_repository(request: Request) -> TargetMySQLRepository:
    """Lazily create the B-database repository on first database API use."""
    cached = getattr(request.app.state, "target_repository", None)
    if cached is not None:
        return cached
    settings = get_settings()
    if not settings.target_database_url:
        raise HTTPException(status_code=503, detail="TARGET_DATABASE_URL is not configured")
    repository = TargetMySQLRepository(settings.target_database_url)
    request.app.state.target_repository = repository
    return repository


def get_database_pipeline(
    request: Request,
    target_repository: TargetMySQLRepository = Depends(get_target_repository),
) -> DatabaseClassificationPipeline:
    """Lazily compose the source repository with the existing classifier service."""
    cached = getattr(request.app.state, "database_pipeline", None)
    if cached is not None:
        return cached
    settings = get_settings()
    if not settings.source_database_url:
        raise HTTPException(status_code=503, detail="SOURCE_DATABASE_URL is not configured")
    classification_service = getattr(request.app.state, "classification_service", None)
    if classification_service is None:
        raise HTTPException(status_code=503, detail="classification service is not ready")
    pipeline = DatabaseClassificationPipeline(
        SourceMySQLRepository(settings.source_database_url),
        target_repository,
        classification_service,
        model_name=settings.deepseek_model,
        knowledge_base_version=settings.knowledge_base_version,
    )
    request.app.state.database_pipeline = pipeline
    return pipeline


@router.post("/pipeline/run", response_model=PipelineSummary)
def run_pipeline(
    pipeline_request: PipelineRequest,
    pipeline: DatabaseClassificationPipeline = Depends(get_database_pipeline),
) -> PipelineSummary:
    return pipeline.run(pipeline_request)


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(
    run_id: UUID,
    repository: TargetMySQLRepository = Depends(get_target_repository),
) -> RunDetail:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="classification run not found")
    return run


@router.get("/results", response_model=list[ClassificationResultRow])
def query_results(
    run_id: UUID | None = None,
    database_name: str | None = None,
    table_name: str | None = None,
    column_name: str | None = None,
    level: Literal["L1", "L2", "L3", "L4"] | None = None,
    category: str | None = None,
    need_review: bool | None = None,
    is_personal: bool | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    repository: TargetMySQLRepository = Depends(get_target_repository),
) -> list[ClassificationResultRow]:
    return repository.query_results(
        run_id=run_id,
        database_name=database_name,
        table_name=table_name,
        column_name=column_name,
        level=level,
        category=category,
        need_review=need_review,
        is_personal=is_personal,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/results/{result_id}/evidence",
    response_model=list[ClassificationEvidenceRow],
)
def get_result_evidence(
    result_id: int = Path(ge=1),
    repository: TargetMySQLRepository = Depends(get_target_repository),
) -> list[ClassificationEvidenceRow]:
    return repository.get_result_evidence(result_id)
