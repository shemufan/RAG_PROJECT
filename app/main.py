"""FastAPI application factory and production dependency lifecycle."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    from app.core.config import get_settings
    from app.repositories.vector_store import VectorStore
    from app.services.classification_service import FieldClassificationService
    from app.services.embedding_service import EmbeddingService
    from app.services.llm_service import LLMService

    settings = get_settings()
    embedding_service = EmbeddingService(model_path=settings.embedding_model_path)
    vector_store = VectorStore(embedding_service, settings=settings)
    if vector_store.count() == 0:
        raise RuntimeError(
            "知识库为空，请先运行 python -m scripts.rebuild_knowledge_base"
        )
    application.state.classification_service = FieldClassificationService(
        vector_store,
        LLMService(settings=settings),
    )
    yield


def create_app(lifespan: Callable = app_lifespan) -> FastAPI:
    application = FastAPI(
        title="Data Compliance RAG API",
        description="企业数据单字段分类分级服务",
        version="1.0.0",
        lifespan=lifespan,
    )
    from app.api.classification import router as classification_router
    from app.api.health import router as health_router

    application.include_router(health_router, prefix="/api")
    application.include_router(classification_router, prefix="/api")
    return application


app = create_app()
