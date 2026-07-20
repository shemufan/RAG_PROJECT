"""FastAPI application factory and service lifecycle."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import MODEL_PATH

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Initialize the baseline dependency graph once per API process."""
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Embedding 模型目录不存在: {MODEL_PATH}")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，请检查 api_key.env")

    from backend.storage.chroma_store import ChromaStore
    from backend.services.embedding_service import EmbeddingService
    from backend.services.llm_service import LLMService
    from backend.services.rag_classifier import RAGClassifier

    logger.info("[1/4] 加载 Embedding 模型: %s", MODEL_PATH)
    embedding_service = EmbeddingService()
    logger.info("[2/4] 打开 Week3 Chroma 知识库")
    chroma_store = ChromaStore(embedding_service)
    document_count = chroma_store.count()
    if document_count == 0:
        raise RuntimeError("Week3 知识库为空，请先运行 python -m backend.scripts.rebuild_kb")
    logger.info("知识库已加载，共 %d 个知识块", document_count)
    logger.info("[3/4] 初始化 DeepSeek LLM")
    llm_service = LLMService()
    logger.info("[4/4] 初始化 RAG 分类器")
    app.state.embedding_service = embedding_service
    app.state.chroma_store = chroma_store
    app.state.classifier = RAGClassifier(chroma_store, llm_service)
    yield


def create_app(lifespan=app_lifespan) -> FastAPI:
    application = FastAPI(
        title="Data Compliance RAG API",
        description="企业数据智能分类分级 RAG 系统 — Week3 Baseline",
        version="0.3.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:7860", "http://localhost:7860"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    from backend.api.classify import router as classify_router
    from backend.api.health import router as health_router

    application.include_router(health_router, prefix="/api")
    application.include_router(classify_router, prefix="/api")
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
