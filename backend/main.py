# backend/main.py
"""FastAPI 应用入口 — 服务生命周期管理 + 路由注册。

启动命令:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

同时提供 get_services() 函数供 Gradio 前端获取已初始化的服务单例。
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, OUTPUT_DIR
)
from backend.services.embedding_service import EmbeddingService
from backend.services.llm_service import LLMService
from backend.services.rag_classifier import RAGClassifier
from backend.db.chroma_store import ChromaStore
from backend.db.mysql import create_mysql_engine
from backend.db.result_store import load_error_rules, ensure_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# ── 模块级服务引用（lifespan 中赋值，get_services() 暴露给 Gradio） ──
_classifier = None
_chroma_store = None
_llm_service = None


def get_services():
    """返回已初始化的服务单例（供 Gradio 前端使用）。

    Returns:
        (RAGClassifier, ChromaStore, LLMService): 三个核心服务实例。
        如果尚未初始化则返回 (None, None, None)。
    """
    return _classifier, _chroma_store, _llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时加载模型/向量库，关闭时清理资源。"""
    global _classifier, _chroma_store, _llm_service

    logger.info("=== 正在初始化 Data Compliance RAG 服务 ===")

    # 1. 加载 Embedding 模型
    logger.info("[1/4] 加载 Embedding 模型...")
    embedding_service = EmbeddingService()

    # 2. 初始化 Chroma 向量库
    logger.info("[2/4] 初始化 Chroma 向量库...")
    _chroma_store = ChromaStore(embedding_service)

    # 3. 初始化 LLM 服务
    logger.info("[3/4] 初始化 LLM 服务...")
    _llm_service = LLMService()

    # 4. 从 MySQL 恢复错题本缓存（失败不影响功能）
    logger.info("[4/4] 恢复错题本缓存...")
    error_cache = {}
    try:
        engine = create_mysql_engine(
            MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
        )
        ensure_tables(engine)
        error_cache.update(load_error_rules(engine))
        engine.dispose()
        logger.info("错题本缓存恢复成功: %d 条规则", len(error_cache))
    except Exception:
        logger.warning("MySQL 错题本缓存恢复失败，使用空缓存启动")

    # 5. 组装 RAG 分类器
    _classifier = RAGClassifier(
        chroma_store=_chroma_store,
        llm_service=_llm_service,
        error_cache=error_cache,
    )

    # 存入 app.state 供 FastAPI 依赖注入
    app.state.classifier = _classifier
    app.state.chroma_store = _chroma_store

    logger.info("=== 所有服务初始化完毕，等待请求 ===")

    yield

    # 关闭时清理
    logger.info("=== 服务关闭中 ===")


app = FastAPI(
    title="Data Compliance RAG API",
    description="企业数据智能分类分级 RAG 系统 — 后端 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from backend.api.classify import router as classify_router
app.include_router(classify_router, prefix="/api")

# 确保 outputs 目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 直接运行入口 ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
