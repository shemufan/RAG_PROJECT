#!/usr/bin/env python
"""启动 Gradio 前端（依赖后端服务已初始化）。

用法:
    python run_gradio.py

此脚本会:
    1. 触发 backend.main 的 lifespan 启动（加载模型、向量库、LLM）
    2. 将已初始化的服务注入 frontend.app
    3. 启动 Gradio Web UI

作为 FastAPI 的替代启动方式，适合开发和演示场景。
"""

import logging
import os
import sys

# 确保项目根在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    # 显式触发服务初始化（与 backend/main.py lifespan 相同的流程）
    from backend.core.config import (
        MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, OUTPUT_DIR
    )
    from backend.services.embedding_service import EmbeddingService
    from backend.services.llm_service import LLMService
    from backend.services.rag_classifier import RAGClassifier
    from backend.db.chroma_store import ChromaStore
    from backend.db.mysql import create_mysql_engine
    from backend.db.result_store import load_error_rules, ensure_tables

    logger.info("=== 正在初始化 Data Compliance RAG 服务 ===")

    # 1. Embedding
    logger.info("[1/4] 加载 Embedding 模型...")
    embedding_service = EmbeddingService()

    # 2. Chroma
    logger.info("[2/4] 初始化 Chroma 向量库...")
    chroma_store = ChromaStore(embedding_service)

    # 3. LLM
    logger.info("[3/4] 初始化 LLM 服务...")
    llm_service = LLMService()

    # 4. Error cache
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

    # 5. RAG Classifier
    classifier = RAGClassifier(
        chroma_store=chroma_store,
        llm_service=llm_service,
        error_cache=error_cache,
    )

    logger.info("=== 所有服务初始化完毕 ===")

    # 注入到 Gradio 前端
    from frontend.app import init_services, build_demo
    init_services(classifier, chroma_store)

    # 构建并启动 Gradio UI
    demo = build_demo(chroma_store=chroma_store)

    # 确保 outputs 目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("启动 Gradio Web UI on http://127.0.0.1:7860")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
