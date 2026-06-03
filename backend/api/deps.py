# backend/api/deps.py
"""FastAPI 依赖注入 — 从 app.state 获取已初始化的服务单例。"""

from fastapi import Request
from backend.services.rag_classifier import RAGClassifier
from backend.db.chroma_store import ChromaStore


def get_rag_classifier(request: Request) -> RAGClassifier:
    """获取 RAGClassifier 实例（在 lifespan 中初始化）。"""
    return request.app.state.classifier


def get_chroma_store(request: Request) -> ChromaStore:
    """获取 ChromaStore 实例（在 lifespan 中初始化）。"""
    return request.app.state.chroma_store
