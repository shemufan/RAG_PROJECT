# backend/services/embedding_service.py
"""Embedding 模型封装 — 管理 sentence-transformer 的加载和访问。"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings
from backend.core.config import MODEL_PATH

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps HuggingFaceEmbeddings for the local sentence-transformer model.

    此类封装了 embedding 模型的初始化，使其可以在 FastAPI lifespan 中
    显式加载，而非在 import 时隐式加载。
    """

    def __init__(self):
        logger.info("正在加载 Embedding 模型 (%s)...", MODEL_PATH)
        self.model = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={"local_files_only": True},
        )

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """返回 HuggingFaceEmbeddings 实例，供 Chroma 等向量库使用。"""
        return self.model
