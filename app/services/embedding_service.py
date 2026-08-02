"""Embedding model construction and dependency injection."""


class EmbeddingService:
    """Expose the embedding implementation used by the vector repository."""

    def __init__(self, *, embeddings=None, model_path=None):
        if embeddings is not None:
            self._embeddings = embeddings
            return
        from langchain_huggingface import HuggingFaceEmbeddings

        if model_path is None:
            from app.core.config import get_settings

            model_path = get_settings().embedding_model_path
        if not model_path:
            raise ValueError("EMBEDDING_MODEL_PATH 未配置")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={"local_files_only": True},
        )

    def get_embeddings(self):
        return self._embeddings
