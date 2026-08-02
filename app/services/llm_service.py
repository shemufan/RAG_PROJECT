"""OpenAI-compatible structured language-model client."""

from langchain_core.messages import BaseMessage

from app.schemas.classification import ClassificationOutput


class LLMService:
    """Return validated structured classification output from an LLM."""

    def __init__(self, *, structured_model=None, settings=None):
        if structured_model is not None:
            self._structured_model = structured_model
            return
        from langchain_openai import ChatOpenAI

        if settings is None:
            from app.core.config import get_settings

            settings = get_settings()
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY 未配置")
        model = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0,
            timeout=settings.deepseek_timeout_seconds,
            max_retries=settings.deepseek_max_retries,
        )
        self._structured_model = model.with_structured_output(
            ClassificationOutput,
            method="function_calling",
        )

    def classify(self, prompt: list[BaseMessage]) -> ClassificationOutput:
        result = self._structured_model.invoke(prompt)
        return ClassificationOutput.model_validate(result)
