# backend/services/llm_service.py
"""LLM 调用封装 — 管理 ChatOpenAI 客户端和结构化输出。"""

import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.schemas.classify_schema import ClassificationOutput
from backend.services.prompt import CLASSIFICATION_SYSTEM

logger = logging.getLogger(__name__)


class LLMService:
    """Wraps ChatOpenAI with structured output for classification.

    LLM 客户端在 __init__ 中创建，with_structured_output 使用
    function_calling 模式确保返回严格的 ClassificationOutput JSON。
    """

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 环境变量未设置。请检查 api_key.env 文件是否存在且包含有效的 API Key。"
            )
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
        )
        self.structured_llm = self.llm.with_structured_output(
            ClassificationOutput, method="function_calling"
        )

    def classify(self, user_message: str) -> ClassificationOutput:
        """调用结构化 LLM 对字段进行分类判定。

        Args:
            user_message: 格式化后的用户提示词（含字段画像 + 检索依据）。

        Returns:
            ClassificationOutput: 结构化的分类结果。
        """
        return self.structured_llm.invoke([
            SystemMessage(content=CLASSIFICATION_SYSTEM),
            HumanMessage(content=user_message),
        ])
