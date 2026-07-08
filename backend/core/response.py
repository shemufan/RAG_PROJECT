# backend/core/response.py
"""统一 API 响应格式。

所有 API 接口统一使用此格式返回，便于前端统一处理。
"""

from typing import Any, Optional
from pydantic import BaseModel


class APIResponse(BaseModel):
    """标准 API 响应包装。

    Fields:
        code:    业务状态码，200 表示成功。
        message: 提示信息。
        data:    响应数据，可为任意类型。
    """

    code: int = 200
    message: str = "success"
    data: Optional[Any] = None


def success(data: Any = None, message: str = "success") -> APIResponse:
    """构建成功响应。"""
    return APIResponse(code=200, message=message, data=data)


def error(code: int, message: str, data: Any = None) -> APIResponse:
    """构建错误响应。"""
    return APIResponse(code=code, message=message, data=data)
