# backend/api/health.py
"""健康检查接口 — 提供服务运行状态探测。"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """服务健康检查。

    Returns:
        dict: code=200, message="backend is running", data=null。
    """
    return {
        "code": 200,
        "message": "backend is running",
        "data": None,
    }
