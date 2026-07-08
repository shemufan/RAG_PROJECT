# backend/main.py
"""FastAPI 应用入口 — 路由注册 + CORS 配置。

启动命令:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

Swagger 文档: http://127.0.0.1:8000/docs
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# ── FastAPI 应用实例 ──────────────────────────────────────

app = FastAPI(
    title="Data Compliance RAG API",
    description="企业数据智能分类分级 RAG 系统 — 后端 API",
    version="1.0.0",
)

# ── CORS 中间件（允许前端跨域访问） ──────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ──────────────────────────────────────────────

from backend.api.health import router as health_router
from backend.api.classify import router as classify_router

app.include_router(health_router, prefix="/api")
app.include_router(classify_router, prefix="/api")

logger.info("=== Data Compliance RAG API 路由已注册 ===")


# ── 直接运行入口 ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
