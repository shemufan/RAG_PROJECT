# backend/api/classify.py
"""分类相关 API 路由 — 单条分类、批量分类、错题本上传、知识库更新。"""

import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from backend.schemas.classify_schema import (
    FieldProfile, ClassificationResult, ClassifyRequest,
    BatchClassifyResponse, MySQLConnectRequest, ErrorLogUploadResponse,
)
from backend.api.deps import get_rag_classifier, get_chroma_store
from backend.services.batch_evaluator import BatchEvaluator
from backend.services.mysql_evaluator import MySQLEvaluator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["classification"])


@router.post("/classify", response_model=ClassificationResult)
async def classify_single(
    request: ClassifyRequest,
    classifier=Depends(get_rag_classifier),
):
    """对单个字段执行 RAG+LLM 分类定级。"""
    field = FieldProfile(**request.model_dump())
    return classifier.classify_field(field)


@router.post("/batch-classify", response_model=BatchClassifyResponse)
async def classify_batch(
    file: UploadFile = File(...),
    classifier=Depends(get_rag_classifier),
):
    """上传 Excel/CSV 测试集，批量分类并返回报告。"""
    suffix = os.path.splitext(file.filename or "test.xlsx")[1] or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        evaluator = BatchEvaluator(classifier)
        log_lines, summary, report_path, df = evaluator.evaluate_file(tmp_path)
        return BatchClassifyResponse(
            total_fields=len(df),
            completed=len(df),
            log="\n".join(log_lines),
            summary=summary,
            report_path=report_path,
        )
    except Exception as e:
        logger.error("批量分类失败: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/error-log", response_model=ErrorLogUploadResponse)
async def upload_error_log(
    file: UploadFile = File(...),
    classifier=Depends(get_rag_classifier),
):
    """上传错题本 Excel/CSV，激活强制拦截规则。"""
    suffix = os.path.splitext(file.filename or "error_log.xlsx")[1] or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        evaluator = BatchEvaluator(classifier)
        cache = evaluator.load_error_log(tmp_path)
        classifier.set_error_log_cache(cache)
        return ErrorLogUploadResponse(
            rules_activated=len(cache),
            message=f"已激活 {len(cache)} 条强制拦截规则。",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("错题本上传失败: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/knowledge-base")
async def upload_knowledge(
    files: list[UploadFile] = File(...),
    chroma_store=Depends(get_chroma_store),
):
    """上传法规 TXT 文件，分块后添加到知识库。

    注意：当前实现需要先将文件保存到磁盘再处理。
    """
    temp_files = []
    try:
        for file in files:
            suffix = os.path.splitext(file.filename or "doc.txt")[1] or ".txt"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            temp_files.append(tmp)

        # 构造 Gradio 兼容的 file_obj 列表（含 .name 属性）
        class FakeFileObj:
            def __init__(self, name):
                self.name = name

        file_objs = [FakeFileObj(t.name) for t in temp_files]
        result = chroma_store.update_knowledge_base(file_objs)
        return {"message": result}
    except Exception as e:
        logger.error("知识库更新失败: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for t in temp_files:
            try:
                os.unlink(t.name)
            except OSError:
                pass


@router.post("/mysql-scan")
async def mysql_scan(
    conn: MySQLConnectRequest,
    classifier=Depends(get_rag_classifier),
):
    """连接 MySQL 并扫描 schema，返回字段列表。

    连接参数通过 POST Body（JSON）传递，避免密码出现在 URL 中。
    """
    evaluator = MySQLEvaluator(classifier)
    fields, log = evaluator.scan(conn.host, conn.port, conn.user, conn.password, conn.database)
    return {
        "fields_count": len(fields),
        "log": log,
        "tables": list(sorted(set(
            f.table_name for f in fields if f.table_name
        ))),
    }


@router.post("/mysql-classify")
async def mysql_classify(
    conn: MySQLConnectRequest,
    classifier=Depends(get_rag_classifier),
):
    """连接 MySQL，扫描全部字段并逐字段分类，生成报告。

    连接参数通过 POST Body（JSON）传递，避免密码出现在 URL 中。
    """
    evaluator = MySQLEvaluator(classifier)
    log_lines, summary, report_path = evaluator.evaluate(
        conn.host, conn.port, conn.user, conn.password, conn.database
    )
    return {
        "log": "\n".join(log_lines),
        "summary": summary,
        "report_path": report_path,
    }
