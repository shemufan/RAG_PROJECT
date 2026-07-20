# backend/storage/result_store.py
"""分类结果持久化 — MySQL 写入层。

所有函数内部 try-except，失败只记 log 不抛异常，
保证 Excel 导出和 LLM 推理不受 DB 写入失败影响。
"""

import logging
import json

logger = logging.getLogger(__name__)

CREATE_RESULTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS classification_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    field_name VARCHAR(255) NOT NULL,
    database_name VARCHAR(255) DEFAULT NULL,
    table_name VARCHAR(255) DEFAULT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(255) DEFAULT NULL,
    level VARCHAR(10) NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    reason TEXT,
    evidence_json JSON,
    need_review BOOLEAN DEFAULT FALSE,
    decision_path VARCHAR(50) DEFAULT 'rag_llm',
    source_system VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_field_name (field_name),
    INDEX idx_level (level),
    INDEX idx_need_review (need_review),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

CREATE_ERROR_RULES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS error_log_rules (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    field_name VARCHAR(255) NOT NULL,
    level VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_field_name (field_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def ensure_tables(engine):
    """自动建表（如不存在）。"""
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text(CREATE_RESULTS_TABLE_SQL))
            conn.execute(text(CREATE_ERROR_RULES_TABLE_SQL))
            conn.commit()
        logger.info("结果表和错题本规则表已就绪")
    except Exception:
        logger.warning("建表失败（可能 MySQL 未连接），跳过", exc_info=True)


def save_results(engine, results, source_system: str = "excel") -> int:
    """批量写入分类结果到 classification_results。

    Returns:
        成功写入的条数，失败时返回 0。
    """
    if not results:
        return 0

    from sqlalchemy import text

    insert_sql = text("""
        INSERT INTO classification_results
            (field_name, database_name, table_name, category, subcategory,
             level, confidence, reason, evidence_json, need_review,
             decision_path, source_system)
        VALUES
            (:field_name, :database_name, :table_name, :category, :subcategory,
             :level, :confidence, :reason, :evidence_json, :need_review,
             :decision_path, :source_system)
    """)

    count = 0
    try:
        with engine.connect() as conn:
            for r in results:
                evidence_json = json.dumps(
                    [e.model_dump() for e in r.evidence],
                    ensure_ascii=False,
                ) if r.evidence else None

                conn.execute(insert_sql, {
                    "field_name": r.field_name,
                    "database_name": r.database_name,
                    "table_name": r.table_name,
                    "category": r.category,
                    "subcategory": r.subcategory,
                    "level": r.level,
                    "confidence": r.confidence,
                    "reason": r.reason,
                    "evidence_json": evidence_json,
                    "need_review": r.need_review,
                    "decision_path": r.decision_path,
                    "source_system": source_system,
                })
                count += 1
            conn.commit()
        logger.info("成功写入 %d 条结果到 MySQL", count)
    except Exception:
        logger.warning("结果写入 MySQL 失败，已通过 Excel 正常导出", exc_info=True)
    return count


def save_error_rules(engine, cache: dict) -> int:
    """将内存 ERROR_LOG_CACHE 写入 error_log_rules（UPSERT）。"""
    if not cache:
        return 0

    from sqlalchemy import text

    upsert_sql = text("""
        INSERT INTO error_log_rules (field_name, level)
        VALUES (:field_name, :level)
        ON DUPLICATE KEY UPDATE level = VALUES(level), updated_at = CURRENT_TIMESTAMP
    """)

    count = 0
    try:
        with engine.connect() as conn:
            for field_name, level in cache.items():
                conn.execute(upsert_sql, {
                    "field_name": str(field_name),
                    "level": str(level),
                })
                count += 1
            conn.commit()
        logger.info("成功持久化 %d 条错题本规则到 MySQL", count)
    except Exception:
        logger.warning("错题本规则写入 MySQL 失败", exc_info=True)
    return count


def load_error_rules(engine) -> dict:
    """从 error_log_rules 表恢复错题本缓存。

    Returns:
        {field_name: level} 字典，无数据或失败时返回空 dict。
    """
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT field_name, level FROM error_log_rules")
            ).mappings().all()
        cache = {row["field_name"]: row["level"] for row in rows}
        if cache:
            logger.info("从 MySQL 恢复 %d 条错题本规则", len(cache))
        return cache
    except Exception:
        logger.warning("从 MySQL 恢复错题本规则失败，使用空缓存", exc_info=True)
        return {}
