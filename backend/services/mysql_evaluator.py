# backend/services/mysql_evaluator.py
"""MySQL 评测服务 — 连接 MySQL、扫描 schema、逐字段分类、生成报告。

从原 core/evaluator.py 的 connect_and_scan_mysql() 和
evaluate_mysql_fields() 提取纯业务逻辑，去除 Gradio generator。
"""

import os
import logging
import pandas as pd
from backend.core.config import OUTPUT_DIR
from backend.db.mysql import create_mysql_engine, scan_mysql_schema
from backend.db.result_store import save_results

logger = logging.getLogger(__name__)


class MySQLEvaluator:
    """MySQL 数据源评测：schema 扫描 + 批量分类 + 报告生成。

    Args:
        rag_classifier: RAGClassifier 实例（已初始化）。
    """

    def __init__(self, rag_classifier):
        self.rag_classifier = rag_classifier

    # ── Schema 扫描 ─────────────────────────────────────────

    def scan(self, host: str, port: int, user: str,
             password: str, database: str):
        """连接 MySQL 并扫描 schema。

        对应原 core/evaluator.py 的 connect_and_scan_mysql()。

        Args:
            host, port, user, password, database: MySQL 连接参数。

        Returns:
            (fields, log_message): FieldProfile 列表和状态日志字符串。
        """
        if not host or not database:
            return [], "请填写数据库主机地址和数据库名。"

        try:
            engine = create_mysql_engine(host, port, user, password, database)
            fields = scan_mysql_schema(engine, database)
            engine.dispose()

            if not fields:
                return [], f"数据库 '{database}' 连接成功，但未扫描到任何字段。"

            tables = sorted(set(f.table_name for f in fields if f.table_name))
            log = (
                f"连接成功！数据库 '{database}' 共 {len(tables)} 张表、"
                f"{len(fields)} 个字段。\n"
            )
            for tbl in tables:
                tbl_fields = [f for f in fields if f.table_name == tbl]
                log += f"  [{tbl}] {len(tbl_fields)} 字段\n"
            return fields, log

        except Exception as e:
            logger.error("MySQL 连接失败: %s", str(e))
            return [], f"连接失败: {str(e)}"

    # ── 批量评测 ────────────────────────────────────────────

    def evaluate(self, host: str, port: int, user: str,
                 password: str, database: str, progress_callback=None):
        """MySQL schema 扫描 + 全字段分类 + Excel 报告。

        对应原 core/evaluator.py 的 evaluate_mysql_fields()，
        去除了 Gradio generator/yield。

        Args:
            host, port, user, password, database: MySQL 连接参数。
            progress_callback: 可选，签名为 (index, total, field_name, level, category, confidence)。

        Returns:
            (log_lines, metrics_summary, report_path):
                log_lines:       逐条处理日志列表。
                metrics_summary: 需复核率摘要。
                report_path:     生成的 Excel 报告路径（无字段时 None）。
        """
        log_lines = ["开始 MySQL 批量评测..."]

        fields, scan_log = self.scan(host, port, user, password, database)
        log_lines.append(scan_log)

        if not fields:
            return log_lines, "无字段可分类", None

        log_lines.append(f"共 {len(fields)} 条字段，开始 RAG+LLM 分类...")

        results = []
        for index, field in enumerate(fields):
            result = self.rag_classifier.classify_field(field)
            results.append(result)
            log_lines.append(
                f"[{index + 1}/{len(fields)}] "
                f"{field.table_name}.{field.field_name} -> {result.level} / "
                f"{result.category} / confidence={result.confidence}"
            )

            if progress_callback:
                progress_callback(
                    index, len(fields),
                    f"{field.table_name}.{field.field_name}",
                    result.level, result.category, result.confidence
                )

        # 构建 DataFrame
        rows = []
        for r, f in zip(results, fields):
            rows.append({
                "数据库名": f.database_name or "",
                "表名": f.table_name or "",
                "字段英文名": f.field_name,
                "字段类型": f.data_type or "",
                "大模型结论": r.level,
                "数据分类": r.category,
                "数据细分类": r.subcategory or "",
                "置信度": r.confidence,
                "是否需要复核": r.need_review,
                "大模型理由": r.reason,
                "业务域": f.business_domain,
            })
        df = pd.DataFrame(rows)

        # 导出 Excel
        report_path = os.path.join(OUTPUT_DIR, "MySQL数据定级评测报告.xlsx")
        with pd.ExcelWriter(report_path) as writer:
            df.to_excel(writer, sheet_name="全量评测记录", index=False)
            review_df = df[df["是否需要复核"] == True]
            review_df.to_excel(writer, sheet_name="需复核清单", index=False)

        review_rate = df["是否需要复核"].mean()
        metrics_summary = f"MySQL 批量分类完成。需复核率: {review_rate * 100:.2f}%"
        log_lines.append("报告生成完毕。")

        # 持久化到 MySQL
        try:
            engine = create_mysql_engine(host, port, user, password, database)
            save_results(engine, results, source_system="mysql")
            engine.dispose()
        except Exception:
            logger.warning("MySQL 结果写入失败，已通过 Excel 正常导出", exc_info=True)

        return log_lines, metrics_summary, report_path
