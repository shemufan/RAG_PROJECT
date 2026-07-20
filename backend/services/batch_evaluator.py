# backend/services/batch_evaluator.py
"""批量评测服务 — 文件加载、字段分类、指标计算、Excel 报告。

从原 core/evaluator.py 提取纯业务逻辑，去除 Gradio generator/yield，
返回结构化数据供前端 (Gradio generator) 和 API (REST) 共用。
"""

import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from backend.core.config import OUTPUT_DIR, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from backend.utils.file_loader import load_field_profiles_from_file, resolve_column_mapping
from backend.storage.mysql import create_mysql_engine
from backend.storage.result_store import save_results, save_error_rules

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """批量字段分类评测。

    Args:
        rag_classifier: RAGClassifier 实例（已初始化）。
    """

    def __init__(self, rag_classifier):
        self.rag_classifier = rag_classifier

    # ── 错题本加载 ──────────────────────────────────────────

    def load_error_log(self, file_path: str) -> dict:
        """从错题本文件加载字段名→级别映射。

        对应原 core/evaluator.py 的 load_error_log()，
        去除了 Gradio file_obj 依赖，改为直接接受文件路径。

        Args:
            file_path: 错题本 Excel/CSV 文件路径。

        Returns:
            dict: {field_name: level} 映射字典。

        Raises:
            ValueError: 文件格式不支持或缺少必要的列。
        """
        df = (
            pd.read_excel(file_path)
            if file_path.endswith(".xlsx")
            else pd.read_csv(file_path)
        )
        col_map = resolve_column_mapping(list(df.columns))
        fn_col = col_map.get("field_name")
        if fn_col is None:
            raise ValueError("未找到字段英文名列（支持的列名: 英文名、name、字段名 等）")

        # 标准答案列直接匹配
        label_col = None
        for alias in ["标准答案", "答案", "级别", "label", "level"]:
            if alias in df.columns:
                label_col = alias
                break
        if label_col is None:
            raise ValueError("未找到标准答案列（支持的列名: 标准答案、答案、级别 等）")

        cache = dict(zip(df[fn_col], df[label_col]))

        # 持久化到 MySQL（失败不影响功能）
        try:
            engine = create_mysql_engine(
                MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
            )
            save_error_rules(engine, cache)
            engine.dispose()
        except Exception:
            logger.warning("错题本规则持久化到 MySQL 失败，仅保留内存缓存", exc_info=True)

        return cache

    # ── 批量评测 ────────────────────────────────────────────

    def evaluate_file(self, file_path: str, progress_callback=None):
        """对文件中的所有字段执行 RAG 分类，生成 Excel 报告。

        对应原 core/evaluator.py 的 batch_evaluate()，
        去除了 Gradio generator/yield，改为通过 progress_callback 报告进度。

        Args:
            file_path:         测试集 Excel/CSV 文件路径。
            progress_callback: 可选，签名为 (index, total, field_name, level, category, confidence)。

        Returns:
            (log_lines, metrics_summary, report_path, df):
                log_lines:       逐条处理日志列表。
                metrics_summary: 准确率或需复核率摘要。
                report_path:     生成的 Excel 报告路径。
                df:              回填了分类结果的 DataFrame。
        """
        log_lines = ["开始执行批量评测任务..."]

        # 1. 读取原始 df，并转换为结构化字段画像
        df, fields = load_field_profiles_from_file(file_path)

        # 2. 检查是否有标准答案
        has_label = "标准答案" in df.columns

        results = []
        log_lines.append(f"读取成功，共 {len(fields)} 条字段，开始 RAG+LLM 分类...")

        # 3. 逐字段推理
        for index, field in enumerate(fields):
            result = self.rag_classifier.classify_field(field)
            results.append(result)

            log_lines.append(
                f"[{index + 1}/{len(fields)}] "
                f"{field.field_name} -> {result.level} / "
                f"{result.category} / confidence={result.confidence}"
            )

            if progress_callback:
                progress_callback(
                    index, len(fields), field.field_name,
                    result.level, result.category, result.confidence
                )

        # 4. 回填结果到 df
        df["大模型结论"] = [r.level for r in results]
        df["数据分类"] = [r.category for r in results]
        df["数据细分类"] = [r.subcategory for r in results]
        df["置信度"] = [r.confidence for r in results]
        df["是否需要复核"] = [r.need_review for r in results]
        df["大模型理由"] = [r.reason for r in results]
        df["结构化JSON结果"] = [r.model_dump_json(ensure_ascii=False) for r in results]

        df["检索到的依据"] = [
            "\n\n".join(
                [f"{e.document_name} {e.hierarchy_level or ''}\n{e.content}" for e in r.evidence]
            )
            for r in results
        ]

        # 5. 计算指标
        if has_label:
            report_dict = classification_report(
                df["标准答案"], df["大模型结论"], output_dict=True, zero_division=0
            )
            acc = accuracy_score(df["标准答案"], df["大模型结论"])
            bad_cases_df = df[df["标准答案"] != df["大模型结论"]]
        else:
            report_dict = {}
            acc = None
            bad_cases_df = df[df["是否需要复核"] == True]

        # 6. 导出 Excel
        report_path = os.path.join(OUTPUT_DIR, "数据定级全量评测报告.xlsx")

        with pd.ExcelWriter(report_path) as writer:
            df.to_excel(writer, sheet_name="全量评测记录", index=False)
            bad_cases_df.to_excel(writer, sheet_name="需复核或错题清单", index=False)

            if has_label:
                metrics_df = pd.DataFrame(report_dict).transpose()
                metrics_df.reset_index(inplace=True)
                metrics_df.rename(columns={
                    "index": "评测维度 (级别)",
                    "precision": "精确率 (Precision)",
                    "recall": "召回率 (Recall)",
                    "f1-score": "F1 分数",
                    "support": "真实数据量",
                }, inplace=True)
                metrics_df.to_excel(writer, sheet_name="量化评估指标", index=False)

        if has_label:
            metrics_summary = f"整体准确率: {acc * 100:.2f}%"
        else:
            review_rate = df["是否需要复核"].mean()
            metrics_summary = f"无标准答案，仅完成批量分类。需复核率: {review_rate * 100:.2f}%"

        log_lines.append("报告生成完毕。")

        # 7. 持久化到 MySQL
        try:
            engine = create_mysql_engine(
                MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
            )
            save_results(engine, results, source_system="excel")
            engine.dispose()
        except Exception:
            logger.warning("结果写入 MySQL 失败，已通过 Excel 正常导出", exc_info=True)

        return log_lines, metrics_summary, report_path, df
