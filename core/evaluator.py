# core/evaluator.py
import logging
import os

import gradio as gr
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from core.config import OUTPUT_DIR
from connectors.file_loader import load_field_profiles_from_file
from core.engine import classify_field
from core.engine import set_error_log_cache

logger = logging.getLogger(__name__)


def load_error_log(file_obj):
    """加载错题本文件，将字段名→标准答案映射写入全局拦截缓存。

    支持 .xlsx 和 .csv 两种格式，表格必须包含「name」列（字段英文名）和「标准答案」列（L1-L4）。

    Args:
        file_obj: Gradio File 组件上传的文件对象，含 .name 属性。

    Returns:
        str: 给前端的加载状态提示。
    """
    if file_obj is None:
        return "未上传文件。"
    try:
        df = (
            pd.read_excel(file_obj.name)
            if file_obj.name.endswith(".xlsx")
            else pd.read_csv(file_obj.name)
        )
        cache = dict(zip(df["name"], df["标准答案"]))
        set_error_log_cache(cache)
        return f"错题集加载成功！已激活 {len(cache)} 条强制拦截规则。"
    except Exception as e:
        return f"格式有误: {str(e)}"


def batch_evaluate(file_obj, progress=gr.Progress()):
    try:
        log_content = "开始执行批量评测任务...\n"
        yield log_content, "处理中...", None

        if file_obj is None:
            yield "请先上传测试集", "未开始", None
            return

        # 1. 读取原始 df，并转换为结构化字段画像
        df, fields = load_field_profiles_from_file(file_obj.name)

        # 2. 检查是否有标准答案
        has_label = "标准答案" in df.columns

        results = []

        log_content += f"读取成功，共 {len(fields)} 条字段，开始 RAG+LLM 分类...\n"
        yield log_content, "推理中...", None

        # 3. 推理阶段只使用 fields
        for index, field in progress.tqdm(
            enumerate(fields), total=len(fields), desc="智能定级推演中"
        ):
            result = classify_field(field)
            results.append(result)

            log_content += (
                f"[{index + 1}/{len(fields)}] "
                f"{field.field_name} -> {result.level} / "
                f"{result.category} / confidence={result.confidence}\n"
            )
            yield log_content, "推理中...", None

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

        # 5. 如果有标准答案，则计算指标；没有则只导出结果
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
                metrics_df.to_excel(writer, sheet_name="量化评估指标")

        if has_label:
            metrics_summary = f"整体准确率: {acc * 100:.2f}%"
        else:
            review_rate = df["是否需要复核"].mean()
            metrics_summary = f"无标准答案，仅完成批量分类。需复核率: {review_rate * 100:.2f}%"

        log_content += "报告生成完毕。\n"
        yield log_content, metrics_summary, report_path

    except Exception as e:
        raise gr.Error(f"系统崩溃了！原因：{str(e)}")
