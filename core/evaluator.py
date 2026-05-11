# core/evaluator.py
import logging
import os
import traceback

import gradio as gr
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from core.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

from core.engine import smart_predict, set_error_log_cache


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
    """批量评测主流程：逐行调用大模型推理 → 计算指标 → 导出多 Sheet Excel 报告。

    这是一个 Gradio 生成器函数，通过 yield 向 Web 界面实时推送处理进度和日志。

    测试集要求：Excel 或 CSV，列名必须为「英文名」「中文名」「业务描述」「标准答案」。

    输出 Excel 包含三个 Sheet：
        - 全量评测记录：每条字段的判定结果 + 推理理由 + 检索法律依据
        - 错题核查清单：大模型结论与标准答案不一致的条目
        - 量化评估指标：整体准确率 + 各级别精确率/召回率/F1

    Args:
        file_obj: Gradio File 组件上传的测试集文件对象。
        progress: Gradio Progress 组件，用于 tqdm 进度条。

    Yields:
        tuple: (log_content, metrics_summary, report_path)
            - log_content:      实时处理控制台的累积日志文本
            - metrics_summary:  准确率与 L4 召回率的摘要文字
            - report_path:      生成的全量评测报告 Excel 文件的本地路径
    """
    try:
        # 定义log_content专门用来存实时日志
        log_content = " 开始执行批量评测任务...\n"
        # 使用yield把当前的日志推送给网页
        yield log_content, "处理中...", None

        # debug1 - 验证函数入口
        logger.info("开始执行批量评测任务")
        if file_obj is None:
            return "请先上传测试集", None

        # debug2 - 验证文件格式
        logger.info("正在读取文件: %s", file_obj.name)
        if file_obj.name.endswith(".xlsx") or file_obj.name.endswith(".xls"):
            df = pd.read_excel(file_obj.name)
        else:
            df = pd.read_csv(file_obj.name)

        # debug3 - 验证表格结构
        logger.debug("读取成功！当前表格的列名有: %s", list(df.columns))
        required_columns = ["英文名", "中文名", "业务描述", "标准答案"]
        for col in required_columns:
            if col not in df.columns:
                # 如果缺列或者列名不匹配，直接触发网页红色报错框
                raise gr.Error(f" 格式不对！找不到名为 '{col}' 的列，请检查 Excel 表头！")

        predictions, reasons, contexts = [], [], []

        log_content += f" 准备就绪，共计 {len(df)} 条数据，开始调用大模型...\n"
        log_content += "-" * 40 + "\n"
        yield log_content, "推理中...", None

        # debug4 - 验证循环逻辑
        logger.info("开始遍历 %d 条数据，准备调用大模型...", len(df))
        for index, row in progress.tqdm(df.iterrows(), total=len(df), desc="智能定级推演中"):
            logger.debug("正在推演第 %d 条数据: %s", index + 1, row["英文名"])
            lvl, rsn, ctx = smart_predict(row["英文名"], row["中文名"], row["业务描述"])
            predictions.append(lvl)
            reasons.append(rsn)
            contexts.append(ctx)

            # 把当前这条的处理结果追加到日志里
            log_content += f" [{index + 1}/{len(df)}] 字段 '{row['英文名']}' -> 判定为: {lvl}\n"
            # 实时推送到网页的日志框里
            yield log_content, "推理中...", None

        log_content += "-" * 40 + "\n"
        log_content += " 大模型推理全部完成，正在计算准确率...\n"
        yield log_content, "结算中...", None

        # debug5 - 验证结果处理
        logger.info("大模型推理全部完成，正在计算准确率...")
        df["大模型结论"] = predictions
        df["大模型理由"] = reasons
        df["检索到的法律依据"] = contexts

        report_dict = classification_report(
            df["标准答案"], df["大模型结论"], output_dict=True, zero_division=0
        )
        acc = accuracy_score(df["标准答案"], df["大模型结论"])

        l4_recall = report_dict.get("L4", {}).get("recall", 0)

        # debug6 - 验证结果导出(检索信息和错题本)
        logger.info("正在生成多维度评测报告...")

        # 1. 准备错题集
        bad_cases_df = df[df["标准答案"] != df["大模型结论"]]

        # 2. 将计算好的评估报告转化为 Pandas 表格，并翻译成中文表头
        metrics_df = pd.DataFrame(report_dict).transpose()
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(
            columns={
                "index": "评测维度 (级别)",
                "precision": "精确率 (Precision)",
                "recall": "召回率 (Recall)",
                "f1-score": "F1 分数",
                "support": "真实数据量",
            },
            inplace=True,
        )

        # 3. 定义导出路径
        report_path = os.path.join(OUTPUT_DIR, "数据定级全量评测报告.xlsx")

        # 4. 使用 ExcelWriter 实现多 Sheet 导出
        with pd.ExcelWriter(report_path) as writer:
            # Sheet 1: 检索信息和大模型推理结果（全量数据）
            df.to_excel(writer, sheet_name="全量评测记录", index=False)
            # Sheet 2: 错题集（方便针对性指正）
            bad_cases_df.to_excel(writer, sheet_name="错题核查清单", index=False)
            # Sheet 3: 量化评测指标（整体准确率和各级别的精确率、召回率、F1分数）
            metrics_df.to_excel(writer, sheet_name="量化评估指标", index=False)

        log_content += "  报告生成完毕，包含全量数据与错题清单。\n"
        metrics_summary = f" 整体准确率: {acc*100:.2f}%\n L4 级别召回率: {l4_recall*100:.2f}%\n(请下载 Excel 查看详细检索依据)"

        # 返回文件路径
        yield log_content, metrics_summary, report_path

    # 捕获所有未知的致命错误
    except Exception as e:
        error_detail = traceback.format_exc()
        error_msg = f"\n================ ❌ 发现致命错误 ================\n{str(e)}\n"
        # 在 Gradio 网页右上角弹出一个红色的报错提示框
        yield log_content + error_msg, "系统崩溃", None
        raise gr.Error(f"系统崩溃了！原因：{str(e)}")
