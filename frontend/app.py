# frontend/app.py
"""Gradio Web UI — 基于重构后的 backend 服务层。

四功能模块:
  1. 规则干预：上传错题本
  2. 知识扩充：上传法规 TXT 入库
  3. 单条语义测试
  4. 批量自动化评测
  + MySQL 数据源
"""

import gradio as gr

from backend.services.rag_classifier import RAGClassifier
from backend.services.batch_evaluator import BatchEvaluator
from backend.services.mysql_evaluator import MySQLEvaluator
from backend.core.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

# ── 服务引用（由 run_gradio.py 在启动时注入） ──
_classifier: RAGClassifier = None
_batch_eval: BatchEvaluator = None
_mysql_eval: MySQLEvaluator = None


def init_services(classifier, chroma_store=None):
    """初始化 Gradio 前端所需的服务引用。

    Args:
        classifier:   RAGClassifier 实例。
        chroma_store: 保留参数，供未来扩展（当前通过 build_demo 闭包传递）。
    """
    global _classifier, _batch_eval, _mysql_eval
    _classifier = classifier
    _batch_eval = BatchEvaluator(classifier)
    _mysql_eval = MySQLEvaluator(classifier)


# ── Gradio 回调函数 ─────────────────────────────────────────


def _load_error_log(file_obj):
    """错题本上传回调 — 加载字段→级别映射并写入缓存。"""
    if _batch_eval is None or _classifier is None:
        return " 服务未初始化，请重启应用。"
    if file_obj is None:
        return " 未上传文件。"
    try:
        cache = _batch_eval.load_error_log(file_obj.name)
        _classifier.set_error_log_cache(cache)
        return f" 错题集加载成功！已激活 {len(cache)} 条强制拦截规则。"
    except ValueError as e:
        return f" 格式有误: {str(e)}"
    except Exception as e:
        return f" 格式有误: {str(e)}"


def _smart_predict(field_en, field_cn, desc):
    """单条测试回调 — 委托给 RAGClassifier.smart_predict()。"""
    if _classifier is None:
        return "未知", "服务未初始化，请重启应用。", ""
    return _classifier.smart_predict(field_en, field_cn, desc)


def _batch_evaluate_generator(file_obj, progress=gr.Progress()):
    """批量评测 Gradio 生成器 — 实时日志 + 进度条。"""
    if _batch_eval is None:
        yield "服务未初始化，请重启应用。", "未开始", None
        return
    if file_obj is None:
        yield "请先上传测试集", "未开始", None
        return

    log_lines = ["开始执行批量评测任务..."]

    def on_progress(index, total, field_name, level, category, confidence):
        line = (
            f"[{index + 1}/{total}] "
            f"{field_name} -> {level} / {category} / confidence={confidence}"
        )
        log_lines.append(line)

    try:
        log_lines_inner, metrics_summary, report_path, df = _batch_eval.evaluate_file(
            file_obj.name,
            progress_callback=on_progress,
        )
        yield "\n".join(log_lines_inner), metrics_summary, report_path
    except Exception as e:
        raise gr.Error(f"系统崩溃了！原因：{str(e)}")


def _connect_and_scan_mysql(host, port, user, password, database):
    """MySQL 扫描回调 — 返回扫描日志。"""
    if _mysql_eval is None:
        return " 服务未初始化，请重启应用。"
    fields, log = _mysql_eval.scan(host, int(port), user, password, database)
    return log


def _evaluate_mysql_generator(host, port, user, password, database,
                              progress=gr.Progress()):
    """MySQL 批量评测 Gradio 生成器 — 实时日志 + 进度条。"""
    if _mysql_eval is None:
        yield "服务未初始化，请重启应用。", "未开始", None
        return

    log_lines = ["开始 MySQL 批量评测..."]

    def on_progress(index, total, field_name, level, category, confidence):
        line = (
            f"[{index + 1}/{total}] "
            f"{field_name} -> {level} / {category} / confidence={confidence}"
        )
        log_lines.append(line)

    try:
        log_lines_inner, metrics_summary, report_path = _mysql_eval.evaluate(
            host, int(port), user, password, database,
            progress_callback=on_progress,
        )
        yield "\n".join(log_lines_inner), metrics_summary, report_path
    except Exception as e:
        raise gr.Error(f"系统崩溃了！原因：{str(e)}")


# ── UI 构建 ─────────────────────────────────────────────────


def build_demo(chroma_store=None):
    """构建 Gradio Blocks UI。

    Args:
        chroma_store: ChromaStore 实例，用于知识库更新功能。
    """
    with gr.Blocks() as demo:
        gr.Markdown("# 🛡️ 智能合规定级引擎 (含人工干预与动态学习)")

        # "错题本"和"知识库更新" 并排放在网页顶部
        with gr.Row():
            with gr.Accordion("⚙️ 规则干预：上传错题本", open=False):
                error_file = gr.File(label="上传错题本 (.xlsx)")
                error_status = gr.Textbox(label="拦截器状态", value="未激活", interactive=False)
                error_file.upload(fn=_load_error_log, inputs=error_file, outputs=error_status)

            with gr.Accordion("📚 知识扩充：上传最新法规", open=False):
                kb_files = gr.File(label="上传法规文本 (.txt)", file_count="multiple")
                kb_btn = gr.Button("🧠 吸收知识入库")
                kb_status = gr.Textbox(label="知识库状态", value="等待投喂...", interactive=False)
                if chroma_store is not None:
                    kb_btn.click(
                        fn=chroma_store.update_knowledge_base,
                        inputs=kb_files,
                        outputs=kb_status,
                    )
                else:
                    kb_btn.click(
                        fn=lambda f: " 服务未初始化，无法更新知识库",
                        inputs=kb_files,
                        outputs=kb_status,
                    )

        # 主体功能区：三个标签页
        with gr.Tabs():
            with gr.TabItem("🔍 单条语义测试"):
                gr.Markdown("输入单个业务字段，测试大模型的思维逻辑。")
                with gr.Row():
                    with gr.Column():
                        input_en = gr.Textbox(label="字段英文名 (如: account_code)")
                        input_cn = gr.Textbox(label="字段中文名")
                        input_desc = gr.Textbox(label="业务描述", lines=3)
                        btn_single = gr.Button("🚀 运行单条判级", variant="primary")
                    with gr.Column():
                        out_level = gr.Textbox(label="判定级别")
                        out_context = gr.Textbox(label="📖 检索到的法律依据 (Top 3)", lines=8)
                        out_reason = gr.Textbox(label="大模型推演理由", lines=5)
                btn_single.click(
                    fn=_smart_predict,
                    inputs=[input_en, input_cn, input_desc],
                    outputs=[out_level, out_reason, out_context],
                )

            with gr.TabItem("📊 批量自动化评测"):
                gr.Markdown("上传包含 `标准答案` 的 Excel 测试集，系统将自动计算准确率并导出错题本。")
                with gr.Row():
                    with gr.Column(scale=1):
                        test_file = gr.File(label="上传测试集 (.xlsx )")
                        btn_batch = gr.Button("🔥 启动批量评测", variant="primary")

                    with gr.Column(scale=2):
                        out_log = gr.Textbox(
                            label=" 实时处理控制台", lines=12, max_lines=15, interactive=False
                        )

                with gr.Row():
                    out_acc = gr.Textbox(label="系统量化指标", lines=2)
                    out_bad_cases = gr.File(label=" 下载错题诊断表")

                btn_batch.click(
                    fn=_batch_evaluate_generator,
                    inputs=test_file,
                    outputs=[out_log, out_acc, out_bad_cases],
                    show_progress="hidden",
                )

            with gr.TabItem(" MySQL 数据源"):
                gr.Markdown("连接 MySQL 数据库，自动扫描全部表结构并逐字段分类定级。")
                with gr.Row():
                    mysql_host = gr.Textbox(label="主机", value=MYSQL_HOST)
                    mysql_port = gr.Number(label="端口", value=MYSQL_PORT, precision=0)
                    mysql_user = gr.Textbox(label="用户名", value=MYSQL_USER)
                    mysql_password = gr.Textbox(label="密码", type="password")
                    mysql_database = gr.Textbox(label="数据库名", placeholder=MYSQL_DATABASE or "test_db")

                with gr.Row():
                    btn_scan = gr.Button(" 连接并扫描", variant="secondary")
                    btn_mysql_run = gr.Button(" 一键分类全部字段", variant="primary")

                mysql_scan_log = gr.Textbox(label="扫描结果", lines=8, interactive=False)

                with gr.Row():
                    mysql_out_log = gr.Textbox(
                        label="实时处理控制台", lines=10, max_lines=15, interactive=False
                    )
                with gr.Row():
                    mysql_out_acc = gr.Textbox(label="量化指标", lines=2)
                    mysql_out_report = gr.File(label="下载评测报告")

                btn_scan.click(
                    fn=lambda h, p, u, pw, db: _connect_and_scan_mysql(h, int(p), u, pw, db),
                    inputs=[mysql_host, mysql_port, mysql_user, mysql_password, mysql_database],
                    outputs=mysql_scan_log,
                )

                btn_mysql_run.click(
                    fn=_evaluate_mysql_generator,
                    inputs=[mysql_host, mysql_port, mysql_user, mysql_password, mysql_database],
                    outputs=[mysql_out_log, mysql_out_acc, mysql_out_report],
                    show_progress="hidden",
                )

    return demo
