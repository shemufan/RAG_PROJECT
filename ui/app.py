# ui/app.py
import gradio as gr

from core.engine import update_knowledge_base, smart_predict
from core.evaluator import load_error_log, batch_evaluate, connect_and_scan_mysql, evaluate_mysql_fields
from core.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ 智能合规定级引擎 (含人工干预与动态学习)")

    # "错题本"和"知识库更新" 并排放在网页顶部
    with gr.Row():
        with gr.Accordion("⚙️ 规则干预：上传错题本", open=False):
            error_file = gr.File(label="上传错题本 (.xlsx)")
            error_status = gr.Textbox(label="拦截器状态", value="未激活", interactive=False)
            error_file.upload(fn=load_error_log, inputs=error_file, outputs=error_status)

        with gr.Accordion("📚 知识扩充：上传最新法规", open=False):
            kb_files = gr.File(label="上传法规文本 (.txt)", file_count="multiple")
            kb_btn = gr.Button("🧠 吸收知识入库")
            kb_status = gr.Textbox(label="知识库状态", value="等待投喂...", interactive=False)
            kb_btn.click(fn=update_knowledge_base, inputs=kb_files, outputs=kb_status)

    # 主体功能区：单条测试和批量评测两个标签页
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
                fn=smart_predict,
                inputs=[input_en, input_cn, input_desc],
                outputs=[out_level, out_reason, out_context],
            )

        with gr.TabItem("📊 批量自动化评测"):
            gr.Markdown("上传包含 `标准答案` 的 Excel 测试集，系统将自动计算准确率并导出错题本。")
            with gr.Row():
                with gr.Column(scale=1):
                    test_file = gr.File(label="上传测试集 (.xlsx )")
                    btn_batch = gr.Button("🔥 启动批量评测", variant="primary")

                # 🌟 右侧“运行监控屏幕”
                with gr.Column(scale=2):
                    out_log = gr.Textbox(
                        label=" 实时处理控制台", lines=12, max_lines=15, interactive=False
                    )

            with gr.Row():
                out_acc = gr.Textbox(label="系统量化指标", lines=2)
                out_bad_cases = gr.File(label=" 下载错题诊断表")

            # 三个输出：实时日志、准确率文本、错题本下载链接
            btn_batch.click(
                fn=batch_evaluate,
                inputs=test_file,
                outputs=[out_log, out_acc, out_bad_cases],
                show_progress="hidden",
            )

        with gr.TabItem(" MySQL 数据源"):
            gr.Markdown("连接 MySQL 数据库，自动扫描全部表结构并逐字段分类定级。")
            with gr.Row():
                mysql_host = gr.Textbox(label="主机", value="localhost")
                mysql_port = gr.Number(label="端口", value=3306, precision=0)
                mysql_user = gr.Textbox(label="用户名", value="root")
                mysql_password = gr.Textbox(label="密码", type="password")
                mysql_database = gr.Textbox(label="数据库名", placeholder="test_db")

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
                fn=lambda h, p, u, pw, db: connect_and_scan_mysql(h, int(p), u, pw, db)[1],
                inputs=[mysql_host, mysql_port, mysql_user, mysql_password, mysql_database],
                outputs=mysql_scan_log,
            )

            btn_mysql_run.click(
                fn=evaluate_mysql_fields,
                inputs=[mysql_host, mysql_port, mysql_user, mysql_password, mysql_database],
                outputs=[mysql_out_log, mysql_out_acc, mysql_out_report],
                show_progress="hidden",
            )
