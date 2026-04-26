# ui/app.py
import gradio as gr

from core.engine import update_knowledge_base, smart_predict
from core.evaluator import load_error_log, batch_evaluate

with gr.Blocks(theme=gr.themes.Soft()) as demo:
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

                # 🌟 新增的右侧“运行监控屏幕”
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
