"""Minimal Gradio client for the Week3 FastAPI baseline."""

import os

import gradio as gr
import httpx

API_URL = os.getenv("CLASSIFY_API_URL", "http://127.0.0.1:8000/api/classify")


def _empty_result(level: str, message: str):
    return level, "—", message, "0%", "暂无可展示依据", "需要人工复核"


def classify_via_api(
    field_name: str,
    field_cn: str,
    field_comment: str,
    sample_values: str,
    *,
    client: httpx.Client | None = None,
):
    """Call FastAPI and format its response for Gradio components."""
    if not (field_name or "").strip():
        return _empty_result("输入有误", "字段英文名不能为空。")

    payload = {
        "field_name": field_name.strip(),
        "field_cn": (field_cn or "").strip() or None,
        "field_comment": (field_comment or "").strip() or None,
        "sample_values": [
            item.strip()
            for item in (sample_values or "").replace("；", ";").replace("\n", ";").split(";")
            if item.strip()
        ],
    }
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0)
    try:
        response = client.post(API_URL, json=payload)
        response.raise_for_status()
        body = response.json()
        data = body.get("data") or {}
        if body.get("code") != 200:
            return _empty_result(data.get("level", "请求失败"), body.get("message", "分类失败"))

        evidence_lines = []
        for index, item in enumerate(data.get("evidence", []), start=1):
            title = " · ".join(
                value for value in [item.get("source"), item.get("article")] if value
            )
            score = item.get("score")
            score_text = f" ｜ 相关度 {score:.0%}" if isinstance(score, (int, float)) else ""
            evidence_lines.append(f"[{index}] {title}{score_text}\n{item.get('content', '')}")

        category = data.get("category", "未知")
        if data.get("subcategory"):
            category += f" / {data['subcategory']}"
        review = "需要人工复核" if data.get("need_review") else "无需人工复核"
        return (
            data.get("level", "UNKNOWN"),
            category,
            data.get("reason", "未返回判定理由"),
            f"{float(data.get('confidence', 0)):.0%}",
            "\n\n".join(evidence_lines) or "知识库未返回依据",
            review,
        )
    except httpx.ConnectError:
        return _empty_result("连接失败", "无法连接后端，请先启动 FastAPI 服务。")
    except httpx.TimeoutException:
        return _empty_result("请求超时", "后端推理超过 60 秒，请稍后重试。")
    except (httpx.HTTPError, ValueError) as exc:
        return _empty_result("请求失败", f"后端响应异常：{exc}")
    finally:
        if owns_client:
            client.close()


CSS = """
:root { --ink: #17211b; --paper: #f3f0e7; --rule: #9b2c2c; --sage: #b7c4af; }
.gradio-container { background: var(--paper) !important; color: var(--ink); max-width: 1180px !important; }
.hero { border-top: 6px solid var(--rule); border-bottom: 1px solid #9a978e; padding: 22px 0 18px; margin-bottom: 18px; }
.hero h1 { font-family: Georgia, 'Noto Serif SC', serif; letter-spacing: .02em; margin: 0; }
.hero p { color: #59625c; margin: 8px 0 0; }
.panel { border: 1px solid #bbb7ac !important; background: rgba(255,255,255,.48) !important; box-shadow: 8px 8px 0 rgba(23,33,27,.07) !important; }
.primary { background: var(--rule) !important; border-color: var(--rule) !important; }
.level input { font-family: Georgia, serif !important; font-size: 2rem !important; font-weight: 700 !important; color: var(--rule) !important; }
"""


def build_demo():
    with gr.Blocks(title="数据合规定级 · Week3 Baseline") as demo:
        gr.HTML(
            "<div class='hero'><h1>数据合规定级工作台</h1>"
            "<p>Week3 Baseline · 单字段语义检索与合规判定</p></div>"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="panel"):
                gr.Markdown("### 字段画像")
                field_name = gr.Textbox(label="字段英文名 *", value="id_card")
                field_cn = gr.Textbox(label="字段中文名", value="身份证号")
                field_comment = gr.Textbox(
                    label="业务描述", value="用户身份证号码", lines=3
                )
                sample_values = gr.Textbox(
                    label="样例值（多值用分号或换行分隔）",
                    value="340************1234",
                    lines=2,
                )
                submit = gr.Button("执行分类定级", variant="primary", elem_classes="primary")
            with gr.Column(scale=7, elem_classes="panel"):
                gr.Markdown("### 判定结论")
                with gr.Row():
                    level = gr.Textbox(label="敏感等级", interactive=False, elem_classes="level")
                    category = gr.Textbox(label="数据分类", interactive=False)
                    confidence = gr.Textbox(label="置信度", interactive=False)
                review = gr.Textbox(label="复核状态", interactive=False)
                reason = gr.Textbox(label="判定理由", lines=7, interactive=False)
                evidence = gr.Textbox(label="检索依据 · Top 3", lines=12, interactive=False)

        submit.click(
            classify_via_api,
            inputs=[field_name, field_cn, field_comment, sample_values],
            outputs=[level, category, reason, confidence, evidence, review],
        )
    return demo
