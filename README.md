# 企业数据智能分类分级服务

## 项目简介

本项目使用法规知识检索和结构化大模型调用，对单个企业数据字段进行分类分级，并返回可追溯依据。

## 当前最小功能

- `GET /api/health`：服务健康检查。
- `POST /api/classify`：接收一个 `FieldProfile` 并返回分类结果。
- 从本地法规知识库检索相关条款。
- 通过 DeepSeek 或兼容 OpenAI 的接口生成结构化结论。
- 使用 Pydantic 校验输入、模型输出和 API 响应。

## 系统流程

```text
FastAPI → FieldProfile → 检索文本 → Chroma → 结构化 LLM → ClassificationResult
```

## 目录结构

```text
app/                    FastAPI、Schema、服务与向量仓库
scripts/                法规知识库重建脚本
data/knowledge/         分类规则与法规原文
tests/unit/             纯单元测试
tests/integration/      API 集成测试
```

## 环境要求

- Python 3.10
- Python `venv`

## 安装

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 配置

复制 `.env.example` 为 `.env`，填写 `DEEPSEEK_API_KEY` 和本地 Embedding 模型路径。`.env` 只保存在本地。如果模型文件已存在，`EMBEDDING_MODEL_PATH` 直接指向该目录，不会重新下载权重。

`sample_values` 会参与检索并发送给配置的 LLM 服务，请只传入已脱敏样例。

## 重建法规知识库

```powershell
python -m scripts.rebuild_knowledge_base
```

## 启动 FastAPI

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger 地址：<http://127.0.0.1:8000/docs>

## 分类请求示例

```powershell
$body = @{
  field_name = "id_card"
  field_cn = "身份证号"
  field_comment = "客户身份证件号码"
  data_type = "varchar(18)"
  sample_values = @("340************1234")
  business_domain = "customer"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/classify `
  -ContentType "application/json" `
  -Body $body
```

成功响应的 `data` 包含 `field_name`、`category`、`subcategory`、`level`、`confidence`、`reason`、`evidence`、`need_review` 和 `decision_path`。输入校验失败返回 HTTP 422；依赖异常时为保持现有合约，HTTP 仍为 200，响应体中 `code=500`、`level=UNKNOWN` 且 `need_review=true`。

文本字段最长 256 字符（`field_comment` 为 1000，`field_name` 为 128）；`sample_values` 最多 5 项，每项最长 256 字符。

## 测试

```powershell
python -m compileall app scripts
ruff check .
pytest -q
```

## 当前阶段边界

当前只处理调用方提交的单字段画像，不扫描业务数据库、不批量评测，也不持久化分类结果。后续数据接入应复用 `FieldProfile`、`FieldClassificationService.classify_field()` 和 `ClassificationResult`。
