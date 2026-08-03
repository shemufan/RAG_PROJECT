# 企业数据智能分类分级服务

## 项目简介

本项目提供一个可查询的企业字段分类分级 Demo。系统从 A 业务数据库读取物理字段元数据和少量脱敏样例，通过 Chroma 检索法规知识、调用结构化 LLM 得出结论，再把字段资产、任务、分类结果和法规依据写入 B 合规数据库。

三个数据存储的职责互相独立：

- A 数据库 `enterprise_source`：保存企业业务数据，不保存分类答案。
- Chroma：保存法规、分类规则和检索向量。
- B 数据库 `compliance_result`：保存扫描任务、字段资产、分类结果和法规依据。

原有单字段接口 `POST /api/classify` 和法规知识库重建脚本继续保留。

## 完整数据流程

```text
enterprise_source
→ SourceMySQLRepository
→ FieldProfile
→ FieldClassificationService
→ Chroma + 结构化 LLM
→ FieldClassificationRecord
→ TargetMySQLRepository
→ compliance_result
→ FastAPI / SQL 查询
```

Pipeline 是同步 Demo，不包含后台队列、定时任务或前端页面。

## 目录结构

```text
app/api/                 FastAPI 路由
app/core/                环境配置
app/schemas/             输入、分类和 Pipeline 模型
app/services/            单字段分类与数据库 Pipeline 编排
app/repositories/        Chroma、A 数据库和 B 数据库适配器
app/rag/                 Prompt 与法规分块
data/knowledge/          分类规则和法规原文
scripts/                 法规知识库重建脚本
sql/                     A/B 建表、虚构数据和查询示例
tests/unit/              不依赖外部服务的单元测试
tests/integration/       API 测试和可选真实 MySQL 测试
```

## FieldProfile 映射

`SourceMySQLRepository` 从 `information_schema.COLUMNS` 和 `information_schema.TABLES` 获取字段信息：

| MySQL 元数据 | FieldProfile |
|---|---|
| `TABLE_SCHEMA` | `database_name` |
| `TABLE_NAME` | `table_name` |
| `TABLE_COMMENT` | `table_comment` |
| `COLUMN_NAME` | `field_name` |
| `COLUMN_COMMENT` | `field_cn`、`field_comment` |
| `COLUMN_TYPE` | `data_type` |
| `IS_NULLABLE` | `is_nullable` |
| `COLUMN_KEY` | `column_key` |

`source_system` 固定为 `mysql`。业务域按表名映射为 `hr`、`customer`、`commerce` 或 `product`，未知表使用 `general`。每个字段最多读取 5 个非空样例；身份证、手机号、银行卡和邮箱在离开 A 数据库前统一脱敏，每个样例最终不超过 50 个字符。

## B 数据库关系表

- `classification_run`：任务状态、模型版本、开始结束时间和统计计数。
- `data_field_asset`：使用稳定 UUID5 标识的物理字段资产。
- `field_classification_result`：可按等级、类别、复核标记等条件查询的分类结论。
- `classification_evidence`：按顺序保存每条分类结果引用的法规依据。

`input_snapshot_json` 和 `raw_output_json` 仅用于审计快照；常用筛选字段均为普通关系字段。

## 环境要求与安装

- Python 3.10
- MySQL 8.x
- Python `venv`

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

已经存在的本地 Sentence Transformers 权重可以直接通过 `EMBEDDING_MODEL_PATH` 引用，不需要重新下载模型文件；Python 环境仍需安装 `sentence-transformers` 运行库。

## 环境配置

复制 `.env.example` 为 `.env`，填写真实配置。数据库密码只存在于服务端环境文件，不通过 API 请求传入。

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL_PATH=G:/AI_Models/sentence-transformer
CHROMA_DB_DIR=.runtime/chroma
CHROMA_COLLECTION=data_classification
KNOWLEDGE_BASE_VERSION=v1
SOURCE_DATABASE_URL=mysql+pymysql://root:password@127.0.0.1:3306/enterprise_source
TARGET_DATABASE_URL=mysql+pymysql://root:password@127.0.0.1:3306/compliance_result
QWEN_OCR_API_KEY=
QWEN_OCR_BASE_URL=https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
QWEN_OCR_MODEL=qwen3.5-ocr
QWEN_OCR_CACHE_DIR=.runtime/ocr_cache
QWEN_OCR_TIMEOUT_SECONDS=180
QWEN_OCR_MAX_RETRIES=2
```

只使用 `/api/classify` 时不要求 MySQL 可用；数据库连接在 Pipeline 或结果查询接口首次调用时才创建。

只有 `data/knowledge/laws/` 中存在 PDF 时才要求配置 `QWEN_OCR_API_KEY` 和
`QWEN_OCR_BASE_URL`。TXT-only 知识库不会创建千问客户端。

## 初始化 A/B 数据库

以下命令使用系统中的 MySQL 客户端，执行时会提示输入密码：

```powershell
cmd /c "mysql -u root -p < sql\source_schema.sql"
cmd /c "mysql -u root -p < sql\source_seed.sql"
cmd /c "mysql -u root -p < sql\target_schema.sql"
```

`source_seed.sql` 为四张业务表各写入 5 条完全虚构的 Demo 数据。也可以在 MySQL Workbench 中按相同顺序执行三个文件。

## 重建法规知识库

法规目录原生支持：

```text
data/knowledge/classification_rules.md
data/knowledge/laws/*.txt
data/knowledge/laws/*.pdf
```

PDF 会先上传至阿里云百炼临时存储，由 `qwen3.5-ocr` 完成文档解析，再进入与 TXT
相同的法规分块、Embedding 和 Chroma 流程。单个 PDF 必须未加密、包含 1–50 页且不超过
100 MB。上传 PDF 前必须确认该文档允许离开本机并由阿里云百炼处理。

OCR 结果按 PDF 内容、模型名称和 Prompt 版本缓存在 `.runtime/ocr_cache/`。文件未变化时
重建不会重复调用付费 OCR；PDF、模型或 Prompt 变化会自动生成新缓存。

更新法规的推荐顺序：

1. 停止 FastAPI。
2. 将允许上传的 TXT/PDF 放入 `data/knowledge/laws/`。
3. 移出已经废止的旧法规，避免新旧标准同时参与检索。
4. 在 `.env` 中增加 `KNOWLEDGE_BASE_VERSION`。
5. 执行重建命令。
6. 重启 FastAPI，并通过结果 Evidence 确认 `document_name` 是新 PDF。

```powershell
python -m scripts.rebuild_knowledge_base
```

所有文档必须先成功读取/OCR，程序才会清空并替换 Chroma collection。任意 PDF 损坏、
超限、配置缺失或 OCR 失败都会终止重建并保留旧知识库，不会静默跳过文件。

## 启动 FastAPI

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger：<http://127.0.0.1:8000/docs>

健康检查：`GET http://127.0.0.1:8000/api/health`

## 运行数据库分类 Pipeline

```powershell
$body = @{
  sample_limit = 3
  table_names = $null
  continue_on_error = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/pipeline/run `
  -ContentType "application/json" `
  -Body $body
```

返回的 `PipelineSummary` 包含 `run_id`、源数据库、总字段数、成功数、复核数、失败数、状态和开始结束时间。分类服务返回 `UNKNOWN` 时，该字段计为失败并更新字段资产，但不会写入一条伪造的 L1–L4 结果。

## 查询 API

```text
GET /api/runs/{run_id}
GET /api/results?run_id=&database_name=&table_name=&column_name=&level=&category=&need_review=&limit=&offset=
GET /api/results/{result_id}/evidence
```

`limit` 范围为 1–200。所有接口使用 Pydantic Response Model，API 层不直接编写 SQL。

手工单字段分类仍可使用：

```powershell
$body = @{
  field_name = "id_card"
  field_cn = "身份证号"
  field_comment = "客户身份证件号码"
  data_type = "varchar(18)"
  sample_values = @("3401**********1234")
  business_domain = "customer"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/classify `
  -ContentType "application/json" `
  -Body $body
```

## SQL 查询示例

[sql/query_examples.sql](sql/query_examples.sql) 提供十类关系查询，包括高敏感字段、表字段分类、等级/类别统计、人工复核、低置信度、法规依据、最近成功任务、按表统计和业务域查询。

## 测试

普通测试不读取真实数据库、LLM、API Key 或本地 Embedding 模型：

```powershell
python -m compileall app scripts
ruff check .
pytest -q -rs
```

真实 MySQL 集成测试只允许专用数据库名中包含 `test` 的连接：

```powershell
$env:MYSQL_TEST_SOURCE_URL = "mysql+pymysql://root:password@127.0.0.1:3306/enterprise_source_test"
$env:MYSQL_TEST_TARGET_URL = "mysql+pymysql://root:password@127.0.0.1:3306/compliance_result_test"
pytest -q tests/integration/test_mysql_integration.py -rs
```

未设置这两个变量时，该测试会明确显示为 skipped，不影响普通测试。

真实千问 OCR 测试是付费且显式启用的，只能使用不含敏感信息、且不超过 5 MB 的测试 PDF：

```powershell
$env:RUN_QWEN_OCR_INTEGRATION = "1"
$env:QWEN_OCR_TEST_PDF = "G:\path\to\non-sensitive-test.pdf"
$env:QWEN_OCR_API_KEY = "你的测试Key"
$env:QWEN_OCR_BASE_URL = "https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:QWEN_OCR_MODEL = "qwen3.5-ocr"
pytest -q tests/integration/test_qwen_ocr_integration.py -rs
```

未显式设置 `RUN_QWEN_OCR_INTEGRATION=1` 时不会发起千问请求。当前临时上传方案适合受控、
低频的同步知识库重建；生产高并发摄取应改用正式 OSS、访问控制和保留策略。

## 当前阶段边界

当前实现面向同步 Demo：一次请求扫描一组字段并逐字段分类写入。生产化阶段仍需根据数据规模补充任务调度、限流、凭据管理、运行恢复和知识库版本切换，但这些能力不属于当前阶段。
