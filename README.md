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

已有 B 库从旧版本升级时，仅执行一次以下迁移；全新执行过当前
`sql/target_schema.sql` 的数据库已经包含该列，不要重复迁移：

```powershell
cmd /c "mysql -u root -p compliance_result < sql\migrations\2026-08-04_add_is_personal.sql"
```

## 个人信息字段 Benchmark

Benchmark 用两份已标注 CSV 检验系统能否在大量非个人信息字段中识别个人信息。CSV
只负责一次性导入；正式评测始终从 A 库逐行读取测试案例，经现有 Chroma + LLM 链路
分类，再将逐案例结果和评分写入 B 库。真实标签不会进入 `FieldProfile`、检索文本或
Prompt。

### 1. 建表

先确保 `enterprise_source` 和 `compliance_result` 已按上文建立，再分别增加 Benchmark
表：

```powershell
cmd /c "mysql -u root -p < sql\benchmark_source_schema.sql"
cmd /c "mysql -u root -p < sql\benchmark_target_schema.sql"
```

- A.`benchmark_field_input`：保存 CSV 原始字段名、最多 5 个脱敏样例和真实标签；同名字段
  不合并。
- B.`benchmark_prediction`：保存每个案例的预测、TP/FP/TN/FN/FAILED、分类详情和法规依据。
- B.`benchmark_run`：保存任务状态、混淆矩阵和最终指标。

### 2. 导入两份 CSV

CSV 保留在仓库外部，不提交 Git。导入会自动尝试 UTF-8 BOM、UTF-8 和 GB18030；第一列
必须为 `字段名`，后续 `样本N` 列作为脱敏样例。两份文件先全部校验，再在同一事务写入 A；
重复执行同一批次只报告 `skipped`，不会删除或覆盖既有案例。

```powershell
python -m scripts.import_benchmark_data `
  --personal "D:\benchmark\个人信息_脱敏.csv" `
  --non-personal "D:\benchmark\非个人信息字段_脱敏.csv" `
  --batch teacher_2026_08
```

### 3. 运行评测

先用同时包含正负样本的小批量验证配置和费用，再决定是否全量运行：

```powershell
# 小批量：20 个个人信息案例 + 80 个非个人信息案例
python -m scripts.run_benchmark `
  --batch teacher_2026_08 `
  --personal-limit 20 `
  --non-personal-limit 80

# 全量新任务（会逐案例调用 LLM，可能耗时并产生费用）
python -m scripts.run_benchmark --batch teacher_2026_08

# 继续中断任务；默认只处理尚未写入 B 的案例
python -m scripts.run_benchmark --resume-run <run_id>

# 继续任务并额外重试失败案例；成功案例不会重复调用
python -m scripts.run_benchmark --resume-run <run_id> --retry-failed
```

Runner 每处理一个案例立即写入 B，单个分类失败会记录为 `FAILED` 并继续。终端输出
run ID、案例数、混淆矩阵、Precision、Recall、F1、Accuracy、Coverage 和 Effective
Recall，不输出脱敏样例。

### 4. 指标与查询

- Precision：预测为个人信息的案例中，有多少确实是个人信息。
- Recall：成功分类的真实个人信息中，有多少被识别。
- F1：Precision 与 Recall 的调和平均。
- Accuracy：成功案例中的总体正确率；类别不平衡时仅作辅助。
- Coverage：成功分类数 / 本次选取总数。
- Effective Recall：TP / 本次选取的全部真实个人信息数，会把执行失败造成的漏识别计入。

启动 FastAPI 后可查看汇总和错误案例：

```text
GET /api/benchmark/runs/{run_id}
GET /api/benchmark/results?run_id={run_id}&outcome=FN&limit=100
GET /api/benchmark/results?run_id={run_id}&outcome=FP&need_review=true
GET /api/benchmark/results?run_id={run_id}&outcome=FAILED
```

Benchmark 长任务只通过 CLI 启动；FastAPI 接口只读，避免让长时间付费调用占用 HTTP
请求。建议演示顺序为：展示 A 输入数量与标签分布 → 运行分层小批量 → 展示终端评分 →
通过 API/Swagger 展示 FN、FP、失败原因、分类理由和法规依据。

## 重建法规知识库

法规目录原生支持：

```text
data/knowledge/classification_rules.md
data/knowledge/laws/*.txt
data/knowledge/laws/*.pdf
```

PDF 会先逐页读取自身文字层。存在有效文字的页面直接使用；没有文字层的页面会在本地
渲染为 JPEG 并检测墨迹，真正的空白页直接跳过，只有非空白扫描页才以 Base64 发送给
`qwen3.5-ocr`。有效页面最后按页码合并，再进入与 TXT 相同的法规分块、Embedding 和
Chroma 流程。单个 PDF 必须未加密、包含 1–50 页且不超过 100 MB。调用 OCR 前必须确认
扫描页允许由阿里云百炼处理。

分页图片保存在 PDF 旁边的同名目录中，例如：

```text
data/knowledge/laws/个人信息安全规范.pdf
data/knowledge/laws/个人信息安全规范/page-0001.jpg
data/knowledge/laws/个人信息安全规范/page-0002.jpg
```

这些图片只会为需要 OCR 的非空白页面保留，是可重新生成的本地运行产物，已被 Git
忽略。每次处理会裁剪历史残留图片，只保留当前实际需要 OCR 的页，不需要 OSS 或公开
文件地址。

完整文档和单页 OCR 结果都按 PDF 内容、模型名称和 Prompt 版本缓存在
`.runtime/ocr_cache/`。文本缓存命中时不会重新解析或调用付费 OCR；中途某页失败时，
已经成功的 OCR 页面会保留，下次重建只继续处理尚未完成的页面。PDF 内容变化时相关缓存
和分页图片会自动失效。

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
GET /api/results?run_id=&database_name=&table_name=&column_name=&level=&category=&need_review=&is_personal=&limit=&offset=
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

未显式设置 `RUN_QWEN_OCR_INTEGRATION=1` 时不会发起千问请求。当前逐页同步 OCR 方案适合
受控、低频的知识库重建；生产高并发摄取仍需补充任务队列、限流和失败恢复。

## 当前阶段边界

当前实现面向受控 Demo：业务字段扫描仍是同步请求；Benchmark 使用可恢复的本地串行
Runner。系统尚未实现后台任务队列、并发限流、Web 前端或生产级凭据管理。全量 Benchmark
会产生真实 LLM 调用费用，应先完成小批量验证并单独确认预算。
