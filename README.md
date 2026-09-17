# 企业数据智能分类分级服务

## 项目介绍

基于 RAG 与 LLM 的企业字段分类分级服务，支持单字段 API、MySQL 字段扫描和 CSV 输入，输出个人信息判定、数据类别、L1–L4 等级、分类理由及法规依据。

项目采用唯一主线：精简 Query 检索冻结法规知识库，再由结构化 LLM 完成分类。A 库保存业务数据，Chroma 保存法规向量，B 库保存分类结果及评分。

## 实现环境

| 组件 | 实现 |
|---|---|
| 运行环境 | Python 3.10、venv；以下命令使用 Windows PowerShell |
| API 与数据模型 | FastAPI、Pydantic |
| RAG 与向量存储 | LangChain、Chroma |
| Embedding | 本地 Sentence Transformers 模型 |
| 分类模型 | DeepSeek，兼容 OpenAI 接口 |
| 关系数据库 | MySQL 8.x、SQLAlchemy、PyMySQL |

## 配置内容

首次部署将 `.env.example` 复制为 `.env`，填写 API Key、本地模型路径和数据库连接：

```dotenv
DEEPSEEK_API_KEY=你的APIKey
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL_PATH=G:/AI_Models/sentence-transformer
CHROMA_DB_DIR=.runtime/chroma
CHROMA_COLLECTION=data_classification__b-rebuild-20260917
KNOWLEDGE_BASE_VERSION=b-rebuild-20260917
SOURCE_DATABASE_URL=mysql+pymysql://root:password@127.0.0.1:3306/enterprise_source
TARGET_DATABASE_URL=mysql+pymysql://root:password@127.0.0.1:3306/compliance_result
```

`SOURCE_DATABASE_URL` 仅用于 MySQL 扫描；CSV 分类及结果持久化需要 `TARGET_DATABASE_URL`。单字段 API 不需要 MySQL。冻结快照恢复不需要 OCR；重新提取 PDF 时需配置 `.env.example` 中的 `QWEN_OCR_*` 参数。`.env` 和本地向量库不提交 Git。

## 运行方式

### 1. 安装依赖

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 初始化数据库

新建演示数据库时，通过 MySQL 客户端或 Workbench 依次执行：

```powershell
cmd /c "mysql -u root -p < sql\source_schema.sql"
cmd /c "mysql -u root -p < sql\source_seed.sql"
cmd /c "mysql -u root -p < sql\target_schema.sql"
cmd /c "mysql -u root -p < sql\benchmark_target_schema.sql"
```

仅使用 CSV 时执行后两条；`source_seed.sql` 提供虚构业务样例。已有旧库的升级脚本位于 `sql/migrations/`。

### 3. 恢复冻结知识库

首次部署到空库时执行；当前工作区已恢复，可直接运行验证命令：

```powershell
python -m scripts.rebuild_frozen_knowledge --snapshot data/knowledge_snapshots/b-rebuild-20260917
python -m scripts.verify_frozen_knowledge
```

冻结库包含分类规则和四份法规，共 260 块。恢复拒绝写入非空库，具体记录见 [RAG_FREEZE.md](docs/RAG_FREEZE.md)。

### 4. 启动与分类

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 [Swagger](http://127.0.0.1:8000/docs)，可直接提交请求：

| 接口 | 用途 |
|---|---|
| `GET /api/health` | 健康检查 |
| `POST /api/classify` | 单字段分类，输入字段名与样例值等画像 |
| `POST /api/pipeline/run` | 扫描 A 库字段并将分类结果写入 B 库 |
| `GET /api/results` | 查询数据库扫描的分类结果 |
| `GET /api/benchmark/results` | 查询 CSV 分类及评分结果 |

CSV 可通过 CLI 直接分类，无需写入 A 库：

```powershell
python -m scripts.run_csv_pipeline --input "D:\data\business.csv" --input-mode auto
```

支持普通业务表格和字段目录两种格式；目录型 CSV 内含标签时，可增加 `--label-column expected_personal` 进行评分。

## 主要流程

```text
单字段 API / MySQL / CSV
→ 统一字段画像 FieldProfile
→ 精简 Query：field_name + sample_values
→ 本地 Embedding → 冻结法规知识库 Top-3
→ 完整字段画像 + 法规依据 → 结构化 LLM
→ 分类结果 → API 返回或写入 B 库
```

检索 Query 只使用字段名与样例值，LLM 接收完整字段画像和检索依据。标签仅用于评分，不进入 Query 或 Prompt；分类失败返回 UNKNOWN 并标记人工复核。

## 主要表结构

A 库 `enterprise_source` 的演示业务表为 `employee`、`customer_account`、`customer_order`、`product`，分别以 `employee_id`、`customer_id`、`order_id`、`product_id` 为主键。

B 库 `compliance_result` 的主要表如下：

| 表 | 主键与关联 | 主要字段 |
|---|---|---|
| `classification_run` | `run_id` | 状态、字段计数、模型、知识库版本、起止时间 |
| `data_field_asset` | `field_id`；源系统/库/表/列组合唯一 | 字段名、注释、类型、业务域 |
| `field_classification_result` | `result_id`；关联 `run_id`、`field_id` | `is_personal`、类别、等级、置信度、理由、复核标记 |
| `classification_evidence` | `evidence_id`；关联 `result_id` | 排序、文档名、条款、正文、检索分数、chunk ID |
| `benchmark_run` | `run_id` | CSV 输入指纹、任务状态、案例计数、模型版本、评分指标 |
| `benchmark_prediction` | `prediction_id`；关联 `run_id`，任务/案例组合唯一 | 字段名、样例值、真实标签、预测、分类详情、法规依据、执行状态 |

完整字段、类型和约束见 [A 库建表](sql/source_schema.sql)、[分类结果建表](sql/target_schema.sql) 和 [CSV 评分建表](sql/benchmark_target_schema.sql)。
