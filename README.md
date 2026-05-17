# 基于 RAG + LLM 的企业数据智能分类分级系统

> Layer1 基础版本 — 可行性验证阶段

利用 RAG（检索增强生成）技术，将《个人信息保护法》《数据安全法》等法规文本向量化存入本地知识库，对企业业务字段自动进行 L1~L4 敏感等级判定，提供全流程可溯源的法律依据。

---

## 一、整体框架设计

系统采用四层架构，自上而下为：展示层 → 逻辑层 → 数据层 → 基础设施层。

```
┌──────────────────────────────────────────────────────┐
│  展示层 (ui/app.py)                                    │
│  Gradio Web → 单条语义测试 / 批量自动化评测             │
│              / 错题本人工干预 / 知识库增量更新          │
├──────────────────────────────────────────────────────┤
│  逻辑层 (core/)                                        │
│  engine.py     → RAG 检索 + LLM 推理 + 级别解析        │
│  evaluator.py  → 批量评测 + 指标计算 + Excel 报告导出   │
│  schemas.py    → Pydantic 结构化数据模型               │
│  prompt.py     → CoT 思维链 Prompt 模板                │
│  config.py     → 全局路径 & 环境变量管理               │
├──────────────────────────────────────────────────────┤
│  数据层                                                │
│  Chroma 向量库 (db/)  ←  4 份法规原文 (data/)         │
│  testdata/            ←  按业务域分类的测试集 Excel    │
├──────────────────────────────────────────────────────┤
│  基础设施                                              │
│  sentence-transformer (MPNet)  →  文本向量化           │
│  DeepSeek API (ChatOpenAI)     →  推理生成             │
│  SQLAlchemy + PyMySQL          →  MySQL 元数据扫描     │
└──────────────────────────────────────────────────────┘
```

### 各层职责

| 层级 | 组件 | 职责 |
|------|------|------|
| 展示层 | `ui/app.py` | Gradio Blocks 构建的 Web UI，包含单条测试、批量评测、错题本上传、知识库更新四个功能模块 |
| 逻辑层 | `core/engine.py` | 核心 RAG 引擎：向量检索 → Prompt 组装 → LLM 推理 → JSON 解析，对外暴露 `smart_predict()` 和 `classify_field()` |
| 逻辑层 | `core/evaluator.py` | 批量评测流水线：文件加载 → 逐字段推理 → 结果回填 → 准确率/F1 计算 → 多 Sheet Excel 导出 |
| 逻辑层 | `core/schemas.py` | 三个 Pydantic v2 模型，定义系统内部数据契约 |
| 逻辑层 | `core/prompt.py` | CoT 思维链 Prompt 模板，引导 LLM 按 Step 1/2/3 逐步推理 |
| 逻辑层 | `core/config.py` | 环境变量加载与全局路径常量，所有路径支持 `os.getenv` 覆盖 |
| 数据层 | `db/` (Chroma) | 持久化向量数据库，存储法规条文的向量嵌入 |
| 数据层 | `data/` | 4 份原始法规 TXT 文件，约 8 万中文字符 |
| 数据层 | `testdata/` | 7 个业务域的测试 Excel，部分含标准答案列用于准确率评测 |
| 基础设施 | `knowledge/chunker.py` | 按「第X章/第X条」结构对法律文本分块 |
| 基础设施 | `connectors/file_loader.py` | Excel/CSV 字段画像加载器 |
| 基础设施 | `connectors/mysql_connectors.py` | MySQL information_schema 扫描器，将数据库元数据转为 FieldProfile |

---

## 二、核心 RAG 推理链路

一次完整的字段定级推理经过 6 个环节：

```
用户输入: 英文名=account_balance, 中文名=账户余额, 业务描述=客户账户资金余额

Step 1 ─ 构建搜索查询
  │ 将 FieldProfile 的字段名、中文名、注释、样例值拼接为语义查询文本
  │
Step 2 ─ 向量检索 (Top 3)
  │ 文本经 HuggingFaceEmbeddings (MPNet) 转为 768 维向量
  │ 在 Chroma 库中余弦相似度检索，取 Top 3 最相关法律条文
  │ → 返回 3 个 Evidence 对象（文档名 + 条文内容 + 章节层级 + chunk_id）
  │
Step 3 ─ 上下文字段组装
  │ 将 3 段法律依据 JSON + 字段画像 JSON 填入 CoT Prompt 模板
  │
Step 4 ─ LLM 思维链推理
  │ DeepSeek 按 Prompt 要求的格式输出 JSON
  │ temperature=0.1 保证推理严谨性
  │
Step 5 ─ JSON 解析
  │ 优先 json.loads，失败则正则匹配 {.*} 提取再解析
  │
Step 6 ─ 返回结构化结果
  │ ClassificationResult: level / category / confidence / reason / evidence / need_review
  └──────────────────────────────────────────────────────
```

---

## 三、向量数据库设计

### 3.1 选型

选用 **ChromaDB** 作为向量存储，原因：

- 轻量级嵌入式部署，无需额外服务进程
- 与 LangChain 生态原生集成 (`langchain-chroma`)
- 支持持久化到本地磁盘 (`chroma.sqlite3` + 分段二进制文件)
- Layer1 阶段数据量小（~100 条条文），无需 Milvus/Qdrant 等分布式方案

### 3.2 存储结构

```
db/
├── chroma.sqlite3          ← 元数据索引（collection 名、文档 ID、元数据字段）
└── {collection_uuid}/
    ├── header.bin          ← 段头信息（向量维度 768、元素数量）
    ├── length.bin          ← 每段长度
    ├── link_lists.bin      ← HNSW 图层链接
    └── data_level0.bin     ← 向量数据（768 维 float32 × N 条记录）
```

### 3.3 Embedding 模型

| 项目 | 配置 |
|------|------|
| 模型 | `sentence-transformers` (MPNet base, 768 维) |
| 加载方式 | `local_files_only=True`，离线本地加载 |
| 路径 | `models/sentence-transformer/`（通过 `EMBEDDING_MODEL_PATH` 环境变量配置） |
| 搜索 | 余弦相似度，`similarity_search(query, k=3)` |

### 3.4 向量库操作

```python
# 检索 — 每次推理调用
docs = vector_db.similarity_search(query, k=3)

# 增量写入 — 知识库更新时调用
vector_db.add_documents(chunks)
vector_db.persist()
```

---

## 四、元数据分块设计

### 4.1 分块策略

法律文本具有天然的结构化层级（章 → 条），因此不采用通用的 `RecursiveCharacterTextSplitter` 滑动窗口切分，而是实现 `split_by_article()` 按语义边界切分。

```
输入文本:
  第一章 总则
  第一条 为了规范...
  第二条 本法所称...

切分算法:
  1. 逐行扫描，遇到第X章 → 更新 current_chapter
  2. 遇到第X条 → flush 前一条，开始新 chunk
  3. 文件末尾 flush 最后一条

输出: List[langchain_core.documents.Document]
```

### 4.2 Chunk 元数据结构

每个 Chunk 携带以下元数据字段：

```json
{
  "chunk_id": "uuid4 唯一标识",
  "document_name": "源法规文件名，如 个人信息保护法.txt",
  "source_type": "法规来源类型 (law | enterprise_rule | national_standard)",
  "domain": "法规所属域 (general | finance | medical ...)",
  "hierarchy_level": "层级路径，如 第一章 总则-第一条",
  "chapter": "当前章节名",
  "article": "当前条款名（截取前20字）",
  "version": "法规版本标识 (2026)",
  "sensitivity_level": "条款内提到的最高敏感等级 (L1-L4，可选)"
}
```

### 4.3 设计要点

- **层级溯源**: `hierarchy_level` 字段使检索结果可追溯到具体章节条款，审计时可精确定位法规依据
- **版本管理**: `version` 字段预留法规更新的版本追踪能力
- **敏感等级检测**: 分块时自动扫描内容中出现的 `L1`/`L2`/`L3`/`L4` 关键词，写入 `sensitivity_level` 元数据，为后续检索过滤提供依据
- **UUID chunk_id**: 每个 chunk 有全局唯一标识，支持未来跨系统引用和缓存

---

## 五、结构化数据模型设计 (Pydantic v2)

### 5.1 FieldProfile — 输入字段画像

定义待定级数据字段的完整描述，是整个系统的统一输入契约。

```python
class FieldProfile(BaseModel):
    database_name: Optional[str] = None       # 来源数据库名
    table_name: Optional[str] = None           # 来源表名
    field_name: str                            # 字段英文名（必填）
    field_comment: Optional[str] = None        # 原始注释
    field_cn: Optional[str] = None             # 字段中文名
    data_type: Optional[str] = None            # 数据类型 (varchar / int / ...)
    sample_values: List[str] = []              # 样例值（最多5条，每条截断50字符）
    business_domain: str = "general"           # 业务域 (general / finance / hr / ...)
    source_system: Optional[str] = None        # 来源系统 (mysql / excel / hive ...)
```

**JSON 序列化示例:**

```json
{
  "database_name": "finance_db",
  "table_name": "t_salary",
  "field_name": "salary_amount",
  "field_comment": "员工税前月薪",
  "field_cn": "工资数额",
  "data_type": "decimal(12,2)",
  "sample_values": ["15000.00", "22000.00", "8500.00"],
  "business_domain": "hr",
  "source_system": "mysql"
}
```

### 5.2 Evidence — 检索法律依据

向量检索返回的法律条文片段，每个 Evidence 对应一条最相关的法规条款。

```python
class Evidence(BaseModel):
    document_name: str                         # 来源文档名
    source_type: str = "unknown"               # law | enterprise_rule | national_standard
    domain: str = "general"                    # 法规域
    hierarchy_level: Optional[str] = None      # 章节-条款层级路径
    content: str                               # 条文原文内容
    score: Optional[float] = None              # 相似度分数（当前 Chroma 未设置）
    chunk_id: Optional[str] = None             # 对应 Chunk 的 uuid4
```

**JSON 序列化示例:**

```json
{
  "document_name": "个人信息保护法.txt",
  "source_type": "law",
  "domain": "personal_data",
  "hierarchy_level": "第二章 个人信息处理规则-第二十八条",
  "content": "敏感个人信息是一旦泄露或者非法使用，容易导致自然人的人格尊严受到侵害或者人身、财产安全受到危害的个人信息，包括……金融账户……行踪轨迹等信息……",
  "score": null,
  "chunk_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### 5.3 ClassificationResult — 输出分类结果

系统最终的输出契约，包含定级结论和完整的推理溯源链。

```python
class ClassificationResult(BaseModel):
    field_name: str                            # 字段名（与输入对应）
    category: str                              # 数据分类（个人信息 / 财务数据 / 员工数据 ...）
    subcategory: Optional[str] = None          # 细分类（个人联系方式 / 薪酬信息 ...）
    level: Literal["L1", "L2", "L3", "L4", "未知"]  # 敏感等级
    confidence: float = 0.0                    # 置信度 [0.0, 1.0]
    reason: str                                # 大模型推理理由
    evidence: List[Evidence]                   # 检索到的法律依据列表
    need_review: bool                          # 是否需要人工复核
    decision_path: str = "rag_llm"             # 决策路径 (rag_llm | rag_llm_error)
    raw_response: Optional[str] = None         # LLM 原始输出（调试用）
```

**JSON 序列化示例:**

```json
{
  "field_name": "salary_amount",
  "category": "个人信息",
  "subcategory": "薪酬信息",
  "level": "L3",
  "confidence": 0.72,
  "reason": "字段为员工工资数额，属于敏感个人信息中的金融账户相关信息。依据《个人信息保护法》第28条，薪酬信息属于一旦泄露可能导致个人权益受损的敏感数据。",
  "evidence": [
    {
      "document_name": "个人信息保护法.txt",
      "source_type": "law",
      "domain": "personal_data",
      "hierarchy_level": "第二章-第二十八条",
      "content": "敏感个人信息是一旦泄露或者非法使用...",
      "score": null,
      "chunk_id": "a1b2c3d4-..."
    }
  ],
  "need_review": false,
  "decision_path": "rag_llm",
  "raw_response": "{\"category\": \"个人信息\", ...}"
}
```

### 5.4 数据流中的格式转换

```
Excel/CSV 行 → FieldProfile (Pydantic)
                    ↓
              build_search_query() → 文本字符串
                    ↓
              vector_db.similarity_search() → List[Document]
                    ↓
              Evidence.from_doc_metadata() → List[Evidence]
                    ↓
              FieldProfile.model_dump_json() + Evidence.model_dump() → Prompt JSON
                    ↓
              LLM.invoke() → 原始字符串
                    ↓
              parse_llm_json_response() → dict → ClassificationResult (Pydantic)
                    ↓
              result.model_dump_json() → 批量评测 Excel 的「结构化JSON结果」列
```

---

## 六、Prompt 工程 — CoT 思维链设计

### 6.1 设计原则

不走「直接问级别」的捷径，而是要求模型按三步推理：

1. **语义分析**: 字段在业务中代表什么实体？
2. **合规比对**: 逐一对照检索条文，每条如何适用？
3. **输出结论**: 基于前两步给出最终定级

### 6.2 Prompt 模板核心结构

```
系统角色设定: 企业数据分类分级专家

输入:
  - 字段画像 JSON (FieldProfile 序列化)
  - 检索依据 JSON (Evidence[] 序列化)

输出约束:
  - 仅输出合法 JSON，不输出 Markdown / 额外解释
  - 依据不足时 confidence ≤ 0.75 且 need_review = true
  - 禁止编造不存在的依据
```

### 6.3 等级定义

| 等级 | 含义 | 判定标准 |
|------|------|----------|
| L1 | 公开数据 | 泄露后基本无影响 |
| L2 | 内部数据 | 仅限企业内部使用 |
| L3 | 敏感数据 | 泄露后可能影响个人权益、企业经营或合规安全 |
| L4 | 核心敏感数据 | 泄露后可能造成严重法律、财产、身份安全或重大经营风险 |

---

## 七、错题本机制 — 规则 + AI 混合架构

### 7.1 设计理念

纯 AI 推理有不确定性，纯规则覆盖不全。采用 **规则优先 + AI 兜底** 的混合架构。

### 7.2 工作流程

```
1. 上传错题本 Excel（name 列 = 字段英文名，标准答案列 = L1-L4）
2. load_error_log() 解析为 {"field_en": "L3", ...} 写入 ERROR_LOG_CACHE
3. 推理时: if field_name in ERROR_LOG_CACHE → 直接返回预设级别（0 成本，100% 准确）
          else → 走完整 RAG + LLM 推理链路
4. 评测后发现新错题 → 导出 → 人工复核 → 下次上传生效（闭环）
```

### 7.3 错题本文件格式

| name | 标准答案 |
|------|----------|
| salary_amount | L3 |
| account_code | L2 |
| id_card_number | L4 |

---

## 八、批量评测与报告导出

### 8.1 测试集格式

Excel/CSV 文件包含以下列（标准答案为可选项）：

| 英文名 | 中文名 | 业务描述 | 字段类型 | 业务域 | 标准答案 |
|--------|--------|----------|----------|--------|----------|
| salary | 工资 | 员工薪资 | decimal | hr | L3 |

### 8.2 评测流水线

```
读取 Excel → 逐行转 FieldProfile → 逐字段 classify_field()
→ 回填 11 列结果到 DataFrame → 与标准答案比对 → 导出多 Sheet Excel
```

### 8.3 导出报告结构

| Sheet | 内容 |
|-------|------|
| 全量评测记录 | 每条字段的级别 / 分类 / 置信度 / 推理理由 / 结构化 JSON / 检索依据 |
| 需复核或错题清单 | 大模型结论 ≠ 标准答案 或 need_review=True 的条目 |
| 量化评估指标 | 整体准确率 + 每级精确率/召回率/F1（仅当有标准答案时） |

---

## 九、知识库增量更新

```
初始构建:
  4 份法规 TXT → split_by_article() → HuggingFaceEmbeddings → Chroma.persist()

运行时增量:
  上传新 TXT → split_by_article() → vector_db.add_documents() → vector_db.persist()
```

优势：无需重建整个向量库，上传即生效。

---

## 十、项目目录结构

```
Data_Compliance_RAG/
├── main.py                   ← 应用入口，启动 Gradio 服务
├── requirements.txt           ← Python 依赖
├── api_key.env                ← API Key / 数据库配置（已 gitignore）
│
├── core/                      ← 核心逻辑层
│   ├── config.py              │ 全局路径 & 环境变量
│   ├── schemas.py             │ Pydantic 数据模型（FieldProfile / Evidence / ClassificationResult）
│   ├── prompt.py              │ CoT 思维链 Prompt 模板
│   ├── engine.py              │ RAG 引擎（检索 + LLM + JSON 解析 + 知识库更新）
│   └── evaluator.py           │ 批量评测（指标计算 + Excel 报告导出）
│
├── knowledge/
│   └── chunker.py             ← 法律文本按「章/条」结构化分块
│
├── connectors/
│   ├── file_loader.py         ← Excel/CSV 字段画像加载器
│   └── mysql_connectors.py    ← MySQL information_schema 扫描器
│
├── ui/
│   └── app.py                 ← Gradio Web UI（单条测试 + 批量评测 + 错题本）
│
├── data/                      ← 法规原文语料（4 份 TXT，约 8 万字）
├── testdata/                  ← 测试数据集（7 个业务域 Excel）
├── db/                        ← Chroma 向量库持久化目录
├── models/                    ← sentence-transformer 本地模型文件
├── outputs/                   ← 评测报告输出目录
│
└── Construct.md               ← 答辩讲解提纲 / 项目设计文档
```

---

## 十一、快速开始

### 环境要求

- Python ≥ 3.10
- 本地已下载 sentence-transformer 模型至 `models/sentence-transformer/`

### 安装

```bash
pip install -r requirements.txt
```

### 配置

编辑 `api_key.env`，填入你的配置：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL_PATH=./models/sentence-transformer
CHROMA_DB_PATH=./db
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database
```

### 构建知识库

将法规 TXT 文件放入 `data/` 目录，首次启动时系统会自动初始化向量库。

### 启动

```bash
python main.py
```

访问 `http://localhost:7860` 打开 Web 界面。

### 使用流程

1. **单条测试** — 输入字段英文名、中文名、业务描述，查看大模型定级结果
2. **批量评测** — 上传含标准答案的 Excel 测试集，获取准确率报告和错题清单
3. **错题本** — 上传错题 Excel 激活强制拦截规则，实现规则优先覆盖
4. **知识库更新** — 上传新法规 TXT，增量扩充向量知识库

---

## 十二、技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 向量数据库 | ChromaDB ≥ 1.5 | 轻量级嵌入式向量存储 |
| Embedding | sentence-transformers ≥ 5.4 (MPNet) | 768 维中文语义向量 |
| LLM | DeepSeek API (ChatOpenAI) | temperature=0.1 保证推理严谨 |
| LLM 框架 | LangChain ≥ 1.2 | 统一检索与模型调用接口 |
| 数据模型 | Pydantic ≥ 2.0 | 类型安全的输入输出契约 |
| Web UI | Gradio ≥ 6.0 | 快速构建交互式 AI 应用 |
| 数据处理 | pandas + openpyxl | Excel 读写与评测指标计算 |
| 指标计算 | scikit-learn | 准确率 / 精确率 / 召回率 / F1 |
| 数据库连接 | SQLAlchemy + PyMySQL | MySQL 元数据扫描 |
| 配置管理 | python-dotenv | api_key.env 环境变量加载 |
