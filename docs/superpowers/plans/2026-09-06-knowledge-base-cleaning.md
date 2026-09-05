# Knowledge Base Cleaning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user requires Inline Execution; do not dispatch subagents.

**Goal:** 建立可审计的法规提取、清洗、质量门禁、标准条款切分和版本化 Chroma 重建流程，修复当前四份 PDF 的纯净度、完整性与超大 Chunk 问题。

**Architecture:** 原始知识源保持只读，提取结果先转换为带页级来源信息的 Pydantic 数据，再经过通用清洗器和质量门禁，只有 `PASS` 文档才能进入结构化切分及候选 Collection。重建采用新 Collection，不在验证前删除或覆盖旧库。

**Tech Stack:** Python 3.11、Pydantic、Pillow、pypdf/pypdfium2、LangChain Document、Chroma、pytest、Ruff。

---

## 文件职责

- 新建 `app/schemas/knowledge_quality.py`：页级文本、清洗审计、文档清单和质量报告模型。
- 新建 `app/services/knowledge_cleaner.py`：只负责确定性文本清洗和删除审计。
- 新建 `app/services/knowledge_quality_service.py`：计算指标、应用门禁并输出报告。
- 修改 `app/services/pdf_image_service.py`：为异常长图提供通用的低墨迹带分段能力。
- 修改 `app/services/ocr_service.py`：输出页级提取结果，分段 OCR，并升级缓存版本。
- 修改 `app/rag/chunker.py`：支持国家标准层级、页码元数据及超长条款二次切分。
- 修改 `app/services/knowledge_service.py`：串联提取、清洗、质量门禁和切分。
- 修改 `app/repositories/vector_store.py`：支持显式候选 Collection，避免重建时覆盖现用库。
- 修改 `scripts/rebuild_knowledge_base.py`：增加检查、候选重建、报告输出和人工批准参数。
- 新增对应单元与集成测试；测试夹具只使用短小合成文本和合成图像，不提交法规副本。

### Task 1：定义质量模型和稳定标识

**Files:**
- Create: `app/schemas/knowledge_quality.py`
- Create: `tests/unit/test_knowledge_quality_schemas.py`

- [ ] **Step 1: 写失败测试**

覆盖：页码必须大于 0、状态枚举只允许 `PASS/REVIEW/FAIL`、报告能序列化为 JSON、`SourceManifest` 包含源哈希和提取器版本。

```python
def test_quality_report_is_json_serializable():
    report = KnowledgeQualityReport(
        document_name="standard.pdf",
        source_sha256="a" * 64,
        status=QualityStatus.PASS,
        pages=[PageQuality(page_number=1, extraction_method="native", character_count=20)],
    )
    assert '"status":"PASS"' in report.model_dump_json()
```

- [ ] **Step 2: 运行并确认测试因模型不存在而失败**

Run: `pytest tests/unit/test_knowledge_quality_schemas.py -q`

- [ ] **Step 3: 用 Pydantic 实现最小模型**

模型必须显式包含 `ExtractedPage`、`SourceManifest`、`PageQuality`、`CleaningAudit`、`KnowledgeQualityReport` 和 `QualityStatus`，禁止使用无约束的顶层 `dict` 作为接口。

- [ ] **Step 4: 运行测试和静态检查**

Run: `pytest tests/unit/test_knowledge_quality_schemas.py -q`

Run: `ruff check app/schemas/knowledge_quality.py tests/unit/test_knowledge_quality_schemas.py`

- [ ] **Step 5: 提交**

```powershell
git add app/schemas/knowledge_quality.py tests/unit/test_knowledge_quality_schemas.py
git commit -m "feat: add knowledge quality models"
```

### Task 2：实现可审计的保守清洗器

**Files:**
- Create: `app/services/knowledge_cleaner.py`
- Create: `tests/unit/test_knowledge_cleaner.py`

- [ ] **Step 1: 写失败测试**

测试以下行为：删除跨页重复页眉页脚、独立页码、`【第 N 页】` 和装饰线；合并多余空白；保留条款号、“应/不得/宜”、附录标题、表头与正文中的普通数字；审计记录规则和删除次数。

```python
def test_cleaner_removes_page_noise_but_preserves_normative_text():
    pages = [
        ExtractedPage(page_number=1, text="GB/T 00000—2026\n1\n1 范围\n处理者应保护数据"),
        ExtractedPage(page_number=2, text="GB/T 00000—2026\n2\n2 要求\n处理者不得泄露数据"),
    ]
    result = KnowledgeCleaner().clean(pages)
    assert "GB/T 00000—2026" not in result.text
    assert "处理者应保护数据" in result.text
    assert "处理者不得泄露数据" in result.text
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/unit/test_knowledge_cleaner.py -q`

- [ ] **Step 3: 实现确定性规则**

页眉页脚必须同时满足“跨页高频 + 页首/页尾位置”；页码必须整行匹配；重复行删除需跳过疑似表头。返回清洗文本、保留的页边界和 `CleaningAudit`，不得改写正文词句。

- [ ] **Step 4: 运行测试与 Ruff**

Run: `pytest tests/unit/test_knowledge_cleaner.py -q`

Run: `ruff check app/services/knowledge_cleaner.py tests/unit/test_knowledge_cleaner.py`

- [ ] **Step 5: 提交**

```powershell
git add app/services/knowledge_cleaner.py tests/unit/test_knowledge_cleaner.py
git commit -m "feat: add auditable knowledge cleaning"
```

### Task 3：为异常长图增加通用分段 OCR

**Files:**
- Modify: `app/services/pdf_image_service.py`
- Modify: `app/services/ocr_service.py`
- Modify: `tests/unit/test_pdf_image_service.py`
- Modify: `tests/unit/test_ocr_service.py`

- [ ] **Step 1: 写长图分段失败测试**

使用 Pillow 生成包含三段内容、两条宽水平空白带的长图，验证分为三个有序片段；普通比例图片保持一个片段；没有可靠空白带时不盲目等距裁切。

- [ ] **Step 2: 运行相关测试并确认失败**

Run: `pytest tests/unit/test_pdf_image_service.py tests/unit/test_ocr_service.py -q`

- [ ] **Step 3: 实现图像结构分段**

在 `PdfImageService` 中增加基于高宽比和水平投影的 `segment_page_image()`。输出文件名包含容器页和片段序号；裁切范围保留少量上下边缘，顺序固定。

- [ ] **Step 4: 将 OCR 改为页级、片段级结果**

`QwenOCRService` 保留 `extract_pdf() -> str` 兼容接口，同时新增返回 `list[ExtractedPage]` 的页级接口。OCR 缓存摘要加入新的提示版本和分段版本；提示增加“表格按阅读顺序逐行转录，禁止解释或补充”。片段失败必须抛出带容器页和片段号的错误。

- [ ] **Step 5: 运行 OCR/PDF 全部单元测试**

Run: `pytest tests/unit/test_pdf_image_service.py tests/unit/test_ocr_service.py tests/unit/test_knowledge_pdf_loading.py -q`

Run: `ruff check app/services/pdf_image_service.py app/services/ocr_service.py tests/unit/test_pdf_image_service.py tests/unit/test_ocr_service.py`

- [ ] **Step 6: 提交**

```powershell
git add app/services/pdf_image_service.py app/services/ocr_service.py tests/unit/test_pdf_image_service.py tests/unit/test_ocr_service.py
git commit -m "feat: segment abnormal pdf pages for ocr"
```

### Task 4：实现质量指标与入库门禁

**Files:**
- Create: `app/services/knowledge_quality_service.py`
- Create: `tests/unit/test_knowledge_quality_service.py`

- [ ] **Step 1: 写失败测试**

分别验证：完整普通文档为 `PASS`；无状态缺页、非空白空文本为 `FAIL`；重复行占比超限或单页长度超过中位数 8 倍为 `REVIEW`；缺少正文起始章节为 `FAIL`；报告包含具体原因和异常页。

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/unit/test_knowledge_quality_service.py -q`

- [ ] **Step 3: 实现纯规则质量服务**

质量服务只计算结构和统计异常，不调用 LLM，不根据文件名改变规则。人工复核状态通过显式审批清单传入，未经审批的 `REVIEW` 保持不可入库。

- [ ] **Step 4: 运行测试与 Ruff**

Run: `pytest tests/unit/test_knowledge_quality_service.py -q`

Run: `ruff check app/services/knowledge_quality_service.py tests/unit/test_knowledge_quality_service.py`

- [ ] **Step 5: 提交**

```powershell
git add app/services/knowledge_quality_service.py tests/unit/test_knowledge_quality_service.py
git commit -m "feat: enforce knowledge quality gates"
```

### Task 5：扩展国家标准结构化切分

**Files:**
- Modify: `app/rag/chunker.py`
- Modify: `tests/unit/test_rag.py`

- [ ] **Step 1: 写失败测试**

覆盖 `1 范围`、`3.1`、`5.2.3`、`附录 A`、`A.1`；验证表、注、示例仍属于当前条款；构造超过 2000 字符的条款，验证按段落形成 800–1600 字符目标块、最大不超过 2000、保留约 150 字符重叠和页码范围。

- [ ] **Step 2: 运行并确认新增测试失败**

Run: `pytest tests/unit/test_rag.py -q`

- [ ] **Step 3: 扩展解析和稳定元数据**

优先级为附录条款、多级数字条款、一级章节、中文章条、Markdown 规则。`chunk_id` 改为由源哈希、层级路径和子块序号生成的稳定 UUID；保留旧调用参数的默认兼容行为。

- [ ] **Step 4: 运行切分与加载回归测试**

Run: `pytest tests/unit/test_rag.py tests/unit/test_knowledge_pdf_loading.py -q`

Run: `ruff check app/rag/chunker.py tests/unit/test_rag.py`

- [ ] **Step 5: 提交**

```powershell
git add app/rag/chunker.py tests/unit/test_rag.py
git commit -m "feat: split standards by numbered clauses"
```

### Task 6：串联清洗、报告与版本化候选 Collection

**Files:**
- Modify: `app/services/knowledge_service.py`
- Modify: `app/repositories/vector_store.py`
- Modify: `scripts/rebuild_knowledge_base.py`
- Modify: `tests/unit/test_knowledge_pdf_loading.py`
- Modify: `tests/unit/test_rebuild_script.py`
- Modify: `tests/unit/test_services.py`

- [ ] **Step 1: 写失败测试**

验证完整链路顺序为提取→清洗→门禁→切分；任一 `FAIL` 或未批准的 `REVIEW` 时不调用向量库写入；候选 Collection 名为 `data_classification__<version>`；重建失败不删除旧 Collection；报告写入指定运行目录。

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/unit/test_knowledge_pdf_loading.py tests/unit/test_rebuild_script.py tests/unit/test_services.py -q`

- [ ] **Step 3: 改造 Loader 和 KnowledgeService**

Loader 为每个源生成 manifest，PDF 使用页级结果，TXT 作为单页结果；清洗和质量报告通过后再切分。`KnowledgeService` 接收候选向量库，不再先对当前库执行 `reset()`。

- [ ] **Step 4: 支持显式 Collection 名称**

`VectorStore` 增加可选 `collection_name` 参数，默认仍使用现有配置，保证 API、Benchmark、CSV 和数据库 Pipeline 行为不变。

- [ ] **Step 5: 扩展重建命令**

支持：

```text
--check-only
--candidate-collection <name>
--report-dir <path>
--approve-review <document-sha256>
```

默认候选名由当前 Collection 和知识库版本生成；`--check-only` 不初始化 Embedding 或 Chroma；报告和清洗文本只写入 `.runtime`。

- [ ] **Step 6: 运行相关测试与 Ruff**

Run: `pytest tests/unit/test_knowledge_pdf_loading.py tests/unit/test_rebuild_script.py tests/unit/test_services.py -q`

Run: `ruff check app/services/knowledge_service.py app/repositories/vector_store.py scripts/rebuild_knowledge_base.py`

- [ ] **Step 7: 提交**

```powershell
git add app/services/knowledge_service.py app/repositories/vector_store.py scripts/rebuild_knowledge_base.py tests/unit/test_knowledge_pdf_loading.py tests/unit/test_rebuild_script.py tests/unit/test_services.py
git commit -m "feat: rebuild validated candidate knowledge collection"
```

### Task 7：增加真实文档检查命令与导入报告

**Files:**
- Modify: `scripts/rebuild_knowledge_base.py`
- Create: `tests/integration/test_knowledge_quality_cli.py`
- Modify: `.gitignore`

- [ ] **Step 1: 写 CLI 失败测试**

验证 `--check-only` 输出每份文档的页数、状态和失败原因；进程在存在 `FAIL` 时返回非零；运行产物位于 `.runtime/knowledge_quality/<version>/`；`.gitignore` 忽略清洗文本、报告和候选 Chroma 数据。

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/integration/test_knowledge_quality_cli.py -q`

- [ ] **Step 3: 完成 CLI 摘要和导入报告**

终端只打印简洁汇总；JSON 报告包含来源数、通过/复核/失败数、Chunk 数、长度最小值/中位数/P95/最大值以及异常清单。任何 Chunk 超过 2000 字符时重建失败。

- [ ] **Step 4: 运行 CLI 和回归测试**

Run: `pytest tests/integration/test_knowledge_quality_cli.py tests/integration/test_benchmark_cli.py tests/integration/test_csv_pipeline_cli.py -q`

Run: `ruff check scripts/rebuild_knowledge_base.py tests/integration/test_knowledge_quality_cli.py`

- [ ] **Step 5: 提交**

```powershell
git add scripts/rebuild_knowledge_base.py tests/integration/test_knowledge_quality_cli.py .gitignore
git commit -m "feat: report knowledge rebuild quality"
```

### Task 8：对四份现有 PDF 执行检查和人工验收

**Files:**
- Runtime only: `.runtime/knowledge_quality/<version>/`
- Runtime only: `.runtime/ocr_cache/`
- Runtime only: `.runtime/chroma/`

- [ ] **Step 1: 运行只读检查**

Run: `python -m scripts.rebuild_knowledge_base --check-only`

Expected: 四份 PDF 均产生 manifest、页级报告、清洗文本和审计日志；存在 `FAIL/REVIEW` 时不写 Chroma。

- [ ] **Step 2: 按文档完成专项核对**

- `GB/T 35273`：确认正文超过旧 TXT 截断位置，并包含后续章节、附录和参考文献。
- TC260：确认 40 个容器页均有状态，目录和页码已从正文排除。
- `GB/T 45574`：逐一查看第 2、4、6、18、19 页，记录是空白页还是提取遗漏。
- `GB/T 41391`：核对首页、普通正文页、旋转表格页、密集表格页和末页；如仍有无关长段文本，保持 `FAIL` 并调整通用分段/OCR，不得直接批准。

- [ ] **Step 3: 重新运行检查并记录批准哈希**

仅对视觉核对通过但仍为 `REVIEW` 的源哈希使用 `--approve-review`。审批记录写入本次运行报告，不修改源文件。

- [ ] **Step 4: 创建候选 Collection**

Run: `python -m scripts.rebuild_knowledge_base --candidate-collection data_classification__<version> --approve-review <sha256>`

Expected: 全部来源通过门禁；候选库中四份法规均形成多个 Chunk；最大 Chunk 不超过 2000 字符；旧 `data_classification` 未改变。

### Task 9：固定查询回归和最终验证

**Files:**
- Create: `tests/integration/test_knowledge_retrieval_regression.py`
- Runtime only: `.runtime/knowledge_quality/<version>/retrieval-comparison.json`

- [ ] **Step 1: 建立不含 Benchmark 答案的固定查询集**

至少覆盖个人信息定义、敏感个人信息、网络数据分类分级、App 最小必要收集、个人信息主体权利、表格条目定位。断言候选库 top-3 中出现正确来源，禁止断言为某个 Benchmark 字段定制的固定 Chunk 文本。

- [ ] **Step 2: 运行新旧 Collection 对比**

使用同一 Embedding 模型和相同 `k=3`，把来源、条款、分数写入 `retrieval-comparison.json`。任何核心查询来源退化时不切换配置。

- [ ] **Step 3: 运行完整验证**

Run: `pytest -q`

Run: `ruff check .`

Run: `python -m compileall app scripts tests`

Expected: 现有测试与新增测试全部通过；Ruff 无错误；compileall 成功。

- [ ] **Step 4: 提交回归测试**

```powershell
git add tests/integration/test_knowledge_retrieval_regression.py
git commit -m "test: add knowledge retrieval regression coverage"
```

- [ ] **Step 5: 人工审批后切换**

向用户提交四份文档质量摘要、Chunk 分布和新旧检索对比。只有用户再次批准，才把运行环境的 `CHROMA_COLLECTION` 指向候选 Collection；本计划不自动修改或删除旧 Collection。

## 实施检查点

- 完成 Task 1–5：提交一次代码与单元测试结果，确认模型、清洗、OCR 和切分边界。
- 完成 Task 6–7：提交一次干运行报告，确认不会覆盖旧库。
- 完成 Task 8：提交四份真实文档的人工抽样结果，未通过则停止。
- 完成 Task 9：提交候选库检索对比，等待最终切换批准。
