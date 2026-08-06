# Benchmark 代码收束设计

日期：2026-08-07  
目标分支：`feature/csv-input-adapters`  
文档状态：待审核，审核通过后再编写实施计划并修改代码

## 1. 背景与目标

项目已经具备通用 CSV 输入能力，包括流式读取、编码回退、CSV 类型识别、字段目录型与普通业务数据型适配、可选标签文件，以及直接进入分类和结果持久化的 CSV Pipeline。

现有 Benchmark 实现早于这套通用能力，内部仍包含独立的 CSV 解析、A 库测试表导入、字段构造和分类流水线。两套实现职责重叠，继续并存会导致相同规则需要维护两次，并使 Benchmark 结果与正式 CSV 流程不能证明使用了同一条输入链路。

本次收束的目标是：

1. Benchmark 复用通用 CSV 读取、字段目录适配和分类流水线；
2. 删除仅用于中转、且已经被通用实现覆盖的专用代码；
3. 保留 Benchmark 独有的标签、指标、运行记录和结果查询能力；
4. 保证新旧版本在输入构造、标签语义和指标计算上具有明确的等价契约；
5. 不影响普通业务 CSV、MySQL A→RAG→B 和历史 Benchmark 结果查询。

## 2. 本次边界

本次只收束 Benchmark 到现有通用 CSV 架构，不修改 RAG 检索、Prompt、LLM 分类规则、知识库、Embedding 或数据库 A 的普通业务扫描流程。

本次不要求 Codex 使用老师的完整数据调用真实模型，也不以一次在线推理结果完全相同作为代码验收条件。实现后由用户使用相同模型、知识库和数据自行运行新旧版本对比。

## 3. 现状审计

### 3.1 已经存在的通用能力

- `CSVReader`：编码回退、逐行读取、表头与行号管理；
- CSV 类型识别：区分字段目录型 CSV 与普通业务数据型 CSV；
- `CatalogCSVAdapter`：将一行字段目录转换为一个 `FieldProfile`；
- `TabularCSVAdapter`：从普通业务数据的列结构和样本构造 `FieldProfile`；
- 独立标签匹配：标签不会写入 `FieldProfile` 或 Prompt；
- `CSVClassificationPipeline`：分类、逐条保存、失败记录、断点续跑和最终指标汇总；
- B 库 Benchmark Repository、指标计算和查询 API。

### 3.2 重复与耦合点

`benchmark_import_service.py` 重复实现了以下通用能力：

- `utf-8-sig`、`utf-8`、`gb18030` 编码尝试；
- CSV 表头和行解析；
- 字段名列校验；
- `样本1`、`样本2` 等列的顺序解析；
- 空样本过滤、最多五个样本和单项长度截断；
- CSV 行到 `FieldProfile` 的转换。

`benchmark_pipeline.py` 与 `csv_pipeline.py` 重复实现了：

- 逐字段分类；
- `UNKNOWN` 结果转失败；
- 单条预测持久化；
- 运行状态更新；
- TP、FP、TN、FN 和派生指标汇总；
- 失败重试和断点续跑。

`benchmark_source.py`、`benchmark_field_input` 和导入脚本只负责把已经结构化的老师 CSV 先写入 A 库，再立即读出并构造相同字段。该中转没有增加业务语义，通用 CSV Pipeline 已能直接完成同一输入转换。

## 4. 代码处置清单

### 4.1 由通用实现替代并删除

| 当前专用代码 | 替代方式 | 处置 |
| --- | --- | --- |
| `app/services/benchmark_import_service.py` | `CSVReader` + `CatalogCSVAdapter` | 删除 |
| `app/repositories/benchmark_source.py` | Benchmark 直接读取两份 CSV，不再经 A 库暂存 | 删除 |
| `app/services/benchmark_pipeline.py` | `CSVClassificationPipeline` | 删除 |
| `scripts/import_benchmark_data.py` | 新版 `scripts/run_benchmark.py` 直接接收两份文件 | 删除 |
| `sql/benchmark_source_schema.sql` | 不再创建 `benchmark_field_input` | 删除 |
| `DatasetLabel` | 文件角色直接生成布尔标签 | 从 Schema 删除 |
| `BenchmarkImportRow` | 通用 CSV 行与适配结果 | 从 Schema 删除 |
| `BenchmarkImportSummary` | 新 CLI 的输入摘要 | 从 Schema 删除 |
| `BenchmarkCase` | 通用 CSV case/label 组合 | 从 Schema 删除 |

对应的旧单元测试不直接保留，而是把仍有价值的行为迁移到通用 CSV 与新版 Benchmark 编排测试中：

- 删除 `tests/unit/test_benchmark_import_service.py`；
- 删除 `tests/unit/test_benchmark_pipeline.py`，其指标与失败行为由通用 Pipeline 测试覆盖；
- 从 `tests/unit/test_benchmark_repositories.py` 移除 A 库 Repository 测试，保留并可重命名 B 库测试；
- 重写 `tests/integration/test_benchmark_smoke.py`，使用临时双文件 fixture；
- 从 `tests/unit/test_sql_assets.py` 移除 Benchmark A 表断言。

### 4.2 必须保留并适度收束

| 代码 | 保留原因 | 调整 |
| --- | --- | --- |
| `app/services/benchmark_evaluator.py` | TP/FP/TN/FN、precision、recall、F1 等是 Benchmark 独有业务 | 保留，继续由通用 Pipeline 调用 |
| `app/services/benchmark_label_service.py` | 标签必须独立于分类输入，防止标签泄漏 | 保留，扩展为双文件标签组装入口 |
| `app/repositories/benchmark_target.py` | 保存运行、预测和指标到 B 库 | 保留 |
| `app/schemas/benchmark.py` | B 库结果、运行摘要、预测和指标响应 | 保留结果类，删除导入/A 库专用类 |
| `app/api/benchmark.py` | 查询历史与新 Benchmark 运行结果 | 保留 |
| `sql/benchmark_target_schema.sql` 及既有迁移 | 新旧版本共用 B 库结果结构 | 保留 |
| `scripts/run_benchmark.py` | 老师的两份文件天然携带标签语义，需提供易复现入口 | 重写为通用能力的薄编排 CLI |

历史记录兼容要求：`BenchmarkRunSummary.source_type` 继续允许旧值 `mysql_benchmark`。即使不再创建新的 MySQL Benchmark 源运行，历史 B 库记录仍必须可以查询和反序列化。

## 5. 方案比较

### 方案 A：保留两套实现

优点是改动最少；缺点是解析、分类、重试和指标逻辑长期重复，Benchmark 不能代表正式 CSV 输入链路。仅适合作为短期过渡，不推荐。

### 方案 B：双文件薄编排器复用通用 CSV 能力

保留老师两份文件的自然使用方式，由一个很薄的 Benchmark 入口完成文件角色赋标、合并与兼容元数据配置，后续全部交给通用 CSV Pipeline。既避免重复代码，又保留可复现的一条命令。推荐采用。

### 方案 C：要求用户手动合并 CSV 并制作独立标签文件

内部最纯粹，只需运行通用 `run_csv_pipeline.py`；但用户每次必须手工合并两份文件、维护标签和校验对应关系，增加人为错误，也不利于重复实验。不推荐作为主要入口，但通用 Pipeline 仍可支持这种高级用法。

## 6. 推荐架构

### 6.1 数据流

```text
个人信息字段 CSV ── CatalogCSVAdapter ── expected_personal=true  ─┐
                                                                  ├─ 双文件组装 ─ CSVInputBatch
非个人信息字段 CSV ─ CatalogCSVAdapter ─ expected_personal=false ─┘
                                                                        │
                                                                        ▼
                                                        CSVClassificationPipeline
                                                  ┌──────────┼──────────┐
                                                  ▼          ▼          ▼
                                                RAG+LLM   B库逐条保存   指标计算
```

老师文件属于字段目录型 CSV，因此两份文件都显式使用 `CatalogCSVAdapter`。不依赖自动模式检测，避免文件格式略有变化时误判为普通业务数据型。

### 6.2 双文件组装职责

在 `benchmark_label_service.py` 增加类似 `prepare_labeled_catalog_benchmark(...)` 的入口，职责仅限：

1. 分别读取个人信息和非个人信息文件；
2. 复用 `CatalogCSVAdapter` 构造字段；
3. 根据文件角色生成 `expected_personal`，而不是从字段内容推断标签；
4. 按“个人信息文件原顺序 → 非个人信息文件原顺序”合并；
5. 分别应用 `personal_limit` 和 `non_personal_limit`；
6. 合并后把 `case_index` 连续重排为 1 到 N；
7. 构造 `CSVInputBatch` 和 `LabelMatchSummary`；
8. 生成双文件组合指纹供断点续跑校验。

该服务不得分类、检索、调用 LLM 或访问数据库。

### 6.3 Pipeline 职责

`CSVClassificationPipeline` 作为唯一 CSV 分类流水线，负责：

- 创建 Benchmark run；
- 逐 case 调用 `FieldClassificationService`；
- 保存预测或失败信息；
- 断点续跑和失败重试；
- 调用 `benchmark_evaluator.py` 计算最终指标；
- 更新运行状态。

新版 `scripts/run_benchmark.py` 只解析参数、调用双文件组装器，再调用 `CSVClassificationPipeline`，不复制业务逻辑。

## 7. 新旧结果等价契约

要使新版本与旧 Benchmark 结果具有合理可比性，代码必须保证以下输入条件不变：

1. 文件未变化时，有效 case 总数一致；当前基线应为 2357；
2. 标签分布一致：个人信息 132、非个人信息 2225；
3. 顺序一致：先个人信息文件，再非个人信息文件，各自保持原行顺序；
4. 字段名保持原值，不自动改名或去重；
5. 样本处理一致：按 `样本N` 数字顺序，过滤空值，只取前五项，每项最多 50 个字符；
6. 标签只进入评估上下文，不进入 `FieldProfile`、检索文本或 Prompt；
7. `UNKNOWN`、异常、失败重试和 coverage 的计算语义一致；
8. 指标公式、B 库字段和查询 API 一致；
9. 使用相同的模型、知识库版本、Prompt、检索参数和代码版本。

尤其需要为通用 `CatalogCSVAdapter` 增加通用的 profile 默认值覆盖机制，例如 `CSVProfileDefaults`，使 Benchmark 明确生成旧版相同的元数据：

```text
source_system = benchmark
database_name = teacher_benchmark
table_name = benchmark_input
data_type = unknown
business_domain = general
```

这些值会进入检索文本或 Prompt 上下文。若直接采用通用适配器当前的 `csv`、`csv_source`、`catalog_input` 默认值，即使字段名和样本相同，模型输入仍然不同，结果差异就不能只归因于模型波动。

以上契约能保证“输入构造和指标计算等价”，但不能保证在线 LLM 的每条预测或最终分数逐位相同。即使 temperature 为 0，供应商模型版本、推理后端和服务行为也可能变化。对比时应同时记录模型名、知识库版本、Prompt 版本或哈希、Git commit 和运行时间。

## 8. 指纹与断点续跑

单文件通用 Pipeline 已有输入指纹。双文件 Benchmark 应生成组合指纹，至少包含：

- 个人信息文件原始内容哈希及 `personal` 角色；
- 非个人信息文件原始内容哈希及 `non_personal` 角色；
- profile 默认值；
- CSV 解析/组合格式版本；
- 两类 limit 参数。

文件角色必须进入指纹，避免交换两份文件后仍被视为相同输入。恢复运行时必须重新提供相同两份文件；任一文件内容、角色或影响 case 集合的参数变化时，应拒绝恢复原 run。

## 9. 收束后的运行方式

### 9.1 前置条件

- B 数据库 Benchmark target schema 和既有迁移已经执行；
- `.env` 中模型、知识库和 `TARGET_DATABASE_URL` 配置有效；
- 不再需要执行 `benchmark_source_schema.sql`；
- Benchmark 不再依赖 `SOURCE_DATABASE_URL`。普通 MySQL A→RAG→B 流程仍继续使用该配置。

### 9.2 小规模试跑

```powershell
python -m scripts.run_benchmark `
  --personal "D:\数据\个人信息_脱敏.csv" `
  --non-personal "D:\数据\非个人信息字段_脱敏.csv" `
  --batch teacher_2026_08_v2 `
  --personal-limit 20 `
  --non-personal-limit 80
```

两个 limit 分别作用于两类数据，避免使用一个总 limit 时因“个人文件在前”造成样本类别失衡。

### 9.3 完整运行

```powershell
python -m scripts.run_benchmark `
  --personal "D:\数据\个人信息_脱敏.csv" `
  --non-personal "D:\数据\非个人信息字段_脱敏.csv" `
  --batch teacher_2026_08_v2
```

该命令直接读取两份文件、赋标签、分类并写入 B 库，不再先运行导入命令。

### 9.4 恢复或重试失败项

```powershell
python -m scripts.run_benchmark `
  --resume-run "<run_id>" `
  --personal "D:\数据\个人信息_脱敏.csv" `
  --non-personal "D:\数据\非个人信息字段_脱敏.csv" `
  --retry-failed
```

恢复时仍要求提供两份原文件，用于重建输入并校验组合指纹。

### 9.5 查看结果

继续使用现有接口，无需改变调用方：

```text
GET /api/benchmark/runs/{run_id}
GET /api/benchmark/results?run_id={run_id}
```

旧版本保存的 run 和 prediction 也必须继续可查。

## 10. 数据库与删除安全

代码收束后，A 库的 `benchmark_field_input` 表不再参与新流程。实施时只删除其 Repository、建表 SQL 和调用代码，不自动执行 `DROP TABLE`，避免破坏用户本地数据。

在新旧结果完成对比并确认不再需要回退后，用户可以自行备份并手动删除旧表。此操作不属于本次代码更新。

B 库中的历史运行、预测和指标不得清理或迁写。新版继续写入相同目标表，以便直接比较不同 run。

## 11. 文档收束

README 中 Benchmark 部分应改为双文件直接运行：

- 删除“创建 Benchmark A 表”和“先导入后运行”步骤；
- 删除 `import_benchmark_data`、`benchmark_source_schema.sql` 和旧 A 表说明；
- 保留 B 库初始化、运行、恢复、查询和指标解释；
- 明确普通业务 CSV 可用 `run_csv_pipeline.py`，老师的双文件 Benchmark 使用 `run_benchmark.py`；
- 明确 `SOURCE_DATABASE_URL` 只服务普通 MySQL A 数据库流程。

旧设计文档由 Git 历史保存。新设计和实施计划成为当前有效文档后，可在实施提交中删除已失效、会误导操作的旧 Benchmark 计划文档；通用 CSV 输入设计文档继续保留。

## 12. 测试设计

实现阶段只使用临时小型 CSV、fake classifier 和测试数据库，不调用真实 DeepSeek、Embedding 或老师完整数据。

必须覆盖：

1. 两份字段目录型 CSV 均通过 `CatalogCSVAdapter`；
2. 新旧输入契约测试：序列化后的 `FieldProfile` 与旧构造规则完全一致；
3. 重复字段名不被错误去重；
4. 样本列排序、过滤、数量和截断规则一致；
5. 文件角色正确生成 true/false 标签；
6. 两类计数、顺序和独立 limit 正确；
7. 标签不进入 `FieldProfile`、检索文本或 Prompt；
8. 文件变化、角色交换或 limit 变化会改变组合指纹；
9. 相同输入可以恢复，变化输入拒绝恢复；
10. TP、FP、TN、FN、FAILED 和派生指标计算准确；
11. 历史 `mysql_benchmark` 运行仍可通过 API 查询；
12. 项目中不再存在 Benchmark A Repository 和旧 Pipeline 的 import；
13. 普通业务 CSV、字段目录 CSV 和 MySQL A→RAG→B 的原有测试继续通过。

实现完成后的代码验收命令：

```powershell
python -m compileall app scripts
ruff check .
pytest -q
```

老师完整文件的结果一致性由用户另行验证，不作为 fake/mock 自动化测试的一部分。

## 13. 人工结果对比建议

用户运行新版完整 Benchmark 后，至少比较：

- total、success、failed；
- 正负标签数量；
- TP、FP、TN、FN；
- precision、recall、F1、accuracy、coverage、effective recall；
- 每个字段的 `expected_personal` 是否一致；
- 同一字段的输入元数据、样本与旧版是否一致；
- 模型名、知识库版本、Prompt 版本或哈希、Git commit。

若输入契约一致但少量预测变化，应先排查模型或知识库版本；若 case 数、顺序、标签或 `FieldProfile` 不一致，则属于收束实现问题，不能视为正常模型波动。

## 14. 实施提交边界

审核通过后再单独编写可执行实施计划。实际更新应分为可审查的提交：

1. 先补充等价契约和双文件组装测试；
2. 增加 profile 默认值覆盖与双文件组装；
3. 让 `run_benchmark.py` 切换到通用 Pipeline；
4. 删除旧 A 导入、Repository 和专用 Pipeline；
5. 更新 Schema、SQL 测试和 README；
6. 完整回归验证。

在通用 Pipeline 接管且测试通过之前，不先删除旧实现。整个过程不自动删除数据库表、不修改历史结果、不 push 远程分支。

## 15. 验收标准

设计实施完成需同时满足：

- Benchmark CSV 解析只存在一套通用实现；
- Benchmark 分类和断点续跑只使用 `CSVClassificationPipeline`；
- 不再需要 Benchmark A 库暂存表和导入命令；
- 标签不泄漏到分类输入；
- 新版双文件入口保持旧版输入等价契约；
- 现有 B 库表和结果 API 可继续使用；
- 历史 `mysql_benchmark` 数据可读取；
- 通用 CSV 和 MySQL 数据库流程不受影响；
- compileall、Ruff 和全部自动化测试通过；
- README 的所有命令与实际入口一致。
