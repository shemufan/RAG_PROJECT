# 多模式 CSV 输入适配器设计

## 1. 目标

将当前项目从“必须先把测试字段写入固定 MySQL A 表”调整为“输入源可插拔、
`FieldProfile` 契约稳定”。首版直接支持两种 CSV：

- 字段目录型 CSV：一行描述一个待分类字段；
- 普通业务数据型 CSV：一列代表一个待分类字段，列内数据作为样例。

CSV 可以直接进入 RAG + LLM 分类链路，不强制写入数据库 A。现有 MySQL A 物理字段扫描、
A 库 Benchmark 表和 A → RAG → B Demo 必须继续可用。

本设计的核心原则是：固定规范化后的输入模型，不固定外部文件格式和数据库表结构。

## 2. 范围

本版本包含：

- `auto`、`catalog`、`tabular` 三种 CSV 输入模式；
- 两个职责独立的 CSV Adapter；
- 一次处理一份 CSV；
- 普通业务 CSV 的可选独立标签文件；
- 无标签分类和有标签 Benchmark 两种运行方式；
- CSV 文件指纹、逐字段结果持久化和可验证恢复；
- 复用现有 `FieldClassificationService`、Chroma、结构化 LLM 和 B 库评分能力；
- 不依赖真实文件、MySQL 或付费 LLM 的测试。

本版本不包含：

- Excel、JSON、Parquet 等新文件类型的具体实现；
- 让 LLM 猜测 CSV 结构或生成列映射；
- 一次运行合并多份普通业务 CSV；
- Web 上传页面、后台任务队列和并发分类；
- 自动生成真实标签；
- 删除现有 MySQL A Pipeline 或老师两份 CSV 的既有导入方式。

后续文件类型只需实现相同输入协议，不应修改分类服务。

## 3. 架构选择

采用“自动识别 + 显式覆盖 + 统一 Schema”的混合方案：

```text
CSV 文件
→ CSV 编码与表头读取
→ 输入模式识别
→ CatalogCSVAdapter / TabularCSVAdapter
→ CSVInputBatch + CSVFieldCase
→ FieldProfile
→ FieldClassificationService
→ Chroma + 结构化 LLM
→ B.benchmark_prediction
→ 可选 BenchmarkEvaluator
→ B.benchmark_run
```

不采用纯自动识别，因为任意 CSV 都可能出现类似 `field_name`、`sample1` 的真实业务字段，
无法做到百分之百无歧义。不采用纯手动模式，因为普通业务 CSV 是更常见输入，默认行为应当
简单。混合方案既提供便利，也允许用户纠正误判。

数据库 A 不再是 CSV 分类的强制中转站，但仍是一个有效输入源。项目中的“A”应理解为
逻辑输入端，而不是某一张固定物理表。

## 4. 统一输入模型

新增以下内部模型：

```text
CSVInputBatch
- source_name：仅保存文件名，不保存绝对路径
- source_fingerprint：原始 CSV 的 SHA-256
- input_mode：catalog 或 tabular
- cases：CSVFieldCase 列表

CSVFieldCase
- case_index：从 1 开始、在同一文件内稳定
- field_profile：传入分类服务的 FieldProfile
- expected_personal：可选真实标签，仅供评分
```

`expected_personal` 必须位于 `FieldProfile` 外层。分类服务只能接收
`field_profile`，不能接收完整 `CSVFieldCase`，从类型边界上防止标签泄漏。

每次适配完成后，所有 `FieldProfile` 都必须经过 Pydantic 校验。输入 Adapter 不调用
Chroma、LLM 或 B Repository。

## 5. CSV 公共读取规则

两个 Adapter 共用底层 CSV Reader：

1. 编码依次尝试 `utf-8-sig`、`utf-8`、`gb18030`；
2. 使用 Python `csv` 模块识别引号、逗号和换行，不用字符串切割；
3. 空文件、无表头、重复表头或空表头直接报错；
4. 表头仅去除首尾空白，保留原始大小写和中文；
5. 样例忽略空值，按原顺序保留前 5 个非空值；
6. 每个样例截断到 50 个字符；
7. 日志只输出文件名、行列位置和计数，不输出样例内容或绝对路径；
8. 默认最大文件为 100 MB、最大 1,000,000 行、最大 10,000 列，防止误读超大文件；
   三项限制允许通过集中配置降低或提高；
9. 普通业务 CSV 使用流式逐行采样，取得每列前 5 个非空值后不再保存后续单元格，避免
   把整份业务数据加载到内存。

首版不做复杂类型推断。普通业务 CSV 的 `data_type` 使用 `unknown`；字段目录型存在明确
类型列时才写入该值。

## 6. 输入模式识别

CLI 提供：

```text
--input-mode auto
--input-mode catalog
--input-mode tabular
```

默认值为 `auto`。

### 6.1 强目录特征

只有同时满足下列条件时，`auto` 才识别为 `catalog`：

1. 存在一个字段名列：`字段名` 或 `field_name`；
2. 存在至少一个样例列：`样本N`、`sampleN` 或 `sample_N`；
3. 样例编号可解析并按数字顺序排列。

不满足强目录特征时默认使用 `tabular`，因为普通业务数据型 CSV 更常见。

### 6.2 歧义处理

当文件满足强目录特征，但除字段名列、样例列和第 7 节列出的目录元数据列之外还存在任意
其他列时，`auto` 不静默猜测，而是返回“输入模式存在歧义”，要求用户显式指定
`catalog` 或 `tabular`。

用户显式指定模式后不再运行自动识别，但 Adapter 仍执行该模式自身的结构校验。选择
`catalog` 却没有字段名列时必须失败，不能退回 `tabular`。显式 `catalog` 允许没有样例
列，此时每个字段的 `sample_values=[]`；只有 `auto` 识别目录型时要求至少一个样例列。

## 7. CatalogCSVAdapter

字段目录型 CSV 一行生成一个 `FieldProfile`。

默认列映射：

| CSV 列 | FieldProfile |
|---|---|
| `字段名` / `field_name` | `field_name` |
| `字段中文名` / `field_cn` | `field_cn` |
| `字段说明` / `field_comment` | `field_comment` |
| `数据类型` / `data_type` | `data_type` |
| `业务域` / `business_domain` | `business_domain` |
| `表名` / `table_name` | `table_name` |
| `数据库名` / `database_name` | `database_name` |
| `来源系统` / `source_system` | `source_system` |
| `样本N` / `sampleN` / `sample_N` | `sample_values` |

缺少可选元数据时使用中性默认值：

```text
source_system = csv
database_name = csv_source
table_name = catalog_input
data_type = unknown
business_domain = general
```

字段名为空时报告原始数据行号并终止整次读取。同名字段按不同数据行保留，不自动去重。

对于列名不符合默认别名的字段目录文件，首版允许通过 CLI 显式指定：

```text
--field-name-column <列名>
--sample-columns <列名1,列名2,...>
```

该显式映射只属于 `catalog` 模式，不影响普通业务 CSV。

## 8. TabularCSVAdapter

普通业务数据型 CSV 一列生成一个 `FieldProfile`：

```text
user_id,reg_ip,product_price,created_at
10001,192.168.*.*,99.00,2026-08-01
10002,10.0.*.*,120.00,2026-08-02
```

生成四个案例：

```text
field_name=user_id, sample_values=[10001, 10002]
field_name=reg_ip, sample_values=[192.168.*.*, 10.0.*.*]
field_name=product_price, sample_values=[99.00, 120.00]
field_name=created_at, sample_values=[2026-08-01, 2026-08-02]
```

统一默认值：

```text
source_system = csv
database_name = csv_source
table_name = tabular_input
field_name = 原始列名去除首尾空白
data_type = unknown
business_domain = general
```

原始文件名只作为 B 库审计元数据，不进入 `FieldProfile`。这样可以防止
`personal.csv`、`非个人信息.csv` 等文件名向 LLM 泄漏标签语义。

所有列都进入分类，包括整列为空的字段；整列为空时 `sample_values=[]`，由字段名独立参与
检索和判断。重复表头无法安全匹配标签和结果，必须拒绝。

## 9. 独立标签文件

普通业务 CSV 可以不提供标签。无标签时系统正常分类并保存结果，但不计算 Precision、
Recall、F1 或 Accuracy。

需要 Benchmark 评分时，使用独立标签文件：

```csv
field_name,expected_personal
reg_ip,true
product_price,false
created_at,false
```

规则：

1. 标签表头固定为 `field_name,expected_personal`；
2. `expected_personal` 只接受 `true`、`false`、`1`、`0`，忽略大小写；
3. 字段名去除首尾空白后进行大小写敏感的精确匹配；
4. 重复标签立即报错；
5. 标签中存在、业务 CSV 中不存在的字段立即报错，防止使用错误标签文件；
6. 业务 CSV 中未标注的字段仍分类，结果标记为 `UNLABELED`，不进入混淆矩阵；
7. 输出 `labeled_cases` 和 `unlabeled_cases`，明确评分覆盖范围；
8. 标签内容、`expected_personal` 不得进入检索文本、Prompt、Evidence 或 LLM 日志。

老师当前两份已经按个人/非个人拆分的字段目录 CSV 仍可继续使用现有导入方式，由文件身份
赋值标签。本设计不强迫把已有 Benchmark 数据改造成普通业务 CSV。

## 10. CLI 与运行流程

新增独立直接运行命令，避免破坏现有 A 库 Runner：

```powershell
# 自动识别，无标签分类
python -m scripts.run_csv_pipeline `
  --input "D:\data\business.csv" `
  --input-mode auto

# 普通业务 CSV Benchmark
python -m scripts.run_csv_pipeline `
  --input "D:\data\business.csv" `
  --input-mode tabular `
  --labels "D:\data\business_labels.csv"

# 字段目录型 CSV，显式列映射
python -m scripts.run_csv_pipeline `
  --input "D:\data\field_catalog.csv" `
  --input-mode catalog `
  --field-name-column "字段名称" `
  --sample-columns "示例1,示例2,示例3"
```

Runner 的处理顺序：

1. 完整读取并校验 CSV；
2. 识别或验证输入模式；
3. 构造所有 `CSVFieldCase`；
4. 如有标签文件，完整读取、校验并匹配；
5. 在任何 LLM 调用前创建 B 运行记录；
6. 按 `case_index` 串行分类，每个案例完成后立即写入；
7. 单案例分类失败写入 `FAILED` 并继续；
8. 从 B 中重新聚合有标签案例的指标；
9. 输出 run ID、模式、总字段数、已标注/未标注数、成功/失败数和可用指标。

CLI 不输出样例值。全量运行会产生真实 LLM 费用，README 必须要求先用小文件验证。

## 11. B 库兼容设计

复用现有 `benchmark_run` 和 `benchmark_prediction`，不新建语义重复的结果表。

`benchmark_run` 增加：

```text
source_type = mysql_benchmark / csv
input_mode = catalog / tabular / null
source_name = 安全文件名
source_fingerprint = 原始输入 SHA-256
label_fingerprint = 可空的标签文件 SHA-256
labeled_cases
unlabeled_cases
```

`benchmark_prediction` 调整：

```text
benchmark_id：对直接 CSV 表示稳定的 case_index；仍兼容原 A 表 benchmark_id
expected_personal：允许 null
outcome：增加 UNLABELED
```

已有 MySQL Benchmark 的 `expected_personal` 始终非空，行为和指标不改变。迁移脚本只增加
列或放宽空值，不删除历史数据。

无标签运行只提供分类成功率和 Coverage，不伪造二分类指标。Precision、Recall、F1、
Accuracy 和 Effective Recall 在 `labeled_cases=0` 时统一返回 `null`，对应 Pydantic 字段
改为可空、B 库指标列允许空值；Coverage 始终按全部输入字段计算。已有历史运行中的数值
保持不变，这项兼容调整必须用回归测试锁定。

## 12. 断点恢复

直接 CSV 不写入 A，因此恢复时必须重新提供原文件：

```powershell
python -m scripts.run_csv_pipeline `
  --resume-run <run_id> `
  --input "D:\data\business.csv" `
  --labels "D:\data\business_labels.csv"
```

恢复前必须验证：

- 输入文件 SHA-256 与运行记录一致；
- 标签文件存在性和 SHA-256 与运行记录一致；
- 输入模式与原运行一致；
- 重新解析得到的 `case_index + field_name` 与已写结果一致。

任一验证失败立即停止，避免把不同文件的结果拼入同一 run。默认跳过已有成功和失败记录；
`--retry-failed` 只重新处理失败案例。现有 A 库 Benchmark 的恢复方式保持不变。

## 13. API

现有 Benchmark 只读 API 继续使用：

```text
GET /api/benchmark/runs/{run_id}
GET /api/benchmark/results?run_id=...
```

响应增加输入来源、模式、已标注数和未标注数。结果筛选增加 `outcome=UNLABELED`。首版不
提供 HTTP 文件上传或启动接口，直接 CSV 任务仍由本地 CLI 发起。

## 14. 错误处理

- 文件不存在、无法解码、超限：在创建 run 和调用 LLM 前失败；
- 表头为空或重复：报告列位置并失败；
- `catalog` 缺少字段名列：失败；显式 `catalog` 允许没有样例列；
- `auto` 结构歧义：要求显式指定模式；
- 标签布尔值非法、重复或包含输入中不存在的字段：失败；
- 某字段分类返回 `UNKNOWN`：写入 `FAILED`，继续下一字段；
- B 单条写入失败：终止运行，保留已提交记录供恢复；
- 恢复文件指纹不一致：拒绝恢复；
- 错误信息不得包含数据库密码、API Key、完整样例或文件绝对路径。

## 15. 测试策略

普通测试全部使用临时 CSV、fake Repository 和 fake LLM，至少覆盖：

1. UTF-8 BOM、UTF-8、GB18030；
2. `auto` 正确识别强目录特征；
3. 无强目录特征默认 `tabular`；
4. 显式模式覆盖和歧义报错；
5. Catalog 一行一个案例及显式列映射；
6. Tabular 一列一个案例、空列和样例上限；
7. 空表头、重复表头、空文件和资源上限；
8. 独立标签正确匹配、非法布尔值、重复标签和多余标签；
9. 未标注字段产生 `UNLABELED` 且不进入指标；
10. `expected_personal` 不进入 `FieldProfile`、Prompt 和分类服务调用；
11. 有标签 CSV 的 TP/FP/TN/FN、Coverage 和 Effective Recall；
12. 无标签运行不伪造评分；
13. 每案例写入、失败继续、恢复跳过和仅重试失败；
14. 输入或标签文件变化后拒绝恢复；
15. Benchmark API 对 CSV 来源和 `UNLABELED` 的查询；
16. 原有单字段 API、MySQL A → RAG → B 和现有老师 Benchmark 回归测试。

真实 CSV、MySQL、Embedding 模型和付费 LLM 不参与普通测试。

## 16. 文件与模块规划

预计新增：

```text
app/schemas/csv_input.py
app/services/csv_reader.py
app/services/csv_mode_detector.py
app/services/catalog_csv_adapter.py
app/services/tabular_csv_adapter.py
app/services/benchmark_label_service.py
app/services/csv_pipeline.py
scripts/run_csv_pipeline.py
sql/migrations/<date>_extend_benchmark_for_csv.sql
tests/unit/test_csv_*.py
tests/integration/test_csv_pipeline.py
```

预计修改：

```text
app/schemas/benchmark.py
app/repositories/benchmark_target.py
app/services/benchmark_evaluator.py
app/api/benchmark.py
sql/benchmark_target_schema.sql
README.md
```

禁止创建 `utils`、`helpers` 或一个同时负责识别、解析、分类、持久化的巨大模块。

## 17. 验收标准

- 同一命令可以处理字段目录型和普通业务数据型 CSV；
- `auto` 有确定规则，并能用显式模式纠正；
- 普通业务 CSV 每列准确生成一个通过 Pydantic 校验的 `FieldProfile`；
- CSV 可直接分类，不要求 A 库连接；
- 有标签时评分准确，无标签时不伪造评分；
- 标签不进入任何模型输入；
- B 中可以查询来源、模式、逐字段结果、未标注字段和指标；
- 中断后只能使用完全相同的输入恢复；
- 原有 MySQL 和老师 Benchmark 链路不回归；
- `compileall`、Ruff、全部非付费测试通过；
- 不提交真实 CSV、标签文件、`.env` 或运行缓存。

## 18. Git 实施边界

本设计文档先提交在现有 `feature/Benchmark-test` 分支供审核，不创建新功能分支。

用户书面批准后，再从包含当前 Benchmark 功能和本设计文档的最新提交创建：

```text
feature/csv-input-adapters
```

后续先编写实施计划，再按 TDD 分批实现。不 push、不修改远程分支，不添加 Codex 或其他
协作者信息。
