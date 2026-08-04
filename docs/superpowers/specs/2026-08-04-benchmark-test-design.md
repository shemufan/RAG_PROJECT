# 个人信息字段 Benchmark 设计

## 1. 目标

在不修改现有 A 库业务表字段名、不增加前置大模型映射的前提下，将老师提供的两份脱敏 CSV 导入 A 库测试输入表，逐行转换为 `FieldProfile`，复用当前 Chroma + 结构化 LLM 分类链路，并将预测和评分写入 B 库。

本版本需要回答：系统能否在大量非个人信息字段干扰下准确识别个人信息字段。主要指标为个人信息召回率、精确率、F1 和端到端覆盖率。

## 2. 范围与边界

本版本包含：

- 一次性 CSV 导入脚本；
- A 库 Benchmark 测试输入表；
- `is_personal` 结构化分类字段；
- 按行读取测试案例的 Repository；
- 可断点续跑的本地 Benchmark Runner；
- B 库运行、逐案例预测和评分记录；
- Benchmark 只读查询 API；
- 不依赖真实 LLM、真实 API Key 的单元测试。

本版本不包含：

- 修改现有 A 库四张业务表；
- 把 CSV 字段映射为现有 A 库物理字段；
- 前置 LLM 清洗或字段匹配；
- Web 前端、任务队列、分布式并发；
- 将老师提供的 CSV 或脱敏样例提交到 Git。

## 3. 方案选择

采用独立 Benchmark Pipeline，而不是复用现有 MySQL 物理字段扫描器。

现有 `SourceMySQLRepository` 从 `information_schema` 扫描物理列。如果直接把 CSV 导入普通表，它只能看到 `field_name`、`sample_values_json` 等表结构，不能把每一行的 `reg_ip`、`email` 当作待分类字段。因此新增 `BenchmarkSourceRepository`，明确按数据行生成测试案例。

CSV 仅在导入阶段读取。分类运行阶段的数据来源始终是 A 库，符合 A → RAG → B 的演示要求。

## 4. 总体数据流

```text
个人信息 CSV + 非个人信息 CSV
→ scripts/import_benchmark_data.py
→ A.benchmark_field_input
→ BenchmarkSourceRepository
→ BenchmarkCase + FieldProfile
→ FieldClassificationService
→ Chroma 法规检索
→ DeepSeek 结构化输出（含 is_personal）
→ BenchmarkClassificationPipeline
→ B.benchmark_prediction
→ BenchmarkEvaluator
→ B.benchmark_run 指标
→ FastAPI 查询结果、误报和漏报
```

真实标签只在导入、持久化和评分阶段使用，不得进入 `FieldProfile`、检索文本或 LLM Prompt。

## 5. A 库数据模型

新增表 `benchmark_field_input`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `benchmark_id` | BIGINT AUTO_INCREMENT | 测试案例主键 |
| `batch_name` | VARCHAR(64) | 数据批次，例如 `teacher_2026_08` |
| `source_dataset` | VARCHAR(32) | `personal` 或 `non_personal` |
| `source_row_number` | INT | CSV 原始数据行号 |
| `field_name` | VARCHAR(128) | 原始待分类字段名 |
| `sample_values_json` | JSON | 有序脱敏样例数组 |
| `expected_personal` | BOOLEAN | 真实二分类标签 |
| `created_at` | DATETIME(6) | 导入时间 |

约束：

- 唯一键为 `(batch_name, source_dataset, source_row_number)`，保证重复导入幂等；
- 不按 `field_name` 去重，保留同名但样例不同的测试案例；
- `source_dataset` 和 `expected_personal` 不传给分类服务；
- CSV 文件路径不写入数据库，只保存稳定批次名和数据集类型。

建表定义放入 `sql/benchmark_source_schema.sql`，不改变现有 `source_schema.sql` 的业务 Demo 职责。

## 6. CSV 导入

新增命令：

```powershell
python -m scripts.import_benchmark_data `
  --personal "D:\path\个人信息_脱敏.csv" `
  --non-personal "D:\path\非个人信息字段_脱敏.csv" `
  --batch "teacher_2026_08"
```

导入规则：

1. 自动尝试 `utf-8-sig`、`utf-8`、`gb18030`；
2. 第一列必须是字段名，其余列按顺序视为样例；
3. 忽略空样例，保留前 5 个非空样例，与现有 `FieldProfile` 上限一致；
4. 单个样例最多保留 50 个字符；
5. 个人信息文件写入 `expected_personal=true`，非个人信息文件写入 `false`；
6. 单行格式错误需报告文件、行号和原因；
7. 两份文件作为同一批次使用一个事务：任一文件任一行校验失败时，整个批次回滚；
8. 输出个人、非个人、总计、空样例行和重复跳过数量，不输出具体脱敏样例。

该脚本属于一次性数据导入工具，不参与后续分类判断。

## 7. 分类模型调整

在成功的 LLM 结构化输出 `ClassificationOutput` 中新增：

```python
is_personal: bool
```

对外编排结果 `ClassificationResult` 使用 `is_personal: bool | None`：成功分类必须为布尔值；现有异常降级结果 `level="UNKNOWN"` 必须为 `None`，不得把无法判断伪装成“非个人信息”。Benchmark 收到 `None` 时将该案例记录为 `FAILED`。

Prompt 明确要求：依据字段名、样例和检索法规判断字段是否属于个人信息，并继续输出现有分类、细分类、L1-L4、置信度、理由和复核标记。

`is_personal` 是评分的唯一预测标签。不得从自由文本 `category`、等级或关键词二次推断。

现有 `/api/classify` 和数据库 Pipeline 也返回该字段。B 库现有 `field_classification_result` 增加可查询的 `is_personal` 列；新安装更新 `sql/target_schema.sql`，已有本地数据库通过一次性迁移脚本增加该列。

## 8. Benchmark 读取模型

新增内部模型 `BenchmarkCase`：

```text
benchmark_id
batch_name
expected_personal
field_profile
```

`BenchmarkSourceRepository` 从 A 库按 `benchmark_id` 升序读取案例，并构造中性 `FieldProfile`：

```text
source_system = benchmark
database_name = teacher_benchmark
table_name = benchmark_input
field_name = CSV 字段名
sample_values = sample_values_json
data_type = unknown
business_domain = general
```

`expected_personal` 保留在 `BenchmarkCase` 外层，不进入 `field_profile`。Benchmark 使用专用 B 表持久化，不借用 `data_field_asset`，因此同名案例不会发生稳定字段 ID 冲突。

## 9. B 库数据模型

### 9.1 `benchmark_run`

保存运行状态和汇总指标：

| 字段组 | 内容 |
|---|---|
| 身份 | `run_id`、`batch_name` |
| 状态 | `status`、`total_cases`、`success_cases`、`failed_cases` |
| 混淆矩阵 | `tp`、`fp`、`tn`、`fn` |
| 指标 | `precision_score`、`recall_score`、`f1_score`、`accuracy_score`、`coverage_score`、`effective_recall_score` |
| 版本 | `model_name`、`knowledge_base_version` |
| 时间与错误 | `started_at`、`finished_at`、`error_message` |

### 9.2 `benchmark_prediction`

保存逐案例结果：

| 字段组 | 内容 |
|---|---|
| 身份 | `prediction_id`、`run_id`、`benchmark_id` |
| 输入快照 | `field_name_snapshot`、`sample_values_json` |
| 标签 | `expected_personal`、`predicted_personal`、`outcome` |
| 分类结果 | `category`、`subcategory`、`level`、`confidence`、`reason`、`need_review`、`decision_path` |
| 依据 | `evidence_json` |
| 执行状态 | `status`、`error_message`、`created_at` |

约束：

- `(run_id, benchmark_id)` 唯一，支持断点续跑；
- `benchmark_id` 是指向 A 库案例的逻辑标识，不建立跨数据库外键；
- `outcome` 只允许 `TP`、`FP`、`TN`、`FN`、`FAILED`；
- 失败案例必须落库，不能静默跳过；
- `evidence_json` 仅保存现有结构化 Evidence 数组，避免为 Benchmark 再复制一套法规依据表。

建表和已有数据库迁移放入独立 SQL 文件，并在 README 中说明执行顺序。

## 10. 运行与断点续跑

Benchmark 首版使用 CLI 运行，避免 2357 次串行 LLM 调用占用一个长时间 HTTP 请求：

```powershell
# 小批量分层验证
python -m scripts.run_benchmark `
  --batch teacher_2026_08 `
  --personal-limit 20 `
  --non-personal-limit 80

# 全量新任务
python -m scripts.run_benchmark --batch teacher_2026_08

# 继续一个中断任务，只处理尚未记录的案例
python -m scripts.run_benchmark --resume-run <run_id>

# 在继续任务时重新尝试已有失败案例
python -m scripts.run_benchmark --resume-run <run_id> --retry-failed
```

运行规则：

- 按 `benchmark_id` 稳定顺序串行处理，首版不增加并发；
- 每个案例完成后立即提交预测记录；
- 单案例失败时记录错误并继续；
- 默认恢复运行时跳过已有成功和失败记录，只处理尚未记录的案例；
- 使用 `--retry-failed` 时覆盖重试已有失败记录，但绝不重复调用成功案例；
- 每次运行结束重新从 `benchmark_prediction` 聚合指标，确保汇总可重建；
- 小批量运行分别限定个人与非个人案例数量，按 `benchmark_id` 取样，保证验证集同时包含正负样本。

## 11. 评分定义

成功分类案例按真实标签与 `is_personal` 计算：

```text
TP：真实个人信息，预测为个人信息
FN：真实个人信息，预测为非个人信息
FP：真实非个人信息，预测为个人信息
TN：真实非个人信息，预测为非个人信息
```

```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × Precision × Recall / (Precision + Recall)
Accuracy = (TP + TN) / 成功案例数
Coverage = 成功案例数 / 总案例数
Effective Recall = TP / 本次选取的全部真实个人信息案例数
```

零分母统一返回 0.0。`Recall` 衡量模型在成功案例中的漏检，`Effective Recall` 会把执行失败的个人信息案例计入端到端损失，是演示时的首要指标。由于个人信息约占数据集 5.6%，Accuracy 只作为辅助指标。

## 12. 查询 API

新增只读接口：

```text
GET /api/benchmark/runs/{run_id}
GET /api/benchmark/results?run_id=&outcome=&predicted_personal=&need_review=&limit=&offset=
```

运行由 CLI 发起，FastAPI 负责展示汇总、TP/FP/TN/FN、失败案例、理由和法规依据。首版不实现后台任务队列和异步启动接口。

## 13. 错误与隐私处理

- CSV 编码、列数或数据校验失败：导入事务回滚并指出文件和行号；
- A/B 数据库不可用：任务不启动，返回明确错误；
- Chroma 无依据或 LLM 返回 `UNKNOWN`：该案例记录为 `FAILED`；
- B 库单条写入失败：保留日志并终止运行，避免任务状态与预测记录失去一致性；
- 日志不得输出 API Key、数据库 URL、完整样例或 CSV 绝对路径；
- CSV、`.env`、缓存和 Benchmark 运行输出继续由 Git 忽略；
- 真实标签不得进入检索文本、Prompt、Evidence 或 LLM 调用。

## 14. 测试策略

单元测试至少覆盖：

1. UTF-8 BOM 与 GB18030 CSV，以及跨文件整批事务回滚；
2. 两份文件混合标签；
3. 空样例、超长样例和格式错误；
4. 同名字段保留、重复导入幂等；
5. `expected_personal` 不进入 `FieldProfile` 和 Prompt；
6. `is_personal` Pydantic 校验；
7. TP、FP、TN、FN 和全部零分母组合；
8. Coverage 与 Effective Recall；
9. 逐案例成功、分类失败和 B 库写入失败；
10. 分层小批量选择、断点续跑和失败案例显式重试；
11. Benchmark 查询 API；
12. 原有单字段分类和数据库 Pipeline 回归测试。

所有普通测试使用 fake Repository、fake Vector Store 和 fake LLM，不读取老师 CSV，不调用真实 DeepSeek，不依赖外部服务。真实 MySQL 和付费 LLM 验证继续显式启用。

## 15. 验收标准

- 两份 CSV 可导入同一批次，行数、标签数与源文件一致；
- 真实标签不会出现在任何模型输入中；
- 每个测试案例都有成功预测或明确失败记录；
- 同名字段不会覆盖；
- 指标可由逐案例记录重新计算并得到相同结果；
- 小批量运行、全量运行和断点续跑命令均真实可用；
- 查询 API 能筛选漏检、误报和失败案例；
- 原有 A → RAG → B、`/api/classify`、知识库重建不回归；
- `compileall`、Ruff 和全部非付费测试通过。

## 16. Git 实施顺序

1. 当前 `feature/database-pipeline` 修改先完成验证并提交；
2. 本设计规格在当前分支单独提交；
3. 用户审阅书面规格后，从最新提交创建 `feature/Benchmark-test`；
4. 使用 `.worktrees/benchmark-test` 隔离实施；
5. 实施过程中不使用多 Agent，不 push，不修改远程分支；
6. 提交使用仓库现有用户身份，不添加 Codex 或其他协作者。
