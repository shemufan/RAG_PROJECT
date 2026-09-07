# 实验 D：去除 RAG Evidence 设计

## 1. 目标

以当前效果最好的 B 实验为基准，验证最终分类效果来自 RAG Evidence，还是主要来自 LLM 对字段画像的直接判断。

- B：`query_strategy=clean`，启用 RAG。
- D：`query_strategy=clean`，关闭 RAG。
- 两者唯一影响分类输入的变量是最终 Prompt 中是否包含检索 Evidence。

## 2. 实验口径

当前 B 的 clean 策略仅约束 Retrieval Query 为 `field_name + sample_values`；最终分类 Prompt 仍接收完整 `FieldProfile`。D 必须保留这一行为，不能额外删减字段画像，否则会同时改变两个变量。

D 仅跳过 `VectorStore.search()`：

```text
FieldProfile
→ 保留原 Query Builder 配置
→ 不调用 VectorStore.search()
→ evidence=[]
→ 原 build_classification_prompt(FieldProfile, evidence)
→ 原 LLMService
→ 原 ClassificationResult 与评测流程
```

Prompt 模板、系统提示词、完整 `FieldProfile`、LLM 参数、输出结构和标签体系均不修改。空 Evidence 继续通过现有 Prompt Builder 序列化为 `[]`，不新增专用 Prompt。

## 3. 接口设计

### 3.1 `FieldClassificationService`

构造函数新增：

```python
use_rag: bool = True
```

- `True`：执行现有 `build_query_text()`、`VectorStore.search(k=3)` 和非空 Evidence 校验。
- `False`：不构造检索 Query、不调用 VectorStore，直接使用空 Evidence。
- 两条路径随后汇合到同一个 Prompt Builder、LLM 调用和结果组装逻辑。

默认值为 `True`，因此 API、数据库 Pipeline、Benchmark 脚本以及已有 A/B/C/C1/C2/E1/E2 行为不变。

### 3.2 CSV CLI

仅在 `scripts/run_csv_pipeline.py` 增加互补布尔参数：

- `--use-rag`：启用 RAG，默认行为。
- `--no-use-rag`：关闭 RAG，实验 D。

为避免产生无意义实验组合，`--no-use-rag` 仅允许与 `--query-strategy clean` 同时使用。该约束不影响任何已有命令。

参数通过 `build_pipeline()` 传入 `FieldClassificationService`。D 仍初始化与 B 相同的 Pipeline 依赖和配置，但单字段分类时不会调用向量检索。

## 4. 错误处理

- B 保留现有规则：检索异常或 Evidence 为空时返回 `UNKNOWN` 并进入失败统计。
- D 不执行检索，也不触发“Evidence 为空”错误；最终 LLM 失败仍沿用现有 `UNKNOWN` 降级。
- D 的 `ClassificationResult.evidence` 固定为空列表，其余字段由原有 LLM 输出映射。

## 5. 测试设计

新增或调整测试，验证：

1. `use_rag=True` 默认路径仍用 clean Query 调用 `VectorStore.search(k=3)`。
2. `use_rag=False` 时 VectorStore 的 `search()` 零调用。
3. D 传给 LLM 的消息仍包含完整 FieldProfile，Evidence 内容为 `[]`。
4. B 与 D 使用同一个 `build_classification_prompt()`，不增加或修改 Prompt 模板。
5. CLI 默认启用 RAG，并接受 `--no-use-rag`。
6. 非 clean 策略与 `--no-use-rag` 的组合被 CLI 拒绝。
7. `build_pipeline()` 正确传递 `use_rag`。
8. CSV Pipeline、Benchmark 评测和既有测试全部通过，指标输出格式不变。

## 6. 运行命令

B：

```powershell
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --input-mode catalog --label-column expected_personal --query-strategy clean --use-rag
```

D：

```powershell
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --input-mode catalog --label-column expected_personal --query-strategy clean --no-use-rag
```

两次实验必须使用新的 `run_id`，不得通过 `--resume-run` 复用既有预测。

## 7. 修改边界

允许修改：

- `app/services/classification_service.py`
- `scripts/run_csv_pipeline.py`
- 对应单元测试和 CLI 集成测试

明确不修改：

- `app/rag/prompt.py`
- Chroma、Embedding、Query Builder 和 `top-k`
- `LLMService` 及结构化分类输出
- CSV/数据库 Pipeline 的执行与持久化逻辑
- Benchmark 数据、指标计算和输出格式
- A/B/C/C1/C2/E1/E2 的默认行为
