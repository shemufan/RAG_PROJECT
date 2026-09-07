# C 实验画像子模式设计

## 1. 目标与范围

在现有 `experiment/query-ablation` 分支的 `profile` Query 策略中增加两个子模式，用于判断 C 实验性能下降主要来自结构画像特征还是候选数据类型：

- `c`：保持现有 C 行为，输出字段名、代表样例、结构特征和候选数据类型；
- `c1`：输出字段名、代表样例和结构特征，不输出候选数据类型；
- `c2`：输出字段名、代表样例和候选数据类型，不输出结构特征。

本次只扩展 `scripts/run_csv_pipeline.py`。现有 `legacy`、`clean`、`profile` 策略及其默认行为不变；API、数据库 Pipeline、`scripts/run_benchmark.py`、Prompt、LLM、Embedding、Chroma、`top-k=3`、CSV 适配、结果落库和指标计算均不修改。

## 2. 参数设计

CSV CLI 新增：

```text
--query-mode c|c1|c2
```

默认值为 `c`。该参数是 `profile` 的子模式，不取代现有 `--query-strategy`：

```powershell
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --label-column expected_personal --query-strategy profile --query-mode c1
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --label-column expected_personal --query-strategy profile --query-mode c2
```

为避免无意义组合，显式指定 `c1` 或 `c2` 时必须同时使用 `--query-strategy profile`。`legacy`、`clean` 搭配默认 `c` 继续合法，因此已有命令不受影响。

## 3. Builder 设计

`app/rag/retrieval_query.py` 增加：

```python
ProfileQueryMode = Literal["c", "c1", "c2"]
```

现有 `RetrievalQueryBuilder` 继续作为唯一画像 Query 实现。构造时接收 `profile_mode="c"`；`build(field)` 仍只调用一次 `ValueProfiler.profile()`，并继续复用现有代表样例选择逻辑。模式只控制两个可选片段：

| 模式 | profiling features | candidate types |
|---|---:|---:|
| `c` | 包含 | 包含 |
| `c1` | 包含 | 排除 |
| `c2` | 排除 | 包含 |

`create_query_builder(strategy, profile_mode="c")` 集中创建 Builder。`legacy` 和 `clean` 的 Builder 不读取画像子模式，也不改变输出。

## 4. 服务与数据流

`FieldClassificationService` 增加默认值为 `c` 的可选 `profile_query_mode`，仅在通过工厂创建 Builder 时传递。显式注入 `query_builder` 的测试和扩展接口保持最高优先级。

完整链路保持：

```text
FieldProfile
→ 选定的 QueryBuilder
→ VectorStore.search(query, k=3)
→ Evidence
→ 原有 Prompt
→ 原有 LLM
→ ClassificationResult
→ 原有 Benchmark 指标
```

C、C1、C2 之间唯一变化是最终 Query 是否包含 `profiling_features` 和 `candidate_types`。

## 5. 兼容性与错误处理

- 不传 `--query-mode` 时等价于当前 `profile`，保证默认行为不变；
- 非法值由 `argparse choices` 在加载模型和数据库前拒绝；
- `legacy/clean + c1/c2` 由 CLI 参数校验拒绝，避免误标实验；
- API、数据库 Pipeline 和其他脚本不传新参数，继续使用完整 `profile`；
- 不修改数据库 Schema，运行结果继续输出 TP、FP、TN、FN、Precision、Recall、F1、Accuracy、Coverage 等现有字段。

## 6. 测试与验收

测试覆盖：

1. 默认 `c` Query 与现有 `profile` Query 精确一致；
2. `c1` 包含结构特征但不包含候选类型；
3. `c2` 包含候选类型但不包含结构特征；
4. 三种模式均只调用一次同一个 ValueProfiler，并使用相同代表样例；
5. C、C1、C2 均调用现有 VectorStore，且 `k=3`；
6. CLI 接受 `c/c1/c2`、默认 `c`、拒绝非法值和非法策略组合；
7. 现有 legacy、clean、profile、CSV、Benchmark、API 和数据库 Pipeline 测试全部通过；
8. Ruff、pytest 和 compileall 全部通过。

实验执行时继续使用同一 CSV、标签、`.env`、Embedding 模型、旧 Chroma Collection、LLM 和 Prompt，只改变 `--query-mode`。
