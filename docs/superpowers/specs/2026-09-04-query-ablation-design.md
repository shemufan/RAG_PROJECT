# Query 消融实验策略切换设计

## 1. 目标

在 `query_rebuild` 的实现基础上增加 `legacy`、`clean`、`profile` 三种 Query 策略，使同一 CSV Benchmark 命令能够只通过 `--query-strategy` 切换检索 Query。Embedding 模型、Chroma 知识库、检索 `top-k=3`、LLM、Prompt、输入数据和指标计算保持不变。

本次参数只接入 `scripts/run_csv_pipeline.py`。API、数据库 Pipeline 和 `scripts/run_benchmark.py` 不增加参数，继续使用默认 `profile` 策略。

## 2. 统一接口

`app/rag/retrieval_query.py` 定义统一的 Builder 协议：

```python
class QueryBuilder(Protocol):
    def build(self, field: FieldProfile) -> str: ...
```

三个实现均只负责 Query 构造：

- `LegacyQueryBuilder`：复现 `query_rebuild` 之前的完整元数据 Query；
- `CleanQueryBuilder`：保留原始字段名和全部有效样例值；
- `RetrievalQueryBuilder`：作为 `profile` 实现，内部调用 `ValueProfiler`，兼容现有类名。

`create_query_builder(strategy)` 集中完成策略到 Builder 的映射与运行时校验。`classification_service.py` 不出现针对三种策略的条件分支。

## 3. 三种策略

### 3.1 legacy

严格沿用旧版序列化顺序和格式：

```text
field_name: ...
field_cn: ...
field_comment: ...
data_type: ...
sample_values: value1、value2
business_domain: ...
table_name: ...
database_name: ...
source_system: ...
```

值为空时省略对应行。这用于复现原始 Baseline，不调整旧版内容和格式。

### 3.2 clean

仅保留：

```text
field_name: ...
sample_values: value1、value2
```

Clean 沿用 legacy 的键名、换行方式、样例顺序和样例数量，使 `legacy → clean` 的主要差异是删除 `field_cn`、注释、库表名、来源、业务域和数据类型等噪声，而不是改变序列化格式。

### 3.3 profile

保持当前 `query_rebuild` 行为：

```text
FieldProfile
→ ValueProfiler.profile(field_name, sample_values)
→ ValueProfile(features, candidate_types)
→ RetrievalQueryBuilder
→ 字段名 + 最多 3 个代表样例 + 结构特征 + 候选类型
```

`RetrievalQueryBuilder` 构造时可注入 `ValueProfiler`，便于验证画像确实被调用。

## 4. 分类服务接入

`FieldClassificationService` 保留两个现有位置参数，并增加可选参数：

```python
def __init__(
    self,
    vector_store,
    llm_service,
    *,
    query_strategy: QueryStrategy = "profile",
    query_builder: QueryBuilder | None = None,
): ...
```

如果显式注入 `query_builder`，优先使用注入对象；否则通过 `create_query_builder(query_strategy)` 创建。`build_query_text(field)` 只调用 `self.query_builder.build(field)`。

分类主流程保持：

```text
FieldProfile
→ QueryBuilder.build(field)
→ VectorStore.search(query, k=3)
→ Evidence
→ 原有 build_classification_prompt(field, evidence)
→ 原有 LLM
→ ClassificationResult
```

不修改异常降级、Evidence、Prompt、LLM 调用或返回结果。

## 5. CSV CLI 接入

`scripts/run_csv_pipeline.py` 新增：

```text
--query-strategy legacy|clean|profile
```

默认值为 `profile`。`argparse` 的 `choices` 在创建模型、知识库或数据库连接之前拒绝非法值。

`build_pipeline(settings, query_strategy="profile")` 将策略传入 `FieldClassificationService`。其他调用方不传值，因此保持 `profile`。

恢复已有运行时，调用者必须使用与原运行一致的 Query 策略；本次不修改数据库 Schema，也不在 Benchmark 结果表新增策略字段。

## 6. 实验隔离条件

三条实验命令只改变 `--query-strategy`。以下条件共享同一配置：

- 150 条 Benchmark 输入文件及标签列；
- Embedding 模型路径；
- Chroma collection 和知识库版本；
- LLM 模型及参数；
- 分类 Prompt；
- `VectorStore.search(..., k=3)`；
- CSV 适配、结果落库、Precision、Recall、F1、Accuracy、TP、FP、TN、FN、Coverage 计算。

不添加针对 Benchmark 字段的规则。

## 7. 错误处理

- CLI 非法策略由 `argparse` 报错并以非零状态退出；
- 代码直接传入非法策略时，Builder 工厂抛出明确的 `ValueError`；
- Builder 或检索异常继续由现有 `classify_field()` 异常边界处理；
- 空样例的 legacy/clean 省略 `sample_values`，profile 保持现有空画像行为。

## 8. 测试与验收

测试至少覆盖：

1. legacy 输出与旧版 Query 精确一致；
2. clean 仅包含 `field_name` 和 `sample_values`；
3. profile 调用注入的 `ValueProfiler` 并输出画像内容；
4. 三种策略均把各自 Query 传给同一 `VectorStore.search(query, k=3)`；
5. CLI 接受三个合法值、默认 profile，并拒绝非法值；
6. CSV、Benchmark、数据库 Pipeline 现有测试继续通过；
7. 全量 compileall、Ruff 和 pytest 通过。

验收时对比分支改动，确认 Chroma、Embedding、LLM、Prompt、CSV Pipeline 和数据库 Pipeline 核心文件没有行为修改。
