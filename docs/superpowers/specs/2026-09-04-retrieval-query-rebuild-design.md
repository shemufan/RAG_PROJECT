# 字段值画像与 Retrieval Query 重构设计

## 1. 目标

将 Retriever Query 从完整 `FieldProfile` 元数据拼接，调整为由字段名、样例值、样例结构特征和候选数据类型组成的短 Query，减少默认数据库名、表名、业务域、未知类型及字段注释带来的检索噪声。

本次只重构 Query 构造链路，不修改 LLM 分类 Prompt、Chroma 检索实现、Evidence 结构、CSV Pipeline、Benchmark Pipeline 或数据库 Pipeline。

## 2. 范围

新增：

- `app/services/value_profiler.py`：纯 Python 字段值画像和候选类型生成；
- `app/rag/retrieval_query.py`：从画像生成干净 Retrieval Query；
- 两个模块的独立单元测试。

修改：

- `app/services/classification_service.py`：接入画像与 Query Builder；
- 分类服务单元测试：验证新 Query 被传给现有 VectorStore 接口。

不包含：

- 新的大模型调用；
- Chroma、Embedding 或相似度算法调整；
- LLM 分类 Prompt 或输出 Schema 调整；
- 针对当前 Benchmark 字段名的专用规则；
- 数据库表结构调整。

## 3. 总体数据流

```text
FieldProfile
  -> ValueProfiler.profile(field_name, sample_values)
  -> ValueProfile(features, candidate_types)
  -> RetrievalQueryBuilder.build(field_name, sample_values, value_profile)
  -> VectorStore.search(query, k=3)
  -> Evidence
  -> 现有 build_classification_prompt(FieldProfile, Evidence)
  -> 现有 LLM 分类
```

完整 `FieldProfile` 仍传给 LLM Prompt。Retriever Query 只使用本设计允许的字段，避免改变分类阶段已有行为。

## 4. ValueProfile 数据契约

`ValueProfile` 使用 Pydantic：

```python
class ValueProfile(BaseModel):
    features: list[str] = Field(default_factory=list)
    candidate_types: list[str] = Field(default_factory=list, max_length=3)
```

`ValueProfiler` 的公开接口：

```python
class ValueProfiler:
    def profile(
        self,
        field_name: str,
        sample_values: list[str],
    ) -> ValueProfile:
        ...
```

输入只包含 `field_name` 和 `sample_values`。模块不接收 `FieldProfile`，因此无法读取 `field_cn`、`field_comment`、Ground Truth 或来源元数据。

## 5. 画像规则

### 5.1 预处理

- 去除空字符串；
- 保留样例原始顺序；
- 规则判断使用去除首尾空白后的值；
- 没有有效样例时返回空候选类型，可保留“无有效样例”结构特征；
- 所有候选类型稳定去重，最多保留 3 个。

### 5.2 通用特征

第一版输出以下可解释特征：

- 字符串典型长度；
- 长度是否一致；
- 数字、字母、中文、十六进制字符的主要组成；
- 多个样例格式是否一致；
- 是否含 `*` 或 `X/x` 脱敏符号；
- 是否含 `@`；
- 明确格式的分隔方式，例如 MAC 的冒号分隔和 UUID 的连字符分组。

特征使用稳定的中文短语，便于直接进入 Retrieval Query 和进行精确单元测试。

### 5.3 具体格式识别

独立检测规则至少覆盖：

- 中国大陆手机号及中间脱敏手机号；
- 邮箱及本地部分脱敏邮箱；
- IPv4，并验证每段为 0～255；
- 6 组十六进制 MAC 地址；
- 18 位身份证号及末位 X、常见中间脱敏形式；
- 常见银行卡号及中间脱敏形式；
- 日期与日期时间；
- 标准 UUID；
- HTTP/HTTPS URL；
- 合法经纬度数值对；
- 15 位 IMEI 类设备编号；
- 无法命中具体格式时的普通文本或普通数字结构描述。

手机号、身份证号、银行卡号和 IMEI 的数字长度存在重叠。规则按格式约束和上下文弱提示排序；无法可靠区分时可以同时返回多个候选，但不得超过 3 个，也不得强行选择唯一类型。

### 5.4 候选类型策略

候选类型以样例结构为主，字段名只作为弱辅助：

- MAC 样例可直接生成 `MAC地址`、`设备标识信息`；
- 手机号样例可直接生成 `手机号码`、`联系方式`；
- 邮箱样例可直接生成 `邮箱`、`联系方式`；
- 只有字段名命中弱语义、样例没有相应结构时，不生成具体敏感类型；
- 普通商品名称只产生普通文本特征，`candidate_types=[]`；
- `attr_01`、`column_x` 等无意义字段名不影响基于样例的识别。

弱字段名提示采用通用概念词，不加入针对 Benchmark 个别字段的名单。

## 6. RetrievalQueryBuilder

公开接口：

```python
class RetrievalQueryBuilder:
    def build(
        self,
        field_name: str,
        sample_values: list[str],
        value_profile: ValueProfile,
    ) -> str:
        ...
```

Builder 不接收完整 `FieldProfile`。输出使用单行、固定顺序：

```text
字段名：device_attr 样例值：A1:B2:C3:D4:E5:F6 数据结构特征：6组十六进制字符、冒号分隔、格式稳定 候选数据类型：MAC地址、设备标识信息
```

### 6.1 代表性样例选择

- 忽略空值；
- 完全相同的值只保留一次；
- 优先保留格式签名不同的样例，再按原始顺序补足；
- 最多输出 3 个样例；
- 不改变原始样例内容，不把 Ground Truth 或文件名加入 Query。

格式签名只描述字符类别和分隔符，不进行新的数据分类。这样可避免前三个重复值占满 Query，同时保持算法简单。

### 6.2 禁止内容

Query 不加入：

- `field_cn`；
- `field_comment`；
- `database_name`；
- `table_name`；
- `source_system`；
- `business_domain`，包括默认值 `general`；
- `data_type`，包括默认值 `unknown`；
- Ground Truth 或 `expected_personal`。

画像没有候选类型时省略“候选数据类型”段，不输出空占位符。

## 7. 分类服务接入

`FieldClassificationService` 构造函数保留现有两个位置参数，并增加可选关键字依赖：

```python
def __init__(
    self,
    vector_store,
    llm_service,
    *,
    value_profiler: ValueProfiler | None = None,
    query_builder: RetrievalQueryBuilder | None = None,
):
    ...
```

没有注入时创建默认实例。现有 CSV、Benchmark、数据库和 API 调用方无需修改。

`build_query_text(field)` 保留，以免破坏已有或外部调用；内部改为：

1. 调用 profiler 生成 `ValueProfile`；
2. 调用 builder 生成 Query；
3. 返回 Query。

`classify_field()` 后续流程保持不变，仍调用 `vector_store.search(query, k=3)`，再将原始完整 `FieldProfile` 和 Evidence 交给现有 LLM Prompt。

## 8. 错误处理

- ValueProfiler 对空样例正常返回，不抛错；
- 无法识别格式时返回普通结构特征和空候选；
- Builder 始终能依靠非空字段名生成 Query；
- profiler 或 builder 的意外异常继续由 `classify_field()` 的现有异常边界捕获，返回 `UNKNOWN`；
- 不在新模块中记录样例日志，避免额外暴露数据。

## 9. 测试设计

新增 `tests/unit/test_value_profiler.py`，覆盖：

1. `138**1234`、`159**5678` 生成手机号码和联系方式候选；
2. `A1:B2:C3:D4:E5:F6` 生成 6 组十六进制、冒号分隔、MAC 地址和设备标识候选；
3. `test***@example.com` 生成邮箱和联系方式候选；
4. 普通商品名称不生成敏感个人信息候选；
5. 无意义字段名仍可依靠样例识别结构；
6. 其他必选格式各有参数化用例；
7. 普通数字、普通文本、空样例和最多 3 个候选的边界。

新增 `tests/unit/test_retrieval_query.py`，覆盖：

1. Query 固定格式和段落顺序；
2. 最多选择 3 个去重、格式有差异的样例；
3. 空候选时省略候选段；
4. Query 不含 `csv_source`、`catalog_input`、`general`、`unknown`、`field_cn` 等噪声。

修改 `tests/unit/test_services.py`：

- 更新旧测试中“Query 包含字段中文名”的断言；
- 验证 VectorStore 收到新 Query；
- 验证 LLM Prompt 仍包含完整字段画像和检索证据；
- 验证现有失败降级行为不变。

开发严格执行 Red-Green-Refactor。最终运行：

```powershell
python -m compileall app scripts
ruff check .
pytest -q
```

## 10. 验收标准

- 新 Query 仅由字段名、最多 3 个代表性样例、结构特征和最多 3 个候选类型构成；
- 用户要求的格式均能用纯 Python 规则识别；
- 不依赖字段中文名、注释、Ground Truth 或来源元数据；
- 普通商品名不会产生敏感个人信息候选；
- 现有 VectorStore、LLM Prompt 和各 Pipeline 接口保持兼容；
- 新增测试先失败后通过；
- compileall、Ruff 和全量测试通过。
