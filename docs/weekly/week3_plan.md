# Week3 任务细化说明：方案 A Baseline 设计

## 1. 本周目标

Week3 的目标是完成方案 A baseline 的最小闭环。

即实现：

```text
输入字段信息
↓
后端接收请求
↓
RAG 检索相关规则或知识
↓
LLM 输出分类结果
↓
前端展示分类等级和依据
```

本周不追求分类效果最优，只要求流程可以跑通，并形成可演示的 baseline。

---

## 2. 本周不做的内容

为了避免任务过大，Week3 暂不实现以下内容：

```text
1. 复杂规则引擎
2. 多知识库检索
3. Redis 缓存
4. 完整日志系统
5. 大规模测试集评估
6. 复杂前端交互
```

这些内容放到后续 Week4、Week5、Week7 之后逐步完成。

---

## 3. 本周统一接口

### 3.1 请求格式

接口路径：

```text
POST /api/classify
```

请求 JSON：

```json
{
  "field_name": "id_card",
  "field_cn": "身份证号",
  "field_comment": "用户身份证号码",
  "data_type": "varchar(18)",
  "sample_values": ["340************1234"],
  "business_domain": "customer",
  "table_name": "user_info",
  "database_name": "user_db"
}
```

字段说明：

| 字段名            | 类型     | 是否必填 | 说明     |
| -------------- | ------ | ---- | ------ |
| field_name     | string | 是    | 字段英文名  |
| field_cn       | string | 否    | 字段中文名  |
| field_comment  | string | 否    | 字段业务描述 |
| data_type      | string | 否    | 数据类型    |
| sample_values  | list   | 否    | 样例值列表   |
| business_domain| string | 否    | 业务域（默认 general） |
| table_name     | string | 否    | 所属表名   |
| database_name  | string | 否    | 所属数据库名 |

---

### 3.2 响应格式

统一响应 JSON（Week3 Baseline 简化版，完整 schema 见 `backend/schemas/classify_schema.py`）：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "level": "L4",
    "category": "敏感个人信息",
    "reason": "字段名称和注释表明该字段为身份证号，属于强身份识别信息。",
    "matched_rules": [
      "身份证号属于个人敏感信息"
    ],
    "references": [
      "企业数据分类分级规则：个人身份识别信息"
    ],
    "confidence": 0.92,
    "need_review": false
  }
}
```

响应字段说明：

| 字段名                | 类型      | 说明                  |
| ------------------ | ------- | ------------------- |
| level              | string  | 分类等级，例如 L1、L2、L3、L4 |
| category           | string  | 数据分类（个人信息/财务数据/...） |
| reason             | string  | 分类原因                |
| matched_rules      | list    | 命中的规则               |
| references         | list    | 参考依据                |
| confidence         | float   | 置信度                 |
| need_review        | boolean | 是否建议人工复核            |

---

## 4. 本周分工

### 后端接口(佘沐繁)

负责分支：

```text
feature/backend-api
```

负责文件：

```text
backend/main.py
backend/api/classify.py
backend/schemas/classify_schema.py
docs/api_contract.md
```

主要任务：

1. 搭建 FastAPI 项目入口。
2. 实现 `/classify` 接口。
3. 定义请求和响应数据结构。
4. 前期可以先返回 mock 数据。
5. 后续接入 RAG 负责人提供的 `classify_field()` 函数。
6. 维护接口文档。

不应修改：

```text
backend/services/rag_classifier.py
backend/db/chroma_store.py
data/
frontend/
```

交付物：

```text
1. 可访问的 /classify 接口
2. Swagger 文档可正常显示
3. 接口请求和响应格式稳定
```

---

### 知识库、规则与测试集(宋张志恒)

负责分支：

```text
feature/kb-data
```

负责文件：

```text
data/sample_fields.csv
data/rules.md
data/knowledge_docs/
backend/db/chroma_store.py
backend/db/mysql.py
```

主要任务：

1. 整理 20-30 条字段测试样例。
2. 整理 5-10 条基础分类规则。
3. 准备知识库文本。
4. 编写或维护 Chroma 向量库写入脚本。
5. 准备简单 MySQL 测试数据。
6. 保证 RAG 模块可以检索到规则文本。

不应修改：

```text
backend/api/
backend/schemas/
backend/services/rag_classifier.py
frontend/
```

交付物：

```text
1. sample_fields.csv
2. rules.md
3. 可被 Chroma 加载的知识库文本
4. 简单向量库构建脚本
```

---

### 成员 C：RAG / LLM 流程(郭佳慧)

负责分支：

```text
feature/rag-baseline
```

负责文件：

```text
backend/services/rag_classifier.py
backend/services/embedding_service.py
backend/services/llm_service.py
```

主要任务：

1. 实现字段信息拼接。
2. 调用 embedding 模型。
3. 从 Chroma 中检索 Top-K 规则。
4. 构造 LLM Prompt。
5. 调用 LLM 输出分类结果。
6. 将结果整理为统一 JSON 格式。
7. 向后端接口负责人提供 `classify_field()` 函数。

建议函数形式：

```python
# 完整签名（最终对齐目标）
from backend.schemas.classify_schema import FieldProfile, ClassificationResult

def classify_field(field: FieldProfile) -> ClassificationResult:
    pass

# Week3 Baseline 可先用简化版 dict 接口，后续迭代对齐
def classify_field_baseline(field_info: dict) -> dict:
    pass
```

输入示例（FieldProfile）：

```python
from backend.schemas.classify_schema import FieldProfile

field = FieldProfile(
    field_name="id_card",
    field_cn="身份证号",
    field_comment="用户身份证号码",
    table_name="user_info",
    sample_values=["340************1234"],
    business_domain="customer",
)
```

输出示例（ClassificationResult）：

```python
{
    "field_name": "id_card",
    "category": "敏感个人信息",
    "level": "L4",
    "confidence": 0.92,
    "reason": "字段名称和注释表明该字段为身份证号，属于强身份识别信息。",
    "evidence": [...],
    "need_review": False,
    "decision_path": "rag_llm"
}
```

不应修改：

```text
backend/api/classify.py
backend/schemas/classify_schema.py
frontend/
data/sample_fields.csv
```

交付物：

```text
1. classify_field() 核心函数
2. 可运行的 RAG baseline 流程
3. 至少 3 个字段的测试结果
```

---

### 成员 D：前端展示(仲亦帆)

负责分支：

```text
feature/frontend-demo
```

负责文件：

```text
frontend/
```

主要任务：

1. 实现字段输入页面。
2. 支持输入字段名、字段注释、表名、示例值。
3. 调用后端 `/classify` 接口。
4. 展示分类等级。
5. 展示分类原因。
6. 展示命中规则。
7. 展示置信度。
8. 展示是否建议人工复核。

前期如果后端接口未完成，可以使用 mock JSON 开发。

不应修改：

```text
backend/
data/
docs/api_contract.md
```

交付物：

```text
1. 字段输入页面
2. 分类结果展示页面
3. 可以调用后端接口
```

---

## 5. Week3 过程安排

### 1：规范与接口确定

```text
1. 确定目录结构
2. 确定接口输入输出格式
3. 创建 Git 分支
4. 每个人明确负责文件
```

---

### 2：各模块独立开发

后端：

```text
完成 /classify 接口，先返回 mock 数据。
```

知识库：

```text
完成 sample_fields.csv 和 rules.md。
```

RAG：

```text
完成 classify_field() 的初版逻辑。
```

前端：

```text
完成输入和结果展示页面，可以先对接 mock 数据。
```

---

### 3：第一次集成

目标：

```text
前端可以调用后端接口。
后端可以调用 RAG 函数。
RAG 可以返回分类结果。
```

允许结果不准确，但流程必须跑通。

---

### 4：修复问题与完善演示

任务：

```text
1. 修复接口调用问题
2. 修复数据格式问题
3. 准备几个演示字段
4. 补充 README 运行说明
```

---

### 5：总结与提交

任务：

```text
1. 整理 Week3 完成情况
2. 记录存在的问题
3. 合并 dev 分支
4. 准备 Week4 评估任务
```

产出：

```text
docs/progress/week3_summary.md
可运行 baseline
```

---

## 6. 本周产出标准

Week3 结束时应达到：

```text
1. 用户可以输入一个字段。
2. 后端可以接收字段信息。
3. 系统可以检索相关规则。
4. LLM 可以输出分类等级和依据。
5. 前端可以展示分类结果。
6. Git 分支合并到 dev 后代码可以运行。
```

---
