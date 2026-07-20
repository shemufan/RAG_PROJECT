# 项目开发规范文档

## 1. 文档目的

本文档用于规范本项目在开发过程中的目录结构、接口设计、代码提交、分支管理、模块边界和团队协作方式。

所有成员在开发过程中应遵守本文档，避免出现重复开发、接口不一致、随意修改他人模块、代码难以合并等问题。

---

## 2. 项目总体协作原则

本项目采用“模块负责人 + 接口契约 + Git 分支协作”的方式进行开发。

基本原则如下：

1. 每个人负责一个相对独立的模块。
2. 模块之间通过明确的接口进行调用。
3. 不直接修改他人负责的核心文件。
4. 所有新功能在独立分支开发，不直接提交到 main 分支。
5. 每次提交代码前，保证本地代码可以正常运行。
6. 合并代码前，需要说明本次修改内容和影响范围。
7. 出现接口变更时，必须同步修改接口文档，并通知相关成员。

---

## 3. 项目目录规范

项目基础目录如下：

```text
project/
├── backend/
│   ├── main.py
│   ├── api/
│   │   └── classify.py
│   ├── services/
│   │   ├── rag_classifier.py
│   │   ├── embedding_service.py
│   │   └── llm_service.py
│   ├── storage/
│   │   ├── mysql.py
│   │   └── chroma_store.py
│   ├── schemas/
│   │   └── classify_schema.py
│   ├── utils/
│   │   ├── file_loader.py
│   │   └── chunker.py
│   └── core/
│       └── config.py
│
├── frontend/
│   └── ...
│
├── data/
│   ├── knowledge_docs/
│   └── testdata/
│
├── docs/
│   ├── development_standard.md
│   ├── git_workflow.md
│   ├── api_contract.md
│   └── weekly/
│
├── README.md
└── requirements.txt
```

---

## 4. 模块边界规范

### 4.1 后端接口模块

负责人主要维护：

```text
backend/main.py
backend/api/
backend/schemas/
```

主要职责：

1. 定义 FastAPI 接口。
2. 接收前端请求。
3. 校验请求参数。
4. 调用 service 层逻辑。
5. 返回统一 JSON 响应。

后端接口负责人不应随意修改：

```text
backend/services/rag_classifier.py
backend/storage/chroma_store.py
data/
frontend/
```

---

### 4.2 RAG / LLM 流程模块

负责人主要维护：

```text
backend/services/rag_classifier.py
backend/services/embedding_service.py
backend/services/llm_service.py
```

主要职责：

1. 将字段信息拼接为查询文本。
2. 调用 embedding 模型完成向量化。
3. 从向量库中检索 Top-K 相关规则。
4. 构造 Prompt。
5. 调用 LLM 生成分类结果。
6. 按统一格式返回分类等级、原因、命中规则、参考依据、置信度和人工复核建议。

RAG 负责人不应随意修改：

```text
backend/api/
frontend/
data/sample_fields.csv
```

如需要新增输入字段或修改输出格式，必须先修改 `docs/api_contract.md` 并通知后端和前端负责人。

---

### 4.3 知识库、规则与测试集模块

负责人主要维护：

```text
data/
backend/storage/chroma_store.py
backend/storage/mysql.py
```

主要职责：

1. 整理分类分级规则。
2. 整理字段词典。
3. 整理测试字段样例。
4. 构建 Chroma 向量库。
5. 维护 MySQL 测试数据。
6. 为后续评估准备人工标注数据。

知识库负责人不应随意修改：

```text
backend/api/
backend/services/rag_classifier.py
frontend/
```

---

### 4.4 前端展示模块

负责人主要维护：

```text
frontend/
```

主要职责：

1. 实现字段输入页面。
2. 调用后端分类接口。
3. 展示分类结果。
4. 展示分类原因、命中规则、置信度、人工复核建议。
5. 后续支持批量上传、日志展示、方案切换等功能。

前端负责人不应随意修改：

```text
backend/services/
backend/storage/
data/
```

如接口不满足展示需求，通知后端接口负责人统一调整。

---

## 5. 接口设计规范

所有接口必须遵守以下规范：

1. 请求参数和响应结果必须写入 `docs/api_contract.md`。
2. 接口路径采用小写英文，单词之间使用下划线或短横线。
3. 接口返回结果必须包含状态码、提示信息和数据主体。
4. 前端不得直接依赖后端内部变量名，只依赖接口文档。
5. 接口变更必须提前通知所有成员。

统一响应格式示例：

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

---

## 6. 代码提交规范

```text
feat: 新增功能
fix: 修复问题
docs: 修改文档
style: 代码格式调整
refactor: 代码重构
test: 增加测试
chore: 其他杂项修改
```

---

## 7. 文件修改规范

每位成员只修改自己负责范围内的文件。

如果需要修改他人负责的文件，应遵守以下流程：

1. 先在群里说明修改原因。
2. 与对应负责人确认。
3. 修改后在提交说明中写清楚影响范围。
4. 合并前由对应负责人检查。

禁止行为：

1. 未沟通直接修改他人核心模块。
2. 为了临时跑通代码随意改接口格式。
3. 将测试代码、临时文件、无关文件提交到仓库。
4. 直接向 main 分支提交代码。
5. 提交无法运行的代码但不说明原因。

---

## 8. 每周交付规范(负责人整理)

每周结束时进行整理总结：

1. 本周完成内容。
2. 本周未完成内容。
3. 当前系统运行方式。
4. 当前存在的问题。
5. 下周计划。
6. 每位成员的实际贡献。

每周总结文档放在：

```text
docs/weekly/
```

---
