# Week3 Baseline API 契约

Base URL：`http://127.0.0.1:8000/api`。所有业务响应统一使用
`{code, message, data}`；HTTP 参数校验仍使用 FastAPI 标准 422 响应。

## 健康检查

`GET /api/health`

```json
{"code": 200, "message": "backend is running", "data": null}
```

## 单字段分类

`POST /api/classify`

请求中仅 `field_name` 必填：

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

成功响应：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "field_name": "id_card",
    "category": "敏感个人信息",
    "subcategory": "身份标识",
    "level": "L4",
    "confidence": 0.92,
    "reason": "依据检索条款得出的分类理由。",
    "evidence": [
      {
        "content": "检索到的原文片段",
        "source": "个人信息保护法.txt",
        "article": "第二十八条",
        "score": 0.91
      }
    ],
    "need_review": false,
    "decision_path": "rag_llm"
  }
}
```

字段约束：

- `level`：`L1`、`L2`、`L3`、`L4` 或失败兜底 `UNKNOWN`。
- `confidence`：0 到 1。
- `need_review`：依据不足、检索失败或模型失败时为 `true`。
- `decision_path`：正常链路为 `rag_llm`；RAG/LLM 失败为 `rag_llm_error`；API 未捕获异常为 `api_error`。
- `evidence`：最多返回 Top-3 检索片段，供前端直接展示。

业务失败仍返回 HTTP 200，便于 Week3 前端统一解析，但 `code=500`、
`data.level="UNKNOWN"` 且 `data.need_review=true`。请求字段不合法时返回 HTTP 422。

## 模块调用约定

API 通过 FastAPI `Depends` 获取 lifespan 中初始化的 `RAGClassifier`，禁止在路由模块中
直接创建空分类器。前端必须调用本接口，不得直接导入后端 RAG 服务。
