# Week3 后端接口规范

## 基础信息

| 项目 | 值 |
|------|-----|
| Base URL | `http://127.0.0.1:8000/api` |
| Content-Type | `application/json` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| 响应格式 | 统一格式 `{code, message, data}` |

所有接口统一返回 JSON，结构为：

```json
{
  "code": 200,
  "message": "success",
  "data": { ... }
}
```

- `code=200` 表示成功，`code=500` 表示服务端异常。
- `message` 为人类可读的提示信息。
- `data` 为具体业务数据，无数据时为 `null`。

---

## 1. 健康检查

### GET /api/health

用途：服务存活探测，供前端或运维监控调用。

请求：无参数。

响应示例：

```json
{
  "code": 200,
  "message": "backend is running",
  "data": null
}
```

---

## 2. 单字段分类

### POST /api/classify

用途：对单个数据库字段执行分类定级，返回敏感等级和判定依据。

### 请求 JSON

```json
{
  "field_name": "user_phone",
  "field_comment": "用户手机号",
  "table_name": "t_user",
  "sample_value": "13800138000"
}
```

### 请求字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `field_name` | string | **是** | 数据库字段名（英文） |
| `field_comment` | string | 否 | 字段注释 / 业务含义说明 |
| `table_name` | string | 否 | 所属表名 |
| `sample_value` | string | 否 | 样例值，用于辅助判断字段语义 |

### 成功响应（code=200）

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "level": "L2",
    "reason": "样例值包含邮箱/网址特征，可能属于个人联系方式。",
    "matched_rules": ["个人信息保护法 第4条"],
    "references": ["《中华人民共和国个人信息保护法》"],
    "confidence": 0.7,
    "need_manual_review": false
  }
}
```

### 失败响应（code=500）

```json
{
  "code": 500,
  "message": "classify failed: 具体错误信息",
  "data": {
    "level": "UNKNOWN",
    "reason": "系统处理失败，建议人工复核。",
    "matched_rules": [],
    "references": [],
    "confidence": 0.0,
    "need_manual_review": true
  }
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `level` | string | 敏感等级：`L1`(公开) / `L2`(内部) / `L3`(敏感) / `L4`(核心敏感) / `UNKNOWN`(异常兜底) |
| `reason` | string | 判定理由简述 |
| `matched_rules` | string[] | 命中的法规条款，如 `["个人信息保护法 第4条"]` |
| `references` | string[] | 引用的法规全称，如 `["《中华人民共和国个人信息保护法》"]` |
| `confidence` | float | 置信度，范围 0.0 ~ 1.0 |
| `need_manual_review` | bool | 是否需要人工复核；`true` 时前端应提示用户 |

---

## 前端调用注意事项

1. **统一响应解析**：所有接口返回 `{code, message, data}`。先判断 `code`，`code === 200` 时读取 `data`，否则展示 `message` 中的错误信息。
2. **兜底展示**：`data.level === "UNKNOWN"` 时，建议显示「系统判定失败，请人工复核」并高亮提醒。
3. **复核标记**：`need_manual_review === true` 时，前端应给该结果添加醒目的复核提示。
4. **置信度展示**：建议将 `confidence` 转为百分比显示（如 `0.7` → `70%`），低于 60% 时标黄。
5. **超时处理**：分类接口可能需要较长时间（后续接入 RAG 后），建议前端设置 30s 超时。

---

## RAG 模块对接说明

RAG 负责人**只需要实现** `backend/services/rag_classifier.py` 中的一个函数：

```python
class RAGClassifier:
    def classify_field(self, field_info: dict) -> dict:
        ...
```

- **输入** `field_info`：字典，键为 `field_name`、`field_comment`、`table_name`、`sample_value`。
- **输出**：字典，键为 `level`、`reason`、`matched_rules`、`references`、`confidence`、`need_manual_review`。
- **异常**：函数内部可抛出异常，API 层会自动捕获并返回 500 兜底响应。
- **不要修改** `backend/api/` 和 `backend/schemas/` 目录下的任何文件。

---

## 接口变更规范

1. 任何字段新增、删除、重命名、类型变更，**必须先更新本文档**。
2. 文档更新后，需在团队群内通知**前端负责人**和 **RAG 负责人**。
3. 前后端以本文档为唯一接口约定，禁止私下协商偏离文档的字段。
4. 变更提交时，commit message 需包含 `docs: api_contract` 前缀。
