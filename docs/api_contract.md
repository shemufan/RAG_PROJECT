# Data Compliance RAG API 接口文档

## 基础信息

- **Base URL**: `http://127.0.0.1:8000/api`
- **Content-Type**: `application/json`
- **Swagger UI**: `http://127.0.0.1:8000/docs`

---

## 1. 单字段分类

**POST** `/api/classify`

请求体:

```json
{
  "field_name": "account_code",
  "field_cn": "科目编码",
  "field_comment": "企业会计科目编码",
  "data_type": "varchar(50)",
  "sample_values": ["1001", "1002"],
  "business_domain": "finance",
  "table_name": "ledger",
  "database_name": "erp_db"
}
```

响应: `ClassificationResult`

```json
{
  "field_name": "account_code",
  "category": "财务数据",
  "subcategory": "会计科目",
  "level": "L2",
  "confidence": 0.85,
  "reason": "Step 1: ... Step 2: ... Step 3: ...",
  "evidence": [...],
  "need_review": false,
  "decision_path": "rag_llm",
  "raw_response": "{...}"
}
```

---

## 2. 批量分类

**POST** `/api/batch-classify`

上传 Excel/CSV 文件 (multipart/form-data)，`file` 字段。

响应: `BatchClassifyResponse`

```json
{
  "total_fields": 50,
  "completed": 50,
  "log": "开始执行批量评测任务...\n...\n报告生成完毕。",
  "summary": "整体准确率: 85.00%",
  "report_path": "outputs/数据定级全量评测报告.xlsx"
}
```

---

## 3. 错题本上传

**POST** `/api/error-log`

上传错题本 Excel/CSV (multipart/form-data)，`file` 字段。

响应: `ErrorLogUploadResponse`

```json
{
  "rules_activated": 12,
  "message": "已激活 12 条强制拦截规则。"
}
```

---

## 4. 知识库更新

**POST** `/api/knowledge-base`

上传法规 TXT 文件 (multipart/form-data)，`files` 字段（可多文件）。

响应:

```json
{
  "message": "知识库更新成功！新增 25 个知识区块并已持久化到本地。"
}
```

---

## 5. MySQL Schema 扫描

**POST** `/api/mysql-scan`

Query 参数: `host`, `port`, `user`, `password`, `database`

响应:

```json
{
  "fields_count": 120,
  "log": "连接成功！数据库 'erp_db' 共 15 张表、120 个字段。\n...",
  "tables": ["employees", "finance", "products", ...]
}
```

---

## 6. MySQL 全量分类

**POST** `/api/mysql-classify`

Query 参数: 同上

响应:

```json
{
  "log": "开始 MySQL 批量评测...\n...\n报告生成完毕。",
  "summary": "MySQL 批量分类完成。需复核率: 15.00%",
  "report_path": "outputs/MySQL数据定级评测报告.xlsx"
}
```
