# schemas：数据模型

使用 Pydantic 定义并校验 API 和业务处理的数据结构。

- `field.py`：统一字段画像 FieldProfile。
- `classification.py`：LLM 分类输出、法规依据及 API 响应。
- `pipeline.py`：数据库扫描请求、任务统计和分类记录。
- `csv_input.py`：CSV 输入批次、字段案例与标签匹配结果。
- `benchmark.py`：预测记录、任务汇总和评分模型。
- `knowledge_quality.py`：法规提取、清洗审计与质量报告模型。
