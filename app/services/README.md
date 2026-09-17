# services：业务处理

编排字段分类、数据输入、知识库处理和评分。

- `classification_service.py`：精简 Query → 法规 Top-3 → 结构化 LLM。
- `database_pipeline.py`、`csv_pipeline.py`：执行分类任务并保存结果。
- `csv_reader.py`、`csv_mode_detector.py`、`*_csv_adapter.py`：读取和识别 CSV，转换为统一字段画像。
- `benchmark_label_service.py`、`benchmark_evaluator.py`：处理评分标签及计算指标。
- `embedding_service.py`、`llm_service.py`：封装 Embedding 和分类模型。
- `knowledge_*.py`、`ocr_service.py`、`pdf_image_service.py`：法规提取、清洗、质量检查和入库。
