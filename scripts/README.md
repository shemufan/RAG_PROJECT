# scripts：命令行工具

在项目根目录使用 `python -m scripts.<模块名>` 运行，可添加 `--help` 查看参数。

- `run_csv_pipeline.py`：直接分类 CSV，可提供标签进行评分。
- `run_benchmark.py`：运行或恢复个人信息／非个人信息双文件评测。
- `rebuild_frozen_knowledge.py`：从冻结快照恢复知识库，仅允许写入空 Collection。
- `verify_frozen_knowledge.py`：检查活动冻结配置、知识块一致性及 Top-3 检索。
- `rebuild_knowledge_base.py`：提取和检查法规，构建独立候选库。
- `freeze_knowledge_snapshot.py`：将已有提取产物导出为新的内容快照。
