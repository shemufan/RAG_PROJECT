# repositories：数据访问

封装数据库和向量库操作，供业务服务调用。

- `source_mysql.py`：读取 A 库字段元数据和脱敏样例。
- `target_mysql.py`：保存、查询数据库扫描任务、字段资产、分类结果和依据。
- `benchmark_target.py`：保存、查询 CSV 分类任务、预测及评分。
- `vector_store.py`：向 Chroma 写入法规知识块并检索依据。
