# app：应用代码

实现字段分类分级 API 和业务处理。`main.py` 创建 FastAPI 应用、加载法规向量库及分类服务。

- `api/`：HTTP 接口。
- `core/`：环境变量和配置加载。
- `rag/`：检索 Query、分类 Prompt 和法规分块。
- `repositories/`：MySQL 与 Chroma 数据访问。
- `schemas/`：输入、输出及内部数据模型。
- `services/`：分类、输入处理、知识库构建与评分逻辑。
