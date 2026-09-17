# rag：检索与提示词

保存法规 RAG 的基础处理逻辑。

- `retrieval_query.py`：只用字段名和样例值构建检索 Query。
- `prompt.py`：将完整字段画像和法规依据组织为结构化分类 Prompt。
- `chunker.py`：按章节、条款等结构切分法规，生成来源元数据及稳定 chunk ID。
