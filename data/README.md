# data：知识库来源与快照

保存用于构建法规知识库的版本化内容。

- `knowledge/classification_rules.md`：字段分类分级规则。
- `knowledge/laws/`：法规与标准原文。
- `knowledge_snapshots/`：冻结内容快照；`manifest.json` 记录版本、哈希和审计信息，`chunks.jsonl` 保存知识块。

当前主线使用 `b-rebuild-20260917` 快照，共 260 块。已冻结快照保持不变；本地向量库保存在根目录 `.runtime/chroma`。冻结记录见 [RAG_FREEZE.md](../docs/RAG_FREEZE.md)。
