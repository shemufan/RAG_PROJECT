# B RAG 冻结记录：b-rebuild-20260917

## 冻结范围

默认分类采用普通法规 RAG：`field_name + sample_values` → 法规 Top-3 → 完整
FieldProfile 与依据 → 结构化 LLM。Web 单字段、数据库扫描、CSV 和旧 Benchmark
入口均继承 clean 默认策略，不执行 Value Profiling 或 Semantic Bridge。
历史实验代码保留用于复现；显式选择 A/C/D/E 时不属于本冻结方案。

采用 knowledge-rebuild 的提取、保守清洗、质量检查和候选库构建实现。
删除目录条目、目录标题和装饰短线，保留孤立标题、短块及规范正文。

## 不可变内容快照

`data/knowledge_snapshots/b-rebuild-20260917/` 保存 manifest.json 和 chunks.jsonl。
包含分类规则与四份标准/指引，共 260 块，最大 1600 字符；manifest 保存源文件
哈希、输入提取文本哈希、清洗审计、来源质量状态及完整清洗文本。

快照复用了 database-pipeline 工作区已有的提取产物，再执行本次目录/短线清洗。
没有重新调用 OCR 或分类 LLM。由于输入是合并后的清洗文本，快照不提供原始页码范围；
从 PDF 重新提取的 rebuild 流程仍支持页码元数据。

## 人工内容验收

项目负责人于 2026-09-17 明确确认已人工核对 GB/T 41391 的 OCR 内容。
manifest 中 human_review.status 已记录为 APPROVED，并绑定源文件 SHA-256。
自动提取的历史 REVIEW 状态与问题记录保留；review_pending 已清空。
其余四个来源的历史报告为 PASS。

恢复脚本读取持久化的人工复核记录，无须再次传入审批哈希：

```powershell
python -m scripts.rebuild_frozen_knowledge `
  --snapshot data/knowledge_snapshots/b-rebuild-20260917
```

脚本验证快照哈希并写入空的 `data_classification__b-rebuild-20260917` 候选库，
不自动替换活动库。使用与冻结时一致的本地 Embedding 模型重新生成向量。
候选库验证后，将 CHROMA_COLLECTION 与 KNOWLEDGE_BASE_VERSION 分别切换至
`data_classification__b-rebuild-20260917`、`b-rebuild-20260917`，并重启服务。

2026-09-17 已完成本工作区的新库重构和 .env 切换，原配置备份位于
`.runtime/knowledge-switch-backup.env`（包含凭据，保持 Git 忽略）。旧 Collection 保留。
260 块文本与快照逐块一致，五组 B Query 检索均返回 Top-3；检索可用性检查
不代表相关性或分类准确率评测。详细运行报告为 `.runtime/frozen_knowledge_verification.json`。

## 验证范围

单元/API 测试覆盖默认 clean Query、目录和短线清洗、规范正文与标题保留、
快照完整性、恢复时的 REVIEW 门槛，以及现有实验功能的兼容性。
未重新执行付费分类 Benchmark，因此不宣称最终 F1 得到提升。
模型、Embedding 路径、凭据及实际向量文件是部署配置，不随 Git 代码自动同步。
