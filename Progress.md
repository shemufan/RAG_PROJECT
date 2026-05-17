# 工作进度报告 — 2026/05/11

## 一、今日工作概览

对「基于 RAG 与 LLM 的企业数据智能分类分级系统」项目进行了全面审计与重构，共发现并修复 **10 项问题**（3 严重 + 4 中等 + 3 轻微），外加编写答辩提纲、函数注释、指标讲解等辅助工作。

## 二、已修改文件清单

| 文件 | 状态 | 变更说明 |
|------|------|----------|
| `core/config.py` | 修改 | MODEL_PATH 改为环境变量优先 + Windows 默认值 |
| `core/engine.py` | 修改 | 废弃 API 迁移、空库保护、正则级别解析、setter 函数、logging、docstring |
| `core/evaluator.py` | 修改 | setter 配合、logging 替代 print、删除多余 pass、docstring |
| `core/__init__.py` | **新增** | 标准 Python 包 |
| `ui/app.py` | 修改 | Gradio 6.0 theme 迁移 + linter 调整 |
| `ui/__init__.py` | **新增** | 标准 Python 包 |
| `main.py` | 修改 | logging.basicConfig + theme 迁移 + linter 调整 |
| `requirements.txt` | 重写 | 219 行 → 13 行直接依赖 |
| `.gitignore` | 修改 | `*.csv/*.xlsx` → `/*.csv/*.xlsx`（仅根目录） |
| `Construct.md` | **新增** | 答辩讲解提纲（九章） |

**合计:** 10 文件变更（+361 行 / -260 行），分布在 2 个 commit 中。

## 三、问题修复明细

### 严重（3 项）

| 问题 | 修复 |
|------|------|
| config.py 硬编码 Linux 路径 `/root/...`，Windows 下不可用 | `os.getenv("SENTENCE_TRANSFORMER_PATH", r"G:\AI_Models\...")` |
| `HuggingFaceEmbeddings` / `Chroma` 使用 langchain_community 废弃 API | 迁移至 `langchain-huggingface` + `langchain-chroma` |
| Chroma 空库时 `similarity_search()` 直接崩溃 | `smart_predict` 入口增加 `_collection.count() == 0` 检查 |

### 中等（4 项）

| 问题 | 修复 |
|------|------|
| evaluator.py 跨模块直接修改 `ERROR_LOG_CACHE` | 新增 `set_error_log_cache()` setter 函数 |
| requirements.txt 是 pip freeze 全量导出（219 行） | 精简为 13 个直接依赖，版本用 `>=` 下限 |
| Gradio 6.0 `gr.Blocks(theme=...)` 已废弃 | theme 移至 `launch(theme=gr.themes.Soft())` |
| 级别解析用 `if "L4" in response` 字符串包含 | 正则优先匹配 `最终定级[：:]\s*(L[1-4])`，回退取末次 `\bL[1-4]\b` |

### 轻微（3 项）

| 问题 | 修复 |
|------|------|
| core/ 和 ui/ 目录缺少 `__init__.py` | 创建两个空文件 |
| .gitignore 中全局 `*.csv/*.xlsx` 会拦截 testdata/ | 改为 `/*.csv` / `/*.xlsx` 仅匹配根目录 |
| 日志全部用 `print()` 无级别无时间戳 | 全部替换为标准 `logging`，main.py 加 `basicConfig` |

## 四、当前分支状态

```
分支: fix/core-improvements（领先 main 2 个 commit）
远程: origin/fix/core-improvements ✓ 已推送

Commit 1 (a7d8b01): fix: 修复核心引擎10项问题
Commit 2 (0afcdcf): feat: 核心函数添加结构化注释 + 项目答辩提纲文档

main 分支: d01661a（未改动）
```

## 五、当前工作状态

- [x] 项目结构：模块化四层架构（ui → core → data → infra），package 规范化
- [x] 兼容性：跨 Windows/Linux 路径，Gradio 6.0 API，新版 langchain-* 包
- [x] 鲁棒性：空数据库保护、全局状态封装、正则级别解析
- [x] 可维护性：标准 logging、精简依赖、docstring、.gitignore 精确化
- [x] 答辩准备：Construct.md 九章提纲
- [ ] 合并到 main：待用户确认后合入
- [ ] 功能测试：尚未在 Gradio 网页上实际跑通全流程

## 六、下次工作建议

### 目标 1：端到端功能验证（优先级最高）

> 启动 `main.py`，在 Gradio Web 界面上跑通以下路径：
>   a) 单条语义测试：输入一个字段，验证检索 + 推理 + 级别解析链路正常
>   b) 批量自动化评测：上传一份 testdata 下的 Excel，验证多 Sheet 报告正确导出
>   c) 错题本拦截：上传 `强制拦截错题本.xlsx`，验证命中规则后跳过 LLM
>   d) 知识库更新：上传一份新 txt，验证向量库增量写入
>
> 这一步能发现代码层面察觉不到的问题（模型路径、API key、Chroma 兼容性等）。

### 目标 2：Prompt 与评测指标基线校准

> 用 7 个业务域的测试集各跑一遍批量评测，记录每个领域的准确率 + L4 召回率，
> 建立当前版本的性能基线。如果某个领域准确率异常低，针对性分析 Prompt 或知识库
> 覆盖是否不足。
