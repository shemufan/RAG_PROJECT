# 贡献指南

- 分支使用 `feature/<name>`、`fix/<name>` 或 `refactor/<name>` 命名。
- commit 使用 `feat:`、`fix:`、`refactor:`、`test:`、`docs:` 或 `chore:` 前缀，并保持一次提交只表达一个目的。
- PR 必须说明变更范围、验证命令和结果；行为变化应包含测试。
- 提交前运行 `ruff check .` 和 `pytest -q`。
- 禁止提交 `.env`、虚拟环境、缓存、日志、运行时数据库、报告及原始导入数据。
- 不得直接向 `main` 推送；通过分支和 PR 合并。
