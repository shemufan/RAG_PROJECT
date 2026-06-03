# Git 团队协作规范

## 1. 分支说明

本项目采用以下分支结构：

```text
main
dev
feature/backend-api
feature/rag-baseline
feature/kb-data
feature/frontend-demo
```

### main 分支

`main` 分支为稳定版本分支，只存放可以正常运行和演示的代码。

禁止直接向 `main` 分支提交代码。

---

### dev 分支

`dev` 分支为开发集成分支。

各成员完成阶段性功能后，将自己的 feature 分支合并到 `dev` 分支。

每周集成和测试主要在 `dev` 分支进行。

---

### feature 分支

每位成员基于自己的任务创建独立功能分支。

命名方式：

```text
feature/模块名称
```

示例：

```text
feature/backend-api
feature/rag-baseline
feature/kb-data
feature/frontend-demo
```

---

## 2. 基本开发流程

每次开始开发前，先同步最新代码：

```bash
git checkout dev
git pull origin dev
```

创建自己的功能分支：

```bash
git checkout -b feature/your-module-name
```

开发完成后提交代码：

```bash
git add .
git commit -m "feat: add your feature"
git push origin feature/your-module-name
```

然后在 GitHub 上向 `dev` 分支发起 Pull Request。

---

## 3. Pull Request 规范

每次 PR 需要说明：

```text
1. 本次修改了什么？
2. 修改了哪些文件？
3. 是否影响其他模块？
4. 如何测试？
5. 是否需要其他成员配合？
```

PR 示例：

```text
标题：feat: add baseline classify api

修改内容：
1. 新增 /classify 接口。
2. 增加 ClassifyRequest 和 ClassifyResponse 数据结构。
3. 暂时使用 mock 数据返回分类结果。

修改文件：
- backend/api/classify.py
- backend/schemas/classify_schema.py
- backend/main.py

影响范围：
- 前端可以开始调用 /classify 接口。
- RAG 模块后续需要接入 classify_field 函数。

测试方式：
- 启动 FastAPI 后访问 /docs。
- 使用 Postman 或 curl 发送测试请求。
```

---

## 4. 合并规范

合并代码前必须确认：

1. 本地代码可以运行。
2. 没有明显报错。
3. 没有提交无关文件。
4. 没有修改他人模块但未说明。
5. 接口变更已经更新文档。

由项目负责人或对应模块负责人审核后再合并。

---

## 5. 建议提交频率

推荐提交粒度：

```text
完成一个小功能就提交一次。
修复一个 bug 就提交一次。
更新一份文档就提交一次。
```

不推荐：

```text
一次提交几十个无关文件。
提交信息写 update、test、aaa。
长时间本地开发但不 push。
```

不要等到一周结束才一次性提交。

---
