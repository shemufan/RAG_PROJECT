# PDF 分页图片 OCR 设计规格

## 目标

优先读取法规 PDF 的原生文字层；无文字层的页面先渲染并检测是否为空白，只有非空白
扫描页才通过 Base64 调用 `qwen3.5-ocr`。分页图片保留在 PDF 同名目录中但不提交到
Git，完整文档、空白页标记与成功的单页 OCR 文本缓存到 `.runtime/ocr_cache/`。

## 范围

本次仅修改法规知识库的 PDF OCR 输入方式，保留以下既有接口和流程：

- `QwenOCRService.extract_pdf(pdf_path)` 调用方式；
- PDF 校验、法规加载、文本分块和 Chroma 重建流程；
- Qwen OCR API Key、模型、超时和重试配置；
- 不依赖真实 API 的测试原则。

本次不引入 OSS，不修改分类服务，也不修改 A/B 数据库流程。

## 文件与职责

### `app/services/pdf_image_service.py`

新增 `PdfImageService`，只负责 PDF 页面渲染和本地图片复用：

- 校验并接收已经确认可处理的 PDF；
- 使用 `pypdfium2` 按 200 DPI 选择性渲染需要 OCR 的页面；
- 以 JPEG 质量 90 输出 `page-0001.jpg` 等顺序文件；
- 将图片写入 PDF 旁边的同名目录；
- 在 `.pdf-pages.json` 记录 PDF SHA-256、页数、DPI、格式和渲染版本；
- 清单匹配且所有页面存在时复用图片；
- PDF 或渲染参数变化时，只清理该同名目录内由本服务生成的分页图片和清单，然后重新渲染；
- 不调用 OCR API，不处理知识库分块。

示例：

```text
data/knowledge/laws/
├── 个人信息安全规范.pdf
└── 个人信息安全规范/
    ├── page-0001.jpg
    ├── page-0002.jpg
    └── .pdf-pages.json
```

### `app/services/ocr_service.py`

保留 `QwenOCRService` 作为 PDF OCR 编排器，职责调整为：

1. 校验 PDF；
2. 优先读取现有 OCR 文本缓存；
3. 缓存未命中时逐页读取 PDF 原生文字；
4. 对无文字的页面调用 `PdfImageService.render_pages()`，同时裁剪历史残留图片；
5. 跳过纯白页，将非空白扫描页构造为 `data:image/jpeg;base64,...`；
6. 通过 OpenAI 兼容的 `/chat/completions` 调用 `qwen3.5-ocr`；
7. 成功后立即写入单页 OCR 缓存；
8. 按页码顺序合并后原子写入完整文档缓存。

删除当前临时上传策略、OSS 上传和 `Responses API input_file` 代码。

## 数据流

```text
PDF
  -> PDF 校验
  -> OCR 文本缓存命中？
       -> 是：直接返回文本
       -> 否：逐页读取原生文字
              -> 无有效文字的页面生成图片并 Base64 OCR
              -> 写入单页 OCR 缓存
              -> 按页合并
              -> 写入文本缓存
  -> 法规分块
  -> Chroma 重建
```

## 缓存规则

- 文本缓存键继续包含 PDF 内容、模型名称和 OCR 提示词版本；
- 文本缓存命中时，不检查或生成分页图片，也不调用 API；
- 单页 OCR 成功后立即缓存，后续页面失败不会丢失已完成结果；
- 只删除完整文档缓存时，复用有效的原生文字和单页 OCR 缓存；
- PDF 内容变化时，分页清单失效并重新渲染；
- 模型或提示词变化时，分页图片仍可复用，但文本缓存失效；
- 只有所有页面 OCR 成功后才写入最终文本缓存。

## API 请求约束

- 每次请求只发送一页，保证页面顺序、错误定位和重试边界清楚；
- 图片使用 Base64 Data URL，不依赖本地路径、OSS 或公开 URL；
- 单张 Base64 数据超过 10 MB 时停止并报告 PDF 名称和页码；
- OCR 输出为空或包含已知的“未提供文档”拒绝文本时视为失败；
- 错误信息包含 PDF 名称、页码、HTTP 状态或服务端错误码，但不包含 API Key 和完整请求体。

## 生成文件与 Git

`.gitignore` 精确加入：

```gitignore
data/knowledge/laws/*/page-*.jpg
data/knowledge/laws/*/.pdf-pages.json
```

法规 PDF 不忽略；不增加 `*.jpg`、`*.json` 等宽泛规则。

## 依赖

运行依赖新增：

- `pypdfium2`：PDF 页面渲染；
- `Pillow`：JPEG 生成和质量控制。

确认 `requests` 未被其他模块使用后删除，因为不再需要手工上传 PDF。
`pypdf` 继续用于页数、加密和 PDF 有效性校验。

## 错误处理

- 加密、损坏、空页数、超过 50 页或超过 100 MB 的 PDF 继续提前拒绝；
- 页面渲染失败时报告具体 PDF 和页码；
- 任一页面 OCR 失败时终止当前 PDF，不写文本缓存；
- 已生成的分页图片保留，下一次重建可以复用；
- 所有知识文档成功加载前不重置 Chroma，保持现有安全顺序。

## 测试与验收

单元测试使用临时 PDF、fake 渲染器和 mock Chat Completions 客户端，覆盖：

1. PDF 同名目录和顺序分页文件生成；
2. 有效清单复用分页图片；
3. PDF 变化后分页图片重建；
4. Base64 图片请求结构；
5. 多页结果顺序合并；
6. 页级错误信息；
7. OCR 失败不写文本缓存；
8. 文本缓存命中时不渲染、不请求；
9. PDF 法规加载和知识库重建保持有效；
10. 真实 Qwen 测试默认跳过，只有显式配置时运行。

最终必须通过：

```powershell
python -m compileall app scripts
ruff check .
pytest -q
```

## 实施边界

- 在 `feature/database-pipeline` worktree 中 Inline Execution；
- 不使用多 Agent；
- 保留当前未提交的法规 PDF、旧 TXT 删除和其他用户修改；
- 不调用真实 OCR 作为自动测试成功条件；
- 不添加 OSS、数据库或其他无关业务代码。
