from app.schemas.knowledge_quality import ExtractedPage
from app.services.knowledge_cleaner import KnowledgeCleaner


def test_cleaner_removes_directory_entries_and_short_decorative_lines():
    page = ExtractedPage(
        page_number=1, extraction_method="native",
        text="4 敏感个人信息识别和界定 2……………………\n"
        "3 数据分类分级原则 ............ 3\n附录 A 示例 …… 12\n"
        "-\n— —\n1 范围\n本文件规定要求。\n4 敏感个人信息识别和界定\nA.1 示例\n短标题",
    )
    result = KnowledgeCleaner().clean([page])
    assert "目录" not in result.text
    assert "……" not in result.text
    assert "............" not in result.text
    assert "-" not in result.text
    assert "—" not in result.text
    for title in ("1 范围", "4 敏感个人信息识别和界定", "A.1 示例", "短标题"):
        assert title in result.text


def test_cleaner_preserves_normative_lines_with_ellipsis():
    page = ExtractedPage(page_number=1, extraction_method="native",
                         text="1 范围\n处理要求……应当执行\n保留期限为 3 年。")
    assert "处理要求……应当执行" in KnowledgeCleaner().clean([page]).text


def test_cleaner_removes_spaced_contents_and_reference_entries():
    page = ExtractedPage(page_number=1, extraction_method="native",
                         text="目  次\n参考文献 11……………………\n前  言\n"
                         "本文件按照规则起草。\n1 范围\n正文\n参考文献")
    result = KnowledgeCleaner().clean([page])
    assert "目  次" not in result.text
    assert "11……………………" not in result.text
    assert "前  言" in result.text
    assert "参考文献" in result.text


def test_cleaner_removes_page_noise_but_preserves_normative_text():
    pages = [
        ExtractedPage(
            page_number=1,
            extraction_method="native",
            text=(
                "GB/T 00000—2026\n【第 1 页】\n1\n1 范围\n"
                "处理者应保护数据\n---"
            ),
        ),
        ExtractedPage(
            page_number=2,
            extraction_method="native",
            text=(
                "GB/T 00000—2026\n2\n2 要求\n"
                "处理者不得泄露数据\n---"
            ),
        ),
    ]

    result = KnowledgeCleaner().clean(pages)

    assert "GB/T 00000—2026" not in result.text
    assert "【第 1 页】" not in result.text
    assert "\n1\n" not in f"\n{result.text}\n"
    assert "处理者应保护数据" in result.text
    assert "处理者不得泄露数据" in result.text
    assert "1 范围" in result.text
    assert {audit.rule for audit in result.audits} >= {
        "repeated_margin",
        "page_marker",
        "standalone_page_number",
        "decorative_separator",
    }


def test_cleaner_keeps_clause_numbers_appendices_and_table_text():
    pages = [
        ExtractedPage(
            page_number=1,
            extraction_method="ocr",
            text=(
                "5.2.3 最小必要\n个人信息处理者宜采取措施\n"
                "附录 A\nA.1 数据类型\n表 1 字段说明\n序号 1 手机号码"
            ),
        )
    ]

    result = KnowledgeCleaner().clean(pages)

    for expected in ("5.2.3", "宜采取措施", "附录 A", "A.1", "表 1", "序号 1"):
        assert expected in result.text


def test_cleaner_does_not_remove_repeated_table_header_from_middle_of_pages():
    pages = [
        ExtractedPage(
            page_number=1,
            extraction_method="ocr",
            text="1 范围\n序号 数据类型 要求\n1 手机号码 应保护\n页尾内容",
        ),
        ExtractedPage(
            page_number=2,
            extraction_method="ocr",
            text="2 要求\n序号 数据类型 要求\n2 邮箱 不得泄露\n另一页尾",
        ),
    ]

    result = KnowledgeCleaner().clean(pages)

    assert result.text.count("序号 数据类型 要求") == 2


def test_cleaner_marks_page_blank_when_it_contains_only_certain_noise():
    pages = [
        ExtractedPage(
            page_number=1,
            extraction_method="native",
            text="GB/T 00000—2026\n1 范围\n正文",
        ),
        ExtractedPage(
            page_number=2,
            extraction_method="native",
            text="GB/T 00000—2026\n----------------\n38",
        ),
    ]

    result = KnowledgeCleaner().clean(pages)

    assert result.pages[1].text == ""
    assert result.pages[1].is_blank is True


def test_cleaner_excludes_table_of_contents_but_keeps_real_preface():
    page = ExtractedPage(
        page_number=1,
        extraction_method="ocr",
        text=(
            "标准封面\n目次\n前言 … III\n1 范围 … 1\n附录 A … 10\n"
            "前言\n本文件按照标准规则起草。\n1 范围\n本文件规定要求。"
        ),
    )

    result = KnowledgeCleaner().clean([page])

    assert "前言 … III" not in result.text
    assert "1 范围 … 1" not in result.text
    assert "本文件按照标准规则起草" in result.text
    assert "1 范围\n本文件规定要求" in result.text
    assert any(audit.rule == "table_of_contents" for audit in result.audits)


def test_cleaner_removes_empty_markdown_table_rows_and_separators():
    page = ExtractedPage(
        page_number=1,
        extraction_method="ocr",
        text=(
            "1 范围\n| 权限 | 说明 |\n| --- | --- |\n"
            "| 读取日历 | 获取日程 |\n|  |  |\n|  |  |"
        ),
    )

    result = KnowledgeCleaner().clean([page])

    assert "| 权限 | 说明 |" in result.text
    assert "| 读取日历 | 获取日程 |" in result.text
    assert "| --- | --- |" not in result.text
    assert "|  |  |" not in result.text


def test_cleaner_removes_truncated_markdown_table_separator():
    page = ExtractedPage(
        page_number=1,
        extraction_method="ocr",
        text="1 范围\n|------|----------|--------------------",
    )

    result = KnowledgeCleaner().clean([page])

    assert result.text == "1 范围"
    assert any(
        audit.rule == "markdown_table_separator" for audit in result.audits
    )


def test_cleaner_removes_generated_markdown_image_placeholders():
    page = ExtractedPage(
        page_number=1,
        extraction_method="ocr",
        text=(
            "B.2 权限要求\n"
            "![页面中的示意图](attachment://figure.png)\n"
            "图 B.1 权限流程"
        ),
    )

    result = KnowledgeCleaner().clean([page])

    assert "attachment://" not in result.text
    assert "B.2 权限要求" in result.text
    assert "图 B.1 权限流程" in result.text
    assert any(
        audit.rule == "generated_image_placeholder" for audit in result.audits
    )
