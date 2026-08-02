from scripts.rebuild_knowledge_base import load_documents


def test_rebuild_script_loads_documents_from_configured_directory(tmp_path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (tmp_path / "classification_rules.md").write_text(
        "### 规则 1：手机号\n- 等级: L3",
        encoding="utf-8",
    )

    documents = load_documents(tmp_path, version="v1")

    assert len(documents) == 1
    assert "手机号" in documents[0].page_content
