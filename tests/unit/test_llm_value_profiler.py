import json
from types import SimpleNamespace

from app.services.llm_value_profiler import LLMValueProfiler
from app.services.value_profiler import ValueProfile


class FakeStructuredModel:
    def __init__(self, output=None, error: Exception | None = None):
        self.output = output
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.output


class BasicStatsOnly:
    def __init__(self):
        self.profile_calls = 0

    def basic_statistics(self, sample_values: list[str]) -> list[str]:
        return ["字符串长度约9位", "存在脱敏字符"]

    def profile(self, field_name: str, sample_values: list[str]):
        self.profile_calls += 1
        raise AssertionError("rule profile fallback must not run")


def fake_settings(tmp_path):
    return SimpleNamespace(
        project_root=tmp_path,
        deepseek_api_key="test-key",
        deepseek_base_url="https://example.test/v1",
        deepseek_model="deepseek-test",
        deepseek_timeout_seconds=30,
        deepseek_max_retries=2,
    )


def test_llm_profiler_uses_only_allowed_inputs_and_normalizes_output(tmp_path):
    model = FakeStructuredModel(
        {
            "features": [" 长度一致 ", "长度一致", "包含脱敏字符"],
            "candidate_types": [" 手机号码 ", "联系方式", "手机号码"],
            "confidence": 0.8,
        }
    )
    profiler = LLMValueProfiler(
        structured_model=model,
        settings=fake_settings(tmp_path),
        cache_dir=tmp_path / "cache",
    )

    result = profiler.profile("contact_attr", ["138**1234", "159**5678"])

    assert result.features == ["长度一致", "包含脱敏字符"]
    assert result.candidate_types == ["手机号码", "联系方式"]
    assert result.confidence == 0.8
    payload = json.loads(model.calls[0][1].content)
    assert set(payload) == {"field_name", "sample_values", "basic_statistics"}
    assert payload["field_name"] == "contact_attr"
    assert "字符串长度约9位" in payload["basic_statistics"]
    for forbidden in (
        "field_cn",
        "field_comment",
        "database_name",
        "table_name",
        "business_domain",
        "expected_personal",
        "evidence",
    ):
        assert forbidden not in model.calls[0][1].content


def test_llm_profiler_reuses_valid_content_hash_cache(tmp_path):
    model = FakeStructuredModel(
        {"features": ["存在脱敏字符"], "candidate_types": ["联系方式"]}
    )
    profiler = LLMValueProfiler(
        structured_model=model,
        settings=fake_settings(tmp_path),
        cache_dir=tmp_path / "cache",
    )
    samples = ["138**1234", "159**5678"]

    first = profiler.profile("contact_attr", samples)
    second = profiler.profile("contact_attr", samples)

    assert first == second
    assert len(model.calls) == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_llm_profiler_failure_returns_empty_without_rule_fallback(
    tmp_path,
    caplog,
):
    model = FakeStructuredModel(error=RuntimeError("service unavailable"))
    basic_profiler = BasicStatsOnly()
    cache_dir = tmp_path / "cache"
    profiler = LLMValueProfiler(
        structured_model=model,
        settings=fake_settings(tmp_path),
        cache_dir=cache_dir,
        basic_profiler=basic_profiler,
    )

    result = profiler.profile("contact_attr", ["138**1234"])

    assert result == ValueProfile()
    assert basic_profiler.profile_calls == 0
    assert not list(cache_dir.glob("*.json"))
    assert "138**1234" not in caplog.text


def test_llm_profiler_replaces_corrupted_cache(tmp_path):
    model = FakeStructuredModel(
        {"features": ["长度一致"], "candidate_types": ["联系方式"]}
    )
    profiler = LLMValueProfiler(
        structured_model=model,
        settings=fake_settings(tmp_path),
        cache_dir=tmp_path / "cache",
    )
    payload = profiler._build_payload("contact_attr", ["138**1234"])
    cache_path = profiler._cache_path(payload)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not-json", encoding="utf-8")

    result = profiler.profile("contact_attr", ["138**1234"])

    assert result.candidate_types == ["联系方式"]
    assert len(model.calls) == 1
    assert ValueProfile.model_validate_json(
        cache_path.read_text(encoding="utf-8")
    ) == result


def test_llm_profiler_skips_call_when_samples_are_empty(tmp_path):
    model = FakeStructuredModel(
        {"features": ["不应使用"], "candidate_types": ["不应使用"]}
    )
    profiler = LLMValueProfiler(
        structured_model=model,
        settings=fake_settings(tmp_path),
        cache_dir=tmp_path / "cache",
    )

    result = profiler.profile("empty_attr", ["", "   "])

    assert result == ValueProfile()
    assert model.calls == []
