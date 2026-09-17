"""Conservative, auditable cleanup for extracted knowledge text."""

import re
from collections import Counter, defaultdict

from app.schemas.knowledge_quality import (
    CleanedKnowledge,
    CleaningAudit,
    ExtractedPage,
)

_PAGE_MARKER = re.compile(r"^【\s*第\s*\d+\s*页\s*】$")
_PAGE_NUMBER = re.compile(r"^(?:第\s*)?[-—–]?\s*\d{1,4}\s*[-—–]?(?:\s*页)?$")
_DECORATIVE = re.compile(r"^[\-—–_=·•]+$")
_CONTENTS_ENTRY = re.compile(
    r"^(?:\d+(?:\.\d+)*\s|[A-ZＡ-Ｚ]\.\d|附录\s|前\s*言|引\s*言|参考文献)"
    r".*(?:\.{3,}|…{2,}|．{3,}|·{3,})\s*(?:\d+|[IVXLCDM]+)?\s*$",
    re.IGNORECASE,
)
_CONTENTS_REVERSED = re.compile(
    r"^(?:\d+(?:\.\d+)*\s|[A-ZＡ-Ｚ]\.\d|附录\s|参考文献)"
    r".*\s\d+\s*(?:\.{3,}|…{2,}|．{3,}|·{3,})\s*$"
)
_TABLE_SEPARATOR = re.compile(r"^\|\s*(?:-+\s*\|\s*)+$")
_TABLE_SEPARATOR_CHARACTERS = re.compile(r"^[\s|:\-]+$")
_GENERATED_IMAGE = re.compile(r"^!\[[^]]*]\(attachment://[^)]+\)$", re.IGNORECASE)
_REAL_SECTION = re.compile(r"^1(?:\.0+)*\s+\S")


class KnowledgeCleaner:
    """Remove only position-aware, structurally certain page noise."""

    def clean(self, pages: list[ExtractedPage]) -> CleanedKnowledge:
        counters: Counter[str] = Counter()
        samples: dict[str, list[str]] = defaultdict(list)
        prepared: list[tuple[ExtractedPage, list[str]]] = []

        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            lines, toc_lines = self._without_table_of_contents(lines)
            for line in toc_lines:
                self._record(counters, samples, "table_of_contents", line)
            retained = []
            for line in lines:
                rule = self._certain_noise_rule(line)
                if rule:
                    self._record(counters, samples, rule, line)
                else:
                    retained.append(line)
            prepared.append((page, retained))

        margin_counts: Counter[str] = Counter()
        for _, lines in prepared:
            for line in set(lines[:1] + lines[-1:]):
                margin_counts[line] += 1
        repeated_margins = {
            line
            for line, count in margin_counts.items()
            if count >= 2 and count / max(len(pages), 1) >= 0.6
        }

        cleaned_pages = []
        for page, lines in prepared:
            retained = []
            last_index = len(lines) - 1
            for index, line in enumerate(lines):
                if line in repeated_margins and index in {0, last_index}:
                    self._record(counters, samples, "repeated_margin", line)
                else:
                    retained.append(line)
            cleaned_text = "\n".join(retained).strip()
            cleaned_pages.append(
                page.model_copy(
                    update={
                        "text": cleaned_text,
                        "is_blank": page.is_blank or (bool(page.text.strip()) and not cleaned_text),
                    }
                )
            )

        audits = [
            CleaningAudit(rule=rule, removed_count=count, samples=samples[rule])
            for rule, count in sorted(counters.items())
        ]
        text = "\n\n".join(page.text for page in cleaned_pages if page.text)
        return CleanedKnowledge(text=text, pages=cleaned_pages, audits=audits)

    @staticmethod
    def _certain_noise_rule(line: str) -> str | None:
        if (
            re.sub(r"\s+", "", line) in {"目次", "目录"}
            or _CONTENTS_ENTRY.fullmatch(line)
            or _CONTENTS_REVERSED.fullmatch(line)
        ):
            return "table_of_contents"
        if _PAGE_MARKER.fullmatch(line):
            return "page_marker"
        if _PAGE_NUMBER.fullmatch(line):
            return "standalone_page_number"
        if _DECORATIVE.fullmatch(line.replace(" ", "")):
            return "decorative_separator"
        if _TABLE_SEPARATOR.fullmatch(line) or (
            "|" in line
            and "-" in line
            and _TABLE_SEPARATOR_CHARACTERS.fullmatch(line)
        ):
            return "markdown_table_separator"
        if line.startswith("|") and not line.replace("|", "").strip():
            return "empty_markdown_table_row"
        if _GENERATED_IMAGE.fullmatch(line):
            return "generated_image_placeholder"
        return None

    @staticmethod
    def _without_table_of_contents(lines: list[str]) -> tuple[list[str], list[str]]:
        try:
            start = next(index for index, line in enumerate(lines)
                         if re.sub(r"\s+", "", line) in {"目次", "目录"})
        except StopIteration:
            return lines, []
        end = None
        for index in range(start + 1, len(lines)):
            line = lines[index]
            if re.sub(r"\s+", "", line) in {"前言", "引言"} or (
                _REAL_SECTION.match(line) and "…" not in line and "..." not in line
            ):
                end = index
                break
        if end is None:
            return lines, []
        return [*lines[:start], *lines[end:]], lines[start:end]

    @staticmethod
    def _record(
        counters: Counter[str],
        samples: dict[str, list[str]],
        rule: str,
        line: str,
    ) -> None:
        counters[rule] += 1
        if line not in samples[rule] and len(samples[rule]) < 5:
            samples[rule].append(line)
