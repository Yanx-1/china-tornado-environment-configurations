"""Audit publication-facing figure text and formal captions for internal terms."""

from __future__ import annotations

import json
import re
from pathlib import Path

from figure_io import CAPTION_DIR, QC_DIR, write_json_atomic, write_text_atomic


REQUIRED_PATTERNS = {
    "Direction A": re.compile(r"\bDirection\s+A\b", re.IGNORECASE),
    "Direction B": re.compile(r"\bDirection\s+B\b", re.IGNORECASE),
    "Frozen": re.compile(r"\bfrozen\b", re.IGNORECASE),
    "Gate": re.compile(r"\bgate(?:\s*[0-9]+)?\b", re.IGNORECASE),
    "PASS_WITH_NONBLOCKING_NOTES": re.compile(
        r"\bPASS_WITH_NONBLOCKING_NOTES\b", re.IGNORECASE
    ),
    "Prototype": re.compile(r"\bprototype\b", re.IGNORECASE),
}
EXTENDED_PATTERNS = {
    **REQUIRED_PATTERNS,
    "Audit status": re.compile(r"\baudit\s+status\b", re.IGNORECASE),
    "Deprecated reporting number": re.compile(
        r"\bdeprecated\s+reporting\s+number\b", re.IGNORECASE
    ),
    "方向A": re.compile(r"方向\s*A", re.IGNORECASE),
    "方向B": re.compile(r"方向\s*B", re.IGNORECASE),
    "冻结": re.compile(r"冻结"),
    "门控": re.compile(r"门控"),
    "原型": re.compile(r"原型"),
    "审计状态": re.compile(r"审计状态"),
    "废止报告数字": re.compile(r"废止报告数字"),
}


def _load_publication_scope() -> tuple[dict[str, str], dict[str, str]]:
    inventory_dir = QC_DIR / "publication_visible_text"
    figure_texts: dict[str, str] = {}
    caption_texts: dict[str, str] = {}
    for number in range(1, 9):
        matches = sorted(inventory_dir.glob(f"Fig{number}_*_visible_text.json"))
        if len(matches) != 1:
            raise AssertionError(
                f"Fig.{number} requires one visible-text inventory; found {matches}"
            )
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        texts = payload.get("texts", [])
        if not isinstance(texts, list):
            raise TypeError(f"Invalid text inventory: {matches[0]}")
        figure_texts[f"Fig{number}"] = "\n".join(str(item) for item in texts)
        for language in ("en", "zh"):
            caption_path = CAPTION_DIR / f"Fig{number}_caption_{language}.md"
            if not caption_path.is_file():
                raise FileNotFoundError(caption_path)
            caption_text = caption_path.read_text(encoding="utf-8")
            if "caption draft" in caption_text.lower() or "图注草案" in caption_text:
                raise AssertionError(
                    f"Publication caption still declares draft status: {caption_path}"
                )
            caption_texts[f"Fig{number}_{language}"] = caption_text
    return figure_texts, caption_texts


def _count_terms(
    scope: dict[str, str],
) -> tuple[dict[str, int], dict[str, list[dict[str, object]]]]:
    counts = {term: 0 for term in EXTENDED_PATTERNS}
    occurrences: dict[str, list[dict[str, object]]] = {
        term: [] for term in EXTENDED_PATTERNS
    }
    for source, text in scope.items():
        for term, pattern in EXTENDED_PATTERNS.items():
            matches = list(pattern.finditer(text))
            counts[term] += len(matches)
            if matches:
                occurrences[term].append(
                    {
                        "source": source,
                        "count": len(matches),
                        "matches": [match.group(0) for match in matches],
                    }
                )
    return counts, occurrences


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def audit() -> dict[str, object]:
    figure_texts, caption_texts = _load_publication_scope()
    figure_counts, figure_occurrences = _count_terms(figure_texts)
    caption_counts, caption_occurrences = _count_terms(caption_texts)
    total_counts = {
        term: figure_counts[term] + caption_counts[term]
        for term in EXTENDED_PATTERNS
    }

    required_visible_text = {
        "Fig1": (
            "Scientific evidence framework",
            "Final tornado sample",
            "Multivariate environmental regimes and their stability",
            "Vertical thermodynamic and kinematic characteristics",
            "Temperature · humidity · environmental wind",
            "Post-hoc weather-type, seasonal, and spatial context",
            "Interpretation restricted to the confirmed-tornado sample",
        ),
        "Fig4": ("Event stability categories",),
        "Fig5": (
            "k=4 cluster",
            "k=3–k=4 structural correspondence (n=790)",
        ),
        "Fig8": ("Sensitivity analyses",),
    }
    missing_required: dict[str, list[str]] = {}
    for figure_id, phrases in required_visible_text.items():
        normalized = _normalize(figure_texts[figure_id])
        missing = [phrase for phrase in phrases if phrase not in normalized]
        if missing:
            missing_required[figure_id] = missing

    passed = all(value == 0 for value in total_counts.values()) and not missing_required
    report = {
        "status": (
            "PASS_INTERNAL_PROJECT_TERMS_REMOVED"
            if passed
            else "FAIL_INTERNAL_PROJECT_TERMS_PRESENT"
        ),
        "scope": {
            "figure_visible_text_inventories": sorted(figure_texts),
            "formal_captions": sorted(caption_texts),
            "excluded": [
                "scripts",
                "logs",
                "manifest",
                "QC prose other than this audit",
                "filenames",
            ],
        },
        "required_counts": {
            term: total_counts[term] for term in REQUIRED_PATTERNS
        },
        "extended_counts": total_counts,
        "figure_counts": figure_counts,
        "caption_counts": caption_counts,
        "occurrences": {
            "figures": {
                key: value for key, value in figure_occurrences.items() if value
            },
            "captions": {
                key: value for key, value in caption_occurrences.items() if value
            },
        },
        "required_replacement_text_missing": missing_required,
        "INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES": passed,
    }
    write_json_atomic(QC_DIR / "GATE3_publication_term_audit.json", report)

    lines = [
        "# Gate 3 Publication-Facing Terminology Audit",
        "",
        "```text",
        f"STATUS = {report['status']}",
        "SCOPE = Fig.1–Fig.8 rendered Text artists + 16 formal caption files",
    ]
    for term in REQUIRED_PATTERNS:
        lines.append(f"{term} count = {total_counts[term]}")
    lines.extend(
        [
            (
                "INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES = "
                f"{str(passed).lower()}"
            ),
            "```",
            "",
            "The audit excludes scripts, logs, manifest records, internal QC prose, "
            "and filenames, as required.",
            "",
            "Extended bilingual internal-term counts:",
            "",
        ]
    )
    lines.extend(f"- {term}: {count}" for term, count in total_counts.items())
    if missing_required:
        lines.extend(
            [
                "",
                "Missing required replacement text:",
                "",
                "```json",
                json.dumps(missing_required, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    write_text_atomic(
        QC_DIR / "GATE3_publication_term_audit.md", "\n".join(lines) + "\n"
    )
    if not passed:
        raise AssertionError(
            "Publication-facing terminology audit failed: "
            f"counts={total_counts}, missing={missing_required}"
        )
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
