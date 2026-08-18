"""Round 2 scientific and publication-facing QC helpers."""

from __future__ import annotations

import re
from pathlib import Path

from round2_io import write_text_atomic


PUBLIC_FORBIDDEN_PATTERNS = {
    "Direction A": re.compile(r"\bDirection\s+A\b", re.IGNORECASE),
    "Direction B": re.compile(r"\bDirection\s+B\b", re.IGNORECASE),
    "Gate": re.compile(r"\bgate(?:\s*[0-9]+)?\b", re.IGNORECASE),
    "PASS": re.compile(r"\bPASS(?:_[A-Z0-9_]+)?\b", re.IGNORECASE),
    "prototype": re.compile(r"\bprototype\b", re.IGNORECASE),
    "frozen": re.compile(r"\bfrozen\b", re.IGNORECASE),
    "audit": re.compile(r"\baudit\b", re.IGNORECASE),
    "deprecated reporting number": re.compile(
        r"\bdeprecated\s+reporting\s+number\b", re.IGNORECASE
    ),
    "tornado corridor": re.compile(r"\btornado\s+corridor\b", re.IGNORECASE),
    "risk corridor": re.compile(r"\brisk\s+corridor\b", re.IGNORECASE),
    "operational threshold": re.compile(
        r"\boperational\s+threshold\b", re.IGNORECASE
    ),
    "best cut point": re.compile(r"\bbest\s+cut\s+point\b", re.IGNORECASE),
    "standard STP": re.compile(r"\bstandard[-\s]+STP\b", re.IGNORECASE),
    "storm-relative hodograph": re.compile(
        r"\bstorm-relative\s+hodograph\b", re.IGNORECASE
    ),
}


def assert_exact(name: str, observed, expected) -> None:
    if observed != expected:
        raise AssertionError(f"{name}: expected {expected!r}, observed {observed!r}")


def assert_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(
            f"{name}: expected {expected} ± {tolerance}, observed {observed}"
        )


def public_term_counts(text: str) -> dict[str, int]:
    return {
        label: len(pattern.findall(text))
        for label, pattern in PUBLIC_FORBIDDEN_PATTERNS.items()
    }


def assert_public_text_clean(text: str, *, source: str) -> None:
    hits = {
        label: count
        for label, count in public_term_counts(text).items()
        if count
    }
    if hits:
        raise AssertionError(f"{source} contains publication-forbidden terms: {hits}")


def write_figure_qc(
    destination: Path,
    *,
    figure_id: str,
    status_label: str,
    scientific_checks: list[str],
    layout_checks: list[str],
    interpretation_checks: list[str],
    deviations: list[str],
    review_notes: list[str],
) -> None:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None."

    text = f"""# {figure_id} Round 2 QC

```text
STATUS_LABEL = {status_label}
OUTPUT_STAGE = 200_DPI_RGB_REVIEW_PROTOTYPE
NEW_SCIENTIFIC_ANALYSIS = FALSE
```

## Scientific-source checks

{bullets(scientific_checks)}

## Layout and visual checks

{bullets(layout_checks)}

## Interpretation boundary

{bullets(interpretation_checks)}

## Recorded layout deviations

{bullets(deviations)}

## Review notes

{bullets(review_notes)}
"""
    write_text_atomic(destination, text)
