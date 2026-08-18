"""Shared scientific and file-level QC helpers for Gate 1 prototypes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from figure_io import sha256_file, write_text_atomic


FORBIDDEN_PHRASES = (
    "tornado probability",
    "tornado risk",
    "most dangerous regime",
    "tornado-favorable ranking",
    "operational threshold",
    "tornado corridor",
    "natural and mutually exclusive classes",
    "independent validation",
    "causal mechanism",
    "low-level jet",
    "upper-level jet",
    "storm-relative hodograph",
    "standard stp",
    "tornado versus non-tornado discrimination",
    "clearly proves",
    "validates",
    "most favorable",
    "risk level",
)


def assert_exact(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise AssertionError(
            f"{name}: expected {expected!r}, observed {observed!r}"
        )


def assert_close(
    name: str, observed: float, expected: float, tolerance: float
) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(
            f"{name}: expected {expected} ± {tolerance}, "
            f"observed {observed}"
        )


def scan_forbidden_language(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    return sorted(
        phrase for phrase in FORBIDDEN_PHRASES if phrase in normalized
    )


def inspect_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", ("", ""))
        return {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "alpha_present": "A" in image.getbands(),
            "dpi_x": dpi[0] if isinstance(dpi, tuple) and dpi else "",
            "dpi_y": dpi[1] if isinstance(dpi, tuple) and len(dpi) > 1 else "",
            "file_size_bytes": path.stat().st_size,
        }


def write_qc_report(
    destination: Path,
    *,
    figure_id: str,
    status: str,
    data_checks: list[str],
    number_checks: list[str],
    interpretation_checks: list[str],
    visual_checks: list[str],
    nonblocking_notes: list[str],
) -> None:
    all_text = "\n".join(
        data_checks
        + number_checks
        + interpretation_checks
        + visual_checks
        + nonblocking_notes
    )
    prohibited = scan_forbidden_language(all_text)
    if prohibited:
        raise AssertionError(
            f"{figure_id} QC text contains prohibited phrases: {prohibited}"
        )

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None."

    report = f"""# {figure_id} Gate 1 QC Report

```text
STATUS = {status}
STAGE = GATE1_LOW_RESOLUTION_SCIENTIFIC_PROTOTYPE
SCIENTIFIC_REANALYSIS = FORBIDDEN
PLOTTING_TRANSFORMATION = ALLOWED_AND_LOGGED
```

## QC-1 Data integrity

{bullets(data_checks)}

## QC-2 Number consistency

{bullets(number_checks)}

## QC-3 Interpretation boundary

{bullets(interpretation_checks)}

## QC-4 Visual prototype review

{bullets(visual_checks)}

## Nonblocking notes

{bullets(nonblocking_notes)}
"""
    write_text_atomic(destination, report)
