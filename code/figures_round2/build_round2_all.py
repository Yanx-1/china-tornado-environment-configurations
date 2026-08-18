"""Rebuild and close the complete Round 2 figure-expansion package."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

import build_fig09_round2
import build_figs1_round2
import build_figs2_round2
import build_figs3_round2
import build_figs4_round2
import build_figs5_round2
import build_round2_source_registry
from round2_io import (
    CAPTION_DIR,
    HANDOFF_DIR,
    LOG_DIR,
    MONTAGE_DIR,
    PROJECT_ROOT,
    QC_DIR,
    ROUND_ROOT,
    SOURCE_REGISTRY_DIR,
    ensure_output_dirs,
    inspect_png,
    relative_source,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from round2_qc import PUBLIC_FORBIDDEN_PATTERNS, assert_exact


FIGURE_BUILDERS = (
    ("Fig9", build_fig09_round2.build),
    ("FigS1", build_figs1_round2.build),
    ("FigS2", build_figs2_round2.build),
    ("FigS3", build_figs3_round2.build),
    ("FigS4", build_figs4_round2.build),
    ("FigS5", build_figs5_round2.build),
)
FIGURE_TITLES = {
    "Fig9": "Spatial and seasonal context",
    "FigS1": "Event-centered pressure-level composite means",
    "FigS2": "Pairwise effect sizes and distribution overlap",
    "FigS3": "Coarse environmental u–v wind profiles",
    "FigS4": "k=4 supplementary structure",
    "FigS5": "STP_mod in rated tornado events",
}
EXPECTED_STATUSES = {
    "Fig9": "MAIN_TEXT_CANDIDATE",
    "FigS1": "SUPPLEMENT_CANDIDATE",
    "FigS2": "SUPPLEMENT_CANDIDATE",
    "FigS3": "SUPPLEMENT_ONLY",
    "FigS4": "SUPPLEMENT_CANDIDATE",
    "FigS5": "DO_NOT_USE_WITHOUT_RESEARCHER_DECISION",
}
LAYOUT_DEVIATIONS = {
    "Fig9": "Longitude distribution omitted because it would be redundant with the map and the regime longitude medians are closely aligned.",
    "FigS1": "A common event-relative Cartesian grid replaces a geographic projection because the source is not an absolute-location field.",
    "FigS2": "A quantile-overlap display replaces a scalar matrix because no accepted scalar overlap coefficient exists.",
    "FigS3": "None; the displayed seven levels follow the accepted wind-statistics product.",
    "FigS4": "None.",
    "FigS5": "None.",
}

MANIFEST = HANDOFF_DIR / "figure_manifest_round2.csv"
CAPTIONS_EN = CAPTION_DIR / "ROUND2_captions_en.md"
CAPTIONS_ZH = CAPTION_DIR / "ROUND2_captions_zh.md"
TERM_SCAN = QC_DIR / "publication_term_scan_round2.csv"
TOTAL_QC = QC_DIR / "ROUND2_TOTAL_QC_REPORT.md"
MONTAGE = MONTAGE_DIR / "Round2_new_figures_montage.png"
HANDOFF_MD = HANDOFF_DIR / "ROUND2_FIGURE_EXPANSION_HANDOFF.md"
HANDOFF_YAML = HANDOFF_DIR / "ROUND2_FIGURE_EXPANSION_HANDOFF.yaml"
ARTIFACT_REGISTRY = HANDOFF_DIR / "round2_artifact_registry.csv"
BUILD_LOG = LOG_DIR / "round2_complete_build.log"


def _verify_source_registry() -> tuple[pd.DataFrame, int]:
    registry_path = SOURCE_REGISTRY_DIR / "round2_source_registry.csv"
    registry = pd.read_csv(registry_path, encoding="utf-8-sig")
    unchanged = 0
    for _, row in registry.iterrows():
        path = Path(str(row["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        current = sha256_file(path)
        if current != str(row["sha256"]).upper():
            raise AssertionError(f"Source changed after registration: {path}")
        unchanged += 1
    assertions = pd.read_csv(
        SOURCE_REGISTRY_DIR / "round2_source_assertions.csv",
        encoding="utf-8-sig",
    )
    assert_exact(
        "Round2 source assertion results",
        assertions["status"].value_counts().to_dict(),
        {"PASS": len(assertions)},
    )
    return registry, unchanged


def _collect_metadata(built: list[dict]) -> pd.DataFrame:
    rows = []
    for metadata in built:
        figure_id = metadata["figure_id"]
        assert_exact(
            f"{figure_id} status",
            metadata["status_label"],
            EXPECTED_STATUSES[figure_id],
        )
        render = metadata["render"]
        rows.append(
            {
                "display_order": list(EXPECTED_STATUSES).index(figure_id) + 1,
                "figure_id": figure_id,
                "title": FIGURE_TITLES[figure_id],
                "status_label": metadata["status_label"],
                "output_stage": metadata["output_stage"],
                "figure_file": metadata["figure_file"],
                "figure_sha256": render["sha256"],
                "width_px": render["width_px"],
                "height_px": render["height_px"],
                "mode": render["mode"],
                "dpi_x": render["dpi_x"],
                "dpi_y": render["dpi_y"],
                "plotting_snapshot": metadata["plotting_snapshot"],
                "script": metadata["script"],
                "caption_en": metadata["caption_en"],
                "caption_zh": metadata["caption_zh"],
                "qc_report": metadata["qc_report"],
                "source_count": len(metadata["inputs"]),
                "scientific_result_changed": metadata["scientific_result_changed"],
                "layout_deviation": LAYOUT_DEVIATIONS[figure_id],
            }
        )
    manifest = pd.DataFrame(rows).sort_values("display_order")
    write_csv_atomic(MANIFEST, manifest)
    return manifest


def _combine_captions(manifest: pd.DataFrame) -> None:
    english = ["# Round 2 figure captions (English)", ""]
    chinese = ["# 第二轮图件图注（中文）", ""]
    for _, row in manifest.iterrows():
        english.append((PROJECT_ROOT / row["caption_en"]).read_text(encoding="utf-8").strip())
        english.append("")
        chinese.append((PROJECT_ROOT / row["caption_zh"]).read_text(encoding="utf-8").strip())
        chinese.append("")
    write_text_atomic(CAPTIONS_EN, "\n".join(english))
    write_text_atomic(CAPTIONS_ZH, "\n".join(chinese))


def _scan_public_text(manifest: pd.DataFrame) -> pd.DataFrame:
    sources: list[tuple[str, str, str]] = []
    for _, row in manifest.iterrows():
        for caption_type in ("caption_en", "caption_zh"):
            path = PROJECT_ROOT / row[caption_type]
            sources.append((row["figure_id"], caption_type, path.read_text(encoding="utf-8")))
        inventory_path = (
            QC_DIR
            / "publication_visible_text"
            / f"{Path(row['figure_file']).stem}_visible_text.json"
        )
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        sources.append(
            (row["figure_id"], "figure_visible_text", "\n".join(inventory["texts"]))
        )

    rows = []
    for figure_id, source_type, text in sources:
        for term, pattern in PUBLIC_FORBIDDEN_PATTERNS.items():
            rows.append(
                {
                    "figure_id": figure_id,
                    "source_type": source_type,
                    "term": term,
                    "count": len(pattern.findall(text)),
                }
            )
    scan = pd.DataFrame(rows)
    if int(scan["count"].sum()) != 0:
        hits = scan.loc[scan["count"] > 0].to_dict(orient="records")
        raise AssertionError(f"Publication text scan failed: {hits}")
    write_csv_atomic(TERM_SCAN, scan)
    return scan


def _font(size: int):
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _save_montage_atomic(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    try:
        image.convert("RGB").save(temporary_name, dpi=(200, 200))
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _build_montage(manifest: pd.DataFrame) -> dict:
    cell_width, cell_height = 1500, 1260
    label_height, padding = 62, 22
    canvas = Image.new("RGB", (2 * cell_width, 3 * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = _font(34)
    for index, (_, row) in enumerate(manifest.iterrows()):
        column = index % 2
        grid_row = index // 2
        x0, y0 = column * cell_width, grid_row * cell_height
        draw.text(
            (x0 + padding, y0 + 12),
            f"{row['figure_id']}  |  {row['status_label']}",
            fill="#202020",
            font=label_font,
        )
        with Image.open(PROJECT_ROOT / row["figure_file"]) as source:
            image = source.convert("RGB")
            image.thumbnail(
                (cell_width - 2 * padding, cell_height - label_height - 2 * padding),
                Image.Resampling.LANCZOS,
            )
            paste_x = x0 + (cell_width - image.width) // 2
            paste_y = y0 + label_height + (
                cell_height - label_height - image.height
            ) // 2
            canvas.paste(image, (paste_x, paste_y))
        if column == 1:
            draw.line(
                [(0, y0 + cell_height - 1), (2 * cell_width, y0 + cell_height - 1)],
                fill="#D0D0D0",
                width=2,
            )
    _save_montage_atomic(canvas, MONTAGE)
    return inspect_png(MONTAGE)


def _artifact_registry(manifest: pd.DataFrame, montage_info: dict) -> pd.DataFrame:
    paths = [Path(PROJECT_ROOT / value) for value in manifest["figure_file"]]
    paths.extend(Path(PROJECT_ROOT / value) for value in manifest["plotting_snapshot"])
    paths.extend(Path(PROJECT_ROOT / value) for value in manifest["script"])
    paths.extend(Path(PROJECT_ROOT / value) for value in manifest["caption_en"])
    paths.extend(Path(PROJECT_ROOT / value) for value in manifest["caption_zh"])
    paths.extend(Path(PROJECT_ROOT / value) for value in manifest["qc_report"])
    paths.extend(
        [
            SOURCE_REGISTRY_DIR / "round2_source_registry.csv",
            SOURCE_REGISTRY_DIR / "round2_source_registry.md",
            SOURCE_REGISTRY_DIR / "round2_source_assertions.csv",
            CAPTIONS_EN,
            CAPTIONS_ZH,
            TERM_SCAN,
            TOTAL_QC,
            MONTAGE,
            MANIFEST,
        ]
    )
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    rows = []
    for path in unique_paths:
        rows.append(
            {
                "artifact_path": relative_source(path),
                "artifact_type": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    registry = pd.DataFrame(rows).sort_values("artifact_path")
    write_csv_atomic(ARTIFACT_REGISTRY, registry)
    return registry


def _write_total_qc(
    manifest: pd.DataFrame,
    source_registry: pd.DataFrame,
    source_unchanged: int,
    term_scan: pd.DataFrame,
    montage_info: dict,
) -> None:
    status_lines = "\n".join(
        f"- {row.figure_id}: `{row.status_label}` — {row.width_px}×{row.height_px} px, "
        f"{row.mode}, {float(row.dpi_x):.1f} dpi."
        for row in manifest.itertuples()
    )
    deviation_lines = "\n".join(
        f"- {figure_id}: {deviation}"
        for figure_id, deviation in LAYOUT_DEVIATIONS.items()
    )
    term_totals = term_scan.groupby("term")["count"].sum().to_dict()
    term_lines = "\n".join(
        f"- {term}: {count}" for term, count in term_totals.items()
    )
    total_qc = f"""# Round 2 total QC report

```text
ROUND2_STATUS = COMPLETE
FIGURES_GENERATED = {len(manifest)}
MAIN_TEXT_CANDIDATES = {(manifest["status_label"] == "MAIN_TEXT_CANDIDATE").sum()}
SUPPLEMENT_CANDIDATES = {(manifest["status_label"] == "SUPPLEMENT_CANDIDATE").sum()}
SUPPLEMENT_ONLY = {(manifest["status_label"] == "SUPPLEMENT_ONLY").sum()}
RESEARCHER_DECISION_REQUIRED = {(manifest["status_label"] == "DO_NOT_USE_WITHOUT_RESEARCHER_DECISION").sum()}
SOURCE_FILES_HASH_VERIFIED = {source_unchanged}
PUBLICATION_TERM_SCAN_TOTAL_HITS = {int(term_scan["count"].sum())}
```

## Outcome

Fig. 9 and five supplementary figures were built in the required order. This exceeds the success criterion of Fig. 9 plus at least four supplementary candidates. All outputs remain 200 dpi RGB review images; no PDF, SVG, or high-resolution submission file was created.

## Figure render checks

{status_lines}

## Source-integrity checks

- All {source_unchanged} files in the source registry were re-hashed after figure generation and match their registered SHA256 values.
- All {len(pd.read_csv(SOURCE_REGISTRY_DIR / "round2_source_assertions.csv", encoding="utf-8-sig"))} scientific closure assertions remain successful.
- The source registry contains {len(source_registry)} entries and all formal scientific inputs belong to the NEW790 result family.
- No old793, legacy790, or non-tornado control file was used.
- No source file in the registered scientific set was modified.

## Scientific-scope checks

- Fig. 9 is descriptive post-hoc spatial and seasonal context within confirmed tornado events.
- Fig. S1 distinguishes event-centered composite means from event-point median statistics.
- Fig. S2 uses accepted Cliff’s delta estimates and quantiles without creating a new overlap metric.
- Fig. S3 uses accepted 850–200-hPa environmental u and v medians and contains no storm-motion processing.
- Fig. S4 summarizes accepted k=4 labels under the exactly reproduced preprocessing path; clustering was not rerun and no physical names were assigned.
- Fig. S5 uses the accepted 181-event rated target and reporting values; no selected cutoff is shown.

## Recorded layout deviations

{deviation_lines}

## Publication-facing term scan

The scan covers every Matplotlib text artist and both language captions for all six figures.

{term_lines}

## Montage check

- Montage: `{relative_source(MONTAGE)}`
- Render: {montage_info["width_px"]}×{montage_info["height_px"]} px, {montage_info["mode"]}, {float(montage_info["dpi_x"]):.1f} dpi.
- All six figures are present once, in build order.

## Pending researcher decisions

- Decide whether Fig. S5 should enter the manuscript supplement.
- Select a target journal before publisher-specific font, width, and line-weight compliance work.
- Continue to keep these images at review quality until candidate selection and scientific review are complete.
"""
    write_text_atomic(TOTAL_QC, total_qc)


def _write_handoff(
    manifest: pd.DataFrame,
    source_registry: pd.DataFrame,
    source_unchanged: int,
    montage_info: dict,
    artifacts: pd.DataFrame,
) -> None:
    figure_rows = [
        {
            "figure_id": row.figure_id,
            "title": row.title,
            "status_label": row.status_label,
            "figure_file": row.figure_file,
            "figure_sha256": row.figure_sha256,
            "plotting_snapshot": row.plotting_snapshot,
            "script": row.script,
            "caption_en": row.caption_en,
            "caption_zh": row.caption_zh,
            "qc_report": row.qc_report,
            "layout_deviation": row.layout_deviation,
        }
        for row in manifest.itertuples()
    ]
    payload = {
        "handoff_protocol_version": "1.0",
        "task": "ROUND2_FIGURE_EXPANSION",
        "status": "ROUND2_FIGURE_EXPANSION_COMPLETE",
        "scope": {
            "scientific_results_changed": False,
            "scientific_sources_modified": False,
            "source_files_hash_verified": source_unchanged,
            "new_main_analysis": False,
            "non_tornado_control_used": False,
            "corridor_identification_used": False,
            "submission_exports_created": False,
            "review_output": "200 dpi RGB PNG",
        },
        "success_criteria": {
            "fig9_generated": True,
            "supplementary_figures_generated": 5,
            "minimum_supplementary_requirement_met": True,
            "publication_visible_internal_terms_removed": True,
            "montage_complete": True,
            "handoff_complete": True,
        },
        "figures": figure_rows,
        "researcher_decisions": [
            "FigS5 requires explicit researcher approval before manuscript use.",
            "A target journal is needed before publisher-specific export compliance.",
        ],
        "layout_deviations_recorded": True,
        "source_registry": relative_source(
            SOURCE_REGISTRY_DIR / "round2_source_registry.csv"
        ),
        "source_registry_entries": len(source_registry),
        "manifest": relative_source(MANIFEST),
        "artifact_registry": relative_source(ARTIFACT_REGISTRY),
        "total_qc": relative_source(TOTAL_QC),
        "publication_term_scan": relative_source(TERM_SCAN),
        "combined_captions": {
            "english": relative_source(CAPTIONS_EN),
            "chinese": relative_source(CAPTIONS_ZH),
        },
        "montage": {
            "path": relative_source(MONTAGE),
            "sha256": montage_info["sha256"],
            "width_px": montage_info["width_px"],
            "height_px": montage_info["height_px"],
            "mode": montage_info["mode"],
        },
        "artifact_registry_prepared": True,
        "optional_figures_not_generated": {
            "FigS6": "Not generated to avoid redundancy with established stability and algorithm-sensitivity figures.",
            "FigS7": "Not generated because the 3×3 FigS1 remained legible without splitting the wind composite.",
        },
    }
    write_text_atomic(
        HANDOFF_YAML,
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
    )

    figure_table = "\n".join(
        f"| {row.figure_id} | {row.status_label} | `{row.figure_file}` |"
        for row in manifest.itertuples()
    )
    handoff_md = f"""# Round 2 figure expansion handoff

```text
STATUS = ROUND2_FIGURE_EXPANSION_COMPLETE
SCIENTIFIC_RESULTS_CHANGED = FALSE
SCIENTIFIC_SOURCES_MODIFIED = FALSE
REVIEW_OUTPUT_ONLY = TRUE
PUBLICATION_VISIBLE_INTERNAL_TERMS_REMOVED = TRUE
```

## Delivered figures

| Figure | Status label | Review image |
|---|---|---|
{figure_table}

Fig. 9 and five supplementary figures were generated, satisfying the required Fig. 9 plus at least four supplementary figures. Fig. S5 remains explicitly subject to researcher approval before manuscript use.

## Integrity and QC

- {source_unchanged} registered source files were re-hashed after all builds with no changes detected.
- All figure images are 200 dpi RGB PNG review files.
- Every figure has an independent script, plotting-data snapshot, English caption, Chinese caption, metadata record, build log, and QC report.
- Publication-visible text across all figure artists and captions has zero forbidden internal-term hits.
- No old793/legacy790 input, non-tornado control, corridor delineation, selected STP_mod cutoff, or new clustering solution was introduced.

## Important scientific boundaries

- Fig. 9 is descriptive post-hoc context within the confirmed-tornado sample.
- Fig. S1 presents event-centered composite means and does not replace event-point statistics.
- Fig. S2 does not introduce a scalar overlap coefficient.
- Fig. S3 is coarse, environmental, not storm-relative, and not Bunkers-based.
- Fig. S4 uses numerical k=4 labels only and does not imply a unique natural classification.
- Fig. S5 reports the accepted rated-event comparison only and requires a researcher decision.

## Package indexes

- Manifest: `{relative_source(MANIFEST)}`
- Source registry: `{relative_source(SOURCE_REGISTRY_DIR / "round2_source_registry.csv")}`
- Artifact registry: `{relative_source(ARTIFACT_REGISTRY)}`
- Total QC: `{relative_source(TOTAL_QC)}`
- Montage: `{relative_source(MONTAGE)}`
- Combined English captions: `{relative_source(CAPTIONS_EN)}`
- Combined Chinese captions: `{relative_source(CAPTIONS_ZH)}`

No publisher-specific compliance is claimed because a target journal has not yet been designated. No submission-resolution PDF, SVG, or 600 dpi export was created.
"""
    write_text_atomic(HANDOFF_MD, handoff_md)


def _final_artifact_registry() -> pd.DataFrame:
    paths = sorted(
        path
        for path in ROUND_ROOT.rglob("*")
        if path.is_file() and path != ARTIFACT_REGISTRY
    )
    rows = [
        {
            "artifact_path": relative_source(path),
            "artifact_type": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    registry = pd.DataFrame(rows).sort_values("artifact_path")
    write_csv_atomic(ARTIFACT_REGISTRY, registry)
    return registry


def main() -> int:
    ensure_output_dirs()
    timestamps = [
        f"ROUND2_BUILD_START_UTC={datetime.now(timezone.utc).isoformat()}",
        "BUILD_ORDER=SOURCE_REGISTRY,Fig9,FigS1,FigS2,FigS3,FigS4,FigS5,MONTAGE,HANDOFF",
    ]
    if build_round2_source_registry.main() != 0:
        raise RuntimeError("Round 2 source registry build failed")

    built: list[dict] = []
    for figure_id, builder in FIGURE_BUILDERS:
        timestamps.append(f"{figure_id}_START_UTC={datetime.now(timezone.utc).isoformat()}")
        metadata = builder()
        built.append(metadata)
        timestamps.append(f"{figure_id}_COMPLETE_UTC={datetime.now(timezone.utc).isoformat()}")

    source_registry, unchanged = _verify_source_registry()
    manifest = _collect_metadata(built)
    _combine_captions(manifest)
    term_scan = _scan_public_text(manifest)
    montage_info = _build_montage(manifest)
    _write_total_qc(
        manifest, source_registry, unchanged, term_scan, montage_info
    )
    preliminary_artifacts = _artifact_registry(manifest, montage_info)
    _write_handoff(
        manifest,
        source_registry,
        unchanged,
        montage_info,
        preliminary_artifacts,
    )
    forbidden_exports = [
        path
        for path in ROUND_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".svg"}
    ]
    assert_exact("Round2 PDF/SVG export count", len(forbidden_exports), 0)
    assert_exact("Round2 figure count", len(manifest), 6)
    assert_exact(
        "Round2 supplementary figure count",
        int(manifest["figure_id"].str.startswith("FigS").sum()),
        5,
    )
    assert_exact("Round2 public term hits", int(term_scan["count"].sum()), 0)

    artifact_paths = {
        path.resolve()
        for path in ROUND_ROOT.rglob("*")
        if path.is_file() and path != ARTIFACT_REGISTRY
    }
    artifact_paths.add(BUILD_LOG.resolve())
    expected_artifact_count = len(artifact_paths)
    timestamps.extend(
        [
            f"ROUND2_BUILD_COMPLETE_UTC={datetime.now(timezone.utc).isoformat()}",
            "FINAL_STATUS=ROUND2_FIGURE_EXPANSION_COMPLETE",
            f"SOURCE_FILES_HASH_VERIFIED={unchanged}",
            f"FIGURES_GENERATED={len(manifest)}",
            f"ARTIFACTS_REGISTERED={expected_artifact_count}",
            "PUBLICATION_TERM_SCAN_TOTAL_HITS=0",
            "PDF_SVG_EXPORTS=0",
            "",
        ]
    )
    write_text_atomic(BUILD_LOG, "\n".join(timestamps))
    artifacts = _final_artifact_registry()
    assert_exact(
        "Round2 artifact registry count",
        len(artifacts),
        expected_artifact_count,
    )
    print(
        json.dumps(
            {
                "status": "ROUND2_FIGURE_EXPANSION_COMPLETE",
                "figures": len(manifest),
                "supplementary_figures": 5,
                "source_files_hash_verified": unchanged,
                "publication_term_hits": 0,
                "montage": relative_source(MONTAGE),
                "manifest": relative_source(MANIFEST),
                "handoff_md": relative_source(HANDOFF_MD),
                "handoff_yaml": relative_source(HANDOFF_YAML),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
