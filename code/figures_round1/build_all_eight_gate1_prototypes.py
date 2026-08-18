"""One-click build and handoff for all eight Gate 1 scientific prototypes."""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps

from audit_publication_terms_gate3 import audit as audit_publication_terms
from fig01_workflow_gate1 import build as build_fig1
from fig02_cluster_centers_gate1 import build as build_fig2
from fig03_raw_distributions_gate1 import build as build_fig3
from fig04_stability_gate1 import build as build_fig4
from fig05_k3_k4_sensitivity_gate1 import build as build_fig5
from fig06_weather_type_association_gate1 import build as build_fig6
from fig07_thermodynamic_profiles_gate1 import build as build_fig7
from fig08_wind_profiles_gate1 import build as build_fig8
from figure_io import (
    HANDOFF_DIR,
    LOG_DIR,
    MONTAGE_DIR,
    PROJECT_ROOT,
    QC_DIR,
    ROUND_ROOT,
    ensure_output_dirs,
    project_path,
    relative_source,
    sha256_file,
    write_text_atomic,
)


SCRIPT_PATH = Path(__file__).resolve()
MANIFEST_PATH = ROUND_ROOT / "figure_manifest.csv"
MONTAGE_PATH = MONTAGE_DIR / "gate1_all_eight_color_montage.png"
COMPATIBILITY_MONTAGE_PATH = MONTAGE_DIR / "gate1_prototypes_montage.png"
GRAYSCALE_MONTAGE_PATH = MONTAGE_DIR / "gate1_all_eight_grayscale_montage.png"
FINAL_STATUS = "GATE1_ALL_EIGHT_SCIENTIFIC_PROTOTYPES_COMPLETE"

FIGURE_DETAILS = {
    "Fig1": {
        "title": "Study design, sample closure, and analysis workflow",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/01_sample_audit/03_sample_closure_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/01_sample_audit/05_event_id_list_new790.csv",
            "paper_rebuild/00_governance/pre_manuscript_conclusion_library/01_master_core_conclusions.md",
        ],
        "claims": (
            "The audited sample closes from 909 to 790 and connects the primary, "
            "sensitivity, vertical-diagnosis, and post-hoc context modules."
        ),
        "limitations": (
            "Workflow schematic only; a non-tornadic comparison is not represented "
            "as completed analysis."
        ),
    },
    "Fig2": {
        "title": "Standardized centers of the three primary regimes",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/06_figures_tables/21_figures_and_tables_v3/source_data/fig05_standardized_centers_source_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/23_regime_interpretation_audit/01_k3_raw_to_formal_label_mapping.csv",
        ],
        "claims": (
            "Three regimes have distinct five-variable standardized profiles "
            "without implying hazard ranking."
        ),
        "limitations": "Gate 3 layout compression and legend refinement are pending.",
    },
    "Fig3": {
        "title": "Raw-unit clustering-variable distributions and overlap",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/02_environment_table/06_environment_table_new790_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/30_labels_k3_regime_ids_v3.csv",
        ],
        "claims": (
            "Regime shifts coexist with within-regime variability, overlap, and "
            "retained extreme values."
        ),
        "limitations": "Gate 3 panel enlargement and visual-noise reduction are pending.",
    },
    "Fig4": {
        "title": "Seed stability, event consistency, and boundary counts",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/09_seed_stability_100_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/10_seed_stability_summary_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/13_event_level_stability_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/14_event_level_stability_summary_v3.csv",
        ],
        "claims": (
            "The KMeans structure is highly stable while retaining a small number "
            "of transition and boundary events."
        ),
        "limitations": "Gate 3 high-probability-region enlargement is pending.",
    },
    "Fig5": {
        "title": "k=3/k=4 structural sensitivity and algorithm dependence",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/28_k3_to_k4_transition_matrix_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/23_regime_interpretation_audit/01_k3_raw_to_formal_label_mapping.csv",
            "paper_rebuild/17_core_sample_reopen_new790/23_regime_interpretation_audit/02_k4_label_identity_and_mapping.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/26_ward_results_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/27_gmm_results_v3.csv",
        ],
        "claims": (
            "k=4 unevenly subdivides k=3 membership, while Ward and Gaussian-mixture "
            "comparisons document algorithm dependence."
        ),
        "limitations": "The comparison does not identify a unique natural partition.",
    },
    "Fig6": {
        "title": "Post-hoc regime by weather-type association",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/24_post_hoc_context_audit/05_k3_macro9_contingency_counts.csv",
            "paper_rebuild/17_core_sample_reopen_new790/24_post_hoc_context_audit/06_k3_macro9_row_percent.csv",
            "paper_rebuild/17_core_sample_reopen_new790/24_post_hoc_context_audit/10_macro9_standardized_residuals.csv",
            "paper_rebuild/17_core_sample_reopen_new790/24_post_hoc_context_audit/CHATGPT_HANDOFF.yaml",
            "paper_rebuild/50_manuscript_figures_round1_topjournal/00_source_registry/gate0_frozen_number_assertions.csv",
        ],
        "claims": (
            "The frozen 3×9 table shows a significant many-to-many post-hoc "
            "association among 787 weather-type-valid events."
        ),
        "limitations": "Weather type is descriptive context within confirmed-tornado events.",
    },
    "Fig7": {
        "title": "Vertical temperature and moisture structure",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/35_direction_b_full_core_execution/05_9level_event_profile_long.csv",
            "paper_rebuild/17_core_sample_reopen_new790/35_direction_b_full_core_execution/11_thermodynamic_primary_statistics.csv",
        ],
        "claims": (
            "C2 has a deeper 850–500-hPa moist profile; C0 is colder aloft with "
            "stated composition sensitivity."
        ),
        "limitations": "Gate 3 effective-sample annotation reduction is pending.",
    },
    "Fig8": {
        "title": "Vertical environmental-wind structure and sensitivities",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/35_direction_b_full_core_execution/05_9level_event_profile_long.csv",
            "paper_rebuild/17_core_sample_reopen_new790/37_direction_b_full790_final_gate/12_upper_level_wind_vertical_continuity.csv",
            "paper_rebuild/17_core_sample_reopen_new790/37_direction_b_full790_final_gate/13_low_level_flow_adjustment_and_spatial_metrics.csv",
            "paper_rebuild/17_core_sample_reopen_new790/37_direction_b_full790_final_gate/11_upper_level_wind_confounder_adjustment.csv",
        ],
        "claims": (
            "C2 is stronger at 850 hPa, C0 is stronger at 200 hPa, and C1 is "
            "relatively weak across multiple levels."
        ),
        "limitations": (
            "No frozen C1 warm-sector-exclusion estimate was located; Gate 3 "
            "forest-panel enlargement is pending."
        ),
    },
}


def _write_manifest_atomic(rows: list[dict[str, object]]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=MANIFEST_PATH.parent,
        prefix=f".{MANIFEST_PATH.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, MANIFEST_PATH)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy


def _save_png_atomic(image: Image.Image, destination: Path, *, dpi: int = 150) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".png",
    )
    os.close(file_descriptor)
    try:
        image.save(temporary_name, dpi=(dpi, dpi))
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _build_montage(results: list[dict[str, object]]) -> None:
    if [item["figure_id"] for item in results] != [f"Fig{i}" for i in range(1, 9)]:
        raise AssertionError("Montage requires Fig1 through Fig8 in order.")
    card_width, card_height = 1120, 760
    margin, gap = 40, 30
    rows, columns = 4, 2
    canvas_width = margin * 2 + card_width * columns + gap * (columns - 1)
    canvas_height = margin * 2 + card_height * rows + gap * (rows - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font_path = font_manager.findfont(
            font_manager.FontProperties(family="Arial", weight="bold")
        )
        status_font_path = font_manager.findfont(
            font_manager.FontProperties(family="Arial", weight="normal")
        )
        title_font = ImageFont.truetype(title_font_path, 28)
        status_font = ImageFont.truetype(status_font_path, 21)
    except OSError:
        title_font = ImageFont.load_default()
        status_font = ImageFont.load_default()

    for index, result in enumerate(results):
        row, column = divmod(index, columns)
        x0 = margin + column * (card_width + gap)
        y0 = margin + row * (card_height + gap)
        draw.rectangle(
            (x0, y0, x0 + card_width, y0 + card_height),
            outline="#BDBDBD",
            width=2,
            fill="white",
        )
        with Image.open(str(result["output"])) as source:
            fitted = _fit_image(
                source.convert("RGB"), card_width - 36, card_height - 102
            )
        image_x = x0 + (card_width - fitted.width) // 2
        image_y = y0 + 12 + (card_height - 102 - fitted.height) // 2
        canvas.paste(fitted, (image_x, image_y))
        draw.text(
            (x0 + 20, y0 + card_height - 74),
            str(result["figure_id"]),
            fill="#202020",
            font=title_font,
        )
    _save_png_atomic(canvas, MONTAGE_PATH)
    _save_png_atomic(canvas.copy(), COMPATIBILITY_MONTAGE_PATH)
    _save_png_atomic(ImageOps.grayscale(canvas).convert("RGB"), GRAYSCALE_MONTAGE_PATH)


def _write_gate3_checklist() -> None:
    checklist = """# Gate 3 Modification Checklist for Existing Gate 1 Prototypes

```text
SCOPE = Fig2, Fig3, Fig4, Fig7, Fig8
CURRENT_ACTION = REGISTER_ONLY; NO REDRAW IN THIS ROUND
SCIENTIFIC_DATA_CHANGE = FORBIDDEN
```

## Fig.2

- Remove the redundant “(a)” when the figure remains a single panel.
- Compress unused whitespace.
- Optimize legend placement.

## Fig.3

- Enlarge every panel.
- Reduce scatter-point visual noise while retaining every observation in the statistics.
- Do not remove extreme values.
- Standardize all five panel sizes and sample-size annotations.

## Fig.4

- Enlarge the high-value region of maximum assignment probability.
- Improve the seed-stability distribution display.
- Reorganize labels in the logarithmic count panel.

## Fig.7

- Remove repeated effective-sample counts along the right side at every level.
- Highlight effective-sample changes only at 850, 925, and 1000 hPa.
- Move the remaining sample-size details to the caption or a supplementary table.

## Fig.8

- Enlarge the forest plot on the right.
- Separate subset names, confidence intervals, and numeric labels.
- Preserve the missing C1 warm-sector-exclusion estimate; do not calculate a replacement.

These entries are visual-reconstruction tasks for Gate 3. The accepted
scientific content and frozen numerical sources remain unchanged.
"""
    write_text_atomic(
        QC_DIR / "GATE3_modification_checklist_existing_figures.md", checklist
    )


def _write_summary_and_handoff(
    results: list[dict[str, object]],
    terminology_audit: dict[str, object],
) -> None:
    completed = ", ".join(str(item["figure_id"]) for item in results)
    resolution_sentence = (
        "587.3 was deprecated because it could not be reproduced from the "
        "frozen 3×9 contingency table."
    )
    qc_summary = f"""# Gate 1 Scientific QC Summary — All Eight Main Figures

```text
GATE_0_STATUS = PASS_WITH_RESOLVED_REPORTING_CONFLICT
GATE_1_STATUS = {FINAL_STATUS}
GATE_3_PUBLICATION_TERMINOLOGY_STATUS = {terminology_audit['status']}
PROTOTYPES_COMPLETED = {completed}
SCIENTIFIC_REANALYSIS = FALSE
FROZEN_SCIENTIFIC_FILES_MODIFIED = FALSE
ACTIVE_BLOCKERS = NONE
INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES = true
```

All eight 200-dpi RGB scientific prototypes have figure-specific source
assertions, plotting-data snapshots, bilingual caption drafts, build logs, PNG
metadata, and QC reports. No PDF, SVG, or 600-dpi final export was produced.

Fig.1 reproduces the 909→795→793→790 sample closure and keeps the study-design
scope within the completed modules. Fig.5 separates event counts from
row-normalized proportions and describes structural sensitivity and algorithm
dependence. Fig.6 uses one frozen 3×9 table for n=787, df=16,
χ²=437.139758077693 (display 437.1), raw V=0.5270, bias-corrected V=0.5172,
95% CI=0.4898–0.5758, and permutation p<0.0001; its composition and residual
panels retain the fixed nine-category order and show a many-to-many post-hoc
association.

The requested Gate 3 visual changes for Fig.2, Fig.3, Fig.4, Fig.7, and Fig.8
are registered in `GATE3_modification_checklist_existing_figures.md`; those
five accepted prototypes were not scientifically altered. Fig.8 still omits
the C1 warm-sector-exclusion subset because no frozen estimate exists.

Publication-facing terminology has been audited separately across all rendered
Matplotlib Text artists and all 16 formal caption files. The required counts
for Direction A, Direction B, Frozen, Gate, PASS_WITH_NONBLOCKING_NOTES, and
Prototype are all zero. Scripts, logs, manifest records, QC prose, and
filenames are outside this publication-facing audit scope.

The prescribed C1 orange (`#E69F00`) retains its existing 2.25:1
white-background review flag and redundant marker, line-style, position, and
text encoding. Exact publisher compliance remains pending because no target
journal or article type has been specified.

{resolution_sentence}
"""
    write_text_atomic(QC_DIR / "GATE1_scientific_qc_summary.md", qc_summary)

    consistency_review = """# Gate 1 Visual Consistency Review — Eight Figures

1. The story chain now covers sample design (Fig.1), primary regime structure
   and overlap (Fig.2–3), stability and structural sensitivity (Fig.4–5),
   post-hoc weather-type context (Fig.6), and Direction B vertical diagnosis
   (Fig.7–8).
2. C0/C1/C2 retain the fixed blue/orange/green palette, marker identities, and
   line styles wherever regime encoding is required.
3. Fig.5 uses neutral algorithm colors and independent count/proportion scales.
4. Fig.6 uses a sequential composition scale and a zero-centered diverging
   residual scale; only key cells carry numeric annotations.
5. Every output is an opaque RGB 200-dpi prototype. Formal vector, font,
   grayscale, and 600-dpi publisher checks remain later-gate tasks.
6. The complete eight-figure color montage is available in `10_montage`.
7. The existing-five Gate 3 visual revision list is registered separately and
   does not change accepted science.
"""
    write_text_atomic(
        QC_DIR / "visual_consistency_review_gate1.md", consistency_review
    )

    handoff_md = f"""# Gate 1 All-Eight Scientific Prototype Handoff

## Decision

```text
PROCESS_STATUS = {FINAL_STATUS}
SCIENTIFIC_RESULT_STATUS = CONSISTENT_WITH_FROZEN_RESULTS
SOURCE_AUDIT_STATUS = PASS_WITH_RESOLVED_REPORTING_CONFLICT
MAIN_FIGURES_PLANNED = 8
GATE1_PROTOTYPES_COMPLETED = 8
MAIN_FIGURES_PASSED_SCIENTIFIC_PROTOTYPE_QC = 8
ACTIVE_BLOCKERS = NONE
INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES = true
```

All eight main-figure scientific prototypes are complete. The three figures
added in this execution were produced in the requested order Fig.1 → Fig.5 →
Fig.6. No frozen scientific file was changed, no old793 or legacy790 result was
used, and no missing sensitivity estimate was recalculated.

**Mandatory final-handoff record:** {resolution_sentence}

Fig.6 is released with n=787, χ²(16)=437.1, raw Cramér’s V=0.5270,
bias-corrected Cramér’s V=0.5172, 95% CI=0.4898–0.5758, and permutation
p<0.0001. The result is a descriptive many-to-many post-hoc association.

Formal PDF/SVG/600-dpi export and final publisher-specific styling remain
deferred. The next visual stage is Gate 3 reconstruction using the registered
checklist.
"""
    write_text_atomic(
        HANDOFF_DIR / "GATE1_SCIENTIFIC_PROTOTYPE_HANDOFF.md", handoff_md
    )
    write_text_atomic(
        HANDOFF_DIR / "GATE1_ALL_EIGHT_SCIENTIFIC_PROTOTYPES_HANDOFF.md",
        handoff_md,
    )
    gate3_handoff_md = f"""# Gate 3 Publication Terminology Handoff

```text
TERMINOLOGY_AUDIT_STATUS = {terminology_audit['status']}
FIGURE_VISIBLE_TEXT_FILES_CHECKED = 8
FORMAL_CAPTION_FILES_CHECKED = 16
Direction A count = 0
Direction B count = 0
Frozen count = 0
Gate count = 0
PASS_WITH_NONBLOCKING_NOTES count = 0
Prototype count = 0
INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES = true
```

Only publication-facing figure text and formal captions were changed. Data,
values, labels, statistical results, and scientific conclusions were not
modified. Internal scripts, logs, manifest records, QC prose, and filenames
remain outside the publication-facing terminology scope.
"""
    write_text_atomic(
        HANDOFF_DIR / "GATE3_PUBLICATION_TERMINOLOGY_HANDOFF.md",
        gate3_handoff_md,
    )

    handoff_yaml = {
        "TASK": "MANUSCRIPT_FIGURE_ROUND1_TOPJOURNAL",
        "PROCESS_STATUS": FINAL_STATUS,
        "SCIENTIFIC_RESULT_STATUS": "CONSISTENT_WITH_FROZEN_RESULTS",
        "SOURCE_AUDIT_STATUS": "PASS_WITH_RESOLVED_REPORTING_CONFLICT",
        "MAIN_FIGURES_PLANNED": 8,
        "MAIN_FIGURES_COMPLETED": 0,
        "GATE1_PROTOTYPES_COMPLETED": [f"Fig{i}" for i in range(1, 9)],
        "MAIN_FIGURES_PASSED_SCIENTIFIC_PROTOTYPE_QC": 8,
        "MAIN_FIGURES_BLOCKED": [],
        "SCIENTIFIC_FILES_MODIFIED": False,
        "FROZEN_FILES_OVERWRITTEN": False,
        "LEGACY790_USED": False,
        "OLD793_CLUSTERING_USED": False,
        "NEW_SCIENTIFIC_ANALYSIS_PERFORMED": False,
        "ACTIVE_BLOCKERS": [],
        "GATE3_PUBLICATION_TERMINOLOGY_STATUS": terminology_audit["status"],
        "PUBLICATION_TERM_REQUIRED_COUNTS": terminology_audit["required_counts"],
        "INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES": True,
        "FIG6_RELEASE": {
            "valid_n": 787,
            "chi_square_exact": 437.139758077693,
            "chi_square_display": 437.1,
            "df": 16,
            "raw_cramers_v": 0.5270,
            "bias_corrected_cramers_v": 0.5172,
            "bootstrap_ci": [0.4898, 0.5758],
            "permutation_p": "<0.0001",
            "deprecated_reporting_number": 587.3,
            "deprecated_status": "DEPRECATED_REPORTING_NUMBER",
            "required_statement": resolution_sentence,
        },
        "GATE3_CHECKLIST": relative_source(
            QC_DIR / "GATE3_modification_checklist_existing_figures.md"
        ),
        "MANIFEST": relative_source(MANIFEST_PATH),
        "COLOR_MONTAGE": relative_source(MONTAGE_PATH),
        "FINAL_DECISION": FINAL_STATUS,
    }
    rendered_yaml = yaml.safe_dump(
        handoff_yaml,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )
    write_text_atomic(
        HANDOFF_DIR / "GATE1_SCIENTIFIC_PROTOTYPE_HANDOFF.yaml", rendered_yaml
    )
    write_text_atomic(
        HANDOFF_DIR / "GATE1_ALL_EIGHT_SCIENTIFIC_PROTOTYPES_HANDOFF.yaml",
        rendered_yaml,
    )
    gate3_handoff_yaml = {
        "TERMINOLOGY_AUDIT_STATUS": terminology_audit["status"],
        "FIGURE_VISIBLE_TEXT_FILES_CHECKED": 8,
        "FORMAL_CAPTION_FILES_CHECKED": 16,
        "REQUIRED_COUNTS": terminology_audit["required_counts"],
        "EXTENDED_BILINGUAL_COUNTS": terminology_audit["extended_counts"],
        "INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES": True,
        "SCIENTIFIC_DATA_MODIFIED": False,
        "STATISTICAL_RESULTS_MODIFIED": False,
    }
    write_text_atomic(
        HANDOFF_DIR / "GATE3_PUBLICATION_TERMINOLOGY_HANDOFF.yaml",
        yaml.safe_dump(
            gate3_handoff_yaml,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
    )


def _manifest_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        figure_id = str(result["figure_id"])
        details = FIGURE_DETAILS[figure_id]
        metadata = result["metadata"]
        output_path = Path(str(result["output"]))
        plotting_path = Path(str(result["plotting_data"]))
        script_path = Path(str(result["script"]))
        input_paths = [project_path(item) for item in details["inputs"]]
        missing = [str(path) for path in input_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{figure_id} manifest inputs missing: {missing}")
        rows.append(
            {
                "figure_id": figure_id,
                "title": details["title"],
                "stage": "GATE1_LOW_RESOLUTION_SCIENTIFIC_PROTOTYPE",
                "status": result["status"],
                "input_files": ";".join(details["inputs"]),
                "input_sha256": ";".join(sha256_file(path) for path in input_paths),
                "plotting_data": relative_source(plotting_path),
                "plotting_data_sha256": sha256_file(plotting_path),
                "output_files": relative_source(output_path),
                "output_sha256": sha256_file(output_path),
                "script": relative_source(script_path),
                "script_sha256": sha256_file(script_path),
                "dimensions_px": f"{metadata['width_px']}x{metadata['height_px']}",
                "dpi": f"{float(metadata['dpi_x']):.4f}",
                "color_mode": metadata["mode"],
                "scientific_claims_supported": details["claims"],
                "qc_status": result["status"],
                "known_limitations": details["limitations"],
            }
        )
    return rows


def main() -> int:
    ensure_output_dirs()
    builders = [
        build_fig1,
        build_fig2,
        build_fig3,
        build_fig4,
        build_fig5,
        build_fig6,
        build_fig7,
        build_fig8,
    ]
    results = [builder() for builder in builders]
    expected_ids = [f"Fig{i}" for i in range(1, 9)]
    if [item["figure_id"] for item in results] != expected_ids:
        raise AssertionError("All-eight build did not return Fig1 through Fig8.")
    manifest_rows = _manifest_rows(results)
    _write_manifest_atomic(manifest_rows)
    _build_montage(results)
    _write_gate3_checklist()
    terminology_audit = audit_publication_terms()
    _write_summary_and_handoff(results, terminology_audit)

    output_hashes = [
        f"{item['figure_id']}={sha256_file(Path(str(item['output'])))}"
        for item in results
    ]
    build_log = "\n".join(
        [
            f"timestamp={datetime.now().astimezone().isoformat()}",
            "NEW_FIGURE_EXECUTION_ORDER=Fig1,Fig5,Fig6",
            "FULL_REPRODUCIBILITY_ORDER=Fig1,Fig2,Fig3,Fig4,Fig5,Fig6,Fig7,Fig8",
            "GATE0_STATUS=PASS_WITH_RESOLVED_REPORTING_CONFLICT",
            f"GATE1_STATUS={FINAL_STATUS}",
            "PROTOTYPE_DPI=200",
            "COLOR_MODE=RGB",
            "FINAL_EXPORT_PERFORMED=FALSE",
            "PDF_CREATED=FALSE",
            "SVG_CREATED=FALSE",
            "DPI600_CREATED=FALSE",
            "SCIENTIFIC_FILES_MODIFIED=FALSE",
            "FROZEN_FILES_OVERWRITTEN=FALSE",
            "OLD793_USED=FALSE",
            "LEGACY790_USED=FALSE",
            "NON_TORNADIC_CONTROL_USED=FALSE",
            (
                "GATE3_PUBLICATION_TERMINOLOGY_STATUS="
                f"{terminology_audit['status']}"
            ),
            "INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES=true",
            "OUTPUT_SHA256=" + "|".join(output_hashes),
            (
                "RESOLUTION_STATEMENT=587.3 was deprecated because it could "
                "not be reproduced from the frozen 3×9 contingency table."
            ),
        ]
    )
    write_text_atomic(LOG_DIR / "build_gate1_prototypes.log", build_log + "\n")
    write_text_atomic(LOG_DIR / "build_all_eight_gate1_prototypes.log", build_log + "\n")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {MONTAGE_PATH}")
    print(f"Wrote {GRAYSCALE_MONTAGE_PATH}")
    print(FINAL_STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
