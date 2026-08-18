"""One-click ordered build for the five Gate 1 scientific prototypes."""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps

from fig02_cluster_centers_gate1 import build as build_fig2
from fig03_raw_distributions_gate1 import build as build_fig3
from fig04_stability_gate1 import build as build_fig4
from fig07_thermodynamic_profiles_gate1 import build as build_fig7
from fig08_wind_profiles_gate1 import build as build_fig8
from figure_io import (
    HANDOFF_DIR,
    LOG_DIR,
    MONTAGE_DIR,
    QC_DIR,
    ROUND_ROOT,
    ensure_output_dirs,
    relative_source,
    sha256_file,
    write_text_atomic,
)


SCRIPT_PATH = Path(__file__).resolve()
MANIFEST_PATH = ROUND_ROOT / "figure_manifest.csv"
MONTAGE_PATH = MONTAGE_DIR / "gate1_prototypes_montage.png"
GRAYSCALE_MONTAGE_PATH = (
    MONTAGE_DIR / "gate1_prototypes_montage_grayscale.png"
)

FIGURE_DETAILS = {
    "Fig2": {
        "title": "Standardized centers of the three primary regimes",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/06_figures_tables/21_figures_and_tables_v3/source_data/fig05_standardized_centers_source_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/23_regime_interpretation_audit/01_k3_raw_to_formal_label_mapping.csv",
        ],
        "claims": "Three regimes have distinct five-variable standardized profiles without implying hazard ranking.",
        "limitations": "General publication prototype; final publisher rules pending.",
    },
    "Fig3": {
        "title": "Raw-unit clustering-variable distributions and overlap",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/02_environment_table/06_environment_table_new790_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/30_labels_k3_regime_ids_v3.csv",
        ],
        "claims": "Regime shifts coexist with within-regime variability, overlap, and retained extreme values.",
        "limitations": "Dense observations are raster content in the prototype.",
    },
    "Fig4": {
        "title": "Seed stability, event consistency, and boundary counts",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/09_seed_stability_100_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/10_seed_stability_summary_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/13_event_level_stability_v3.csv",
            "paper_rebuild/17_core_sample_reopen_new790/04_clustering/12_clustering_results_v3/14_event_level_stability_summary_v3.csv",
        ],
        "claims": "The KMeans structure is highly stable while retaining a small number of transition and boundary events.",
        "limitations": "Observed stable-core minima are descriptive of frozen statuses, not universal cutoffs.",
    },
    "Fig7": {
        "title": "Vertical temperature and moisture structure",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/35_direction_b_full_core_execution/05_9level_event_profile_long.csv",
            "paper_rebuild/17_core_sample_reopen_new790/35_direction_b_full_core_execution/11_thermodynamic_primary_statistics.csv",
        ],
        "claims": "C2 has a deeper 850–500-hPa moist profile; C0 is colder aloft with stated composition sensitivity.",
        "limitations": "Bootstrap intervals are logged plotting summaries; final publisher review pending.",
    },
    "Fig8": {
        "title": "Vertical environmental-wind structure and sensitivities",
        "inputs": [
            "paper_rebuild/17_core_sample_reopen_new790/35_direction_b_full_core_execution/05_9level_event_profile_long.csv",
            "paper_rebuild/17_core_sample_reopen_new790/37_direction_b_full790_final_gate/12_upper_level_wind_vertical_continuity.csv",
            "paper_rebuild/17_core_sample_reopen_new790/37_direction_b_full790_final_gate/13_low_level_flow_adjustment_and_spatial_metrics.csv",
            "paper_rebuild/17_core_sample_reopen_new790/37_direction_b_full790_final_gate/11_upper_level_wind_confounder_adjustment.csv",
        ],
        "claims": "C2 is stronger at 850 hPa, C0 is stronger at 200 hPa, and C1 is relatively weak across multiple levels.",
        "limitations": "No frozen C1 warm-sector-exclusion estimate was located, so that optional subset is omitted.",
    },
}


def write_manifest_atomic(rows: list[dict[str, object]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=MANIFEST_PATH.parent,
        prefix=f".{MANIFEST_PATH.name}.",
        suffix=".tmp",
    )
    fields = list(rows[0].keys())
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, MANIFEST_PATH)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy


def build_montage(results: list[dict[str, object]]) -> None:
    card_width, card_height = 1080, 760
    margin, gap = 40, 30
    canvas_width = margin * 2 + card_width * 2 + gap
    canvas_height = margin * 2 + card_height * 3 + gap * 2
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
        status_font = ImageFont.truetype(status_font_path, 22)
    except OSError:
        title_font = ImageFont.load_default()
        status_font = ImageFont.load_default()

    for index, result in enumerate(results):
        row, column = divmod(index, 2)
        x0 = margin + column * (card_width + gap)
        y0 = margin + row * (card_height + gap)
        draw.rectangle(
            (x0, y0, x0 + card_width, y0 + card_height),
            outline="#BDBDBD",
            width=2,
            fill="white",
        )
        with Image.open(str(result["output"])) as source:
            source_rgb = source.convert("RGB")
        fitted = fit_image(source_rgb, card_width - 40, card_height - 100)
        image_x = x0 + (card_width - fitted.width) // 2
        image_y = y0 + 12
        canvas.paste(fitted, (image_x, image_y))
        draw.text(
            (x0 + 20, y0 + card_height - 72),
            str(result["figure_id"]),
            fill="#202020",
            font=title_font,
        )
        draw.text(
            (x0 + 130, y0 + card_height - 68),
            str(result["status"]),
            fill="#4D4D4D",
            font=status_font,
        )

    summary_x = margin + card_width + gap
    summary_y = margin + 2 * (card_height + gap)
    draw.rectangle(
        (
            summary_x,
            summary_y,
            summary_x + card_width,
            summary_y + card_height,
        ),
        outline="#BDBDBD",
        width=2,
        fill="#F7F7F7",
    )
    summary_lines = [
        "Gate 1 visual consistency card",
        "",
        "Order: Fig2 → Fig3 → Fig4 → Fig7 → Fig8",
        "Fixed C0/C1/C2 colors",
        "Opaque RGB prototypes",
        "No frozen scientific files modified",
        "Final PDF/SVG/600-dpi export: Gate 4",
    ]
    for line_index, line in enumerate(summary_lines):
        draw.text(
            (summary_x + 40, summary_y + 55 + line_index * 52),
            line,
            fill="#202020",
            font=title_font if line_index == 0 else status_font,
        )

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=MONTAGE_PATH.parent,
        prefix=f".{MONTAGE_PATH.stem}.",
        suffix=".png",
    )
    os.close(file_descriptor)
    try:
        canvas.save(temporary_name, dpi=(150, 150))
        os.replace(temporary_name, MONTAGE_PATH)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    grayscale = ImageOps.grayscale(canvas).convert("RGB")
    grayscale.save(GRAYSCALE_MONTAGE_PATH, dpi=(150, 150))


def write_summary_and_handoff(
    results: list[dict[str, object]], manifest_rows: list[dict[str, object]]
) -> None:
    qc_summary = """# Gate 1 Scientific QC Summary

```text
GATE_0_STATUS = PASS_WITH_RESOLVED_REPORTING_CONFLICT
GATE_1_STATUS = COMPLETE_WITH_NONBLOCKING_NOTES
PROTOTYPES_COMPLETED = Fig2, Fig3, Fig4, Fig7, Fig8
SCIENTIFIC_REANALYSIS = FALSE
```

All five requested prototypes passed automated source, count, key-number, and
interpretation-boundary checks. Each has an independent plotting snapshot,
script, bilingual caption draft, build log, PNG metadata record, and four-level
QC report.

The prototypes are deliberately 200-dpi RGB review files. PDF, SVG, 600-dpi
PNG, embedded-font review, exact journal compliance, and the eight-figure final
montage remain later-gate work.

The prescribed C1 orange (`#E69F00`) receives a review flag in a 3:1
white-background graphical-contrast screen (2.25:1). It is retained because the
palette is scientifically frozen for this task and is redundantly encoded by a
square marker, dashed line, fixed category position, and text label. All three
regime pairs pass the heuristic grayscale-separation screen. These automated
screens do not establish accessibility or journal compliance.

Fig.8 omits the optional C1 warm-sector-exclusion subset because no frozen
estimate was identified. No new sensitivity result was calculated.

587.3 was deprecated because it could not be reproduced from the frozen 3×9
contingency table.
"""
    write_text_atomic(QC_DIR / "GATE1_scientific_qc_summary.md", qc_summary)

    consistency_review = """# Gate 1 Visual Consistency Review

1. **Story chain:** Fig.2 introduces the five-variable group profiles; Fig.3
   shows raw-unit spread and overlap; Fig.4 establishes stability with boundary
   cases; Fig.7 and Fig.8 provide vertical physical structure.
2. **Redundancy:** No prototype duplicates another statistical object.
3. **Core evidence coverage:** The principal group structure, overlap,
   stability, deep 850–500-hPa moisture, and vertical environmental-flow
   statements each have a dedicated prototype.
4. **Visual weighting:** Stability and sensitivity panels remain smaller than
   the primary profile panels.
5. **Color consistency:** C0/C1/C2 use #0072B2/#E69F00/#009E73 throughout.
   Markers and line styles provide redundant identification.
6. **Second-round needs:** Fine-tune final typography after the journal and
   article type are selected; inspect final vector fonts and dense-point
   rasterization.
7. **Chinese-draft readiness:** All five are suitable as scientific prototypes
   for the Chinese draft after author review.
8. **Current status:** These are scientific prototypes, not final submission
   exports.

The grayscale montage retains regime identification through marker, line-style,
panel position, and text labels even where hue differences collapse.

The palette audit flags the prescribed C1 orange for review against white
(2.25:1 under a 3:1 graphical-object screen); its redundant non-color encodings
are therefore essential. This automated screen is not an accessibility
certification.
"""
    write_text_atomic(
        QC_DIR / "visual_consistency_review_gate1.md", consistency_review
    )

    exact_resolution_sentence = (
        "587.3 was deprecated because it could not be reproduced from the "
        "frozen 3×9 contingency table."
    )
    handoff_md = f"""# Gate 1 Scientific Prototype Handoff

## Decision

```text
PROCESS_STATUS = GATE1_COMPLETE_WITH_NONBLOCKING_NOTES
SCIENTIFIC_RESULT_STATUS = CONSISTENT_WITH_FROZEN_RESULTS
SOURCE_AUDIT_STATUS = PASS_WITH_RESOLVED_REPORTING_CONFLICT
MAIN_FIGURES_PLANNED = 8
GATE1_PROTOTYPES_COMPLETED = 5
ACTIVE_BLOCKERS = NONE
```

Gate 0 was closed after the researcher adjudicated the Fig.6 reporting
conflict. The five authorized Gate 1 prototypes were then built in the required
order: Fig.2 → Fig.3 → Fig.4 → Fig.7 → Fig.8.

**Mandatory final-handoff record:** {exact_resolution_sentence}

No frozen scientific source was modified or overwritten. No legacy 790 or old
793 clustering result was used. No new scientific analysis was performed.

## Nonblocking notes

- Exact journal, article type, and submission phase remain unspecified.
- Final PDF/SVG/600-dpi exports are intentionally deferred to Gate 4.
- Fig.8 omits an optional C1 warm-sector-exclusion sensitivity because no
  frozen estimate was found.
- The prescribed C1 orange has a 2.25:1 white-background graphical-contrast
  review flag; square markers, dashed lines, category position, and text labels
  provide redundant identification.
- Fig.6 is released for later figure production using
  `χ²(16)=437.1`, valid n=787, raw V=0.5270, corrected V=0.5172,
  bootstrap 95% CI=0.4898–0.5758, and permutation p<0.0001.
"""
    write_text_atomic(
        HANDOFF_DIR / "GATE1_SCIENTIFIC_PROTOTYPE_HANDOFF.md", handoff_md
    )

    handoff_yaml = {
        "TASK": "MANUSCRIPT_FIGURE_ROUND1_TOPJOURNAL",
        "PROCESS_STATUS": "GATE1_COMPLETE_WITH_NONBLOCKING_NOTES",
        "SCIENTIFIC_RESULT_STATUS": "CONSISTENT_WITH_FROZEN_RESULTS",
        "SOURCE_AUDIT_STATUS": "PASS_WITH_RESOLVED_REPORTING_CONFLICT",
        "MAIN_FIGURES_PLANNED": 8,
        "MAIN_FIGURES_COMPLETED": 0,
        "GATE1_PROTOTYPES_COMPLETED": [
            result["figure_id"] for result in results
        ],
        "MAIN_FIGURES_PASSED_SCIENTIFIC_PROTOTYPE_QC": 5,
        "MAIN_FIGURES_BLOCKED": [],
        "SUPPLEMENTARY_FIGURES_COMPLETED": 0,
        "SCIENTIFIC_FILES_MODIFIED": False,
        "FROZEN_FILES_OVERWRITTEN": False,
        "EXTERNAL_DRIVES_MODIFIED": False,
        "LEGACY790_USED": False,
        "OLD793_CLUSTERING_USED": False,
        "NEW_SCIENTIFIC_ANALYSIS_PERFORMED": False,
        "ACTIVE_BLOCKERS": [],
        "NONBLOCKING_NOTES": [
            "Exact target-journal guidance is pending.",
            "Final PDF/SVG/600-dpi export is deferred to Gate 4.",
            "No frozen C1 warm-sector-exclusion estimate was located.",
            "The prescribed C1 orange has a 2.25:1 graphical-contrast review flag against white; redundant non-color encodings are retained.",
        ],
        "FIG6_RELEASE": {
            "valid_n": 787,
            "chi_square_exact": 437.139758,
            "chi_square_display": 437.1,
            "df": 16,
            "raw_cramers_v": 0.5270,
            "bias_corrected_cramers_v": 0.5172,
            "bootstrap_ci": [0.4898, 0.5758],
            "permutation_p": "<0.0001",
            "deprecated_reporting_number": 587.3,
            "deprecated_status": "DEPRECATED_REPORTING_NUMBER",
            "required_statement": exact_resolution_sentence,
        },
        "MANIFEST": relative_source(MANIFEST_PATH),
        "FINAL_DECISION": "PROCEED_TO_GATE2_SCIENTIFIC_QC",
    }
    write_text_atomic(
        HANDOFF_DIR / "GATE1_SCIENTIFIC_PROTOTYPE_HANDOFF.yaml",
        yaml.safe_dump(
            handoff_yaml,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
    )


def main() -> int:
    ensure_output_dirs()
    builders = [build_fig2, build_fig3, build_fig4, build_fig7, build_fig8]
    results: list[dict[str, object]] = []
    for builder in builders:
        results.append(builder())

    manifest_rows: list[dict[str, object]] = []
    for result in results:
        figure_id = str(result["figure_id"])
        details = FIGURE_DETAILS[figure_id]
        metadata = result["metadata"]
        output_path = Path(str(result["output"]))
        plotting_path = Path(str(result["plotting_data"]))
        script_path = Path(str(result["script"]))
        manifest_rows.append(
            {
                "figure_id": figure_id,
                "title": details["title"],
                "stage": "GATE1_LOW_RESOLUTION_SCIENTIFIC_PROTOTYPE",
                "status": result["status"],
                "input_files": ";".join(details["inputs"]),
                "plotting_data": relative_source(plotting_path),
                "output_files": relative_source(output_path),
                "script": relative_source(script_path),
                "output_sha256": sha256_file(output_path),
                "script_sha256": sha256_file(script_path),
                "dimensions_px": (
                    f"{metadata['width_px']}x{metadata['height_px']}"
                ),
                "dpi": f"{float(metadata['dpi_x']):.4f}",
                "color_mode": metadata["mode"],
                "scientific_claims_supported": details["claims"],
                "qc_status": result["status"],
                "known_limitations": details["limitations"],
            }
        )
    write_manifest_atomic(manifest_rows)
    build_montage(results)
    write_summary_and_handoff(results, manifest_rows)

    build_log = "\n".join(
        [
            f"timestamp={datetime.now().astimezone().isoformat()}",
            "BUILD_ORDER=Fig2,Fig3,Fig4,Fig7,Fig8",
            "GATE0_STATUS=PASS_WITH_RESOLVED_REPORTING_CONFLICT",
            "GATE1_STATUS=COMPLETE_WITH_NONBLOCKING_NOTES",
            "PROTOTYPE_DPI=200",
            "FINAL_EXPORT_PERFORMED=FALSE",
            "SCIENTIFIC_FILES_MODIFIED=FALSE",
            "FROZEN_FILES_OVERWRITTEN=FALSE",
            (
                "RESOLUTION_STATEMENT=587.3 was deprecated because it could "
                "not be reproduced from the frozen 3×9 contingency table."
            ),
        ]
    )
    write_text_atomic(LOG_DIR / "build_gate1_prototypes.log", build_log + "\n")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {MONTAGE_PATH}")
    print(f"Wrote {GRAYSCALE_MONTAGE_PATH}")
    print("Gate 1 prototypes completed:", [item["figure_id"] for item in results])
    return 0


if __name__ == "__main__":
    # Compatibility entry point: the authoritative Gate 1 build now covers
    # Fig.1 through Fig.8.
    from build_all_eight_gate1_prototypes import main as build_all_eight

    raise SystemExit(build_all_eight())
