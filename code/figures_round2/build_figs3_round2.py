"""Build Fig. S3: coarse environmental u-v wind profiles."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from round2_io import (
    CAPTION_DIR,
    LOG_DIR,
    PLOTTING_DATA_DIR,
    PROJECT_ROOT,
    QC_DIR,
    SUPP_FIGURE_DIR,
    ensure_output_dirs,
    inspect_png,
    read_csv_checked,
    relative_source,
    save_figure_atomic,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from round2_qc import (
    assert_close,
    assert_exact,
    assert_public_text_clean,
    write_figure_qc,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_configs"))
from round2_style import (  # noqa: E402
    FIGURE_SIZES_MM,
    PROTOTYPE_DPI,
    REGIME_COLORS,
    REGIME_ORDER,
    manuscript_style,
    mm_to_inches,
    panel_label,
    style_axis,
)


PROFILE_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/35_direction_b_full_core_execution/"
    "05_9level_event_profile_long.csv"
)
ACCEPTED_STATS_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/35_direction_b_full_core_execution/"
    "14_wind_primary_statistics.csv"
)
CLAIM_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/39_direction_b_2d_resume_full790/"
    "09_direction_b_final_claim_adjudication_v6.csv"
)

OUTPUT = SUPP_FIGURE_DIR / "FigS3_environmental_wind_profiles_ROUND2_review.png"
SNAPSHOT = PLOTTING_DATA_DIR / "FigS3_plotting_snapshot.csv"
CAPTION_EN = CAPTION_DIR / "FigS3_caption_en.md"
CAPTION_ZH = CAPTION_DIR / "FigS3_caption_zh.md"
METADATA = QC_DIR / "FigS3_metadata.json"
QC_REPORT = QC_DIR / "FigS3_QC.md"
BUILD_LOG = LOG_DIR / "FigS3_build.log"

PRESSURE_LEVELS = (850, 700, 500, 400, 300, 250, 200)
SELECTED_LABELS = (850, 500, 200)
REGIME_N = {"C0": 131, "C1": 307, "C2": 352}


def _load_and_summarize() -> pd.DataFrame:
    raw = read_csv_checked(
        PROFILE_SOURCE,
        required_columns=(
            "event_id",
            "regime",
            "level_hPa",
            "u_ms",
            "v_ms",
            "below_ground",
            "valid",
        ),
    )
    accepted = read_csv_checked(
        ACCEPTED_STATS_SOURCE,
        required_columns=("variable", "level_hPa", "regime", "n_valid", "median"),
    )
    assert_exact("FigS3 raw rows", len(raw), 7110)
    assert_exact("FigS3 raw unique events", raw["event_id"].nunique(), 790)
    assert_exact(
        "FigS3 raw pressure levels",
        tuple(sorted(raw["level_hPa"].unique(), reverse=True)),
        (1000, 925, 850, 700, 500, 400, 300, 250, 200),
    )

    filtered = raw.loc[
        raw["valid"].eq(True)
        & raw["below_ground"].eq(False)
        & raw["level_hPa"].isin(PRESSURE_LEVELS)
    ].copy()
    summary = (
        filtered.groupby(["regime", "level_hPa"], observed=True)
        .agg(
            n_valid=("event_id", "nunique"),
            median_u_ms=("u_ms", "median"),
            median_v_ms=("v_ms", "median"),
        )
        .reset_index()
    )
    assert_exact("FigS3 summary rows", len(summary), 21)

    for component, column in (("u_ms", "median_u_ms"), ("v_ms", "median_v_ms")):
        check = accepted.loc[
            accepted["variable"].eq(component)
            & accepted["level_hPa"].isin(PRESSURE_LEVELS)
        ].copy()
        merged = summary.merge(
            check[["regime", "level_hPa", "n_valid", "median"]],
            on=["regime", "level_hPa"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_accepted"),
        )
        assert_exact(f"FigS3 accepted {component} rows", len(merged), 21)
        for _, row in merged.iterrows():
            assert_exact(
                f"FigS3 {component} n {row['regime']} {row['level_hPa']}",
                int(row["n_valid"]),
                int(row["n_valid_accepted"]),
            )
            assert_close(
                f"FigS3 {component} median {row['regime']} {row['level_hPa']}",
                row[column],
                row["median"],
                0.011,
            )
    summary["aggregation"] = "component-wise event median"
    summary["profile_scope"] = "850–200 hPa environmental flow"
    return summary.sort_values(["regime", "level_hPa"], ascending=[True, False])


def _draw(summary: pd.DataFrame) -> None:
    width, height = FIGURE_SIZES_MM["FigS3"]
    common_limit = (-5.0, 27.0)
    label_offsets = {
        850: (5, 5, "left", "bottom"),
        500: (5, 4, "left", "bottom"),
        200: (-5, 5, "right", "bottom"),
    }
    with manuscript_style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=mm_to_inches(width, height),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        for index, (ax, regime) in enumerate(zip(axes, REGIME_ORDER)):
            rows = (
                summary.loc[summary["regime"] == regime]
                .set_index("level_hPa")
                .loc[list(PRESSURE_LEVELS)]
                .reset_index()
            )
            ax.plot(
                rows["median_u_ms"],
                rows["median_v_ms"],
                color=REGIME_COLORS[regime],
                linewidth=1.7,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=REGIME_COLORS[regime],
                markeredgewidth=0.9,
                markersize=4.5,
                zorder=3,
            )
            for level in SELECTED_LABELS:
                row = rows.loc[rows["level_hPa"] == level].iloc[0]
                dx, dy, horizontal, vertical = label_offsets[level]
                ax.annotate(
                    f"{level}",
                    xy=(row["median_u_ms"], row["median_v_ms"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=horizontal,
                    va=vertical,
                    fontsize=6.6,
                    color="#333333",
                )
            ax.axhline(0, color="#777777", linewidth=0.65, linestyle=":")
            ax.axvline(0, color="#777777", linewidth=0.65, linestyle=":")
            ax.set_xlim(common_limit)
            ax.set_ylim(common_limit)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([-5, 0, 5, 10, 15, 20, 25])
            ax.set_yticks([-5, 0, 5, 10, 15, 20, 25])
            ax.set_title(regime, color=REGIME_COLORS[regime], fontweight="bold", pad=4)
            ax.set_xlabel("Median u (m s$^{-1}$)")
            if index == 0:
                ax.set_ylabel("Median v (m s$^{-1}$)")
            style_axis(ax, grid_axis="both")
            panel_label(ax, f"({chr(97 + index)})", x=-0.18, y=1.035)
        fig.suptitle("Coarse environmental u–v wind profiles", fontsize=9.2)
        save_figure_atomic(fig, OUTPUT, dpi=PROTOTYPE_DPI)
        plt.close(fig)


def _write_caption_and_qc(summary: pd.DataFrame) -> None:
    caption_en = """# Fig. S3 caption (English)

**Figure S3. Coarse environmental u–v wind profiles for C0, C1, and C2.** Each curve connects the component-wise event medians at 850, 700, 500, 400, 300, 250, and 200 hPa; selected pressure levels are labeled in hPa. The three panels use identical u and v limits and equal axis scales. These profiles have coarse vertical resolution, describe environmental rather than storm-motion-adjusted flow, are not storm-relative, and are not based on a Bunkers storm-motion estimate. They are intended only as a qualitative supplementary view of vertical wind-vector structure.
"""
    caption_zh = """# 图S3图注（中文）

**图S3. C0、C1和C2的粗分辨率环境u–v风廓线。** 每条曲线连接850、700、500、400、300、250和200 hPa各层u、v分量的事件中位数，图中仅标注部分代表性气压层（hPa）。三个面板采用完全一致的u、v范围和等比例坐标。这些廓线的垂直分辨率较粗，描述的是环境风而非经风暴移动订正的气流，不属于风暴相对风，也不基于Bunkers风暴移动估计；其用途仅限于定性补充展示垂直风矢量结构。
"""
    assert_public_text_clean(caption_en, source=str(CAPTION_EN))
    assert_public_text_clean(caption_zh, source=str(CAPTION_ZH))
    write_text_atomic(CAPTION_EN, caption_en)
    write_text_atomic(CAPTION_ZH, caption_zh)

    visible_inventory = (
        QC_DIR
        / "publication_visible_text"
        / f"{OUTPUT.stem}_visible_text.json"
    )
    visible_text = "\n".join(json.loads(visible_inventory.read_text(encoding="utf-8"))["texts"])
    assert_public_text_clean(visible_text, source=str(OUTPUT))

    level_n = (
        summary.pivot(index="level_hPa", columns="regime", values="n_valid")
        .sort_index(ascending=False)
        .to_dict()
    )
    write_figure_qc(
        QC_REPORT,
        figure_id="Fig. S3",
        status_label="SUPPLEMENT_ONLY",
        scientific_checks=[
            "The source contains 7,110 rows for 790 unique events and nine available pressure levels.",
            "The displayed curves use only the seven accepted environmental-wind levels from 850 to 200 hPa.",
            "All 21 displayed u medians and all 21 displayed v medians reproduce the accepted primary-statistics table within 0.011 m s−1.",
            f"Level-specific valid sample sizes were retained in the plotting snapshot: {level_n}.",
            "No storm motion, storm-relative transformation, interpolation, or additional vector diagnostic was calculated.",
        ],
        layout_checks=[
            "The figure uses three equal-width panels in C0, C1, C2 order.",
            "Every panel has identical x and y limits and an equal data aspect ratio.",
            "Only 850, 500, and 200 hPa are labeled to reduce crowding; all seven accepted levels remain plotted.",
            "The file is a 200 dpi RGB review image with no alpha channel.",
            "All publication-visible text and both captions passed the internal-term search.",
        ],
        interpretation_checks=[
            "The caption explicitly states coarse vertical resolution, environmental scope, and the absence of storm-relative or Bunkers processing.",
            "The curves are presented as qualitative geometry, not as a convective-storm motion diagnostic.",
            "Component-wise medians are identified on both axes and in the caption.",
        ],
        deviations=[],
        review_notes=[
            "The available 1000- and 925-hPa raw levels are not shown because the accepted wind-statistics product and adjudicated coarse profile use the seven 850–200-hPa levels.",
            "No publisher-specific font or line-width compliance is claimed because a target journal has not yet been designated.",
        ],
    )


def build() -> dict:
    ensure_output_dirs()
    summary = _load_and_summarize()
    write_csv_atomic(SNAPSHOT, summary)
    _draw(summary)
    _write_caption_and_qc(summary)

    image_info = inspect_png(OUTPUT)
    assert_exact("FigS3 mode", image_info["mode"], "RGB")
    assert_exact("FigS3 alpha", image_info["alpha_present"], False)
    metadata = {
        "figure_id": "FigS3",
        "status_label": "SUPPLEMENT_ONLY",
        "output_stage": "200_DPI_RGB_REVIEW_PROTOTYPE",
        "figure_file": relative_source(OUTPUT),
        "plotting_snapshot": relative_source(SNAPSHOT),
        "caption_en": relative_source(CAPTION_EN),
        "caption_zh": relative_source(CAPTION_ZH),
        "qc_report": relative_source(QC_REPORT),
        "script": relative_source(Path(__file__)),
        "inputs": [
            {"path": PROFILE_SOURCE, "sha256": sha256_file(PROJECT_ROOT / PROFILE_SOURCE)},
            {
                "path": ACCEPTED_STATS_SOURCE,
                "sha256": sha256_file(PROJECT_ROOT / ACCEPTED_STATS_SOURCE),
            },
            {"path": CLAIM_SOURCE, "sha256": sha256_file(PROJECT_ROOT / CLAIM_SOURCE)},
        ],
        "render": image_info,
        "scientific_result_changed": False,
        "pressure_levels_displayed_hPa": list(PRESSURE_LEVELS),
        "storm_motion_calculated": False,
    }
    write_json_atomic(METADATA, metadata)
    write_text_atomic(
        BUILD_LOG,
        "\n".join(
            [
                "FIGURE_ID=FigS3",
                "STATUS=BUILT_AND_QC_COMPLETE",
                f"OUTPUT={relative_source(OUTPUT)}",
                f"SHA256={image_info['sha256']}",
                "DISPLAY_LEVELS_HPA=850,700,500,400,300,250,200",
                "AGGREGATION=COMPONENT_WISE_EVENT_MEDIAN",
                "STORM_RELATIVE=FALSE",
                "BUNKERS_BASED=FALSE",
                "STATUS_LABEL=SUPPLEMENT_ONLY",
                "",
            ]
        ),
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
