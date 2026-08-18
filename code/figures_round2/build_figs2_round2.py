"""Build Fig. S2 from accepted Cliff's delta and quantile summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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


EFFECT_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/24_post_hoc_context_audit/"
    "01_k3_pairwise_cliffs_delta_new790.csv"
)
OVERLAP_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/24_post_hoc_context_audit/"
    "02_k3_distribution_overlap_new790.csv"
)

OUTPUT = SUPP_FIGURE_DIR / "FigS2_effect_sizes_overlap_ROUND2_review.png"
SNAPSHOT = PLOTTING_DATA_DIR / "FigS2_plotting_snapshot.csv"
CAPTION_EN = CAPTION_DIR / "FigS2_caption_en.md"
CAPTION_ZH = CAPTION_DIR / "FigS2_caption_zh.md"
METADATA = QC_DIR / "FigS2_metadata.json"
QC_REPORT = QC_DIR / "FigS2_QC.md"
BUILD_LOG = LOG_DIR / "FigS2_build.log"

VARIABLE_ORDER = (
    "MLCAPE_Jkg",
    "MLLCL_m",
    "ERA5_d2m_K",
    "SHR6_ms",
    "SRH1_m2s2",
)
VARIABLE_LABELS = {
    "MLCAPE_Jkg": "MLCAPE (J kg$^{-1}$)",
    "MLLCL_m": "MLLCL (m)",
    "ERA5_d2m_K": "2-m dewpoint (K)",
    "SHR6_ms": "0–6-km shear (m s$^{-1}$)",
    "SRH1_m2s2": "0–1-km SRH (m$^2$ s$^{-2}$)",
}
COMPARISON_ORDER = ("C0_vs_C1", "C0_vs_C2", "C1_vs_C2")
COMPARISON_LABELS = {
    "C0_vs_C1": "C0 − C1",
    "C0_vs_C2": "C0 − C2",
    "C1_vs_C2": "C1 − C2",
}
COMPARISON_COLORS = {
    "C0_vs_C1": "#8C6D31",
    "C0_vs_C2": "#2B6F8A",
    "C1_vs_C2": "#7A4E8A",
}
COMPARISON_MARKERS = {"C0_vs_C1": "o", "C0_vs_C2": "s", "C1_vs_C2": "^"}
REGIME_N = {"C0": 131, "C1": 307, "C2": 352}


def _load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    effect = read_csv_checked(
        EFFECT_SOURCE,
        required_columns=(
            "comparison",
            "variable",
            "n_a",
            "n_b",
            "cliffs_delta",
            "ci_low",
            "ci_high",
        ),
    )
    overlap = read_csv_checked(
        OVERLAP_SOURCE,
        required_columns=(
            "variable",
            "regime",
            "n",
            "p10",
            "p25",
            "median",
            "p75",
            "p90",
        ),
    )
    assert_exact("FigS2 effect rows", len(effect), 15)
    assert_exact("FigS2 overlap rows", len(overlap), 15)
    assert_exact(
        "FigS2 comparisons",
        tuple(effect["comparison"].drop_duplicates()),
        COMPARISON_ORDER,
    )
    assert_exact(
        "FigS2 effect variables",
        set(effect["variable"]),
        set(VARIABLE_ORDER),
    )
    assert_exact(
        "FigS2 overlap variables",
        set(overlap["variable"]),
        set(VARIABLE_ORDER),
    )
    for regime in REGIME_ORDER:
        observed_n = overlap.loc[overlap["regime"] == regime, "n"].unique()
        assert_exact(f"FigS2 {regime} n values", observed_n.tolist(), [REGIME_N[regime]])
    if not (
        (effect["ci_low"] <= effect["cliffs_delta"])
        & (effect["cliffs_delta"] <= effect["ci_high"])
    ).all():
        raise AssertionError("FigS2 effect estimates are not enclosed by every CI")
    if not (
        (overlap["p10"] <= overlap["p25"])
        & (overlap["p25"] <= overlap["median"])
        & (overlap["median"] <= overlap["p75"])
        & (overlap["p75"] <= overlap["p90"])
    ).all():
        raise AssertionError("FigS2 quantile ordering failed")
    return effect, overlap


def _write_snapshot(effect: pd.DataFrame, overlap: pd.DataFrame) -> None:
    effect_snapshot = effect.copy()
    effect_snapshot.insert(0, "record_type", "pairwise_cliffs_delta")
    overlap_snapshot = overlap.copy()
    overlap_snapshot.insert(0, "record_type", "distribution_quantiles")
    write_csv_atomic(
        SNAPSHOT,
        pd.concat([effect_snapshot, overlap_snapshot], ignore_index=True, sort=False),
    )


def _draw(effect: pd.DataFrame, overlap: pd.DataFrame) -> None:
    width, height = FIGURE_SIZES_MM["FigS2"]
    with manuscript_style():
        fig = plt.figure(figsize=mm_to_inches(width, height))
        outer = fig.add_gridspec(
            2,
            1,
            height_ratios=[1.02, 1.62],
            left=0.205,
            right=0.985,
            bottom=0.075,
            top=0.94,
            hspace=0.33,
        )
        ax_effect = fig.add_subplot(outer[0, 0])
        bottom = outer[1, 0].subgridspec(5, 1, hspace=0.72)
        quantile_axes = [fig.add_subplot(bottom[index, 0]) for index in range(5)]

        base_positions = np.arange(len(VARIABLE_ORDER))
        offsets = {"C0_vs_C1": -0.22, "C0_vs_C2": 0.0, "C1_vs_C2": 0.22}
        for comparison in COMPARISON_ORDER:
            rows = (
                effect.loc[effect["comparison"] == comparison]
                .set_index("variable")
                .loc[list(VARIABLE_ORDER)]
            )
            y = base_positions + offsets[comparison]
            x = rows["cliffs_delta"].to_numpy()
            xerr = np.vstack(
                [
                    x - rows["ci_low"].to_numpy(),
                    rows["ci_high"].to_numpy() - x,
                ]
            )
            ax_effect.errorbar(
                x,
                y,
                xerr=xerr,
                fmt=COMPARISON_MARKERS[comparison],
                color=COMPARISON_COLORS[comparison],
                ecolor=COMPARISON_COLORS[comparison],
                elinewidth=1.0,
                capsize=2.2,
                capthick=0.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
                markersize=4.6,
                label=COMPARISON_LABELS[comparison],
                zorder=3,
            )
        ax_effect.axvline(0, color="#555555", linewidth=0.8, linestyle=":")
        ax_effect.set_xlim(-1.03, 1.03)
        ax_effect.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
        ax_effect.set_yticks(base_positions, [VARIABLE_LABELS[v] for v in VARIABLE_ORDER])
        ax_effect.invert_yaxis()
        ax_effect.set_xlabel("Cliff’s delta (first regime minus second)")
        ax_effect.set_title(
            "Pairwise effect sizes with 95% confidence intervals",
            loc="left",
            pad=5,
        )
        ax_effect.legend(
            loc="upper right",
            ncol=3,
            frameon=False,
            handletextpad=0.45,
            columnspacing=1.4,
        )
        style_axis(ax_effect, grid_axis="x")
        panel_label(ax_effect, "(a)", x=-0.185, y=1.03)

        quantile_legend = [
            Line2D(
                [],
                [],
                color=REGIME_COLORS[regime],
                marker="o",
                markerfacecolor="white",
                markeredgewidth=0.8,
                linewidth=2.2,
                label=f"{regime} (n={REGIME_N[regime]})",
            )
            for regime in REGIME_ORDER
        ]
        for index, (variable, ax) in enumerate(zip(VARIABLE_ORDER, quantile_axes)):
            rows = overlap.loc[overlap["variable"] == variable].set_index("regime")
            for position, regime in enumerate(REGIME_ORDER):
                row = rows.loc[regime]
                ax.plot(
                    [row["p10"], row["p90"]],
                    [position, position],
                    color=REGIME_COLORS[regime],
                    linewidth=1.25,
                    alpha=0.65,
                    solid_capstyle="round",
                )
                ax.plot(
                    [row["p25"], row["p75"]],
                    [position, position],
                    color=REGIME_COLORS[regime],
                    linewidth=4.0,
                    alpha=0.72,
                    solid_capstyle="butt",
                )
                ax.plot(
                    row["median"],
                    position,
                    marker="o",
                    color=REGIME_COLORS[regime],
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    markersize=4.1,
                    zorder=3,
                )
            values = rows[["p10", "p90"]].to_numpy()
            value_range = float(np.nanmax(values) - np.nanmin(values))
            padding = max(0.08 * value_range, 0.2)
            ax.set_xlim(float(np.nanmin(values) - padding), float(np.nanmax(values) + padding))
            ax.set_ylim(-0.55, 2.55)
            ax.set_yticks([0, 1, 2], REGIME_ORDER)
            ax.invert_yaxis()
            ax.set_xlabel(VARIABLE_LABELS[variable], labelpad=1.5)
            ax.tick_params(axis="x", labelsize=6.5, pad=1)
            ax.tick_params(axis="y", labelsize=6.6)
            style_axis(ax, grid_axis="x")
            if index == 0:
                ax.set_title(
                    "Quantile-based distribution overlap on variable-specific raw scales",
                    loc="left",
                    pad=16,
                )
                ax.legend(
                    handles=quantile_legend,
                    loc="lower right",
                    bbox_to_anchor=(1.0, 1.11),
                    ncol=3,
                    frameon=False,
                    handlelength=2.4,
                    columnspacing=1.6,
                )
                panel_label(ax, "(b)", x=-0.185, y=1.38)

        save_figure_atomic(fig, OUTPUT, dpi=PROTOTYPE_DPI)
        plt.close(fig)


def _write_caption_and_qc() -> None:
    caption_en = """# Fig. S2 caption (English)

**Figure S2. Pairwise separation and distribution overlap for the five clustering variables.** (a) Cliff’s delta and 95% confidence intervals for each regime comparison. A positive value indicates that the first regime named in the legend tends to have larger values than the second. (b) Accepted quantile summaries displayed on variable-specific raw scales: thin lines span the 10th–90th percentiles, thick lines span the interquartile range, and open circles mark medians. The direct overlap of these intervals visualizes distributional overlap without introducing a scalar overlap coefficient. Sample sizes are C0, 131; C1, 307; and C2, 352.
"""
    caption_zh = """# 图S2图注（中文）

**图S2. 五个聚类变量的两两分离程度与分布重叠。**（a）各环境型比较的Cliff’s delta及95%置信区间；正值表示图例中前一个环境型的取值整体倾向于高于后一个环境型。（b）在各变量原始量纲上展示已接受的分位数汇总：细线覆盖第10–90百分位，粗线表示四分位距，空心圆表示中位数。区间的直接重叠用于展示分布重叠，不引入新的标量重叠系数。样本量为C0 131例、C1 307例、C2 352例。
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

    write_figure_qc(
        QC_REPORT,
        figure_id="Fig. S2",
        status_label="SUPPLEMENT_CANDIDATE",
        scientific_checks=[
            "Panel (a) uses all 15 accepted Cliff’s delta estimates and their existing confidence intervals.",
            "Panel (b) uses all 15 accepted p10, p25, median, p75, and p90 summaries.",
            "Sample sizes close at C0=131, C1=307, and C2=352 for every variable.",
            "Every confidence interval encloses its accepted effect estimate, and every quantile row is monotonically ordered.",
            "No effect-size or scalar overlap metric was calculated.",
        ],
        layout_checks=[
            "The figure contains two vertically stacked scientific panels.",
            "Panel (a) uses a zero reference and the full theoretical Cliff’s delta range.",
            "Panel (b) keeps each variable in its original unit and uses a common regime encoding.",
            "The file is a 200 dpi RGB review image with no alpha channel.",
            "All publication-visible text and both captions passed the internal-term search.",
        ],
        interpretation_checks=[
            "The sign convention for Cliff’s delta is stated in the x-axis and caption.",
            "Distribution overlap is shown directly through accepted quantile intervals, without converting them to a new score.",
            "Variable-specific scales are explicitly identified, so distances are not compared across physical units.",
        ],
        deviations=[
            "A scalar color-matrix representation was replaced by a five-row quantile-overlap display because no accepted scalar overlap coefficient exists. Constructing such a matrix would require a new metric and would violate the scientific scope.",
        ],
        review_notes=[
            "Magnitude classifications remain available in the source table but were not redundantly printed on the figure face.",
            "No publisher-specific font or line-width compliance is claimed because a target journal has not yet been designated.",
        ],
    )


def build() -> dict:
    ensure_output_dirs()
    effect, overlap = _load_sources()
    _write_snapshot(effect, overlap)
    _draw(effect, overlap)
    _write_caption_and_qc()

    image_info = inspect_png(OUTPUT)
    assert_exact("FigS2 mode", image_info["mode"], "RGB")
    assert_exact("FigS2 alpha", image_info["alpha_present"], False)
    metadata = {
        "figure_id": "FigS2",
        "status_label": "SUPPLEMENT_CANDIDATE",
        "output_stage": "200_DPI_RGB_REVIEW_PROTOTYPE",
        "figure_file": relative_source(OUTPUT),
        "plotting_snapshot": relative_source(SNAPSHOT),
        "caption_en": relative_source(CAPTION_EN),
        "caption_zh": relative_source(CAPTION_ZH),
        "qc_report": relative_source(QC_REPORT),
        "script": relative_source(Path(__file__)),
        "inputs": [
            {"path": EFFECT_SOURCE, "sha256": sha256_file(PROJECT_ROOT / EFFECT_SOURCE)},
            {"path": OVERLAP_SOURCE, "sha256": sha256_file(PROJECT_ROOT / OVERLAP_SOURCE)},
        ],
        "render": image_info,
        "scientific_result_changed": False,
        "new_metric_calculated": False,
    }
    write_json_atomic(METADATA, metadata)
    write_text_atomic(
        BUILD_LOG,
        "\n".join(
            [
                "FIGURE_ID=FigS2",
                "STATUS=BUILT_AND_QC_COMPLETE",
                f"OUTPUT={relative_source(OUTPUT)}",
                f"SHA256={image_info['sha256']}",
                "PAIRWISE_EFFECT_ROWS=15",
                "QUANTILE_SUMMARY_ROWS=15",
                "NEW_EFFECT_SIZE=NONE",
                "NEW_OVERLAP_METRIC=NONE",
                "LAYOUT_DEVIATION=SCALAR_MATRIX_REPLACED_BY_ACCEPTED_QUANTILE_DISPLAY",
                "",
            ]
        ),
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
