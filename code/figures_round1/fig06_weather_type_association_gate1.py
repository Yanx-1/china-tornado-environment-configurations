"""Gate 1 prototype for Fig. 6: post-hoc weather-type association."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


SCRIPT_PATH = Path(__file__).resolve()
ROUND_ROOT = SCRIPT_PATH.parent.parent
sys.path.insert(0, str(ROUND_ROOT / "03_config"))

from figure_style_config import (  # noqa: E402
    FIGURE_SIZES_MM,
    PROTOTYPE_DPI,
    manuscript_style,
    mm_to_inches,
    panel_label,
)
from figure_io import (  # noqa: E402
    CAPTION_DIR,
    LOG_DIR,
    PLOTTING_DATA_DIR,
    PNG_DIR,
    QC_DIR,
    ensure_output_dirs,
    project_path,
    read_csv_checked,
    relative_source,
    save_figure_atomic,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from figure_qc import (  # noqa: E402
    assert_close,
    assert_exact,
    inspect_png,
    scan_forbidden_language,
    write_qc_report,
)


COUNT_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "24_post_hoc_context_audit/05_k3_macro9_contingency_counts.csv"
)
ROW_PERCENT_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "24_post_hoc_context_audit/06_k3_macro9_row_percent.csv"
)
RESIDUAL_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "24_post_hoc_context_audit/10_macro9_standardized_residuals.csv"
)
HANDOFF_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "24_post_hoc_context_audit/CHATGPT_HANDOFF.yaml"
)
ASSERTION_SOURCE = ROUND_ROOT / "00_source_registry" / "gate0_frozen_number_assertions.csv"
REGIME_ORDER = ("C0", "C1", "C2")
WEATHER_ORDER = (
    "QLCS/飑线",
    "TC",
    "其他",
    "冷涡",
    "华北对流",
    "暖区",
    "气旋/冷锋",
    "西南对流",
    "超单(未分类)",
)
WEATHER_DISPLAY = (
    "QLCS",
    "TC",
    "Other",
    "Cold vortex",
    "N-China conv.",
    "Warm sector",
    "Cyclone / CF",
    "SW-China conv.",
    "Supercell (uncl.)",
)
REGIME_COUNTS = {"C0": 130, "C1": 306, "C2": 351}
CHI_SQUARE_EXACT = 437.139758077693
CHI_SQUARE_DISPLAY = 437.1
RAW_V = 0.526996571456
CORRECTED_V = 0.517249716557
CI_LOW = 0.489776578475
CI_HIGH = 0.575757611897
PERMUTATION_P = 0.000099990001


def _assert_registered_numbers() -> None:
    assertions = pd.read_csv(ASSERTION_SOURCE, encoding="utf-8-sig").set_index("check")

    def observed(check: str) -> str:
        return str(assertions.loc[check, "observed"])

    assert_close(
        "registered chi-square exact",
        float(observed("weather Pearson chi-square")),
        CHI_SQUARE_EXACT,
        5e-7,
    )
    assert_close(
        "registered chi-square display",
        float(observed("weather Pearson chi-square display value")),
        CHI_SQUARE_DISPLAY,
        1e-12,
    )
    assert_exact(
        "registered df",
        int(float(observed("weather chi-square degrees of freedom"))),
        16,
    )
    assert_close(
        "registered raw V",
        float(observed("weather raw Cramér's V")),
        RAW_V,
        5e-12,
    )
    assert_close(
        "registered corrected V",
        float(observed("weather bias-corrected Cramér's V")),
        CORRECTED_V,
        5e-12,
    )
    assert_close(
        "registered CI lower",
        float(observed("weather bootstrap raw-V CI lower bound")),
        CI_LOW,
        5e-12,
    )
    assert_close(
        "registered CI upper",
        float(observed("weather bootstrap raw-V CI upper bound")),
        CI_HIGH,
        5e-12,
    )
    permutation_observed = observed("weather permutation p")
    if "0.000099990001" not in permutation_observed:
        raise AssertionError(
            f"registered permutation p is not the frozen value: {permutation_observed}"
        )
    statuses = assertions.loc[
        [
            "weather Pearson chi-square",
            "weather Pearson chi-square display value",
            "weather chi-square degrees of freedom",
            "weather raw Cramér's V",
            "weather bias-corrected Cramér's V",
            "weather bootstrap raw-V CI lower bound",
            "weather bootstrap raw-V CI upper bound",
            "weather permutation p",
        ],
        "status",
    ].tolist()
    assert_exact("registered weather-statistic statuses", statuses, ["PASS"] * 8)


def _format_weather_axis(ax) -> None:
    ax.set_xticks(range(len(WEATHER_ORDER)), WEATHER_DISPLAY)
    ax.tick_params(axis="x", labelsize=5.6, pad=1.5)
    ax.tick_params(axis="y", length=0)
    for label in ax.get_xticklabels():
        label.set_rotation(48)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#606060")


def build() -> dict[str, object]:
    ensure_output_dirs()
    counts = read_csv_checked(COUNT_SOURCE, required_columns=("k3_formal", *WEATHER_ORDER))
    row_percent_frozen = read_csv_checked(
        ROW_PERCENT_SOURCE, required_columns=("k3_formal", *WEATHER_ORDER)
    )
    residual_frozen = read_csv_checked(
        RESIDUAL_SOURCE, required_columns=("k3_formal", *WEATHER_ORDER)
    )
    counts = counts.set_index("k3_formal").loc[list(REGIME_ORDER), list(WEATHER_ORDER)].astype(int)
    row_percent_frozen = (
        row_percent_frozen.set_index("k3_formal")
        .loc[list(REGIME_ORDER), list(WEATHER_ORDER)]
        .astype(float)
    )
    residual_frozen = (
        residual_frozen.set_index("k3_formal")
        .loc[list(REGIME_ORDER), list(WEATHER_ORDER)]
        .astype(float)
    )

    assert_exact("weather column order", tuple(counts.columns), WEATHER_ORDER)
    assert_exact("regime sample sizes", counts.sum(axis=1).astype(int).to_dict(), REGIME_COUNTS)
    assert_exact("weather contingency closure", int(counts.to_numpy().sum()), 787)
    row_proportion = counts.div(counts.sum(axis=1), axis=0)
    assert_close(
        "frozen row-percent rounding",
        float(np.max(np.abs(row_proportion.to_numpy() - row_percent_frozen.to_numpy()))),
        0.0,
        5.1e-5,
    )

    chi_square, asymptotic_p, df, expected = chi2_contingency(
        counts.to_numpy(dtype=float), correction=False
    )
    residual_calculated = (counts.to_numpy(dtype=float) - expected) / np.sqrt(expected)
    assert_close("chi-square from table", chi_square, CHI_SQUARE_EXACT, 5e-7)
    assert_close("chi-square display", round(chi_square, 1), CHI_SQUARE_DISPLAY, 1e-12)
    assert_exact("degrees of freedom from table", int(df), 16)
    assert_exact("valid n from table", int(counts.to_numpy().sum()), 787)
    if not asymptotic_p < 0.0001:
        raise AssertionError(f"asymptotic p does not meet reporting threshold: {asymptotic_p}")
    assert_close(
        "standardized residual source",
        float(np.max(np.abs(residual_calculated - residual_frozen.to_numpy()))),
        0.0,
        1e-12,
    )

    n = float(counts.to_numpy().sum())
    rows, columns = counts.shape
    raw_v_calculated = np.sqrt(chi_square / (n * min(rows - 1, columns - 1)))
    phi_squared = chi_square / n
    phi_squared_corrected = max(
        0.0, phi_squared - ((columns - 1) * (rows - 1)) / (n - 1)
    )
    # Preserve the accepted audit's numerator-only finite-sample correction.
    # This is intentionally not replaced by an alternative corrected-dimension
    # convention, because that would change the frozen reporting value.
    corrected_v_calculated = np.sqrt(
        phi_squared_corrected / min(rows - 1, columns - 1)
    )
    assert_close("raw V from table", raw_v_calculated, RAW_V, 5e-12)
    assert_close(
        "bias-corrected V from table",
        corrected_v_calculated,
        CORRECTED_V,
        5e-12,
    )
    _assert_registered_numbers()

    handoff_text = project_path(HANDOFF_SOURCE).read_text(encoding="utf-8")
    required_handoff_fragments = (
        "WEATHER_TYPE_VALID_N: 787",
        "MACRO9_DF: 16",
        "MACRO9_CRAMERS_V: 0.527",
        "MACRO9_CORRECTED_V: 0.5172",
        "MACRO9_CI_LOW: 0.4898",
        "MACRO9_CI_HIGH: 0.5758",
        "MACRO9_PERMUTATION_P: <0.0001",
    )
    missing_handoff = [
        fragment for fragment in required_handoff_fragments if fragment not in handoff_text
    ]
    assert_exact("handoff statistic fragments missing", missing_handoff, [])

    source_counts = relative_source(project_path(COUNT_SOURCE))
    source_row_percent = relative_source(project_path(ROW_PERCENT_SOURCE))
    source_residual = relative_source(project_path(RESIDUAL_SOURCE))
    source_assertion = relative_source(ASSERTION_SOURCE)
    records: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        for weather_type in WEATHER_ORDER:
            proportion = float(row_proportion.loc[regime, weather_type])
            residual = float(residual_frozen.loc[regime, weather_type])
            records.append(
                {
                    "regime": regime,
                    "regime_n": REGIME_COUNTS[regime],
                    "weather_type": weather_type,
                    "weather_order": WEATHER_ORDER.index(weather_type) + 1,
                    "count": int(counts.loc[regime, weather_type]),
                    "row_proportion": proportion,
                    "standardized_residual": residual,
                    "proportion_annotated": proportion >= 0.25,
                    "residual_annotated": abs(residual) >= 4.0,
                    "valid_n": 787,
                    "chi_square_exact": CHI_SQUARE_EXACT,
                    "chi_square_display": CHI_SQUARE_DISPLAY,
                    "df": 16,
                    "raw_cramers_v": RAW_V,
                    "bias_corrected_cramers_v": CORRECTED_V,
                    "bootstrap_ci_low": CI_LOW,
                    "bootstrap_ci_high": CI_HIGH,
                    "permutation_p": PERMUTATION_P,
                    "source_count_file": source_counts,
                    "source_row_percent_file": source_row_percent,
                    "source_residual_file": source_residual,
                    "source_assertion_file": source_assertion,
                    "transformation_note": (
                        "row proportion recalculated from the frozen table and "
                        "checked against the frozen rounded source"
                    ),
                }
            )
    plotting = pd.DataFrame(records)
    assert_exact("Fig.6 plotting rows", len(plotting), 27)
    data_path = PLOTTING_DATA_DIR / "Fig6_plotting_data.csv"
    output_path = PNG_DIR / "Fig6_weather_type_association_GATE1_prototype.png"
    write_csv_atomic(data_path, plotting)

    with manuscript_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=mm_to_inches(*FIGURE_SIZES_MM["Fig6"]),
            layout="constrained",
            gridspec_kw={"width_ratios": [1, 1]},
        )
        ax_a, ax_b = axes
        fig.suptitle(
            "Post-hoc association · n=787 · χ²(16)=437.1 · raw V=0.5270 · "
            "bias-corrected V=0.5172\n"
            "95% CI=0.4898–0.5758 · permutation p<0.0001",
            fontsize=7.0,
            fontweight="normal",
        )

        proportion_values = row_proportion.to_numpy(dtype=float)
        image_a = ax_a.imshow(
            proportion_values,
            cmap="Blues",
            vmin=0,
            vmax=0.5,
            aspect="auto",
            interpolation="none",
        )
        for row in range(proportion_values.shape[0]):
            for column in range(proportion_values.shape[1]):
                value = float(proportion_values[row, column])
                if value >= 0.25:
                    ax_a.text(
                        column,
                        row,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        color="white" if value >= 0.37 else "#202020",
                        fontweight="bold",
                    )
        _format_weather_axis(ax_a)
        ax_a.set_yticks(
            range(len(REGIME_ORDER)),
            [f"{regime} (n={REGIME_COUNTS[regime]})" for regime in REGIME_ORDER],
        )
        ax_a.set_title("Within-regime composition")
        colorbar_a = fig.colorbar(image_a, ax=ax_a, fraction=0.036, pad=0.025)
        colorbar_a.set_label("Row proportion")
        colorbar_a.outline.set_linewidth(0.55)
        panel_label(ax_a, "(a)")

        residual_values = residual_frozen.to_numpy(dtype=float)
        limit = 10.0
        image_b = ax_b.imshow(
            residual_values,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
            interpolation="none",
        )
        for row in range(residual_values.shape[0]):
            for column in range(residual_values.shape[1]):
                value = float(residual_values[row, column])
                if abs(value) >= 4.0:
                    ax_b.text(
                        column,
                        row,
                        f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=6.1,
                        color="white" if abs(value) >= 5.6 else "#202020",
                        fontweight="bold",
                    )
        _format_weather_axis(ax_b)
        ax_b.set_yticks(
            range(len(REGIME_ORDER)),
            [f"{regime} (n={REGIME_COUNTS[regime]})" for regime in REGIME_ORDER],
        )
        ax_b.set_title("Standardized residuals")
        colorbar_b = fig.colorbar(image_b, ax=ax_b, fraction=0.036, pad=0.025)
        colorbar_b.set_label("Standardized residual")
        colorbar_b.outline.set_linewidth(0.55)
        panel_label(ax_b, "(b)")

        save_figure_atomic(fig, output_path, dpi=PROTOTYPE_DPI)
        plt.close(fig)

    caption_en = """# Fig. 6 caption (English)

Post-hoc association between the three environmental regimes and the nine
weather-type categories among 787 events with non-missing weather type.
(a) Weather-type composition normalized within each regime; only proportions
of at least 25% are printed, and regime sample sizes are shown on the ordinate.
(b) Pearson standardized residuals from the same 3×9 contingency table;
only cells with |residual| at least 4 are labeled. Positive and negative
residuals indicate over- and under-representation relative to independence,
respectively. The association is χ²(16)=437.1, raw Cramér’s V=0.5270,
bias-corrected Cramér’s V=0.5172, 95% bootstrap CI=0.4898–0.5758, and
permutation p<0.0001. The distributed cell patterns show a many-to-many
relationship and are interpreted descriptively within the confirmed-tornado
sample.
"""
    caption_zh = """# Fig. 6 图注（中文）

在天气型非缺失的787个事件中，3个环境组与9类天气型之间的事后关联。
（a）各环境组内部归一化的天气型构成；仅标注不低于25%的比例，纵轴同时给出
各组样本量。（b）由同一3×9列联表得到的Pearson标准化残差；仅标注
|残差|不低于4的单元格。正、负残差分别表示相对于独立性预期的偏多和偏少。
关联统计量为χ²(16)=437.1，原始Cramér’s V=0.5270，偏差校正Cramér’s
V=0.5172，95% bootstrap CI=0.4898–0.5758，置换p<0.0001。分散于多个
单元格的结构体现多对多关系，仅在已确认龙卷样本内部作描述性解释。
"""
    prohibited = scan_forbidden_language(caption_en + caption_zh)
    if prohibited:
        raise AssertionError(f"Fig.6 caption contains prohibited text: {prohibited}")
    if "587.3" in caption_en + caption_zh:
        raise AssertionError("Deprecated reporting number entered the Fig.6 caption.")
    write_text_atomic(CAPTION_DIR / "Fig6_caption_en.md", caption_en)
    write_text_atomic(CAPTION_DIR / "Fig6_caption_zh.md", caption_zh)

    metadata = inspect_png(output_path)
    write_json_atomic(QC_DIR / "Fig6_file_metadata.json", metadata)
    write_qc_report(
        QC_DIR / "Fig6_qc_report.md",
        figure_id="Fig.6",
        status="PASS_WITH_NONBLOCKING_NOTES",
        data_checks=[
            "Counts, row percentages, and standardized residuals retain the fixed nine-category order.",
            "All panels use the same accepted frozen 3×9 table; no missing event is imputed.",
            "The three displayed regime sample sizes are 130, 306, and 351 and close to n=787.",
        ],
        number_checks=[
            "Direct Pearson recomputation yields chi-square 437.139758077693, df=16, and display value 437.1.",
            "Direct effect-size checks yield raw V 0.5270 and bias-corrected V 0.5172 at reporting precision.",
            "The accepted bootstrap interval 0.4898–0.5758 and permutation p below 0.0001 match the Gate 0 registry and handoff.",
        ],
        interpretation_checks=[
            "The caption explicitly identifies a descriptive post-hoc association.",
            "The distribution of highlighted cells is described as a many-to-many relationship.",
            "Weather type is contextual information and does not alter the formal environmental labels.",
        ],
        visual_checks=[
            f"Prototype PNG is {metadata['width_px']}×{metadata['height_px']} px in RGB mode.",
            "Only proportions at least 25% and residual magnitudes at least 4 are printed.",
            "Sequential and diverging panels use separate units and color scales.",
        ],
        nonblocking_notes=[
            "Final physical-size typography, vector fonts, and publisher-specific rules remain for Gate 4.",
        ],
    )

    write_text_atomic(
        LOG_DIR / "Fig6_build.log",
        f"""FIGURE=Fig6
STATUS=PASS_WITH_NONBLOCKING_NOTES
SOURCE_COUNTS={project_path(COUNT_SOURCE)}
SOURCE_ROW_PERCENT={project_path(ROW_PERCENT_SOURCE)}
SOURCE_RESIDUALS={project_path(RESIDUAL_SOURCE)}
SOURCE_HANDOFF={project_path(HANDOFF_SOURCE)}
SOURCE_ASSERTIONS={ASSERTION_SOURCE}
PLOTTING_DATA={data_path}
OUTPUT={output_path}
TABLE_SHAPE=3x9
VALID_N=787
CHI_SQUARE_EXACT={chi_square:.12f}
CHI_SQUARE_DISPLAY=437.1
DF=16
RAW_CRAMERS_V={raw_v_calculated:.12f}
BIAS_CORRECTED_CRAMERS_V={corrected_v_calculated:.12f}
BOOTSTRAP_CI=0.4898-0.5758
PERMUTATION_P=<0.0001
SCIENTIFIC_REANALYSIS=FALSE
""",
    )
    return {
        "figure_id": "Fig6",
        "status": "PASS_WITH_NONBLOCKING_NOTES",
        "output": str(output_path),
        "plotting_data": str(data_path),
        "script": str(SCRIPT_PATH),
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(build())
