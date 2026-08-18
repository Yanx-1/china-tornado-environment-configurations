"""Gate 1 prototype for Fig. 7: temperature and humidity profiles."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter


SCRIPT_PATH = Path(__file__).resolve()
ROUND_ROOT = SCRIPT_PATH.parent.parent
sys.path.insert(0, str(ROUND_ROOT / "03_config"))

from figure_style_config import (  # noqa: E402
    FIGURE_SIZES_MM,
    PROTOTYPE_DPI,
    REGIME_COLORS,
    REGIME_LINESTYLES,
    REGIME_MARKERS,
    REGIME_ORDER,
    manuscript_style,
    mm_to_inches,
    panel_label,
    style_axis,
)
from figure_io import (  # noqa: E402
    CAPTION_DIR,
    LOG_DIR,
    PLOTTING_DATA_DIR,
    PNG_DIR,
    QC_DIR,
    bootstrap_median_ci,
    ensure_output_dirs,
    project_path,
    read_csv_checked,
    relative_source,
    save_figure_atomic,
    stable_seed,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from figure_qc import (  # noqa: E402
    assert_exact,
    inspect_png,
    scan_forbidden_language,
    write_qc_report,
)


PROFILE_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "35_direction_b_full_core_execution/05_9level_event_profile_long.csv"
)
FROZEN_STATS_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "35_direction_b_full_core_execution/11_thermodynamic_primary_statistics.csv"
)
LEVELS = (1000, 925, 850, 700, 500, 400, 300, 250, 200)
VARIABLES = {
    "t_K": {"display": "Temperature", "xlabel": "Temperature (K)", "unit": "K"},
    "rh_pct": {
        "display": "Relative humidity",
        "xlabel": "Relative humidity (%)",
        "unit": "%",
    },
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BASE_SEED = 20260804


def build() -> dict[str, object]:
    ensure_output_dirs()
    profile = read_csv_checked(
        PROFILE_SOURCE,
        required_columns=(
            "event_id",
            "regime",
            "level_hPa",
            "t_K",
            "rh_pct",
            "below_ground",
            "valid",
        ),
        dtype={"event_id": str},
    )
    frozen_stats = read_csv_checked(
        FROZEN_STATS_SOURCE,
        required_columns=(
            "variable",
            "level_hPa",
            "regime",
            "n_valid",
            "median",
            "ci_low",
            "ci_high",
        ),
    )
    assert_exact("profile rows", len(profile), 790 * 9)
    assert_exact("profile unique event_id", profile["event_id"].nunique(), 790)
    assert_exact(
        "levels per event",
        profile.groupby("event_id").size().unique().tolist(),
        [9],
    )
    assert_exact(
        "profile levels",
        sorted(profile["level_hPa"].astype(int).unique().tolist(), reverse=True),
        list(LEVELS),
    )
    assert_exact(
        "profile regime counts",
        profile.groupby("regime")["event_id"].nunique().reindex(REGIME_ORDER).astype(int).tolist(),
        [131, 307, 352],
    )

    valid_mask = (
        profile["valid"].astype(str).str.lower().eq("true")
        & ~profile["below_ground"].astype(str).str.lower().eq("true")
    )
    records: list[dict[str, object]] = []
    source_display = relative_source(project_path(PROFILE_SOURCE))
    for variable, metadata in VARIABLES.items():
        for regime in REGIME_ORDER:
            for level in LEVELS:
                values = profile.loc[
                    valid_mask
                    & (profile["regime"] == regime)
                    & (profile["level_hPa"].astype(int) == level),
                    variable,
                ].dropna()
                median, ci_low, ci_high, n_valid = bootstrap_median_ci(
                    values,
                    seed=stable_seed(
                        BOOTSTRAP_BASE_SEED,
                        f"Fig7:{variable}:{regime}:{level}",
                    ),
                    n_boot=BOOTSTRAP_REPLICATES,
                )
                records.append(
                    {
                        "aggregation_level": "regime × pressure level",
                        "group_label": regime,
                        "variable": variable,
                        "variable_display": metadata["display"],
                        "level_hPa": level,
                        "median": median,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "unit": metadata["unit"],
                        "statistic": "median with percentile bootstrap 95% CI",
                        "sample_size": n_valid,
                        "source_file": source_display,
                        "transformation_note": (
                            "frozen valid=True and below_ground=False mask; "
                            f"{BOOTSTRAP_REPLICATES} event bootstrap replicates; "
                            "no smoothing or clipping"
                        ),
                    }
                )
    plotting = pd.DataFrame(records)
    assert_exact("Fig.7 plotting rows", len(plotting), 2 * 3 * 9)

    comparison = plotting.merge(
        frozen_stats[
            ["variable", "level_hPa", "regime", "n_valid", "median"]
        ],
        left_on=["variable", "level_hPa", "group_label"],
        right_on=["variable", "level_hPa", "regime"],
        how="inner",
        validate="one_to_one",
        suffixes=("_plot", "_frozen"),
    )
    assert_exact("frozen-stat comparison rows", len(comparison), 42)
    assert_exact(
        "all frozen n reproduced",
        bool(
            (
                comparison["sample_size"].astype(int)
                == comparison["n_valid"].astype(int)
            ).all()
        ),
        True,
    )
    maximum_median_difference = float(
        (
            comparison["median_plot"].astype(float)
            - comparison["median_frozen"].astype(float)
        )
        .abs()
        .max()
    )
    if maximum_median_difference > 0.011:
        raise AssertionError(
            "Fig.7 median reproduction exceeds rounded-table tolerance: "
            f"{maximum_median_difference}"
        )

    key_expected = {
        ("rh_pct", 500, "C0"): 54.3,
        ("rh_pct", 500, "C1"): 57.0,
        ("rh_pct", 500, "C2"): 87.7,
        ("t_K", 500, "C0"): 259.3,
        ("t_K", 500, "C1"): 268.7,
        ("t_K", 500, "C2"): 269.5,
    }
    for (variable, level, regime), expected_value in key_expected.items():
        observed_value = float(
            plotting.loc[
                (plotting["variable"] == variable)
                & (plotting["level_hPa"] == level)
                & (plotting["group_label"] == regime),
                "median",
            ].iloc[0]
        )
        if abs(observed_value - expected_value) > 0.06:
            raise AssertionError(
                f"Fig.7 key value mismatch for {variable}/{level}/{regime}: "
                f"{observed_value} versus {expected_value}"
            )

    data_path = PLOTTING_DATA_DIR / "Fig7_plotting_data.csv"
    n_path = PLOTTING_DATA_DIR / "Fig7_effective_n.csv"
    output_path = PNG_DIR / "Fig7_thermodynamic_profiles_GATE1_prototype.png"
    write_csv_atomic(data_path, plotting)
    write_csv_atomic(
        n_path,
        plotting[
            ["variable", "level_hPa", "group_label", "sample_size", "source_file"]
        ].copy(),
    )

    with manuscript_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=mm_to_inches(*FIGURE_SIZES_MM["Fig7"]),
            sharey=True,
            layout="constrained",
        )
        for panel_index, (variable, metadata) in enumerate(VARIABLES.items()):
            ax = axes[panel_index]
            ax.axhspan(850, 500, color="#EFEFEF", alpha=0.85, zorder=0)
            for regime in REGIME_ORDER:
                subset = (
                    plotting.loc[
                        (plotting["variable"] == variable)
                        & (plotting["group_label"] == regime)
                    ]
                    .set_index("level_hPa")
                    .reindex(LEVELS)
                    .reset_index()
                )
                pressure = subset["level_hPa"].to_numpy(dtype=float)
                median = subset["median"].to_numpy(dtype=float)
                ci_low = subset["ci_low"].to_numpy(dtype=float)
                ci_high = subset["ci_high"].to_numpy(dtype=float)
                ax.fill_betweenx(
                    pressure,
                    ci_low,
                    ci_high,
                    color=REGIME_COLORS[regime],
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    median,
                    pressure,
                    color=REGIME_COLORS[regime],
                    linestyle=REGIME_LINESTYLES[regime],
                    marker=REGIME_MARKERS[regime],
                    markersize=4.1,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    linewidth=1.4,
                    label=regime,
                    zorder=3,
                )
            ax.set_yscale("log")
            ax.set_ylim(1000, 200)
            ax.set_yticks(LEVELS)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(metadata["xlabel"])
            ax.set_title(metadata["display"])
            style_axis(ax, grid_axis="both")
            panel_label(ax, f"({'ab'[panel_index]})")

        axes[0].set_ylabel("Pressure (hPa)")
        axes[0].text(
            0.03,
            0.28,
            "850–500 hPa",
            transform=axes[0].transAxes,
            fontsize=6.5,
            color="#595959",
            ha="left",
            va="center",
        )
        axes[0].legend(frameon=False, loc="lower left", handlelength=2.6)
        axes[1].set_xlim(left=0)

        count_transform = mtransforms.blended_transform_factory(
            axes[1].transAxes, axes[1].transData
        )
        axes[1].text(
            1.015,
            1.02,
            "valid n: C0/C1/C2",
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=6.0,
            clip_on=False,
        )
        for level in LEVELS:
            n_values = [
                int(
                    plotting.loc[
                        (plotting["variable"] == "rh_pct")
                        & (plotting["level_hPa"] == level)
                        & (plotting["group_label"] == regime),
                        "sample_size",
                    ].iloc[0]
                )
                for regime in REGIME_ORDER
            ]
            axes[1].text(
                1.015,
                level,
                "/".join(str(value) for value in n_values),
                transform=count_transform,
                ha="left",
                va="center",
                fontsize=5.7,
                color="#4D4D4D",
                clip_on=False,
            )

        save_figure_atomic(fig, output_path, dpi=PROTOTYPE_DPI)
        plt.close(fig)

    caption_en = """# Fig. 7 caption (English)

Vertical thermodynamic and moisture profiles for the three environmental
regimes within the formal 790-event confirmed-tornado sample: (a) temperature
and (b) relative humidity. Points show level-wise medians and shading shows
deterministic 2,000-replicate percentile-bootstrap 95% confidence intervals for
the median. Lines connect only the nine sampled pressure levels and are not
smoothed. Pressure decreases upward on a logarithmic axis. The light-gray band
marks 850–500 hPa, where C2 is characterized by a deeper moist profile. C0 is
colder in the mid-troposphere, while the magnitude of this difference is partly
associated with latitude, month, and weather-type composition. Predefined
validity and below-ground masks are applied at each level; valid C0/C1/C2
sample sizes are printed beside panel (b) and supplied in Fig7_effective_n.csv. Key event-point
medians at 500 hPa are RH=54.3/57.0/87.7% and T=259.3/268.7/269.5 K for
C0/C1/C2, respectively.
"""
    caption_zh = """# Fig. 7 图注（中文）

正式790例已确认龙卷样本中三个环境组的垂直热力与水汽廓线：（a）温度和
（b）相对湿度。点表示各层中位数，阴影表示中位数的确定性2000次百分位
bootstrap 95%置信区间。连线仅连接九个实际压力层，未进行平滑。压力轴采用
对数尺度并由下向上递减。淡灰色带标记850–500 hPa层，C2在该层结范围内
表现出更深厚的湿润廓线。C0中层温度较低，但差异幅度部分与纬度、月份和
天气型构成相关。每层均使用预先定义的有效值和地下层掩膜；C0/C1/C2有效
样本量列于（b）面板右侧，并写入Fig7_effective_n.csv。500 hPa事件点中位数
分别为RH=54.3/57.0/87.7%和T=259.3/268.7/269.5 K（C0/C1/C2）。
"""
    prohibited = scan_forbidden_language(caption_en + caption_zh)
    if prohibited:
        raise AssertionError(f"Fig.7 caption contains prohibited text: {prohibited}")
    write_text_atomic(CAPTION_DIR / "Fig7_caption_en.md", caption_en)
    write_text_atomic(CAPTION_DIR / "Fig7_caption_zh.md", caption_zh)

    metadata = inspect_png(output_path)
    write_json_atomic(QC_DIR / "Fig7_file_metadata.json", metadata)
    write_qc_report(
        QC_DIR / "Fig7_qc_report.md",
        figure_id="Fig.7",
        status="PASS_WITH_NONBLOCKING_NOTES",
        data_checks=[
            "The frozen long profile contains 7,110 rows, 790 unique events, and nine pressure levels per event.",
            "Frozen valid and below-ground flags are applied before each level-wise summary.",
            "C0/C1/C2 event membership remains 131/307/352.",
        ],
        number_checks=[
            "All 42 available frozen 850–200-hPa medians and effective sample sizes reproduce within rounded-table tolerance.",
            "RH500 reproduces 54.3/57.0/87.7% and T500 reproduces 259.3/268.7/269.5 K at display precision.",
            f"Plotting intervals use {BOOTSTRAP_REPLICATES} deterministic event bootstrap replicates and are explicitly logged as display transformations.",
        ],
        interpretation_checks=[
            "The highlighted layer is restricted to 850–500 hPa.",
            "The C0 temperature difference is qualified by latitude, month, and weather-type composition.",
            "Fixed-level profiles are not interpreted as a physical process attribution.",
        ],
        visual_checks=[
            f"Prototype PNG is {metadata['width_px']}×{metadata['height_px']} px in RGB mode.",
            "Pressure decreases upward on an explicitly logarithmic axis with all nine requested levels labeled.",
            "Color, marker, and line style redundantly identify regimes; uncertainty remains subordinate to medians.",
        ],
        nonblocking_notes=[
            "Bootstrap intervals at 1000 and 925 hPa are plotting summaries derived from frozen valid event values; no new hypothesis test is introduced.",
            "Publisher-specific export and final vector-font inspection remain for Gate 4.",
        ],
    )

    write_text_atomic(
        LOG_DIR / "Fig7_build.log",
        f"""FIGURE=Fig7
STATUS=PASS_WITH_NONBLOCKING_NOTES
SOURCE_PROFILE={project_path(PROFILE_SOURCE)}
SOURCE_FROZEN_STATS={project_path(FROZEN_STATS_SOURCE)}
PLOTTING_DATA={data_path}
EFFECTIVE_N_DATA={n_path}
OUTPUT={output_path}
MASK=valid=True AND below_ground=False AND finite variable value
AGGREGATION=level-wise median
UNCERTAINTY=percentile bootstrap 95% CI, B={BOOTSTRAP_REPLICATES}, deterministic stable seeds from base {BOOTSTRAP_BASE_SEED}
SMOOTHING=NONE
CLIPPING=NONE
SCIENTIFIC_REANALYSIS=FALSE
MAX_FROZEN_MEDIAN_ABS_DIFF={maximum_median_difference:.12f}
""",
    )
    return {
        "figure_id": "Fig7",
        "status": "PASS_WITH_NONBLOCKING_NOTES",
        "output": str(output_path),
        "plotting_data": str(data_path),
        "script": str(SCRIPT_PATH),
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(build())
