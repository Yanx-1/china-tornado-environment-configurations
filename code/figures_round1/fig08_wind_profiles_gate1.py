"""Gate 1 prototype for Fig. 8: vertical environmental-wind structure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter


SCRIPT_PATH = Path(__file__).resolve()
ROUND_ROOT = SCRIPT_PATH.parent.parent
sys.path.insert(0, str(ROUND_ROOT / "03_config"))

from figure_style_config import (  # noqa: E402
    FIGURE_SIZES_MM,
    PROTOTYPE_DPI,
    REFERENCE_COLOR,
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
CONTINUITY_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "37_direction_b_full790_final_gate/12_upper_level_wind_vertical_continuity.csv"
)
LOW_SENSITIVITY_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "37_direction_b_full790_final_gate/13_low_level_flow_adjustment_and_spatial_metrics.csv"
)
UPPER_SENSITIVITY_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "37_direction_b_full790_final_gate/11_upper_level_wind_confounder_adjustment.csv"
)
ENVIRONMENT_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "02_environment_table/06_environment_table_new790_v3.csv"
)
WEATHER_SOURCE = "data/events/tornado_synoptic_type.csv"

LEVELS = (1000, 925, 850, 700, 500, 400, 300, 250, 200)
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
            "u_ms",
            "v_ms",
            "below_ground",
            "valid",
        ),
        dtype={"event_id": str},
    )
    continuity = read_csv_checked(
        CONTINUITY_SOURCE,
        required_columns=("regime", "level_hPa", "ws_median"),
    )
    low_sensitivity = read_csv_checked(
        LOW_SENSITIVITY_SOURCE,
        required_columns=("regime", "adjustment", "n", "ws850_median"),
    )
    upper_sensitivity = read_csv_checked(
        UPPER_SENSITIVITY_SOURCE,
        required_columns=(
            "regime",
            "adjustment",
            "n",
            "ws200_median",
        ),
    )
    environment = read_csv_checked(
        ENVIRONMENT_SOURCE,
        required_columns=(
            "event_id",
            "date_utc",
            "latitude",
            "tc_related",
        ),
        dtype={"event_id": str},
    )[["event_id", "date_utc", "latitude", "tc_related"]]
    weather = read_csv_checked(
        WEATHER_SOURCE,
        required_columns=("event_id", "synoptic_class"),
        dtype={"event_id": str},
    )[["event_id", "synoptic_class"]]

    assert_exact("profile rows", len(profile), 790 * 9)
    assert_exact("profile unique event_id", profile["event_id"].nunique(), 790)
    assert_exact(
        "profile group counts",
        profile.groupby("regime")["event_id"].nunique().reindex(REGIME_ORDER).astype(int).tolist(),
        [131, 307, 352],
    )
    valid_mask = (
        profile["valid"].astype(str).str.lower().eq("true")
        & ~profile["below_ground"].astype(str).str.lower().eq("true")
        & profile["u_ms"].notna()
        & profile["v_ms"].notna()
    )
    wind_events = profile.loc[
        valid_mask,
        ["event_id", "regime", "level_hPa", "u_ms", "v_ms"],
    ].copy()
    wind_events["wind_speed_ms"] = np.hypot(
        wind_events["u_ms"].astype(float),
        wind_events["v_ms"].astype(float),
    )
    wind_events = wind_events.merge(
        environment, on="event_id", how="left", validate="many_to_one"
    ).merge(weather, on="event_id", how="left", validate="many_to_one")
    if wind_events[["date_utc", "latitude", "tc_related"]].isna().any().any():
        raise ValueError("Fig.8 is missing required environment metadata.")
    wind_events["month"] = pd.to_datetime(
        wind_events["date_utc"], errors="raise"
    ).dt.month
    wind_events["is_tc"] = (
        wind_events["tc_related"].astype(str).str.lower().eq("true")
    )

    source_profile = relative_source(project_path(PROFILE_SOURCE))
    profile_records: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        for level in LEVELS:
            values = wind_events.loc[
                (wind_events["regime"] == regime)
                & (wind_events["level_hPa"].astype(int) == level),
                "wind_speed_ms",
            ]
            median, ci_low, ci_high, n_valid = bootstrap_median_ci(
                values,
                seed=stable_seed(
                    BOOTSTRAP_BASE_SEED,
                    f"Fig8:profile:{regime}:{level}",
                ),
                n_boot=BOOTSTRAP_REPLICATES,
            )
            profile_records.append(
                {
                    "record_type": "vertical_profile",
                    "group_label": regime,
                    "variable": "environmental wind speed",
                    "level_hPa": level,
                    "adjustment": "full",
                    "median": median,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "unit": "m s−1",
                    "statistic": "median with percentile bootstrap 95% CI",
                    "sample_size": n_valid,
                    "source_file": source_profile,
                    "transformation_note": (
                        "wind speed=hypot(u,v); frozen valid/below-ground mask; "
                        f"{BOOTSTRAP_REPLICATES} event bootstrap replicates"
                    ),
                }
            )
    profile_stats = pd.DataFrame(profile_records)
    assert_exact("wind profile summary rows", len(profile_stats), 27)

    comparison = profile_stats.merge(
        continuity,
        left_on=["group_label", "level_hPa"],
        right_on=["regime", "level_hPa"],
        how="inner",
        validate="one_to_one",
    )
    assert_exact("frozen continuity comparisons", len(comparison), 21)
    max_continuity_difference = float(
        (comparison["median"] - comparison["ws_median"]).abs().max()
    )
    if max_continuity_difference > 0.051:
        raise AssertionError(
            "Fig.8 wind medians exceed rounded frozen-table tolerance: "
            f"{max_continuity_difference}"
        )

    key_expected = {
        ("C0", 850): 7.7,
        ("C1", 850): 6.0,
        ("C2", 850): 13.9,
        ("C0", 200): 27.9,
        ("C1", 200): 11.8,
        ("C2", 200): 18.5,
    }
    for (regime, level), expected_value in key_expected.items():
        observed = float(
            profile_stats.loc[
                (profile_stats["group_label"] == regime)
                & (profile_stats["level_hPa"] == level),
                "median",
            ].iloc[0]
        )
        if abs(observed - expected_value) > 0.06:
            raise AssertionError(
                f"Fig.8 key median mismatch for {regime}/{level}: "
                f"{observed} versus {expected_value}"
            )

    sensitivity_definitions = [
        {
            "group": "C2",
            "level": 850,
            "adjustment": "full",
            "display": "C2 850: full",
            "mask": lambda frame: (frame["regime"] == "C2")
            & (frame["level_hPa"].astype(int) == 850),
            "expected_n": 352,
            "expected_median": 13.9,
            "source": LOW_SENSITIVITY_SOURCE,
        },
        {
            "group": "C2",
            "level": 850,
            "adjustment": "noTC",
            "display": "C2 850: no TC",
            "mask": lambda frame: (frame["regime"] == "C2")
            & (frame["level_hPa"].astype(int) == 850)
            & (~frame["is_tc"]),
            "expected_n": 250,
            "expected_median": 12.6,
            "source": LOW_SENSITIVITY_SOURCE,
        },
        {
            "group": "C0",
            "level": 200,
            "adjustment": "full",
            "display": "C0 200: full",
            "mask": lambda frame: (frame["regime"] == "C0")
            & (frame["level_hPa"].astype(int) == 200),
            "expected_n": 131,
            "expected_median": 27.9,
            "source": UPPER_SENSITIVITY_SOURCE,
        },
        {
            "group": "C0",
            "level": 200,
            "adjustment": "30-40N",
            "display": "C0 200: 30–40°N",
            "mask": lambda frame: (frame["regime"] == "C0")
            & (frame["level_hPa"].astype(int) == 200)
            & (frame["latitude"].astype(float) >= 30)
            & (frame["latitude"].astype(float) < 40),
            "expected_n": 39,
            "expected_median": 32.9,
            "source": UPPER_SENSITIVITY_SOURCE,
        },
        {
            "group": "C0",
            "level": 200,
            "adjustment": "JJA",
            "display": "C0 200: JJA",
            "mask": lambda frame: (frame["regime"] == "C0")
            & (frame["level_hPa"].astype(int) == 200)
            & (frame["month"].isin([6, 7, 8])),
            "expected_n": 83,
            "expected_median": 27.1,
            "source": UPPER_SENSITIVITY_SOURCE,
        },
        {
            "group": "C0",
            "level": 200,
            "adjustment": "C0_noCV",
            "display": "C0 200: no cold vortex",
            "mask": lambda frame: (frame["regime"] == "C0")
            & (frame["level_hPa"].astype(int) == 200)
            & (frame["synoptic_class"].ne("冷涡")),
            "expected_n": 76,
            "expected_median": 27.7,
            "source": UPPER_SENSITIVITY_SOURCE,
        },
    ]
    sensitivity_records: list[dict[str, object]] = []
    for definition in sensitivity_definitions:
        selected = wind_events.loc[
            definition["mask"](wind_events), "wind_speed_ms"
        ]
        median, ci_low, ci_high, n_selected = bootstrap_median_ci(
            selected,
            seed=stable_seed(
                BOOTSTRAP_BASE_SEED,
                f"Fig8:sensitivity:{definition['display']}",
            ),
            n_boot=BOOTSTRAP_REPLICATES,
        )
        assert_exact(
            f"{definition['display']} n",
            n_selected,
            definition["expected_n"],
        )
        if abs(median - definition["expected_median"]) > 0.06:
            raise AssertionError(
                f"{definition['display']} median mismatch: "
                f"{median} versus {definition['expected_median']}"
            )
        sensitivity_records.append(
            {
                "record_type": "sensitivity_subset",
                "group_label": definition["group"],
                "variable": "environmental wind speed",
                "level_hPa": definition["level"],
                "adjustment": definition["adjustment"],
                "display_label": definition["display"],
                "median": median,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "unit": "m s−1",
                "statistic": "frozen subset median with plotting bootstrap 95% CI",
                "sample_size": n_selected,
                "source_file": relative_source(
                    project_path(definition["source"])
                ),
                "transformation_note": (
                    "subset definition reproduces frozen n and median; "
                    f"{BOOTSTRAP_REPLICATES} event bootstrap replicates for display"
                ),
            }
        )
    sensitivity_stats = pd.DataFrame(sensitivity_records)

    # Cross-check the medians and n against the frozen sensitivity tables.
    for _, row in sensitivity_stats.iterrows():
        if row["group_label"] == "C2":
            frozen_row = low_sensitivity.loc[
                (low_sensitivity["regime"] == row["group_label"])
                & (low_sensitivity["adjustment"] == row["adjustment"])
            ].iloc[0]
            frozen_median = float(frozen_row["ws850_median"])
        else:
            frozen_row = upper_sensitivity.loc[
                (upper_sensitivity["regime"] == row["group_label"])
                & (upper_sensitivity["adjustment"] == row["adjustment"])
            ].iloc[0]
            frozen_median = float(frozen_row["ws200_median"])
        assert_exact(
            f"{row['display_label']} frozen n",
            int(row["sample_size"]),
            int(frozen_row["n"]),
        )
        if abs(float(row["median"]) - frozen_median) > 0.06:
            raise AssertionError(
                f"{row['display_label']} failed frozen sensitivity check."
            )

    plotting = pd.concat(
        [profile_stats, sensitivity_stats], ignore_index=True, sort=False
    )
    data_path = PLOTTING_DATA_DIR / "Fig8_plotting_data.csv"
    n_path = PLOTTING_DATA_DIR / "Fig8_effective_n.csv"
    output_path = PNG_DIR / "Fig8_wind_profiles_GATE1_prototype.png"
    write_csv_atomic(data_path, plotting)
    write_csv_atomic(
        n_path,
        profile_stats[
            ["level_hPa", "group_label", "sample_size", "source_file"]
        ].copy(),
    )

    with manuscript_style():
        fig, axes = plt.subplot_mosaic(
            [["profile", "key"], ["profile", "sensitivity"]],
            figsize=mm_to_inches(*FIGURE_SIZES_MM["Fig8"]),
            layout="constrained",
            gridspec_kw={"width_ratios": [1.02, 1.18]},
        )
        ax_profile = axes["profile"]
        for reference_level in (850, 200):
            ax_profile.axhline(
                reference_level,
                color=REFERENCE_COLOR,
                linestyle=(0, (2, 2)),
                linewidth=0.75,
                alpha=0.8,
                zorder=0,
            )
        for regime in REGIME_ORDER:
            subset = (
                profile_stats.loc[profile_stats["group_label"] == regime]
                .set_index("level_hPa")
                .reindex(LEVELS)
                .reset_index()
            )
            pressure = subset["level_hPa"].to_numpy(dtype=float)
            ax_profile.fill_betweenx(
                pressure,
                subset["ci_low"].to_numpy(dtype=float),
                subset["ci_high"].to_numpy(dtype=float),
                color=REGIME_COLORS[regime],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
            ax_profile.plot(
                subset["median"].to_numpy(dtype=float),
                pressure,
                color=REGIME_COLORS[regime],
                linestyle=REGIME_LINESTYLES[regime],
                marker=REGIME_MARKERS[regime],
                markersize=4.1,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=1.45,
                label=regime,
                zorder=3,
            )
        ax_profile.set_yscale("log")
        ax_profile.set_ylim(1000, 200)
        ax_profile.set_yticks(LEVELS)
        ax_profile.yaxis.set_major_formatter(ScalarFormatter())
        ax_profile.yaxis.set_minor_formatter(NullFormatter())
        ax_profile.set_xlabel(r"Environmental wind speed (m s$^{-1}$)")
        ax_profile.set_ylabel("Pressure (hPa)")
        ax_profile.set_title("Vertical profile")
        ax_profile.legend(frameon=False, loc="lower right")
        style_axis(ax_profile, grid_axis="both")
        panel_label(ax_profile, "(a)")

        ax_key = axes["key"]
        level_y = {850: 0.0, 200: 1.0}
        group_offsets = {"C0": -0.19, "C1": 0.0, "C2": 0.19}
        for regime in REGIME_ORDER:
            for level in (850, 200):
                row = profile_stats.loc[
                    (profile_stats["group_label"] == regime)
                    & (profile_stats["level_hPa"] == level)
                ].iloc[0]
                y_value = level_y[level] + group_offsets[regime]
                ax_key.errorbar(
                    float(row["median"]),
                    y_value,
                    xerr=np.array(
                        [
                            [float(row["median"]) - float(row["ci_low"])],
                            [float(row["ci_high"]) - float(row["median"])],
                        ]
                    ),
                    fmt=REGIME_MARKERS[regime],
                    color=REGIME_COLORS[regime],
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    markersize=4.4,
                    elinewidth=0.85,
                    capsize=2.0,
                    label=regime if level == 850 else None,
                    zorder=3,
                )
                ax_key.annotate(
                    f"{float(row['median']):.1f}",
                    (float(row["median"]), y_value),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=6.0,
                    ha="left",
                    va="center",
                )
        ax_key.set_yticks([0, 1], ["850 hPa", "200 hPa"])
        ax_key.set_xlabel(r"Wind speed (m s$^{-1}$)")
        ax_key.set_title("Key-level event estimates")
        ax_key.legend(frameon=False, ncol=3, loc="lower right")
        style_axis(ax_key, grid_axis="x")
        panel_label(ax_key, "(b)")

        ax_sensitivity = axes["sensitivity"]
        sensitivity_plot = sensitivity_stats.iloc[::-1].reset_index(drop=True)
        y_positions = np.arange(len(sensitivity_plot))
        for y_position, (_, row) in zip(
            y_positions, sensitivity_plot.iterrows()
        ):
            regime = row["group_label"]
            ax_sensitivity.errorbar(
                float(row["median"]),
                y_position,
                xerr=np.array(
                    [
                        [float(row["median"]) - float(row["ci_low"])],
                        [float(row["ci_high"]) - float(row["median"])],
                    ]
                ),
                fmt=REGIME_MARKERS[regime],
                color=REGIME_COLORS[regime],
                markerfacecolor="white",
                markeredgewidth=0.9,
                markersize=4.3,
                elinewidth=0.85,
                capsize=2.0,
                zorder=3,
            )
            ax_sensitivity.annotate(
                f"{float(row['median']):.1f}",
                (float(row["median"]), y_position),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=5.9,
                ha="left",
                va="center",
            )
        ax_sensitivity.set_yticks(
            y_positions, sensitivity_plot["display_label"].tolist()
        )
        ax_sensitivity.set_xlabel(r"Wind speed (m s$^{-1}$)")
        ax_sensitivity.set_title("Sensitivity analyses")
        style_axis(ax_sensitivity, grid_axis="x")
        panel_label(ax_sensitivity, "(c)")

        save_figure_atomic(fig, output_path, dpi=PROTOTYPE_DPI)
        plt.close(fig)

    caption_en = """# Fig. 8 caption (English)

Vertical environmental-wind structure for the three regimes within the formal
790-event confirmed-tornado sample. (a) Median wind-speed profiles derived as
the vector magnitude of the u and v components; shading shows deterministic
2,000-replicate percentile-bootstrap 95% confidence intervals for the median.
Predefined validity and below-ground masks are applied at every pressure level.
Horizontal reference lines mark 850 and 200 hPa. (b) Event-point estimates at
the two key levels: WS850=7.7/6.0/13.9 m s−1 and
WS200=27.9/11.8/18.5 m s−1 for C0/C1/C2. (c) Sensitivity analyses with
plotting confidence intervals: C2 retains stronger 850-hPa flow after excluding
tropical-cyclone cases (12.6 m s−1), while the C0 200-hPa estimate remains
similar across latitude, summer-month, and cold-vortex exclusions. C1 has
relatively weak environmental flow at multiple pressure levels. A C1
warm-sector-exclusion estimate was unavailable and is therefore not shown.
"""
    caption_zh = """# Fig. 8 图注（中文）

正式790例已确认龙卷样本中三个环境组的垂直环境风结构。（a）由u、v风分量
计算矢量模得到的风速中位数廓线；阴影表示中位数的确定性2000次百分位
bootstrap 95%置信区间。每个压力层均应用预先定义的有效值和地下层掩膜。水平参考
线标记850和200 hPa。（b）两个关键层的事件点估计：C0/C1/C2的WS850分别为
7.7/6.0/13.9 m s−1，WS200分别为27.9/11.8/18.5 m s−1。（c）敏感性
分析及绘图置信区间：排除热带气旋案例后，C2在850 hPa仍保持较强环境风
（12.6 m s−1）；C0的200 hPa估计在纬度、夏季月份和冷涡排除条件下方向一致。
C1在多个压力层的环境风相对较弱。C1暖区排除估计不可用，因此本图不展示该项。
"""
    prohibited = scan_forbidden_language(caption_en + caption_zh)
    if prohibited:
        raise AssertionError(f"Fig.8 caption contains prohibited text: {prohibited}")
    write_text_atomic(CAPTION_DIR / "Fig8_caption_en.md", caption_en)
    write_text_atomic(CAPTION_DIR / "Fig8_caption_zh.md", caption_zh)

    metadata = inspect_png(output_path)
    write_json_atomic(QC_DIR / "Fig8_file_metadata.json", metadata)
    write_qc_report(
        QC_DIR / "Fig8_qc_report.md",
        figure_id="Fig.8",
        status="PASS_WITH_NONBLOCKING_NOTES",
        data_checks=[
            "The frozen long profile contains 7,110 rows and 790 unique events; wind speed is calculated from frozen u and v only after valid/below-ground masking.",
            "Formal C0/C1/C2 membership remains 131/307/352.",
            "Sensitivity subsets reproduce frozen membership counts and medians before display intervals are added.",
        ],
        number_checks=[
            "All 21 frozen 850–200-hPa wind-speed medians reproduce within rounded-table tolerance.",
            "WS850 reproduces 7.7/6.0/13.9 m s−1 and WS200 reproduces 27.9/11.8/18.5 m s−1.",
            "The C2 no-tropical-cyclone estimate reproduces n=250 and 12.6 m s−1; the C0 no-cold-vortex estimate reproduces n=76 and 27.7 m s−1.",
        ],
        interpretation_checks=[
            "All wind quantities are fixed-pressure environmental wind speeds.",
            "The display supports relative vertical-flow tendencies within the confirmed-tornado sample.",
            "No unavailable sensitivity estimate is synthesized.",
        ],
        visual_checks=[
            f"Prototype PNG is {metadata['width_px']}×{metadata['height_px']} px in RGB mode.",
            "Pressure decreases upward on an explicit logarithmic axis; 850 and 200 hPa are marked without oversized emphasis.",
            "Color, marker, and line style redundantly identify regimes; sensitivity estimates use points and intervals rather than bars.",
        ],
        nonblocking_notes=[
            "No frozen C1 warm-sector-exclusion estimate was located, so that optional sensitivity is omitted instead of being newly derived.",
            "Bootstrap intervals are plotting transformations around frozen event values and subsets; no new hypothesis test is introduced.",
            "Final vector export and publisher-specific review remain for Gate 4.",
        ],
    )

    write_text_atomic(
        LOG_DIR / "Fig8_build.log",
        f"""FIGURE=Fig8
STATUS=PASS_WITH_NONBLOCKING_NOTES
SOURCE_PROFILE={project_path(PROFILE_SOURCE)}
SOURCE_CONTINUITY={project_path(CONTINUITY_SOURCE)}
SOURCE_LOW_SENSITIVITY={project_path(LOW_SENSITIVITY_SOURCE)}
SOURCE_UPPER_SENSITIVITY={project_path(UPPER_SENSITIVITY_SOURCE)}
SOURCE_ENVIRONMENT={project_path(ENVIRONMENT_SOURCE)}
SOURCE_WEATHER={project_path(WEATHER_SOURCE)}
PLOTTING_DATA={data_path}
EFFECTIVE_N_DATA={n_path}
OUTPUT={output_path}
DERIVED_VARIABLE=wind_speed_ms=hypot(u_ms,v_ms)
MASK=valid=True AND below_ground=False AND finite u/v
UNCERTAINTY=percentile bootstrap 95% CI, B={BOOTSTRAP_REPLICATES}, deterministic stable seeds from base {BOOTSTRAP_BASE_SEED}
FROZEN_SENSITIVITY_REPRODUCED=C2 noTC; C0 30-40N; C0 JJA; C0 no cold vortex
C1_WARM_SECTOR_EXCLUSION=OMITTED_NO_FROZEN_ESTIMATE
SCIENTIFIC_REANALYSIS=FALSE
MAX_CONTINUITY_MEDIAN_ABS_DIFF={max_continuity_difference:.12f}
""",
    )
    return {
        "figure_id": "Fig8",
        "status": "PASS_WITH_NONBLOCKING_NOTES",
        "output": str(output_path),
        "plotting_data": str(data_path),
        "script": str(SCRIPT_PATH),
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(build())
