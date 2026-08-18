"""Build Fig. 9: event locations, seasonality, and latitude distributions."""

from __future__ import annotations

import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from round2_io import (
    CAPTION_DIR,
    LOG_DIR,
    MAIN_FIGURE_DIR,
    PLOTTING_DATA_DIR,
    PROJECT_ROOT,
    QC_DIR,
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
    REGIME_LINESTYLES,
    REGIME_MARKERS,
    REGIME_ORDER,
    manuscript_style,
    mm_to_inches,
    panel_label,
    style_axis,
)


EVENTS_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/06_figures_tables/"
    "21_figures_and_tables_v3/source_data/fig01_event_distribution_source_v3.csv"
)
LABELS_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/30_labels_k3_regime_ids_v3.csv"
)
MONTHLY_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/05_regime_interpretation/"
    "19_seasonal_analysis_v3/01_monthly_counts_by_regime_v3.csv"
)
SPATIAL_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/06_figures_tables/"
    "21_figures_and_tables_v3/source_data/fig09_spatial_source_v3.csv"
)

OUTPUT = MAIN_FIGURE_DIR / "Fig9_spatial_seasonal_context_ROUND2_review.png"
SNAPSHOT = PLOTTING_DATA_DIR / "Fig9_plotting_snapshot.csv"
CAPTION_EN = CAPTION_DIR / "Fig9_caption_en.md"
CAPTION_ZH = CAPTION_DIR / "Fig9_caption_zh.md"
METADATA = QC_DIR / "Fig9_metadata.json"
QC_REPORT = QC_DIR / "Fig9_QC.md"
BUILD_LOG = LOG_DIR / "Fig9_build.log"


def _load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = read_csv_checked(
        EVENTS_SOURCE,
        required_columns=("event_id", "date_utc", "longitude", "latitude"),
    )
    labels = read_csv_checked(
        LABELS_SOURCE,
        required_columns=("event_id", "regime_id"),
    )[["event_id", "regime_id"]]
    monthly_wide = read_csv_checked(
        MONTHLY_SOURCE,
        required_columns=("regime_id", "3", "4", "5", "6", "7", "8", "9", "10"),
    )
    monthly = monthly_wide.melt(
        id_vars="regime_id",
        value_vars=[str(month) for month in range(3, 11)],
        var_name="month",
        value_name="count",
    )
    monthly["month"] = monthly["month"].astype(int)
    monthly["count"] = monthly["count"].astype(int)
    spatial = read_csv_checked(
        SPATIAL_SOURCE,
        required_columns=("regime_id", "n", "median_latitude", "median_longitude"),
    )

    merged = events.merge(labels, on="event_id", how="inner", validate="one_to_one")
    merged["date_utc"] = pd.to_datetime(merged["date_utc"])
    merged["month"] = merged["date_utc"].dt.month

    assert_exact("Fig9 merged rows", len(merged), 790)
    assert_exact("Fig9 unique events", merged["event_id"].nunique(), 790)
    assert_exact(
        "Fig9 regime counts",
        merged["regime_id"].value_counts().sort_index().to_dict(),
        {"C0": 131, "C1": 307, "C2": 352},
    )
    assert_exact("Fig9 month range", sorted(merged["month"].unique()), list(range(3, 11)))
    assert_exact(
        "Fig9 coordinate missing count",
        int(merged[["longitude", "latitude"]].isna().sum().sum()),
        0,
    )

    expected_monthly = (
        merged.groupby(["regime_id", "month"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["regime_id", "month"])
        .reset_index(drop=True)
    )
    observed_monthly = (
        monthly[["regime_id", "month", "count"]]
        .sort_values(["regime_id", "month"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(expected_monthly, observed_monthly, check_dtype=False)

    observed_spatial = spatial.set_index("regime_id")
    for regime in REGIME_ORDER:
        subset = merged.loc[merged["regime_id"] == regime]
        assert_exact(f"{regime} spatial n", int(observed_spatial.loc[regime, "n"]), len(subset))
        assert_close(
            f"{regime} latitude median",
            float(observed_spatial.loc[regime, "median_latitude"]),
            float(subset["latitude"].median()),
            1e-10,
        )
        assert_close(
            f"{regime} longitude median",
            float(observed_spatial.loc[regime, "median_longitude"]),
            float(subset["longitude"].median()),
            1e-10,
        )

    monthly = monthly.copy()
    denominators = merged["regime_id"].value_counts()
    monthly["within_regime_pct"] = [
        100.0 * row["count"] / denominators[row["regime_id"]]
        for _, row in monthly.iterrows()
    ]
    return merged, monthly, spatial


def _write_snapshot(
    events: pd.DataFrame, monthly: pd.DataFrame, spatial: pd.DataFrame
) -> None:
    event_snapshot = events[
        ["event_id", "date_utc", "month", "longitude", "latitude", "regime_id"]
    ].copy()
    event_snapshot.insert(0, "record_type", "event")

    monthly_snapshot = monthly[
        ["regime_id", "month", "count", "within_regime_pct"]
    ].copy()
    monthly_snapshot.insert(0, "record_type", "monthly_summary")

    spatial_snapshot = spatial[
        ["regime_id", "n", "median_latitude", "median_longitude"]
    ].copy()
    spatial_snapshot.insert(0, "record_type", "spatial_summary")

    columns = [
        "record_type",
        "event_id",
        "date_utc",
        "month",
        "longitude",
        "latitude",
        "regime_id",
        "count",
        "within_regime_pct",
        "n",
        "median_latitude",
        "median_longitude",
    ]
    snapshot = pd.concat(
        [event_snapshot, monthly_snapshot, spatial_snapshot],
        ignore_index=True,
        sort=False,
    ).reindex(columns=columns)
    write_csv_atomic(SNAPSHOT, snapshot)


def _draw(events: pd.DataFrame, monthly: pd.DataFrame) -> None:
    width, height = FIGURE_SIZES_MM["Fig9"]
    with manuscript_style():
        fig = plt.figure(figsize=mm_to_inches(width, height), layout="constrained")
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.65, 1.00],
            height_ratios=[1.00, 1.00],
        )
        ax_map = fig.add_subplot(grid[:, 0], projection=ccrs.PlateCarree())
        ax_month = fig.add_subplot(grid[0, 1])
        ax_lat = fig.add_subplot(grid[1, 1])

        ax_map.set_extent([84, 135, 17, 53], crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F5F5F2", zorder=0)
        ax_map.add_feature(
            cfeature.OCEAN.with_scale("50m"), facecolor="#EDF4F7", zorder=0
        )
        ax_map.add_feature(
            cfeature.COASTLINE.with_scale("50m"),
            linewidth=0.55,
            edgecolor="#555555",
            zorder=2,
        )
        ax_map.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linewidth=0.45,
            edgecolor="#777777",
            zorder=2,
        )
        province_lines = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none",
        )
        ax_map.add_feature(
            province_lines, linewidth=0.25, edgecolor="#B0B0B0", zorder=1
        )
        gl = ax_map.gridlines(
            draw_labels=True,
            xlocs=np.arange(90, 131, 10),
            ylocs=np.arange(20, 56, 10),
            linewidth=0.4,
            color="#B8B8B8",
            alpha=0.55,
            linestyle=":",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 6.8}
        gl.ylabel_style = {"size": 6.8}

        legend_handles: list[Line2D] = []
        for regime in REGIME_ORDER:
            subset = events.loc[events["regime_id"] == regime]
            ax_map.scatter(
                subset["longitude"],
                subset["latitude"],
                s=11,
                marker=REGIME_MARKERS[regime],
                c=REGIME_COLORS[regime],
                alpha=0.58,
                linewidths=0.18,
                edgecolors="white",
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker=REGIME_MARKERS[regime],
                    color="none",
                    markerfacecolor=REGIME_COLORS[regime],
                    markeredgecolor="white",
                    markeredgewidth=0.3,
                    markersize=5.3,
                    label=f"{regime} (n={len(subset)})",
                )
            )
        ax_map.legend(
            handles=legend_handles,
            title="Environmental regime",
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#B0B0B0",
            framealpha=0.92,
            title_fontsize=6.8,
        )
        ax_map.set_title("Event locations", pad=5)
        panel_label(ax_map, "(a)", x=-0.04, y=1.025)

        for regime in REGIME_ORDER:
            subset = monthly.loc[monthly["regime_id"] == regime].sort_values("month")
            ax_month.plot(
                subset["month"],
                subset["within_regime_pct"],
                color=REGIME_COLORS[regime],
                marker=REGIME_MARKERS[regime],
                linestyle=REGIME_LINESTYLES[regime],
                markerfacecolor="white",
                markeredgewidth=0.8,
            )
        ax_month.set_xlim(2.7, 10.3)
        ax_month.set_xticks(range(3, 11))
        ax_month.set_xticklabels(["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"])
        ax_month.set_ylabel("Within-regime events (%)")
        ax_month.set_title("March–October occurrence", pad=4)
        style_axis(ax_month, grid_axis="y")
        panel_label(ax_month, "(b)", x=-0.16, y=1.04)

        rng = np.random.default_rng(790)
        positions = np.arange(len(REGIME_ORDER))
        latitude_data = [
            events.loc[events["regime_id"] == regime, "latitude"].to_numpy()
            for regime in REGIME_ORDER
        ]
        box = ax_lat.boxplot(
            latitude_data,
            vert=False,
            positions=positions,
            widths=0.44,
            whis=(10, 90),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#202020", "linewidth": 1.1},
            whiskerprops={"color": "#666666", "linewidth": 0.7},
            capprops={"color": "#666666", "linewidth": 0.7},
        )
        for patch, regime in zip(box["boxes"], REGIME_ORDER):
            patch.set_facecolor(REGIME_COLORS[regime])
            patch.set_alpha(0.30)
            patch.set_edgecolor(REGIME_COLORS[regime])
            patch.set_linewidth(0.8)
        for position, regime, values in zip(positions, REGIME_ORDER, latitude_data):
            jitter = rng.uniform(-0.16, 0.16, len(values))
            ax_lat.scatter(
                values,
                position + jitter,
                s=5.5,
                c=REGIME_COLORS[regime],
                marker=REGIME_MARKERS[regime],
                alpha=0.22,
                linewidths=0,
                zorder=1,
            )
        ax_lat.set_yticks(positions, REGIME_ORDER)
        ax_lat.set_xlabel("Latitude (°N)")
        ax_lat.set_title("Latitude distributions", pad=4)
        ax_lat.invert_yaxis()
        style_axis(ax_lat, grid_axis="x")
        panel_label(ax_lat, "(c)", x=-0.16, y=1.04)

        save_figure_atomic(fig, OUTPUT, dpi=PROTOTYPE_DPI)
        plt.close(fig)


def _write_caption_and_qc(events: pd.DataFrame) -> None:
    caption_en = """# Fig. 9 caption (English)

**Figure 9. Spatial and seasonal context of the three environmental regimes.** (a) Locations of the 790 confirmed-tornado events during March–October 2006–2024, colored by environmental regime. (b) Monthly counts expressed as percentages within each regime. (c) Latitude distributions; boxes show the interquartile range, center lines show medians, whiskers span the 10th–90th percentiles, and translucent points show all events. Regime sample sizes are C0, 131; C1, 307; and C2, 352. These descriptive patterns provide post-hoc context within the confirmed-tornado sample and do not represent event climatology for the general population.
"""
    caption_zh = """# 图9图注（中文）

**图9. 三类环境型的空间与季节背景。**（a）2006–2024年3–10月790个已确认龙卷事件的位置，颜色表示环境型。（b）各月份事件数占相应环境型总样本的百分比。（c）纬度分布；箱体表示四分位距，中线表示中位数，须线覆盖第10–90百分位，半透明点表示全部事件。各环境型样本量为C0 131例、C1 307例、C2 352例。这些描述性分布仅提供已确认龙卷样本内部的事后背景，不代表一般总体的事件气候分布。
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
        figure_id="Fig. 9",
        status_label="MAIN_TEXT_CANDIDATE",
        scientific_checks=[
            "The map contains exactly 790 unique formal-sample events.",
            "Regime counts close exactly at C0=131, C1=307, and C2=352.",
            "The March–October counts were reproduced from event dates and match the accepted monthly table exactly.",
            "Latitude and longitude medians match the accepted spatial summary within 1e-10.",
            "All latitude observations remain in the distribution panel; no extreme values were removed.",
        ],
        layout_checks=[
            "The prescribed 2×2 GridSpec uses width ratios 1.65:1.00 and equal row heights.",
            "The map spans both rows; monthly percentages and latitude distributions occupy the right column.",
            "Colors, markers, line styles, and typography follow the established publication system.",
            "The file is a 200 dpi RGB review image with no alpha channel.",
            "All publication-visible text and both captions passed the internal-term search.",
        ],
        interpretation_checks=[
            "Monthly values are explicitly within-regime percentages, not pooled percentages.",
            "The map uses point locations only; no density contours or boundary inference are shown.",
            "The caption limits interpretation to descriptive post-hoc context within confirmed tornado events.",
        ],
        deviations=[
            "The optional longitude-distribution panel was omitted because regime longitude medians are closely aligned and the map already displays longitudinal information.",
        ],
        review_notes=[
            "Confirm whether the target journal prefers administrative boundaries to be removed from the final map.",
            "No publisher-specific font or line-width compliance is claimed because a target journal has not yet been designated.",
        ],
    )


def build() -> dict:
    ensure_output_dirs()
    events, monthly, spatial = _load_sources()
    _write_snapshot(events, monthly, spatial)
    _draw(events, monthly)
    _write_caption_and_qc(events)

    image_info = inspect_png(OUTPUT)
    assert_exact("Fig9 mode", image_info["mode"], "RGB")
    assert_exact("Fig9 alpha", image_info["alpha_present"], False)
    metadata = {
        "figure_id": "Fig9",
        "status_label": "MAIN_TEXT_CANDIDATE",
        "output_stage": "200_DPI_RGB_REVIEW_PROTOTYPE",
        "figure_file": relative_source(OUTPUT),
        "plotting_snapshot": relative_source(SNAPSHOT),
        "caption_en": relative_source(CAPTION_EN),
        "caption_zh": relative_source(CAPTION_ZH),
        "qc_report": relative_source(QC_REPORT),
        "script": relative_source(Path(__file__)),
        "inputs": [
            {"path": EVENTS_SOURCE, "sha256": sha256_file(PROJECT_ROOT / EVENTS_SOURCE)},
            {"path": LABELS_SOURCE, "sha256": sha256_file(PROJECT_ROOT / LABELS_SOURCE)},
            {"path": MONTHLY_SOURCE, "sha256": sha256_file(PROJECT_ROOT / MONTHLY_SOURCE)},
            {"path": SPATIAL_SOURCE, "sha256": sha256_file(PROJECT_ROOT / SPATIAL_SOURCE)},
        ],
        "render": image_info,
        "scientific_result_changed": False,
    }
    write_json_atomic(METADATA, metadata)
    write_text_atomic(
        BUILD_LOG,
        "\n".join(
            [
                "FIGURE_ID=Fig9",
                "STATUS=BUILT_AND_QC_COMPLETE",
                f"OUTPUT={relative_source(OUTPUT)}",
                f"SHA256={image_info['sha256']}",
                "FORMAL_SAMPLE_N=790",
                "REGIME_COUNTS=C0:131,C1:307,C2:352",
                "LAYOUT=2x2_MAP_SPANS_ROWS_WIDTH_RATIO_1.65_1.00",
                "LONGITUDE_PANEL=OMITTED_AS_REDUNDANT",
                "",
            ]
        ),
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
