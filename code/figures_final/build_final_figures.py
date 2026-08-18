"""Build the 9 main and 3 supplementary final candidate figures.

Only accepted Round 1/Round 2 plotting snapshots are read.  All numerical
assertions are checked before drawing.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

from final_common import (
    FINAL_ROOT,
    HOLD_DIR,
    K4_COLORS,
    K4_DISPLAY,
    K4_LINESTYLES,
    K4_MARKERS,
    K4_ORDER,
    LOG_DIR,
    PROJECT_ROOT,
    QC_DIR,
    REGIME_COLORS,
    REGIME_LINESTYLES,
    REGIME_MARKERS,
    REGIME_N,
    REGIME_ORDER,
    ROUND1_ROOT,
    ROUND2_ROOT,
    SNAPSHOT_DIR,
    ensure_directories,
    export_formal_figure,
    mm_to_in,
    panel_label,
    publication_style,
    regime_legend_handles,
    relative_to_final,
    sha256_file,
    style_axis,
    write_json,
    write_text,
)


R1_DATA = ROUND1_ROOT / "01_plotting_data"
R2_DATA = ROUND2_ROOT / "01_plotting_data"

FIGURE_STEMS = {
    "Fig1": "Fig01_workflow_final",
    "Fig2": "Fig02_cluster_centers_final",
    "Fig3": "Fig03_raw_distributions_final",
    "Fig4": "Fig04_stability_final",
    "Fig5": "Fig05_k3_k4_correspondence_final",
    "Fig6": "Fig06_weather_type_association_final",
    "Fig7": "Fig07_thermodynamic_moisture_profiles_final",
    "Fig8": "Fig08_environmental_wind_profiles_final",
    "Fig9": "Fig09_spatial_seasonal_context_final",
    "FigS1": "FigS01_event_centered_composites_final",
    "FigS2": "FigS02_effect_sizes_quantile_ranges_final",
    "FigS3": "FigS03_k4_structure_final",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _assert_close(label: str, value: float, expected: float, tolerance: float) -> None:
    if not math.isclose(float(value), float(expected), abs_tol=tolerance, rel_tol=0):
        raise AssertionError(f"{label}: {value} != {expected} within {tolerance}")


def _copy_source_snapshots() -> list[dict]:
    ensure_directories()
    source_map = {
        "Fig01_source_plotting_snapshot.csv": R1_DATA / "Fig1_plotting_data.csv",
        "Fig02_source_plotting_snapshot.csv": R1_DATA / "Fig2_plotting_data.csv",
        "Fig03_source_plotting_snapshot.csv": R1_DATA / "Fig3_plotting_data.csv",
        "Fig04_source_plotting_snapshot.csv": R1_DATA / "Fig4_plotting_data.csv",
        "Fig05_source_plotting_snapshot.csv": R1_DATA / "Fig5_plotting_data.csv",
        "Fig06_source_plotting_snapshot.csv": R1_DATA / "Fig6_plotting_data.csv",
        "Fig07_source_plotting_snapshot.csv": R1_DATA / "Fig7_plotting_data.csv",
        "Fig08_source_plotting_snapshot.csv": R1_DATA / "Fig8_plotting_data.csv",
        "Fig09_source_plotting_snapshot.csv": R2_DATA / "Fig9_plotting_snapshot.csv",
        "FigS01_source_plotting_snapshot.csv": R2_DATA / "FigS1_plotting_snapshot.csv",
        "FigS02_source_plotting_snapshot.csv": R2_DATA / "FigS2_plotting_snapshot.csv",
        "FigS03_source_plotting_snapshot.csv": R2_DATA / "FigS4_plotting_snapshot.csv",
    }
    rows = []
    for final_name, source in source_map.items():
        destination = SNAPSHOT_DIR / final_name
        shutil.copy2(source, destination)
        rows.append(
            {
                "snapshot": relative_to_final(destination),
                "source": source.resolve().relative_to(PROJECT_ROOT).as_posix(),
                "source_sha256": sha256_file(source),
                "copied_sha256": sha256_file(destination),
                "identical": sha256_file(source) == sha256_file(destination),
            }
        )
    write_json(SNAPSHOT_DIR / "source_snapshot_copy_record.json", rows)
    return rows


def _box(ax, xy, width, height, text, face="#F4F7F9", edge="#52606D", size=7.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.85,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=size,
        linespacing=1.18,
    )


def _arrow(ax, start, end, text=None, text_xy=None):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": "#66717D", "lw": 0.85},
    )
    if text:
        x, y = text_xy if text_xy else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.5,
            color="#58636E",
        )


def build_fig1() -> dict:
    source = _read_csv(R1_DATA / "Fig1_plotting_data.csv")
    observed = (
        source.query("record_type == 'node' and panel == 'a'")
        .set_index("node_id")["count"]
        .astype(int)
        .to_dict()
    )
    expected = {"sample_909": 909, "sample_795": 795, "sample_793": 793, "sample_790": 790}
    if observed != expected:
        raise AssertionError(f"Fig1 sample closure mismatch: {observed}")

    with publication_style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(mm_to_in(178), mm_to_in(102)),
            gridspec_kw={"width_ratios": [0.88, 1.08, 1.14]},
        )
        fig.subplots_adjust(left=0.035, right=0.99, bottom=0.045, top=0.92, wspace=0.18)
        for ax in axes:
            ax.set_axis_off()

        ax = axes[0]
        ax.set_title("(a) Sample closure", loc="left", pad=8)
        y_values = [0.77, 0.55, 0.33, 0.11]
        labels = [
            "Initial records\nn=909",
            "After excluding 2025\nn=795",
            "After duplicate removal\nn=793",
            "March–October sample\nn=790",
        ]
        for y, label in zip(y_values, labels):
            _box(ax, (0.12, y), 0.76, 0.115, label, face="#F5F8FA")
        for idx, edge_text in enumerate(("−114", "−2", "−3")):
            _arrow(
                ax,
                (0.50, y_values[idx]),
                (0.50, y_values[idx + 1] + 0.115),
                edge_text,
                (0.54, (y_values[idx] + y_values[idx + 1] + 0.115) / 2),
            )

        ax = axes[1]
        ax.set_title("(b) Analysis workflow", loc="left", pad=8)
        stages = [
            (0.74, "Event database and\nsample screening"),
            (0.55, "ERA5 environmental\nextraction"),
            (0.36, "Five-variable clustering\nMLCAPE · MLLCL · Td2m\nSHR6 · SRH1"),
            (0.17, "Stability analysis"),
        ]
        for y, label in stages:
            _box(ax, (0.12, y), 0.76, 0.125, label)
        for upper, lower in zip(stages[:-1], stages[1:]):
            _arrow(ax, (0.50, upper[0]), (0.50, lower[0] + 0.125))
        _box(ax, (0.08, 0.015), 0.38, 0.085, "k=3 primary\nsolution", face="#E9F3FA", edge="#0072B2")
        _box(ax, (0.54, 0.015), 0.38, 0.085, "k=4 sensitivity\nanalysis", face="#F7F2FB", edge="#7A4E8A")
        _arrow(ax, (0.43, 0.17), (0.28, 0.10))
        _arrow(ax, (0.57, 0.17), (0.73, 0.10))

        ax = axes[2]
        ax.set_title("(c) Scientific evidence framework", loc="left", pad=8, fontsize=8.1)
        evidence = [
            ("Final tornado sample\n2006–2024, March–October (n=790)", "#E9F3FA", "#0072B2"),
            ("Multivariate environmental regimes\nand their stability", "#F4F7F9", "#52606D"),
            (
                "Vertical thermodynamic and\nkinematic characteristics\n"
                "Temperature · humidity ·\nenvironmental wind",
                "#EEF8F4",
                "#009E73",
            ),
            ("Post-hoc weather-type,\nseasonal, and spatial context", "#FFF7E8", "#B87700"),
            (
                "Interpretation restricted to\nthe confirmed-tornado sample",
                "#F7F3F2",
                "#8C5A52",
            ),
        ]
        ys = [0.79, 0.60, 0.38, 0.19, 0.02]
        heights = [0.12, 0.12, 0.175, 0.12, 0.12]
        for y, height, (label, face, edge) in zip(ys, heights, evidence):
            _box(ax, (0.08, y), 0.84, height, label, face=face, edge=edge, size=6.9)
        for idx in range(len(ys) - 1):
            _arrow(ax, (0.50, ys[idx]), (0.50, ys[idx + 1] + heights[idx + 1]))

    return export_formal_figure(fig, FIGURE_STEMS["Fig1"], "main")


def build_fig2() -> dict:
    data = _read_csv(R1_DATA / "Fig2_plotting_data.csv")
    if len(data) != 15:
        raise AssertionError("Fig2 requires 15 center values")
    variable_order = ["MLCAPE", "MLLCL", "2-m dew point", "0–6-km shear", "0–1-km SRH"]
    with publication_style():
        fig, ax = plt.subplots(figsize=(mm_to_in(178), mm_to_in(78)))
        fig.subplots_adjust(left=0.105, right=0.965, bottom=0.22, top=0.90)
        x = np.arange(len(variable_order))
        for regime in REGIME_ORDER:
            values = (
                data.loc[data["group_label"] == regime]
                .set_index("variable")
                .loc[variable_order, "value"]
                .to_numpy(float)
            )
            ax.plot(
                x,
                values,
                color=REGIME_COLORS[regime],
                marker=REGIME_MARKERS[regime],
                linestyle=REGIME_LINESTYLES[regime],
                markerfacecolor="white",
                markeredgecolor="#855A00" if regime == "C1" else REGIME_COLORS[regime],
                markeredgewidth=1.0,
                markersize=5.7,
                label=f"{regime} (n={REGIME_N[regime]})",
            )
        ax.axhline(0, color="#4F4F4F", linewidth=0.8, linestyle=":")
        ax.set_xticks(x, variable_order)
        ax.get_xticklabels()[-1].set_ha("right")
        ax.set_xlim(-0.12, 4.12)
        ax.set_ylabel("Standardized cluster center")
        ax.set_title("Standardized centers of the three environmental regimes", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.02), ncol=3, frameon=False)
        style_axis(ax, "y")
    return export_formal_figure(fig, FIGURE_STEMS["Fig2"], "main")


def build_fig3() -> dict:
    data = _read_csv(R1_DATA / "Fig3_plotting_data.csv")
    if len(data) != 3950 or data["event_id"].nunique() != 790:
        raise AssertionError("Fig3 source must contain 790 events × 5 variables")
    variable_order = ["MLCAPE", "MLLCL", "2-m dew point", "0–6-km shear", "0–1-km SRH"]
    units = {
        "MLCAPE": "J kg$^{-1}$",
        "MLLCL": "m",
        "2-m dew point": "K",
        "0–6-km shear": "m s$^{-1}$",
        "0–1-km SRH": "m$^2$ s$^{-2}$",
    }
    rng = np.random.default_rng(20240804)
    with publication_style():
        fig = plt.figure(figsize=(mm_to_in(178), mm_to_in(135)))
        outer = fig.add_gridspec(
            2, 1, height_ratios=[1, 1], left=0.085, right=0.985, bottom=0.095, top=0.92, hspace=0.39
        )
        top = outer[0].subgridspec(1, 3, wspace=0.36)
        bottom = outer[1].subgridspec(1, 3, wspace=0.36)
        axes = [
            fig.add_subplot(top[0, 0]),
            fig.add_subplot(top[0, 1]),
            fig.add_subplot(top[0, 2]),
            fig.add_subplot(bottom[0, 0]),
            fig.add_subplot(bottom[0, 1]),
        ]
        legend_ax = fig.add_subplot(bottom[0, 2])
        legend_ax.set_axis_off()
        for panel_index, (ax, variable) in enumerate(zip(axes, variable_order)):
            distributions = [
                data.loc[
                    (data["variable"] == variable) & (data["group_label"] == regime),
                    "value",
                ].to_numpy(float)
                for regime in REGIME_ORDER
            ]
            boxes = ax.boxplot(
                distributions,
                positions=np.arange(3),
                widths=0.54,
                whis=(5, 95),
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#202020", "linewidth": 1.45},
                whiskerprops={"color": "#656565", "linewidth": 0.7},
                capprops={"color": "#656565", "linewidth": 0.7},
            )
            for patch, regime in zip(boxes["boxes"], REGIME_ORDER):
                patch.set_facecolor(REGIME_COLORS[regime])
                patch.set_edgecolor(REGIME_COLORS[regime])
                patch.set_alpha(0.23)
                patch.set_linewidth(1.05)
            for position, regime, values in zip(np.arange(3), REGIME_ORDER, distributions):
                jitter = rng.uniform(-0.19, 0.19, len(values))
                ax.scatter(
                    position + jitter,
                    values,
                    s=4.0,
                    alpha=0.11,
                    color=REGIME_COLORS[regime],
                    edgecolors="none",
                    rasterized=True,
                    zorder=1,
                )
                q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
                ax.plot([position - 0.22, position + 0.22], [median, median], color="#151515", lw=1.5, zorder=4)
                ax.plot([position, position], [q25, q75], color=REGIME_COLORS[regime], lw=2.4, zorder=3)
            ax.set_xticks(np.arange(3), REGIME_ORDER)
            ax.set_ylabel(units[variable])
            ax.set_title(variable, loc="left")
            if variable == "2-m dew point":
                ax.set_ylim(268, 302)
                ax.set_yticks([270, 280, 290, 300])
            style_axis(ax, "y")
            panel_label(ax, f"({chr(97 + panel_index)})", x=-0.18, y=1.03)
        legend_ax.legend(
            handles=regime_legend_handles(include_n=True),
            loc="upper left",
            frameon=False,
            title="Confirmed-tornado sample",
            title_fontsize=7.8,
        )
        legend_ax.text(
            0,
            0.54,
            "All observations are retained.\nPoints show individual events;\nboxes emphasize the median\nand interquartile range.",
            transform=legend_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#4C4C4C",
            linespacing=1.35,
        )
        fig.suptitle("Original-unit distributions of the five clustering variables", x=0.07, ha="left")
    return export_formal_figure(fig, FIGURE_STEMS["Fig3"], "main")


def build_fig4() -> dict:
    data = _read_csv(R1_DATA / "Fig4_plotting_data.csv")
    assignments = data.loc[data["record_type"] == "event_assignment_consistency"].copy()
    counts = data.loc[data["record_type"] == "stability_count"].copy()
    observed = {
        int(k): group.set_index("category")["value"].astype(int).to_dict()
        for k, group in counts.groupby("k")
    }
    expected = {
        3: {"STABLE_CORE": 775, "MODERATE": 8, "BOUNDARY_EVENT": 7},
        4: {"STABLE_CORE": 763, "MODERATE": 21, "BOUNDARY_EVENT": 6},
    }
    if observed != expected:
        raise AssertionError(f"Fig4 stability counts mismatch: {observed}")
    stable_min = {
        int(k): group.loc[group["category"] == "STABLE_CORE", "value"].min()
        for k, group in assignments.groupby("k")
    }
    _assert_close("k3 stable threshold", stable_min[3], 0.80223880597, 1e-10)
    _assert_close("k4 stable threshold", stable_min[4], 0.80512820513, 1e-10)

    k_colors = {3: "#3366A8", 4: "#7A4E8A"}
    with publication_style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(mm_to_in(178), mm_to_in(83)),
            gridspec_kw={"width_ratios": [0.90, 1.17, 1.05]},
        )
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.85, wspace=0.38)

        ax = axes[0]
        seed = data.loc[data["record_type"] == "seed_ari"]
        distributions = [seed.loc[seed["k"] == k, "value"].to_numpy(float) for k in (3, 4)]
        violins = ax.violinplot(distributions, positions=[0, 1], widths=0.72, showextrema=False, showmedians=True)
        for body, k in zip(violins["bodies"], (3, 4)):
            body.set_facecolor(k_colors[k]); body.set_edgecolor(k_colors[k]); body.set_alpha(0.22)
        violins["cmedians"].set_color("#202020"); violins["cmedians"].set_linewidth(1.3)
        rng = np.random.default_rng(44)
        for pos, k, values in zip((0, 1), (3, 4), distributions):
            ax.scatter(pos + rng.uniform(-0.16, 0.16, len(values)), values, s=5, alpha=0.22, color=k_colors[k], edgecolors="none")
        ax.set_xticks([0, 1], ["k=3", "k=4"])
        ax.set_ylabel("Adjusted Rand index")
        ax.set_ylim(0.25, 1.025)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.set_title("Seed stability", loc="left")
        style_axis(ax, "y")
        panel_label(ax, "(a)", x=-0.20)

        ax = axes[1]
        bins = np.linspace(0.55, 1.0001, 24)
        for k in (3, 4):
            values = assignments.loc[assignments["k"] == k, "value"].to_numpy(float)
            ax.hist(values, bins=bins, histtype="step", lw=1.35, color=k_colors[k], label=f"k={k}")
            ax.axvline(stable_min[k], color=k_colors[k], linestyle="--" if k == 3 else ":", lw=1.0)
        below = {k: int((assignments.loc[assignments["k"] == k, "value"] < 0.55).sum()) for k in (3, 4)}
        ax.text(
            0.02,
            0.95,
            f"Values <0.55: k=3, {below[3]}; k=4, {below[4]}\nStable-core minima: 0.802 and 0.805",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#4B4B4B",
        )
        ax.set_xlim(0.55, 1.005)
        ax.set_xlabel("Maximum assignment probability")
        ax.set_ylabel("Events")
        ax.set_title("Event assignment consistency", loc="left")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.72))
        style_axis(ax, "y")
        panel_label(ax, "(b)", x=-0.17)

        ax = axes[2]
        categories = ("STABLE_CORE", "MODERATE", "BOUNDARY_EVENT")
        labels = ("Stable core", "Moderate", "Boundary")
        x = np.arange(3)
        width = 0.34
        for offset, k in ((-width / 2, 3), (width / 2, 4)):
            values = [observed[k][category] for category in categories]
            bars = ax.bar(x + offset, values, width, color=k_colors[k], alpha=0.78, label=f"k={k}")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value}", ha="center", va="bottom", fontsize=6.8)
        ax.set_yscale("log")
        ax.set_ylim(4, 1600)
        ax.set_yticks([10, 100, 1000])
        ax.minorticks_off()
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylabel("Event count (log scale)")
        ax.set_title("Event stability categories", loc="left")
        ax.legend(frameon=False, loc="upper right")
        style_axis(ax, "y")
        panel_label(ax, "(c)", x=-0.17)
    return export_formal_figure(fig, FIGURE_STEMS["Fig4"], "main")


def _matrix(data: pd.DataFrame, value_column: str = "value") -> np.ndarray:
    return (
        data.pivot(index="k3_regime", columns="k4_cluster", values=value_column)
        .reindex(index=REGIME_ORDER, columns=K4_ORDER)
        .to_numpy(float)
    )


def build_fig5() -> dict:
    data = _read_csv(R1_DATA / "Fig5_plotting_data.csv")
    correspondence = data.loc[data["record_type"] == "k3_k4_correspondence"].copy()
    count_rows = correspondence.loc[correspondence["statistic"] == "count"]
    if count_rows["value"].sum() != 790:
        raise AssertionError("Fig5 count heatmap does not sum to 790")
    counts = _matrix(count_rows)
    proportions = counts / counts.sum(axis=1, keepdims=True)
    algorithm = data.loc[data["record_type"] == "alternative_algorithm_ari"].copy()

    with publication_style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(mm_to_in(178), mm_to_in(96)),
            gridspec_kw={"width_ratios": [1.20, 1.20, 0.78]},
        )
        fig.subplots_adjust(left=0.07, right=0.965, bottom=0.34, top=0.76, wspace=0.50)
        labels = [K4_DISPLAY[value] for value in K4_ORDER]

        ax = axes[0]
        image = ax.imshow(counts, cmap="Blues", vmin=0, vmax=counts.max(), aspect="auto")
        for i in range(3):
            for j in range(4):
                color = "white" if counts[i, j] > counts.max() * 0.53 else "#202020"
                ax.text(j, i, f"{int(counts[i, j])}", ha="center", va="center", color=color, fontsize=7.3)
        ax.set_xticks(range(4), labels, rotation=25, ha="right")
        ax.set_yticks(range(3), REGIME_ORDER)
        ax.set_xlabel("k=4 cluster")
        ax.set_ylabel("k=3 regime")
        ax.set_title("Event counts", loc="left")
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.36, aspect=24)
        cbar.set_label("Events")
        cbar.ax.tick_params(labelsize=6.5)
        panel_label(ax, "(a)", x=-0.28)

        ax = axes[1]
        image = ax.imshow(proportions, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(4):
                value = proportions[i, j]
                color = "white" if value > 0.57 else "#202020"
                ax.text(j, i, f"{100 * value:.0f}%", ha="center", va="center", color=color, fontsize=7.3)
        ax.set_xticks(range(4), labels, rotation=25, ha="right")
        ax.set_yticks(range(3), [f"{r} (n={REGIME_N[r]})" for r in REGIME_ORDER])
        ax.set_xlabel("k=4 cluster")
        ax.set_title("Row-normalized correspondence", loc="left", fontsize=8.2)
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.36, aspect=24)
        cbar.set_label("Within-row fraction")
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["0", "0.5", "1"])
        cbar.ax.tick_params(labelsize=6.5)
        panel_label(ax, "(b)", x=-0.35)

        ax = axes[2]
        styles = {
            "Ward": ("#666666", "o", "-"),
            "Gaussian mixture": ("#8C6D31", "s", "--"),
        }
        for name, rows in algorithm.groupby("algorithm", sort=False):
            rows = rows.sort_values("k")
            color, marker, linestyle = styles[name]
            ax.plot(rows["k"], rows["value"], color=color, marker=marker, linestyle=linestyle, markerfacecolor="white", markersize=4, label=name)
            endpoint_offset = 0.012 if name == "Ward" else -0.014
            endpoint_label = "Gaussian\nmixture" if name == "Gaussian mixture" else name
            ax.text(
                6.30,
                float(rows["value"].iloc[-1]) + endpoint_offset,
                endpoint_label,
                ha="left",
                va="center",
                fontsize=6.4,
                color=color,
                linespacing=0.95,
            )
        ax.set_xticks([2, 3, 4, 5, 6])
        ax.set_xlim(1.8, 8.65)
        ax.set_ylim(0, 0.62)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("ARI relative to k-means")
        ax.set_title("Algorithm\ndependence", loc="left", fontsize=8.2)
        style_axis(ax, "y")
        panel_label(ax, "(c)", x=-0.30)
        fig.suptitle(
            "k=3–k=4 structural correspondence",
            x=0.07,
            y=0.965,
            ha="left",
        )
    return export_formal_figure(fig, FIGURE_STEMS["Fig5"], "main")


WEATHER_LABELS = {
    "QLCS/飑线": "QLCS",
    "TC": "TC",
    "其他": "Other",
    "冷涡": "Cold vortex",
    "华北对流": "N China",
    "暖区": "Warm sector",
    "气旋/冷锋": "Cyclone/CF",
    "西南对流": "SW China",
    "超单(未分类)": "Supercell*",
}


def build_fig6() -> dict:
    data = _read_csv(R1_DATA / "Fig6_plotting_data.csv").sort_values(["regime", "weather_order"])
    first = data.iloc[0]
    assertions = {
        "valid_n": (first["valid_n"], 787, 0),
        "chi_square_exact": (first["chi_square_exact"], 437.139758, 1e-6),
        "df": (first["df"], 16, 0),
        "raw_cramers_v": (first["raw_cramers_v"], 0.5269965715, 1e-10),
        "bias_corrected_cramers_v": (first["bias_corrected_cramers_v"], 0.5172497166, 1e-10),
        "bootstrap_ci_low": (first["bootstrap_ci_low"], 0.4897765785, 1e-10),
        "bootstrap_ci_high": (first["bootstrap_ci_high"], 0.5757576119, 1e-10),
    }
    for label, (value, expected, tolerance) in assertions.items():
        _assert_close(label, value, expected, tolerance)
    weather_order = (
        data[["weather_type", "weather_order"]]
        .drop_duplicates()
        .sort_values("weather_order")["weather_type"]
        .tolist()
    )
    proportions = (
        data.pivot(index="regime", columns="weather_type", values="row_proportion")
        .reindex(index=REGIME_ORDER, columns=weather_order)
        .to_numpy(float)
    )
    residuals = (
        data.pivot(index="regime", columns="weather_type", values="standardized_residual")
        .reindex(index=REGIME_ORDER, columns=weather_order)
        .to_numpy(float)
    )
    prop_flags = (
        data.pivot(index="regime", columns="weather_type", values="proportion_annotated")
        .reindex(index=REGIME_ORDER, columns=weather_order)
        .to_numpy(bool)
    )
    resid_flags = (
        data.pivot(index="regime", columns="weather_type", values="residual_annotated")
        .reindex(index=REGIME_ORDER, columns=weather_order)
        .to_numpy(bool)
    )

    with publication_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(mm_to_in(178), mm_to_in(107)),
            gridspec_kw={"width_ratios": [1, 1]},
        )
        fig.subplots_adjust(left=0.125, right=0.945, bottom=0.30, top=0.70, wspace=0.45)
        xlabels = [WEATHER_LABELS[value] for value in weather_order]

        ax = axes[0]
        image = ax.imshow(proportions, cmap="YlGnBu", vmin=0, vmax=max(0.55, proportions.max()), aspect="auto")
        for i in range(3):
            for j in range(9):
                if prop_flags[i, j]:
                    color = "white" if proportions[i, j] > 0.31 else "#202020"
                    ax.text(j, i, f"{100 * proportions[i, j]:.0f}%", ha="center", va="center", color=color, fontsize=6.8)
        ax.set_xticks(range(9), xlabels, rotation=50, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=5.8)
        ax.tick_params(axis="y", labelsize=6.7)
        ax.set_yticks(range(3), [f"{r} (n={int(data.loc[data.regime == r, 'regime_n'].iloc[0])})" for r in REGIME_ORDER])
        ax.set_title("Within-regime weather-type composition", loc="left", fontsize=8.1)
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("Row proportion")
        cbar.set_ticks([0.0, 0.2, 0.4])
        panel_label(ax, "(a)", x=-0.16)

        ax = axes[1]
        limit = 10.0
        image = ax.imshow(residuals, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), aspect="auto")
        for i in range(3):
            for j in range(9):
                if resid_flags[i, j]:
                    color = "white" if abs(residuals[i, j]) > limit * 0.56 else "#202020"
                    ax.text(j, i, f"{residuals[i, j]:+.1f}", ha="center", va="center", color=color, fontsize=6.8)
        ax.set_xticks(range(9), xlabels, rotation=50, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=5.8)
        ax.tick_params(axis="y", labelsize=6.7)
        ax.set_yticks(range(3), [f"{r} (n={int(data.loc[data.regime == r, 'regime_n'].iloc[0])})" for r in REGIME_ORDER])
        ax.set_title("Standardized residuals", loc="left", fontsize=8.1)
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("Residual")
        cbar.set_ticks([-8, -4, 0, 4, 8])
        panel_label(ax, "(b)", x=-0.16)

        fig.text(
            0.125,
            0.955,
            "Post-hoc weather-type association",
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
        )
        fig.text(
            0.125,
            0.885,
            "n=787; χ²(16)=437.1; raw Cramér’s V=0.5270; "
            "bias-corrected V=0.5172",
            ha="left",
            va="top",
            fontsize=7.3,
        )
        fig.text(
            0.125,
            0.825,
            "95% CI=0.4898–0.5758; permutation p<0.0001",
            ha="left",
            va="top",
            fontsize=7.3,
        )
        fig.text(
            0.945,
            0.028,
            "Sparse labels identify selected large proportions or residuals; the association remains many-to-many.",
            ha="right",
            va="bottom",
            fontsize=6.3,
            color="#4F4F4F",
        )
    return export_formal_figure(fig, FIGURE_STEMS["Fig6"], "main")


def build_fig7() -> dict:
    data = _read_csv(R1_DATA / "Fig7_plotting_data.csv")
    levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200]
    if len(data) != 54:
        raise AssertionError("Fig7 requires 3 regimes × 9 levels × 2 variables")
    key_n = {}
    for level in (1000, 925, 850):
        key_n[level] = [
            int(
                data.loc[
                    (data["variable"] == "t_K")
                    & (data["level_hPa"] == level)
                    & (data["group_label"] == regime),
                    "sample_size",
                ].iloc[0]
            )
            for regime in REGIME_ORDER
        ]
    if key_n != {1000: [11, 118, 139], 925: [84, 291, 350], 850: [109, 299, 352]}:
        raise AssertionError(f"Fig7 lower-level effective n mismatch: {key_n}")

    specifications = [
        ("t_K", "Temperature", "Temperature (K)"),
        ("rh_pct", "Relative humidity", "Relative humidity (%)"),
    ]
    with publication_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(mm_to_in(178), mm_to_in(112)),
            sharey=True,
        )
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.22, top=0.82, wspace=0.20)
        for panel_index, (ax, (variable, title, xlabel)) in enumerate(zip(axes, specifications)):
            ax.axhspan(500, 850, color="#D9D9D9", alpha=0.26, zorder=0)
            for regime in REGIME_ORDER:
                rows = (
                    data.loc[
                        (data["variable"] == variable)
                        & (data["group_label"] == regime)
                    ]
                    .set_index("level_hPa")
                    .loc[levels]
                    .reset_index()
                )
                ax.fill_betweenx(
                    rows["level_hPa"],
                    rows["ci_low"],
                    rows["ci_high"],
                    color=REGIME_COLORS[regime],
                    alpha=0.10,
                    linewidth=0,
                )
                ax.plot(
                    rows["median"],
                    rows["level_hPa"],
                    color=REGIME_COLORS[regime],
                    marker=REGIME_MARKERS[regime],
                    linestyle=REGIME_LINESTYLES[regime],
                    markerfacecolor="white",
                    markeredgecolor="#855A00" if regime == "C1" else REGIME_COLORS[regime],
                    markeredgewidth=0.85,
                    markersize=4.2,
                    label=regime,
                )
            ax.set_yscale("log")
            ax.set_ylim(1050, 180)
            ax.set_yticks(levels, [str(level) for level in levels])
            ax.minorticks_off()
            ax.set_xlabel(xlabel)
            ax.set_title(title, loc="left")
            if variable == "rh_pct":
                ax.set_xlim(0, 102)
                ax.set_xticks([20, 40, 60, 80, 100])
            style_axis(ax, "both")
            panel_label(ax, f"({chr(97 + panel_index)})", x=-0.16)
        axes[0].set_ylabel("Pressure (hPa)")
        fig.legend(
            handles=regime_legend_handles(include_n=True),
            loc="upper center",
            bbox_to_anchor=(0.72, 0.925),
            ncol=3,
            frameon=False,
        )
        fig.suptitle(
            "Vertical thermodynamic and moisture structures",
            x=0.11,
            y=0.985,
            ha="left",
        )
        fig.text(
            0.11,
            0.045,
            "Effective n (C0/C1/C2): 1000 hPa, 11/118/139; "
            "925 hPa, 84/291/350; 850 hPa, 109/299/352.\n"
            "Shading denotes 95% bootstrap confidence intervals.",
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#4C4C4C",
            linespacing=1.35,
        )
    return export_formal_figure(fig, FIGURE_STEMS["Fig7"], "main")


def build_fig8() -> dict:
    data = _read_csv(R1_DATA / "Fig8_plotting_data.csv")
    profile = data.loc[data["record_type"] == "vertical_profile"].copy()
    sensitivity = data.loc[data["record_type"] != "vertical_profile"].copy()
    if len(profile) != 27 or len(sensitivity) != 6:
        raise AssertionError("Fig8 requires 27 profile rows and 6 registered sensitivity rows")
    levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200]

    with publication_style():
        fig = plt.figure(figsize=(mm_to_in(178), mm_to_in(124)))
        outer = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.10, 0.90],
            left=0.085,
            right=0.955,
            bottom=0.11,
            top=0.89,
            wspace=0.38,
        )
        ax_profile = fig.add_subplot(outer[0, 0])
        right = outer[0, 1].subgridspec(2, 1, height_ratios=[0.40, 0.60], hspace=0.48)
        ax_key = fig.add_subplot(right[0, 0])
        lower_right = right[1, 0].subgridspec(
            1, 2, width_ratios=[0.64, 0.36], wspace=0.05
        )
        ax_sens = fig.add_subplot(lower_right[0, 0])
        ax_numbers = fig.add_subplot(lower_right[0, 1], sharey=ax_sens)

        for regime in REGIME_ORDER:
            rows = (
                profile.loc[profile["group_label"] == regime]
                .set_index("level_hPa")
                .loc[levels]
                .reset_index()
            )
            ax_profile.fill_betweenx(
                rows["level_hPa"],
                rows["ci_low"],
                rows["ci_high"],
                color=REGIME_COLORS[regime],
                alpha=0.10,
                linewidth=0,
            )
            ax_profile.plot(
                rows["median"],
                rows["level_hPa"],
                color=REGIME_COLORS[regime],
                marker=REGIME_MARKERS[regime],
                linestyle=REGIME_LINESTYLES[regime],
                markerfacecolor="white",
                markeredgecolor="#855A00" if regime == "C1" else REGIME_COLORS[regime],
                markeredgewidth=0.85,
                markersize=4.4,
                label=regime,
            )
        ax_profile.set_yscale("log")
        ax_profile.set_ylim(1050, 180)
        ax_profile.set_yticks(levels, [str(level) for level in levels])
        ax_profile.minorticks_off()
        ax_profile.set_xlabel("Environmental wind speed (m s$^{-1}$)")
        ax_profile.set_ylabel("Pressure (hPa)")
        ax_profile.set_title("Vertical environmental-wind profiles", loc="left")
        ax_profile.legend(handles=regime_legend_handles(include_n=True), frameon=False, loc="lower right")
        style_axis(ax_profile, "both")
        panel_label(ax_profile, "(a)", x=-0.16)

        key_rows = profile.loc[profile["level_hPa"].isin([850, 200])].copy()
        base_y = {"850 hPa": 0, "200 hPa": 1}
        offsets = {"C0": -0.18, "C1": 0.0, "C2": 0.18}
        for regime in REGIME_ORDER:
            rows = key_rows.loc[key_rows["group_label"] == regime].sort_values("level_hPa", ascending=False)
            for _, row in rows.iterrows():
                label = f"{int(row['level_hPa'])} hPa"
                y = base_y[label] + offsets[regime]
                ax_key.errorbar(
                    row["median"],
                    y,
                    xerr=[[row["median"] - row["ci_low"]], [row["ci_high"] - row["median"]]],
                    fmt=REGIME_MARKERS[regime],
                    color=REGIME_COLORS[regime],
                    ecolor=REGIME_COLORS[regime],
                    markerfacecolor="white",
                    markeredgecolor="#855A00" if regime == "C1" else REGIME_COLORS[regime],
                    capsize=2,
                    markersize=4.2,
                )
        ax_key.set_yticks([0, 1], ["850 hPa", "200 hPa"])
        ax_key.set_xlim(4, 32)
        ax_key.set_xticks([10, 20, 30])
        ax_key.set_xlabel("Median and 95% CI (m s$^{-1}$)")
        ax_key.set_title("Key pressure levels", loc="left")
        style_axis(ax_key, "x")
        panel_label(ax_key, "(b)", x=-0.28)

        display_labels = [
            "C2 full",
            "C2 no TC",
            "C0 full",
            "C0 30–40°N",
            "C0 JJA",
            "C0 no cold vortex",
        ]
        sensitivity = sensitivity.reset_index(drop=True)
        y = np.arange(len(sensitivity))
        for position, (_, row), label in zip(y, sensitivity.iterrows(), display_labels):
            regime = row["group_label"]
            ax_sens.errorbar(
                row["median"],
                position,
                xerr=[[row["median"] - row["ci_low"]], [row["ci_high"] - row["median"]]],
                fmt=REGIME_MARKERS[regime],
                color=REGIME_COLORS[regime],
                ecolor=REGIME_COLORS[regime],
                markerfacecolor="white",
                markeredgecolor="#855A00" if regime == "C1" else REGIME_COLORS[regime],
                capsize=2,
                markersize=4.2,
            )
            ax_numbers.text(
                0.02,
                position,
                f"{row['median']:.1f} [{row['ci_low']:.1f}, {row['ci_high']:.1f}]",
                transform=ax_numbers.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=5.8,
                clip_on=False,
            )
        ax_sens.set_yticks(y, display_labels)
        ax_sens.invert_yaxis()
        ax_sens.set_xlim(10, 36)
        ax_sens.set_xticks([10, 20, 30])
        ax_sens.tick_params(axis="y", labelsize=6.6, pad=2)
        ax_sens.set_xlabel("Median wind speed (m s$^{-1}$)")
        ax_sens.set_title("Sensitivity analyses", loc="left")
        style_axis(ax_sens, "x")
        panel_label(ax_sens, "(c)", x=-0.28)
        ax_numbers.set_ylim(ax_sens.get_ylim())
        ax_numbers.set_axis_off()
        ax_numbers.text(
            0.02,
            1.03,
            "Estimate [95% CI]",
            transform=ax_numbers.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
        )
        fig.suptitle("Vertical environmental-wind structures", x=0.085, ha="left")
    return export_formal_figure(fig, FIGURE_STEMS["Fig8"], "main")


def build_fig9() -> dict:
    data = _read_csv(R2_DATA / "Fig9_plotting_snapshot.csv")
    events = data.loc[data["record_type"] == "event"].copy()
    monthly = data.loc[data["record_type"] == "monthly_summary"].copy()
    if len(events) != 790 or events["event_id"].nunique() != 790:
        raise AssertionError("Fig9 event snapshot must contain 790 unique events")
    observed = events["regime_id"].value_counts().sort_index().to_dict()
    if observed != REGIME_N:
        raise AssertionError(f"Fig9 regime counts mismatch: {observed}")

    with publication_style():
        fig = plt.figure(figsize=(mm_to_in(178), mm_to_in(116)))
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.40, 1.00],
            height_ratios=[1, 1],
            left=0.055,
            right=0.985,
            bottom=0.105,
            top=0.92,
            wspace=0.26,
            hspace=0.42,
        )
        ax_map = fig.add_subplot(grid[:, 0], projection=ccrs.PlateCarree())
        ax_month = fig.add_subplot(grid[0, 1])
        ax_lat = fig.add_subplot(grid[1, 1])

        ax_map.set_extent([84, 135, 17, 53], crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F5F5F2", zorder=0)
        ax_map.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#EDF4F7", zorder=0)
        ax_map.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="#555555", zorder=2)
        ax_map.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="#777777", zorder=2)
        provinces = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none",
        )
        ax_map.add_feature(provinces, linewidth=0.25, edgecolor="#B0B0B0", zorder=1)
        gl = ax_map.gridlines(
            draw_labels=True,
            xlocs=np.arange(90, 131, 10),
            ylocs=np.arange(20, 51, 10),
            linewidth=0.4,
            color="#B8B8B8",
            alpha=0.55,
            linestyle=":",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 6.8}
        gl.ylabel_style = {"size": 6.8}
        for regime in REGIME_ORDER:
            rows = events.loc[events["regime_id"] == regime]
            ax_map.scatter(
                rows["longitude"],
                rows["latitude"],
                transform=ccrs.PlateCarree(),
                s=8,
                alpha=0.42,
                marker=REGIME_MARKERS[regime],
                facecolor=REGIME_COLORS[regime],
                edgecolor="white",
                linewidth=0.15,
                label=f"{regime} (n={REGIME_N[regime]})",
                zorder=3,
            )
        ax_map.set_title("Spatial distribution", loc="left", pad=7)
        ax_map.legend(loc="lower left", frameon=True, framealpha=0.88, borderpad=0.35)
        panel_label(ax_map, "(a)", x=-0.10, y=1.02)

        for regime in REGIME_ORDER:
            rows = monthly.loc[monthly["regime_id"] == regime].sort_values("month")
            ax_month.plot(
                rows["month"],
                rows["within_regime_pct"],
                color=REGIME_COLORS[regime],
                marker=REGIME_MARKERS[regime],
                linestyle=REGIME_LINESTYLES[regime],
                markerfacecolor="white",
                markeredgecolor="#855A00" if regime == "C1" else REGIME_COLORS[regime],
                markersize=4.1,
            )
        ax_month.set_xticks(range(3, 11), ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"])
        ax_month.tick_params(axis="x", rotation=30)
        ax_month.set_ylabel("Within-regime frequency (%)")
        ax_month.set_title("March–October\nwithin-regime frequency", loc="left", fontsize=8.1)
        style_axis(ax_month, "y")
        panel_label(ax_month, "(b)", x=-0.19)

        rng = np.random.default_rng(909)
        distributions = [
            events.loc[events["regime_id"] == regime, "latitude"].to_numpy(float)
            for regime in REGIME_ORDER
        ]
        boxes = ax_lat.boxplot(
            distributions,
            positions=np.arange(3),
            widths=0.52,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#171717", "linewidth": 1.45},
            whiskerprops={"color": "#626262", "linewidth": 0.7},
            capprops={"color": "#626262", "linewidth": 0.7},
        )
        for patch, regime in zip(boxes["boxes"], REGIME_ORDER):
            patch.set_facecolor(REGIME_COLORS[regime])
            patch.set_edgecolor(REGIME_COLORS[regime])
            patch.set_alpha(0.25)
            patch.set_linewidth(1.0)
        for pos, regime, values in zip(np.arange(3), REGIME_ORDER, distributions):
            ax_lat.scatter(
                pos + rng.uniform(-0.18, 0.18, len(values)),
                values,
                s=3.5,
                alpha=0.08,
                color=REGIME_COLORS[regime],
                edgecolors="none",
                rasterized=True,
            )
        ax_lat.set_xticks(np.arange(3), [f"{r}\n(n={REGIME_N[r]})" for r in REGIME_ORDER])
        ax_lat.set_ylabel("Latitude (°N)")
        ax_lat.set_title("Latitude distributions", loc="left")
        style_axis(ax_lat, "y")
        panel_label(ax_lat, "(c)", x=-0.19)
    return export_formal_figure(fig, FIGURE_STEMS["Fig9"], "main")


def build_figs1() -> dict:
    data = _read_csv(R2_DATA / "FigS1_plotting_snapshot.csv")
    if len(data) != 59049:
        raise AssertionError("FigS1 requires 3 × 3 × 81 × 81 grid cells")
    field_order = ("Z500", "RH500", "WS200")
    specs = {
        "Z500": ("500-hPa geopotential height", "Composite mean (gpm)", "viridis", 5600, 5900),
        "RH500": ("500-hPa relative humidity", "Composite mean (%)", "YlGnBu", 40, 75),
        "WS200": (
            "Magnitude of composite-mean\nwind vector (200 hPa)",
            "Magnitude of composite-mean vector (m s$^{-1}$)",
            "magma",
            0,
            32,
        ),
    }
    with publication_style():
        fig, axes = plt.subplots(
            3,
            3,
            figsize=(mm_to_in(178), mm_to_in(168)),
            sharex=True,
            sharey=True,
        )
        fig.subplots_adjust(
            left=0.14,
            right=0.96,
            bottom=0.15,
            top=0.81,
            wspace=0.16,
            hspace=0.18,
        )
        images = {}
        for row_index, regime in enumerate(REGIME_ORDER):
            for col_index, field in enumerate(field_order):
                ax = axes[row_index, col_index]
                rows = data.loc[(data["regime"] == regime) & (data["field"] == field)]
                grid = (
                    rows.pivot(
                        index="relative_latitude_deg",
                        columns="relative_longitude_deg",
                        values="composite_mean_value",
                    )
                    .sort_index()
                    .sort_index(axis=1)
                )
                x = grid.columns.to_numpy(float)
                y = grid.index.to_numpy(float)
                _, _, cmap, vmin, vmax = specs[field]
                image = ax.pcolormesh(
                    x,
                    y,
                    grid.to_numpy(float),
                    shading="nearest",
                    cmap=cmap,
                    norm=Normalize(vmin, vmax),
                    rasterized=False,
                )
                images[field] = image
                ax.axhline(0, color="white", linewidth=0.45, alpha=0.6)
                ax.axvline(0, color="white", linewidth=0.45, alpha=0.6)
                ax.plot(0, 0, marker="+", color="white", markersize=6, markeredgewidth=1.0)
                ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_aspect("equal")
                ax.set_xticks([-10, -5, 0, 5, 10])
                ax.set_yticks([-10, -5, 0, 5, 10])
                if row_index == 2:
                    ax.set_xlabel("Relative longitude (°)")
                if col_index == 0:
                    ax.set_ylabel("Relative latitude (°)")
                ax.set_title(
                    f"({chr(97 + row_index * 3 + col_index)})",
                    loc="left",
                    pad=3,
                )
        for col_index, field in enumerate(field_order):
            position = axes[0, col_index].get_position()
            fig.text(
                (position.x0 + position.x1) / 2,
                0.875,
                specs[field][0],
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                linespacing=1.05,
            )
        for row_index, regime in enumerate(REGIME_ORDER):
            position = axes[row_index, 0].get_position()
            fig.text(
                0.035,
                (position.y0 + position.y1) / 2,
                f"{regime}\n(n={REGIME_N[regime]})",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                linespacing=1.12,
            )
        for col_index, field in enumerate(field_order):
            cbar = fig.colorbar(
                images[field],
                ax=axes[:, col_index].tolist(),
                orientation="horizontal",
                fraction=0.035,
                pad=0.14,
                aspect=28,
            )
            cbar.set_label(specs[field][1], fontsize=6.8)
            cbar.ax.tick_params(labelsize=6.8)
            if field == "Z500":
                cbar.set_ticks([5600, 5700, 5800, 5900])
            elif field == "RH500":
                cbar.set_ticks([40, 50, 60, 70])
            else:
                cbar.set_ticks([0, 8, 16, 24, 32])
        fig.suptitle(
            "Event-centered two-dimensional composite fields",
            x=0.14,
            y=0.985,
            ha="left",
        )
    return export_formal_figure(fig, FIGURE_STEMS["FigS1"], "supplementary")


S2_VARIABLE_ORDER = ("MLCAPE_Jkg", "MLLCL_m", "ERA5_d2m_K", "SHR6_ms", "SRH1_m2s2")
S2_VARIABLE_LABELS = {
    "MLCAPE_Jkg": "MLCAPE\n(J kg$^{-1}$)",
    "MLLCL_m": "MLLCL\n(m)",
    "ERA5_d2m_K": "2-m dew point\n(K)",
    "SHR6_ms": "0–6-km shear\n(m s$^{-1}$)",
    "SRH1_m2s2": "0–1-km SRH\n(m$^2$ s$^{-2}$)",
}
S2_R1_NAMES = {
    "MLCAPE_Jkg": "MLCAPE",
    "MLLCL_m": "MLLCL",
    "ERA5_d2m_K": "2-m dew point",
    "SHR6_ms": "0–6-km shear",
    "SRH1_m2s2": "0–1-km SRH",
}


def _s2_quantiles(source: pd.DataFrame) -> pd.DataFrame:
    raw = _read_csv(R1_DATA / "Fig3_plotting_data.csv")
    rows = []
    for variable in S2_VARIABLE_ORDER:
        source_rows = source.loc[
            (source["record_type"] == "distribution_quantiles")
            & (source["variable"] == variable)
        ].set_index("regime")
        for regime in REGIME_ORDER:
            values = raw.loc[
                (raw["variable"] == S2_R1_NAMES[variable])
                & (raw["group_label"] == regime),
                "value",
            ].to_numpy(float)
            p5, p95 = np.quantile(values, [0.05, 0.95])
            reference = source_rows.loc[regime]
            rows.append(
                {
                    "variable": variable,
                    "regime": regime,
                    "n": int(reference["n"]),
                    "p5": p5,
                    "p25": float(reference["p25"]),
                    "median": float(reference["median"]),
                    "p75": float(reference["p75"]),
                    "p95": p95,
                    "display_derivation": "p5/p95 descriptive quantiles from the accepted 790-event raw plotting snapshot; p25/median/p75 copied from the registered Round 2 snapshot",
                }
            )
    derived = pd.DataFrame(rows)
    destination = SNAPSHOT_DIR / "FigS02_display_quantiles_snapshot.csv"
    derived.to_csv(destination, index=False, encoding="utf-8-sig")
    return derived


def build_figs2() -> dict:
    source = _read_csv(R2_DATA / "FigS2_plotting_snapshot.csv")
    effect = source.loc[source["record_type"] == "pairwise_cliffs_delta"].copy()
    quantiles = _s2_quantiles(source)
    if len(effect) != 15 or len(quantiles) != 15:
        raise AssertionError("FigS2 requires 15 effect rows and 15 quantile rows")
    comparisons = ["C0_vs_C1", "C0_vs_C2", "C1_vs_C2"]
    comp_labels = {"C0_vs_C1": "C0 − C1", "C0_vs_C2": "C0 − C2", "C1_vs_C2": "C1 − C2"}
    comp_colors = {"C0_vs_C1": "#8C6D31", "C0_vs_C2": "#2B6F8A", "C1_vs_C2": "#7A4E8A"}
    comp_markers = {"C0_vs_C1": "o", "C0_vs_C2": "s", "C1_vs_C2": "^"}

    with publication_style():
        fig = plt.figure(figsize=(mm_to_in(178), mm_to_in(160)))
        outer = fig.add_gridspec(
            2,
            1,
            height_ratios=[0.60, 0.40],
            left=0.20,
            right=0.95,
            bottom=0.10,
            top=0.91,
            hspace=0.72,
        )
        ax_effect = fig.add_subplot(outer[0, 0])
        base = np.arange(len(S2_VARIABLE_ORDER))
        offsets = {"C0_vs_C1": -0.21, "C0_vs_C2": 0.0, "C1_vs_C2": 0.21}
        for comparison in comparisons:
            rows = (
                effect.loc[effect["comparison"] == comparison]
                .set_index("variable")
                .loc[list(S2_VARIABLE_ORDER)]
            )
            values = rows["cliffs_delta"].to_numpy(float)
            ax_effect.errorbar(
                values,
                base + offsets[comparison],
                xerr=np.vstack([values - rows["ci_low"].to_numpy(float), rows["ci_high"].to_numpy(float) - values]),
                fmt=comp_markers[comparison],
                color=comp_colors[comparison],
                ecolor=comp_colors[comparison],
                markerfacecolor="white",
                markeredgewidth=0.9,
                capsize=2.2,
                markersize=4.7,
                label=comp_labels[comparison],
            )
        ax_effect.axvline(0, color="#555555", linewidth=0.8, linestyle=":")
        ax_effect.set_xlim(-1.02, 1.02)
        ax_effect.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
        ax_effect.set_yticks(base, [S2_VARIABLE_LABELS[v].replace("\n", " ") for v in S2_VARIABLE_ORDER])
        ax_effect.invert_yaxis()
        ax_effect.set_xlabel("Cliff’s δ (first regime minus second)")
        ax_effect.set_title("Pairwise Cliff’s δ with 95% confidence intervals", loc="left")
        ax_effect.legend(
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.015),
            ncol=3,
        )
        style_axis(ax_effect, "x")
        panel_label(ax_effect, "(a)", x=-0.19)

        bottom = outer[1, 0].subgridspec(1, 5, wspace=0.60)
        quantile_axes = [fig.add_subplot(bottom[0, index]) for index in range(5)]
        for index, (ax, variable) in enumerate(zip(quantile_axes, S2_VARIABLE_ORDER)):
            rows = quantiles.loc[quantiles["variable"] == variable].set_index("regime")
            for pos, regime in enumerate(REGIME_ORDER):
                row = rows.loc[regime]
                ax.plot([row["p5"], row["p95"]], [pos, pos], color=REGIME_COLORS[regime], lw=0.8)
                ax.plot([row["p25"], row["p75"]], [pos, pos], color=REGIME_COLORS[regime], lw=3.1, solid_capstyle="round")
                ax.plot(row["median"], pos, marker=REGIME_MARKERS[regime], color=REGIME_COLORS[regime], markerfacecolor="white", markeredgecolor="#855A00" if regime == "C1" else REGIME_COLORS[regime], markersize=4.0)
            ax.set_yticks(range(3), REGIME_ORDER if index == 0 else [])
            ax.set_ylim(2.55, -0.55)
            ax.set_title(S2_VARIABLE_LABELS[variable], fontsize=6.8, pad=3)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis="x", labelsize=6.4)
            style_axis(ax, "x")
        quantile_top = max(ax.get_position().y1 for ax in quantile_axes)
        panel_b_y = quantile_top + 0.065
        fig.text(0.20, panel_b_y, "(b)", ha="left", va="bottom", fontsize=8.7, fontweight="bold")
        fig.text(
            0.235,
            panel_b_y,
            "Regime-specific quantile ranges in original units",
            ha="left",
            va="bottom",
            fontsize=8.7,
            fontweight="bold",
        )
        fig.legend(
            handles=regime_legend_handles(include_n=True),
            loc="lower center",
            bbox_to_anchor=(0.59, 0.015),
            ncol=3,
            frameon=False,
        )
    return export_formal_figure(fig, FIGURE_STEMS["FigS2"], "supplementary")


def build_figs3() -> dict:
    data = _read_csv(R2_DATA / "FigS4_plotting_snapshot.csv")
    events = data.loc[data["record_type"] == "event_raw_values"].copy()
    centers = data.loc[data["record_type"] == "k4_standardized_center"].copy()
    if len(events) != 790 or len(centers) != 4:
        raise AssertionError("FigS3 requires 790 event rows and 4 k=4 centers")
    counts = events["k4_label"].value_counts().sort_index().to_dict()
    expected = {"K4_C0": 125, "K4_C1": 206, "K4_C2": 249, "K4_C3": 210}
    if counts != expected:
        raise AssertionError(f"FigS3 k4 counts mismatch: {counts}")
    features = ("MLCAPE_Jkg", "MLLCL_m", "ERA5_d2m_K", "SHR6_ms", "SRH1_m2s2")
    center_columns = tuple(f"{feature}_z_center" for feature in features)
    center_labels = ("MLCAPE", "log(1+MLLCL)", "2-m dew point", "0–6-km shear", "0–1-km SRH")
    raw_labels = {
        "MLCAPE_Jkg": "MLCAPE (J kg$^{-1}$)",
        "MLLCL_m": "MLLCL (m)",
        "ERA5_d2m_K": "2-m dew point (K)",
        "SHR6_ms": "0–6-km shear (m s$^{-1}$)",
        "SRH1_m2s2": "0–1-km SRH (m$^2$ s$^{-2}$)",
    }

    with publication_style():
        fig = plt.figure(figsize=(mm_to_in(178), mm_to_in(188)))
        outer = fig.add_gridspec(
            3,
            1,
            height_ratios=[1.18, 1, 1],
            left=0.10,
            right=0.975,
            bottom=0.07,
            top=0.84,
            hspace=0.52,
        )
        ax_center = fig.add_subplot(outer[0, 0])
        x = np.arange(5)
        indexed = centers.set_index("k4_label")
        for label in K4_ORDER:
            ax_center.plot(
                x,
                indexed.loc[label, list(center_columns)].to_numpy(float),
                color=K4_COLORS[label],
                marker=K4_MARKERS[label],
                linestyle=K4_LINESTYLES[label],
                markerfacecolor="white",
                markeredgewidth=0.9,
                markersize=5.0,
                label=f"{K4_DISPLAY[label]} (n={expected[label]})",
            )
        ax_center.axhline(0, color="#555555", linewidth=0.8, linestyle=":")
        ax_center.set_xticks(x, center_labels)
        ax_center.set_ylabel("Standardized center")
        ax_center.set_title("k=4 standardized centers", loc="left")
        handles, labels = ax_center.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.61, 0.905),
            ncol=4,
        )
        style_axis(ax_center, "y")
        panel_label(ax_center, "(a)", x=-0.08)

        second = outer[1, 0].subgridspec(1, 3, wspace=0.32)
        third = outer[2, 0].subgridspec(1, 2, wspace=0.30)
        axes = [fig.add_subplot(second[0, index]) for index in range(3)] + [fig.add_subplot(third[0, index]) for index in range(2)]
        rng = np.random.default_rng(403)
        for panel_index, (ax, feature) in enumerate(zip(axes, features), start=1):
            distributions = [events.loc[events["k4_label"] == label, feature].to_numpy(float) for label in K4_ORDER]
            boxes = ax.boxplot(
                distributions,
                positions=np.arange(4),
                widths=0.52,
                whis=(5, 95),
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#202020", "linewidth": 1.25},
                whiskerprops={"color": "#646464", "linewidth": 0.65},
                capprops={"color": "#646464", "linewidth": 0.65},
            )
            for patch, label in zip(boxes["boxes"], K4_ORDER):
                patch.set_facecolor(K4_COLORS[label]); patch.set_edgecolor(K4_COLORS[label]); patch.set_alpha(0.23)
            for pos, label, values in zip(np.arange(4), K4_ORDER, distributions):
                ax.scatter(pos + rng.uniform(-0.17, 0.17, len(values)), values, s=3.5, alpha=0.08, color=K4_COLORS[label], edgecolors="none", rasterized=True)
            ax.set_xticks(range(4), [K4_DISPLAY[label] for label in K4_ORDER], rotation=24, ha="right")
            ax.set_ylabel(raw_labels[feature])
            if feature == "SHR6_ms":
                ax.set_ylim(-1.0, 40.5)
                ax.set_yticks([0, 10, 20, 30, 40])
            ax.set_title(raw_labels[feature].split(" (")[0], loc="left")
            style_axis(ax, "y")
            panel_label(ax, f"({chr(97 + panel_index)})", x=-0.17)
        fig.suptitle(
            "Complete k=4 supplementary structure",
            x=0.10,
            y=0.965,
            ha="left",
        )
    return export_formal_figure(fig, FIGURE_STEMS["FigS3"], "supplementary")


def preserve_held_assets() -> list[dict]:
    """Copy complete Round 2 source bundles for the two held figures."""

    specifications = [
        {
            "hold_id": "HOLD_S01_environmental_uv_profiles",
            "source_id": "Round2 FigS3",
            "status": "ARCHIVED_OPTIONAL_SUPPLEMENT",
            "snapshot": R2_DATA / "FigS3_plotting_snapshot.csv",
            "script": ROUND2_ROOT / "02_scripts" / "build_figs3_round2.py",
            "caption_en": ROUND2_ROOT / "06_caption_drafts" / "FigS3_caption_en.md",
            "caption_zh": ROUND2_ROOT / "06_caption_drafts" / "FigS3_caption_zh.md",
            "qc": ROUND2_ROOT / "07_qc_reports" / "FigS3_QC.md",
            "review_glob": "*FigS3*review*.png",
        },
        {
            "hold_id": "HOLD_S02_stpmod_rated_events",
            "source_id": "Round2 FigS5",
            "status": "RESEARCHER_APPROVAL_REQUIRED",
            "snapshot": R2_DATA / "FigS5_plotting_snapshot.csv",
            "script": ROUND2_ROOT / "02_scripts" / "build_figs5_round2.py",
            "caption_en": ROUND2_ROOT / "06_caption_drafts" / "FigS5_caption_en.md",
            "caption_zh": ROUND2_ROOT / "06_caption_drafts" / "FigS5_caption_zh.md",
            "qc": ROUND2_ROOT / "07_qc_reports" / "FigS5_QC.md",
            "review_glob": "*FigS5*review*.png",
        },
    ]
    records = []
    for spec in specifications:
        root = HOLD_DIR / spec["hold_id"]
        for child in ("data", "script", "captions", "qc", "review"):
            (root / child).mkdir(parents=True, exist_ok=True)
        files = {
            "data": spec["snapshot"],
            "script": spec["script"],
            "caption_en": spec["caption_en"],
            "caption_zh": spec["caption_zh"],
            "qc": spec["qc"],
        }
        review_candidates = list(ROUND2_ROOT.rglob(spec["review_glob"]))
        if len(review_candidates) != 1:
            raise AssertionError(f"Expected one held review PNG for {spec['source_id']}, found {review_candidates}")
        files["review"] = review_candidates[0]
        destinations = {
            "data": root / "data" / files["data"].name,
            "script": root / "script" / files["script"].name,
            "caption_en": root / "captions" / files["caption_en"].name,
            "caption_zh": root / "captions" / files["caption_zh"].name,
            "qc": root / "qc" / files["qc"].name,
            "review": root / "review" / files["review"].name,
        }
        for key, source in files.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destinations[key])
        readme = (
            f"# {spec['hold_id']}\n\n"
            f"- Source: {spec['source_id']}\n"
            f"- Status: `{spec['status']}`\n"
            "- This bundle is retained for possible future use and has no active formal supplementary-figure number.\n"
            "- The copied plotting snapshot, source script, bilingual captions, review image, and QC report are preserved unchanged.\n"
        )
        write_text(root / "README.md", readme)
        records.append(
            {
                "hold_id": spec["hold_id"],
                "source_id": spec["source_id"],
                "status": spec["status"],
                "files": {key: relative_to_final(value) for key, value in destinations.items()},
                "sha256": {key: sha256_file(value) for key, value in destinations.items()},
            }
        )
    write_json(HOLD_DIR / "held_asset_registry.json", records)
    return records


def build_all_figures() -> dict:
    ensure_directories()
    _copy_source_snapshots()
    builders = [
        ("Fig1", build_fig1),
        ("Fig2", build_fig2),
        ("Fig3", build_fig3),
        ("Fig4", build_fig4),
        ("Fig5", build_fig5),
        ("Fig6", build_fig6),
        ("Fig7", build_fig7),
        ("Fig8", build_fig8),
        ("Fig9", build_fig9),
        ("FigS1", build_figs1),
        ("FigS2", build_figs2),
        ("FigS3", build_figs3),
    ]
    metadata = {}
    log_lines = []
    for figure_id, builder in builders:
        metadata[figure_id] = builder()
        log_lines.append(
            f"{figure_id}\t{metadata[figure_id]['stem']}\t"
            f"{metadata[figure_id]['image']['pixel_width']}x{metadata[figure_id]['image']['pixel_height']}"
        )
    metadata["held"] = preserve_held_assets()
    write_json(QC_DIR / "formal_figure_build_metadata.json", metadata)
    write_text(LOG_DIR / "final_figure_build.log", "\n".join(log_lines) + "\n")
    return metadata


if __name__ == "__main__":
    build_all_figures()
