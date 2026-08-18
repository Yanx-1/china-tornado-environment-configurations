"""Gate 1 prototype for Fig. 4: seed and event-level stability."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROUND_ROOT = SCRIPT_PATH.parent.parent
sys.path.insert(0, str(ROUND_ROOT / "03_config"))

from figure_style_config import (  # noqa: E402
    FIGURE_SIZES_MM,
    K_LINESTYLES,
    K_MARKERS,
    K_NEUTRAL_COLORS,
    PROTOTYPE_DPI,
    STABILITY_COLORS,
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
    assert_close,
    assert_exact,
    inspect_png,
    scan_forbidden_language,
    write_qc_report,
)


SEED_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/09_seed_stability_100_v3.csv"
)
SEED_SUMMARY_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/10_seed_stability_summary_v3.csv"
)
EVENT_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/13_event_level_stability_v3.csv"
)
EVENT_SUMMARY_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/14_event_level_stability_summary_v3.csv"
)
SCATTER_BASE_SEED = 20260804
STATUS_ORDER = ("STABLE_CORE", "MODERATE", "BOUNDARY_EVENT")
STATUS_DISPLAY = {
    "STABLE_CORE": "Stable core",
    "MODERATE": "Moderate",
    "BOUNDARY_EVENT": "Boundary",
}


def build() -> dict[str, object]:
    ensure_output_dirs()
    seed = read_csv_checked(
        SEED_SOURCE, required_columns=("k", "seed", "ari_vs_primary")
    )
    seed_summary = read_csv_checked(
        SEED_SUMMARY_SOURCE, required_columns=("k", "median_ari")
    )
    events = read_csv_checked(
        EVENT_SOURCE,
        required_columns=(
            "event_id",
            "k",
            "max_assignment_probability",
            "stability_status",
        ),
        dtype={"event_id": str},
    )
    event_summary = read_csv_checked(
        EVENT_SUMMARY_SOURCE,
        required_columns=("k", "stability_status", "n_events"),
    )
    seed = seed.loc[seed["k"].isin([3, 4])].copy()
    seed_summary = seed_summary.loc[seed_summary["k"].isin([3, 4])].copy()
    events = events.loc[events["k"].isin([3, 4])].copy()
    event_summary = event_summary.loc[event_summary["k"].isin([3, 4])].copy()

    assert_exact(
        "seed replicate counts",
        seed.groupby("k").size().reindex([3, 4]).astype(int).tolist(),
        [100, 100],
    )
    assert_exact(
        "event stability rows",
        events.groupby("k").size().reindex([3, 4]).astype(int).tolist(),
        [790, 790],
    )
    expected_counts = {
        3: {"STABLE_CORE": 775, "MODERATE": 8, "BOUNDARY_EVENT": 7},
        4: {"STABLE_CORE": 763, "MODERATE": 21, "BOUNDARY_EVENT": 6},
    }
    for k in (3, 4):
        observed = (
            event_summary.loc[event_summary["k"] == k]
            .set_index("stability_status")["n_events"]
            .astype(int)
            .to_dict()
        )
        assert_exact(f"k={k} stability counts", observed, expected_counts[k])
        assert_exact(f"k={k} stability closure", sum(observed.values()), 790)

    median_ari = (
        seed_summary.set_index("k")["median_ari"].astype(float).to_dict()
    )
    assert_close("k=3 median ARI", median_ari[3], 0.978, 0.0005)
    assert_close("k=4 median ARI", median_ari[4], 0.974, 0.0005)
    observed_stable_min = {
        k: float(
            events.loc[
                (events["k"] == k)
                & (events["stability_status"] == "STABLE_CORE"),
                "max_assignment_probability",
            ].min()
        )
        for k in (3, 4)
    }
    assert_close(
        "k=3 observed stable-core minimum",
        observed_stable_min[3],
        0.8022,
        0.0001,
    )
    assert_close(
        "k=4 observed stable-core minimum",
        observed_stable_min[4],
        0.8051,
        0.0001,
    )

    source_seed = relative_source(project_path(SEED_SOURCE))
    source_event = relative_source(project_path(EVENT_SOURCE))
    source_summary = relative_source(project_path(EVENT_SUMMARY_SOURCE))
    records: list[dict[str, object]] = []
    for _, row in seed.iterrows():
        records.append(
            {
                "record_type": "seed_ari",
                "event_id": "",
                "k": int(row["k"]),
                "seed": int(row["seed"]),
                "category": "",
                "value": float(row["ari_vs_primary"]),
                "unit": "ARI",
                "statistic": "ARI versus primary solution",
                "sample_size": 100,
                "source_file": source_seed,
                "transformation_note": "subset frozen seed runs to k=3 and k=4",
            }
        )
    for _, row in events.iterrows():
        records.append(
            {
                "record_type": "event_assignment_consistency",
                "event_id": row["event_id"],
                "k": int(row["k"]),
                "seed": "",
                "category": row["stability_status"],
                "value": float(row["max_assignment_probability"]),
                "unit": "proportion",
                "statistic": "maximum assignment probability",
                "sample_size": 790,
                "source_file": source_event,
                "transformation_note": "frozen event-level value; no reclassification",
            }
        )
    for _, row in event_summary.iterrows():
        records.append(
            {
                "record_type": "stability_count",
                "event_id": "",
                "k": int(row["k"]),
                "seed": "",
                "category": row["stability_status"],
                "value": int(row["n_events"]),
                "unit": "events",
                "statistic": "frozen status count",
                "sample_size": 790,
                "source_file": source_summary,
                "transformation_note": "frozen categories retained verbatim",
            }
        )
    plotting = pd.DataFrame(records)
    assert_exact("Fig.4 plotting rows", len(plotting), 200 + 1580 + 6)
    data_path = PLOTTING_DATA_DIR / "Fig4_plotting_data.csv"
    output_path = PNG_DIR / "Fig4_stability_GATE1_prototype.png"
    write_csv_atomic(data_path, plotting)

    with manuscript_style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=mm_to_inches(*FIGURE_SIZES_MM["Fig4"]),
            layout="constrained",
            gridspec_kw={"width_ratios": [0.95, 1.25, 1.05]},
        )
        ax_a, ax_b, ax_c = axes

        seed_values = [
            seed.loc[seed["k"] == k, "ari_vs_primary"].to_numpy(dtype=float)
            for k in (3, 4)
        ]
        violin = ax_a.violinplot(
            seed_values,
            positions=[0, 1],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, k in zip(violin["bodies"], (3, 4)):
            body.set_facecolor(K_NEUTRAL_COLORS[k])
            body.set_edgecolor(K_NEUTRAL_COLORS[k])
            body.set_alpha(0.16)
        box = ax_a.boxplot(
            seed_values,
            positions=[0, 1],
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "linewidth": 0.7},
            capprops={"color": "#555555", "linewidth": 0.7},
            boxprops={"linewidth": 0.8},
        )
        for patch, k in zip(box["boxes"], (3, 4)):
            patch.set_facecolor(K_NEUTRAL_COLORS[k])
            patch.set_alpha(0.24)
            patch.set_edgecolor(K_NEUTRAL_COLORS[k])
        for position, (k, values) in enumerate(zip((3, 4), seed_values)):
            rng = np.random.default_rng(
                stable_seed(SCATTER_BASE_SEED, f"seed-ari-k{k}")
            )
            ax_a.scatter(
                position + rng.uniform(-0.12, 0.12, size=len(values)),
                values,
                s=5,
                marker=K_MARKERS[k],
                color=K_NEUTRAL_COLORS[k],
                alpha=0.28,
                edgecolors="none",
                zorder=1,
            )
            ax_a.text(
                position,
                0.735,
                f"median={median_ari[k]:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        ax_a.set_xticks([0, 1], ["k=3", "k=4"])
        ax_a.set_ylabel("ARI versus primary solution")
        ax_a.set_ylim(0.72, 1.012)
        ax_a.set_title("Seed stability")
        style_axis(ax_a, grid_axis="y")
        panel_label(ax_a, "(a)")

        for k in (3, 4):
            values = np.sort(
                events.loc[
                    events["k"] == k, "max_assignment_probability"
                ].to_numpy(dtype=float)
            )
            ecdf = np.arange(1, len(values) + 1) / len(values)
            ax_b.plot(
                values,
                ecdf,
                color=K_NEUTRAL_COLORS[k],
                linestyle=K_LINESTYLES[k],
                marker=None,
                linewidth=1.45,
                label=f"k={k}",
            )
            ax_b.axvline(
                observed_stable_min[k],
                color=K_NEUTRAL_COLORS[k],
                linestyle=(0, (2, 2)),
                linewidth=0.8,
                alpha=0.8,
            )
        ax_b.text(
            0.807,
            0.60,
            "observed stable-core\nminima: 0.8022 / 0.8051",
            fontsize=6.2,
            color="#4D4D4D",
            ha="left",
            va="center",
        )
        ax_b.set_xlim(0.0, 1.005)
        ax_b.set_ylim(0, 1.01)
        ax_b.set_xlabel("Maximum assignment probability")
        ax_b.set_ylabel("Cumulative fraction")
        ax_b.set_title("Event-level consistency")
        ax_b.legend(frameon=False, loc="upper left")
        style_axis(ax_b, grid_axis="both")
        panel_label(ax_b, "(b)")

        y_base = np.arange(len(STATUS_ORDER))
        offsets = {3: -0.12, 4: 0.12}
        for k in (3, 4):
            for status_index, status in enumerate(STATUS_ORDER):
                count = expected_counts[k][status]
                ax_c.scatter(
                    count,
                    status_index + offsets[k],
                    s=30,
                    marker=K_MARKERS[k],
                    facecolor=STABILITY_COLORS[status],
                    edgecolor="#202020",
                    linewidth=0.55,
                    label=f"k={k}" if status_index == 0 else None,
                    zorder=3,
                )
                ax_c.annotate(
                    str(count),
                    (count, status_index + offsets[k]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=6.5,
                )
        ax_c.set_xscale("log")
        ax_c.set_xlim(4.5, 1000)
        ax_c.set_yticks(y_base, [STATUS_DISPLAY[item] for item in STATUS_ORDER])
        ax_c.invert_yaxis()
        ax_c.set_xlabel("Event count (log scale)")
        ax_c.set_title("Event stability categories")
        ax_c.legend(frameon=False, loc="lower right")
        style_axis(ax_c, grid_axis="x")
        panel_label(ax_c, "(c)")

        save_figure_atomic(fig, output_path, dpi=PROTOTYPE_DPI)
        plt.close(fig)

    caption_en = """# Fig. 4 caption (English)

Stability of the k=3 primary solution and k=4 structural sensitivity
within the KMeans framework. (a) Distribution of adjusted Rand index (ARI)
relative to the primary solution across 100 random-seed runs; median ARI is
0.978 for k=3 and 0.974 for k=4. (b) Empirical cumulative distributions of the
event-level maximum assignment probability. Dashed lines mark the observed
minima among events in the stable-core category
(approximately 0.8022 and 0.8051); they are not universal cutoffs and are not
used here to reclassify events. (c) Counts of stable-core, moderate, and
boundary events on a logarithmic count axis: 775/8/7 for k=3 and 763/21/6 for
k=4. Stable core denotes consistently assigned events, moderate denotes
intermediate assignment stability, and boundary denotes transition cases
rather than errors.
"""
    caption_zh = """# Fig. 4 图注（中文）

KMeans框架下k=3主要方案和k=4结构敏感性的稳定性。（a）100次随机种子运行
相对于主要方案的调整兰德指数（ARI）分布；k=3和k=4的ARI中位数分别为
0.978和0.974。（b）事件级最大归属概率的经验累积分布。虚线表示稳定核心
类别中事件的观测最小值（约0.8022和0.8051）；它们不是通用截点，本图也未
据此重新分类事件。（c）稳定核心、中等稳定和边界事件的计数，计数轴采用
对数尺度：k=3为775/8/7，k=4为763/21/6。稳定核心表示归属一致性较高，
中等稳定表示介于两者之间，边界表示过渡情形而非错误事件。
"""
    prohibited = scan_forbidden_language(caption_en + caption_zh)
    if prohibited:
        raise AssertionError(f"Fig.4 caption contains prohibited text: {prohibited}")
    write_text_atomic(CAPTION_DIR / "Fig4_caption_en.md", caption_en)
    write_text_atomic(CAPTION_DIR / "Fig4_caption_zh.md", caption_zh)

    metadata = inspect_png(output_path)
    write_json_atomic(QC_DIR / "Fig4_file_metadata.json", metadata)
    write_qc_report(
        QC_DIR / "Fig4_qc_report.md",
        figure_id="Fig.4",
        status="PASS_WITH_NONBLOCKING_NOTES",
        data_checks=[
            "Frozen seed, seed-summary, event-level, and event-summary files are used without changing statuses.",
            "Each of k=3 and k=4 has 100 seed runs and 790 event-level records.",
            "No event is reclassified from an observed probability value.",
        ],
        number_checks=[
            "Median seed ARI reproduces 0.978 for k=3 and 0.974 for k=4 at three-decimal display precision.",
            "Frozen status counts reproduce 775/8/7 and 763/21/6 and each closes to 790.",
            "Observed stable-core minima reproduce approximately 0.8022 and 0.8051.",
        ],
        interpretation_checks=[
            "Boundary status is described as a transition category, not a data-quality judgment.",
            "The observed stable-core minima are not presented as universal rules.",
        ],
        visual_checks=[
            f"Prototype PNG is {metadata['width_px']}×{metadata['height_px']} px in RGB mode.",
            "k=3 and k=4 use neutral colors and different markers or line styles.",
            "The count panel declares its logarithmic axis and labels every count.",
        ],
        nonblocking_notes=[
            "Final physical-size typography, vector fonts, and publisher-specific rules remain for Gate 4.",
        ],
    )

    write_text_atomic(
        LOG_DIR / "Fig4_build.log",
        f"""FIGURE=Fig4
STATUS=PASS_WITH_NONBLOCKING_NOTES
SOURCE_SEEDS={project_path(SEED_SOURCE)}
SOURCE_SEED_SUMMARY={project_path(SEED_SUMMARY_SOURCE)}
SOURCE_EVENTS={project_path(EVENT_SOURCE)}
SOURCE_EVENT_SUMMARY={project_path(EVENT_SUMMARY_SOURCE)}
PLOTTING_DATA={data_path}
OUTPUT={output_path}
TRANSFORMATION=subset frozen records to k=3/k=4; ECDF sorting; exact count lookup
JITTER=deterministic stable seeds from base {SCATTER_BASE_SEED}
STATUS_RECLASSIFICATION=NONE
SCIENTIFIC_REANALYSIS=FALSE
""",
    )
    return {
        "figure_id": "Fig4",
        "status": "PASS_WITH_NONBLOCKING_NOTES",
        "output": str(output_path),
        "plotting_data": str(data_path),
        "script": str(SCRIPT_PATH),
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(build())
