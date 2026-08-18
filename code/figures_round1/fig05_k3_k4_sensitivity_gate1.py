"""Gate 1 prototype for Fig. 5: k=3/k=4 structural sensitivity."""

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


TRANSITION_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/28_k3_to_k4_transition_matrix_v3.csv"
)
K3_MAPPING_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "23_regime_interpretation_audit/01_k3_raw_to_formal_label_mapping.csv"
)
K4_MAPPING_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "23_regime_interpretation_audit/02_k4_label_identity_and_mapping.csv"
)
WARD_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/26_ward_results_v3.csv"
)
GMM_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/27_gmm_results_v3.csv"
)
K3_ORDER = ("C0", "C1", "C2")
K4_ORDER = ("K4_C0", "K4_C1", "K4_C2", "K4_C3")


def _annotate_matrix(ax, matrix: np.ndarray, *, kind: str) -> None:
    maximum = float(np.nanmax(matrix))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            normalized = value / maximum if maximum else 0.0
            text = f"{int(value)}" if kind == "count" else f"{value:.0%}"
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.6,
                color="white" if normalized >= 0.56 else "#202020",
                fontweight="bold" if normalized >= 0.56 else "normal",
            )


def _format_heatmap_axis(ax) -> None:
    ax.set_xticks(range(len(K4_ORDER)), [item.replace("_", " ") for item in K4_ORDER])
    ax.set_yticks(range(len(K3_ORDER)), K3_ORDER)
    ax.set_xlabel("k=4 cluster")
    ax.set_ylabel("k=3 formal regime")
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#606060")


def build() -> dict[str, object]:
    ensure_output_dirs()
    transition = read_csv_checked(
        TRANSITION_SOURCE, required_columns=("k3_cluster_raw", "0", "1", "2", "3")
    )
    k3_mapping = read_csv_checked(
        K3_MAPPING_SOURCE,
        required_columns=("raw_kmeans_label", "formal_label", "event_count"),
    )
    k4_mapping = read_csv_checked(
        K4_MAPPING_SOURCE,
        required_columns=("raw_kmeans_label", "formal_label", "event_count"),
    )
    ward = read_csv_checked(
        WARD_SOURCE, required_columns=("k", "ari_vs_kmeans")
    )
    gmm = read_csv_checked(
        GMM_SOURCE, required_columns=("k", "ari_vs_kmeans")
    )

    k3_map = dict(
        zip(
            k3_mapping["raw_kmeans_label"].astype(int),
            k3_mapping["formal_label"].astype(str),
        )
    )
    k4_map = dict(
        zip(
            k4_mapping["raw_kmeans_label"].astype(int),
            k4_mapping["formal_label"].astype(str),
        )
    )
    count_frame = transition.set_index("k3_cluster_raw")[["0", "1", "2", "3"]]
    count_frame.index = [k3_map[int(value)] for value in count_frame.index]
    count_frame.columns = [k4_map[int(value)] for value in count_frame.columns]
    count_frame = count_frame.loc[list(K3_ORDER), list(K4_ORDER)].astype(int)
    row_proportion = count_frame.div(count_frame.sum(axis=1), axis=0)

    assert_exact("k=3 row totals", count_frame.sum(axis=1).tolist(), [131, 307, 352])
    assert_exact("k=4 column totals", count_frame.sum(axis=0).tolist(), [125, 206, 249, 210])
    assert_exact("transition closure", int(count_frame.to_numpy().sum()), 790)
    assert_exact(
        "mapping k=3 counts",
        k3_mapping.set_index("formal_label").loc[list(K3_ORDER), "event_count"].astype(int).tolist(),
        [131, 307, 352],
    )
    assert_exact(
        "mapping k=4 counts",
        k4_mapping.set_index("formal_label").loc[list(K4_ORDER), "event_count"].astype(int).tolist(),
        [125, 206, 249, 210],
    )
    assert_close(
        "row proportion closure",
        float(np.max(np.abs(row_proportion.sum(axis=1).to_numpy() - 1.0))),
        0.0,
        1e-12,
    )
    assert_exact("Ward k sequence", ward["k"].astype(int).tolist(), [2, 3, 4, 5, 6])
    assert_exact("GMM k sequence", gmm["k"].astype(int).tolist(), [2, 3, 4, 5, 6])

    source_transition = relative_source(project_path(TRANSITION_SOURCE))
    source_ward = relative_source(project_path(WARD_SOURCE))
    source_gmm = relative_source(project_path(GMM_SOURCE))
    records: list[dict[str, object]] = []
    for statistic, matrix, unit in (
        ("count", count_frame, "events"),
        ("row proportion", row_proportion, "proportion"),
    ):
        for k3_label in K3_ORDER:
            for k4_label in K4_ORDER:
                records.append(
                    {
                        "panel": "a" if statistic == "count" else "b",
                        "record_type": "k3_k4_correspondence",
                        "k3_regime": k3_label,
                        "k4_cluster": k4_label,
                        "algorithm": "",
                        "k": "",
                        "value": (
                            int(matrix.loc[k3_label, k4_label])
                            if statistic == "count"
                            else float(matrix.loc[k3_label, k4_label])
                        ),
                        "unit": unit,
                        "statistic": statistic,
                        "sample_size": 790,
                        "source_file": source_transition,
                        "transformation_note": (
                            "raw labels reordered by accepted mapping; no count change"
                            if statistic == "count"
                            else "count divided by its k=3 row total"
                        ),
                    }
                )
    for algorithm, frame, source in (
        ("Ward", ward, source_ward),
        ("Gaussian mixture", gmm, source_gmm),
    ):
        for _, row in frame.iterrows():
            records.append(
                {
                    "panel": "c",
                    "record_type": "alternative_algorithm_ari",
                    "k3_regime": "",
                    "k4_cluster": "",
                    "algorithm": algorithm,
                    "k": int(row["k"]),
                    "value": float(row["ari_vs_kmeans"]),
                    "unit": "ARI",
                    "statistic": "ARI versus KMeans at the same k",
                    "sample_size": 790,
                    "source_file": source,
                    "transformation_note": "frozen algorithm-comparison value",
                }
            )
    plotting = pd.DataFrame(records)
    assert_exact("Fig.5 plotting rows", len(plotting), 34)
    data_path = PLOTTING_DATA_DIR / "Fig5_plotting_data.csv"
    output_path = PNG_DIR / "Fig5_k3_k4_sensitivity_GATE1_prototype.png"
    write_csv_atomic(data_path, plotting)

    with manuscript_style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=mm_to_inches(*FIGURE_SIZES_MM["Fig5"]),
            layout="constrained",
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]},
        )
        ax_a, ax_b, ax_c = axes

        count_values = count_frame.to_numpy(dtype=float)
        image_a = ax_a.imshow(
            count_values,
            cmap="Blues",
            vmin=0,
            vmax=float(np.max(count_values)),
            aspect="auto",
            interpolation="none",
        )
        _annotate_matrix(ax_a, count_values, kind="count")
        _format_heatmap_axis(ax_a)
        ax_a.set_title("k=3–k=4 structural\ncorrespondence (n=790)")
        colorbar_a = fig.colorbar(image_a, ax=ax_a, fraction=0.050, pad=0.03)
        colorbar_a.set_label("Count (events)")
        colorbar_a.outline.set_linewidth(0.55)
        panel_label(ax_a, "(a)")

        proportion_values = row_proportion.to_numpy(dtype=float)
        image_b = ax_b.imshow(
            proportion_values,
            cmap="Blues",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="none",
        )
        _annotate_matrix(ax_b, proportion_values, kind="proportion")
        _format_heatmap_axis(ax_b)
        ax_b.set_title("Row-normalized\ncorrespondence")
        colorbar_b = fig.colorbar(image_b, ax=ax_b, fraction=0.050, pad=0.03)
        colorbar_b.set_label("Row proportion")
        colorbar_b.set_ticks([0, 0.5, 1.0], labels=["0", "0.5", "1.0"])
        colorbar_b.outline.set_linewidth(0.55)
        panel_label(ax_b, "(b)")

        for algorithm, frame, k_value in (
            ("Ward", ward, 3),
            ("Gaussian mixture", gmm, 4),
        ):
            ax_c.plot(
                frame["k"].astype(int),
                frame["ari_vs_kmeans"].astype(float),
                label=algorithm,
                color=K_NEUTRAL_COLORS[k_value],
                marker=K_MARKERS[k_value],
                linestyle=K_LINESTYLES[k_value],
                linewidth=1.35,
                markersize=4.2,
            )
        ax_c.set_xticks([2, 3, 4, 5, 6])
        ax_c.set_xlim(1.8, 6.2)
        ax_c.set_ylim(0, 0.58)
        ax_c.set_xlabel("Number of clusters (k)")
        ax_c.set_ylabel("ARI versus KMeans")
        ax_c.set_title("Algorithm dependence")
        ax_c.legend(frameon=False, loc="lower right")
        style_axis(ax_c, grid_axis="both")
        panel_label(ax_c, "(c)")

        save_figure_atomic(fig, output_path, dpi=PROTOTYPE_DPI)
        plt.close(fig)

    caption_en = """# Fig. 5 caption (English)

Structural sensitivity of the 790-event classification. (a) Event-count
correspondence between the formal k=3 regimes and the four k=4 clusters; each
cell is an integer count and the matrix sums to n=790. (b) The same
correspondence normalized within each k=3 row, shown with a separate proportion
scale. The k=4 solution subdivides the k=3 memberships unevenly, which
documents structural sensitivity without implying a unique natural partition.
(c) Adjusted Rand index (ARI) of Ward and Gaussian-mixture solutions relative
to KMeans at the same k. These comparisons describe algorithm dependence and
are not used to change the formal labels.
"""
    caption_zh = """# Fig. 5 图注（中文）

790例分类结果对结构设定的敏感性。（a）正式k=3环境组与4个k=4簇之间的事件
计数对应；每个单元格均为整数计数，整张矩阵合计n=790。（b）同一对应关系按
k=3各行归一化，并使用独立的比例色标。k=4方案对k=3成员进行了不均匀细分，
反映结构敏感性，但不表示存在唯一的自然划分。（c）Ward和高斯混合方案与相同
k值下KMeans结果之间的调整兰德指数（ARI）。这些比较用于描述算法依赖性，
不改变正式标签。
"""
    prohibited = scan_forbidden_language(caption_en + caption_zh)
    if prohibited:
        raise AssertionError(f"Fig.5 caption contains prohibited text: {prohibited}")
    write_text_atomic(CAPTION_DIR / "Fig5_caption_en.md", caption_en)
    write_text_atomic(CAPTION_DIR / "Fig5_caption_zh.md", caption_zh)

    metadata = inspect_png(output_path)
    write_json_atomic(QC_DIR / "Fig5_file_metadata.json", metadata)
    write_qc_report(
        QC_DIR / "Fig5_qc_report.md",
        figure_id="Fig.5",
        status="PASS_WITH_NONBLOCKING_NOTES",
        data_checks=[
            "The accepted raw-to-formal mappings reorder labels without changing membership or counts.",
            "The frozen k=3 by k=4 transition matrix closes exactly to 790 events.",
            "Ward and Gaussian-mixture ARI values are read from frozen algorithm-comparison files.",
        ],
        number_checks=[
            "Formal k=3 row totals reproduce 131, 307, and 352.",
            "k=4 column totals reproduce 125, 206, 249, and 210.",
            "Every normalized k=3 row sums to one within floating-point precision.",
        ],
        interpretation_checks=[
            "Panels are described as structural sensitivity and algorithm dependence.",
            "No singular or privileged natural partition is inferred from the comparison.",
            "The alternative-algorithm results do not modify the formal cluster labels.",
        ],
        visual_checks=[
            f"Prototype PNG is {metadata['width_px']}×{metadata['height_px']} px in RGB mode.",
            "Count and row-proportion panels have separate titles, units, and color scales.",
            "The alluvial option was omitted because the compact heatmap already carries the correspondence.",
        ],
        nonblocking_notes=[
            "Final physical-size typography, vector fonts, and publisher-specific rules remain for Gate 4.",
        ],
    )

    write_text_atomic(
        LOG_DIR / "Fig5_build.log",
        f"""FIGURE=Fig5
STATUS=PASS_WITH_NONBLOCKING_NOTES
SOURCE_TRANSITION={project_path(TRANSITION_SOURCE)}
SOURCE_K3_MAPPING={project_path(K3_MAPPING_SOURCE)}
SOURCE_K4_MAPPING={project_path(K4_MAPPING_SOURCE)}
SOURCE_WARD={project_path(WARD_SOURCE)}
SOURCE_GMM={project_path(GMM_SOURCE)}
PLOTTING_DATA={data_path}
OUTPUT={output_path}
TRANSFORMATION=formal-label reordering; row normalization for panel b only
COUNT_PROPORTION_SCALES=SEPARATE
ALGORITHM_COMPARISON_TERM=algorithm dependence
SCIENTIFIC_REANALYSIS=FALSE
""",
    )
    return {
        "figure_id": "Fig5",
        "status": "PASS_WITH_NONBLOCKING_NOTES",
        "output": str(output_path),
        "plotting_data": str(data_path),
        "script": str(SCRIPT_PATH),
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(build())
