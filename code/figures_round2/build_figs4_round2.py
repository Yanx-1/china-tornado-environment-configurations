"""Build Fig. S4: k=4 standardized centers and raw-variable distributions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

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
    K4_COLORS,
    K4_LINESTYLES,
    K4_MARKERS,
    K4_ORDER,
    PROTOTYPE_DPI,
    manuscript_style,
    mm_to_inches,
    panel_label,
    style_axis,
)


ENV_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/02_environment_table/"
    "06_environment_table_new790_v3.csv"
)
K4_LABEL_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/08_primary_labels_k4_new790.csv"
)
K4_MAPPING_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/23_regime_interpretation_audit/"
    "02_k4_label_identity_and_mapping.csv"
)
K3_LABEL_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/30_labels_k3_regime_ids_v3.csv"
)
K3_CENTER_REFERENCE = (
    "paper_rebuild/17_core_sample_reopen_new790/06_figures_tables/"
    "21_figures_and_tables_v3/source_data/fig05_standardized_centers_source_v3.csv"
)

OUTPUT = SUPP_FIGURE_DIR / "FigS4_k4_structure_ROUND2_review.png"
SNAPSHOT = PLOTTING_DATA_DIR / "FigS4_plotting_snapshot.csv"
CAPTION_EN = CAPTION_DIR / "FigS4_caption_en.md"
CAPTION_ZH = CAPTION_DIR / "FigS4_caption_zh.md"
METADATA = QC_DIR / "FigS4_metadata.json"
QC_REPORT = QC_DIR / "FigS4_QC.md"
BUILD_LOG = LOG_DIR / "FigS4_build.log"

FEATURES = ("MLCAPE_Jkg", "MLLCL_m", "ERA5_d2m_K", "SHR6_ms", "SRH1_m2s2")
CENTER_LABELS = (
    "MLCAPE",
    "log(1+MLLCL)",
    "2-m dewpoint",
    "0–6-km shear",
    "0–1-km SRH",
)
RAW_LABELS = {
    "MLCAPE_Jkg": "MLCAPE (J kg$^{-1}$)",
    "MLLCL_m": "MLLCL (m)",
    "ERA5_d2m_K": "2-m dewpoint (K)",
    "SHR6_ms": "0–6-km shear (m s$^{-1}$)",
    "SRH1_m2s2": "0–1-km SRH (m$^2$ s$^{-2}$)",
}
K4_COUNTS = {"K4_C0": 125, "K4_C1": 206, "K4_C2": 249, "K4_C3": 210}


def _load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    env = read_csv_checked(
        ENV_SOURCE,
        required_columns=("event_id", *FEATURES),
    )[["event_id", *FEATURES]].copy()
    labels = read_csv_checked(
        K4_LABEL_SOURCE,
        required_columns=("event_id", "cluster_raw"),
    )[["event_id", "cluster_raw"]]
    mapping = read_csv_checked(
        K4_MAPPING_SOURCE,
        required_columns=("raw_kmeans_label", "formal_label", "event_count"),
    )
    assert_exact("FigS4 environment rows", len(env), 790)
    assert_exact("FigS4 environment events", env["event_id"].nunique(), 790)
    assert_exact("FigS4 k4 label rows", len(labels), 790)
    assert_exact("FigS4 feature missing values", int(env[list(FEATURES)].isna().sum().sum()), 0)
    assert_exact(
        "FigS4 mapping labels",
        tuple(mapping.sort_values("raw_kmeans_label")["formal_label"]),
        K4_ORDER,
    )

    joined = env.merge(labels, on="event_id", how="inner", validate="one_to_one")
    map_dict = mapping.set_index("raw_kmeans_label")["formal_label"].to_dict()
    joined["k4_label"] = joined["cluster_raw"].map(map_dict)
    assert_exact("FigS4 unmapped labels", int(joined["k4_label"].isna().sum()), 0)
    assert_exact(
        "FigS4 k4 counts",
        joined["k4_label"].value_counts().sort_index().to_dict(),
        K4_COUNTS,
    )

    transformed = joined[list(FEATURES)].copy()
    transformed["MLLCL_m"] = np.log1p(transformed["MLLCL_m"])
    standardized = pd.DataFrame(
        StandardScaler().fit_transform(transformed),
        columns=FEATURES,
        index=joined.index,
    )
    standardized["event_id"] = joined["event_id"]
    standardized["k4_label"] = joined["k4_label"]
    centers = (
        standardized.groupby("k4_label", observed=True)[list(FEATURES)]
        .mean()
        .reindex(K4_ORDER)
        .reset_index()
    )

    # Reproduce the already accepted k=3 center table to prove that the exact
    # established transformation and scaling path was used.
    k3_labels = read_csv_checked(
        K3_LABEL_SOURCE,
        required_columns=("event_id", "k3_cluster_raw"),
    )[["event_id", "k3_cluster_raw"]]
    k3_reference = read_csv_checked(
        K3_CENTER_REFERENCE,
        required_columns=("k3_cluster_raw",),
    ).set_index("k3_cluster_raw")
    k3_check = standardized.drop(columns="k4_label").merge(
        k3_labels, on="event_id", how="inner", validate="one_to_one"
    )
    k3_observed = (
        k3_check.groupby("k3_cluster_raw", observed=True)[list(FEATURES)]
        .mean()
        .sort_index()
    )
    reference_columns = [f"{feature}_z_center" for feature in FEATURES]
    max_difference = float(
        np.max(
            np.abs(
                k3_observed.to_numpy()
                - k3_reference[reference_columns].sort_index().to_numpy()
            )
        )
    )
    assert_close("FigS4 preprocessing k3 center reproduction", max_difference, 0.0, 1e-12)
    centers.attrs["k3_reproduction_max_abs_difference"] = max_difference
    return joined, centers


def _write_snapshot(joined: pd.DataFrame, centers: pd.DataFrame) -> None:
    raw_snapshot = joined[
        ["event_id", "cluster_raw", "k4_label", *FEATURES]
    ].copy()
    raw_snapshot.insert(0, "record_type", "event_raw_values")
    center_snapshot = centers.rename(
        columns={feature: f"{feature}_z_center" for feature in FEATURES}
    ).copy()
    center_snapshot.insert(0, "record_type", "k4_standardized_center")
    write_csv_atomic(
        SNAPSHOT,
        pd.concat([raw_snapshot, center_snapshot], ignore_index=True, sort=False),
    )


def _draw(joined: pd.DataFrame, centers: pd.DataFrame) -> None:
    width, height = FIGURE_SIZES_MM["FigS4"]
    with manuscript_style():
        fig, axes = plt.subplots(
            2,
            3,
            figsize=mm_to_inches(width, height),
            layout="constrained",
        )
        axes_flat = axes.ravel()
        ax_center = axes_flat[0]
        x_positions = np.arange(len(FEATURES))
        center_indexed = centers.set_index("k4_label")
        for label in K4_ORDER:
            ax_center.plot(
                x_positions,
                center_indexed.loc[label, list(FEATURES)].to_numpy(dtype=float),
                color=K4_COLORS[label],
                marker=K4_MARKERS[label],
                linestyle=K4_LINESTYLES[label],
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=4.2,
            )
        ax_center.axhline(0, color="#666666", linewidth=0.75, linestyle=":")
        ax_center.set_xticks(x_positions, CENTER_LABELS, rotation=27, ha="right")
        ax_center.set_ylabel("Standardized center")
        ax_center.set_title("k=4 standardized centers", pad=4)
        style_axis(ax_center, grid_axis="y")
        panel_label(ax_center, "(a)", x=-0.20, y=1.035)

        rng = np.random.default_rng(4)
        for panel_index, (ax, feature) in enumerate(
            zip(axes_flat[1:], FEATURES), start=1
        ):
            distributions = [
                joined.loc[joined["k4_label"] == label, feature].to_numpy()
                for label in K4_ORDER
            ]
            box = ax.boxplot(
                distributions,
                positions=np.arange(4),
                widths=0.50,
                whis=(10, 90),
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#202020", "linewidth": 1.0},
                whiskerprops={"color": "#666666", "linewidth": 0.65},
                capprops={"color": "#666666", "linewidth": 0.65},
            )
            for patch, label in zip(box["boxes"], K4_ORDER):
                patch.set_facecolor(K4_COLORS[label])
                patch.set_edgecolor(K4_COLORS[label])
                patch.set_alpha(0.27)
                patch.set_linewidth(0.8)
            for position, label, values in zip(
                np.arange(4), K4_ORDER, distributions
            ):
                jitter = rng.uniform(-0.17, 0.17, len(values))
                ax.scatter(
                    position + jitter,
                    values,
                    s=4.5,
                    color=K4_COLORS[label],
                    alpha=0.16,
                    linewidths=0,
                    rasterized=True,
                    zorder=1,
                )
            ax.set_xticks(np.arange(4), ["C0", "C1", "C2", "C3"])
            ax.set_xlabel("k=4 cluster")
            ax.set_ylabel(RAW_LABELS[feature])
            ax.set_title(f"Raw {RAW_LABELS[feature]}", pad=4)
            style_axis(ax, grid_axis="y")
            panel_label(ax, f"({chr(97 + panel_index)})", x=-0.20, y=1.035)

        legend_handles = [
            Line2D(
                [],
                [],
                color=K4_COLORS[label],
                marker=K4_MARKERS[label],
                linestyle=K4_LINESTYLES[label],
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=f"{label.replace('_', ' ')} (n={K4_COUNTS[label]})",
            )
            for label in K4_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            loc="outside upper center",
            ncol=4,
            frameon=False,
            columnspacing=1.4,
            handlelength=2.2,
        )
        save_figure_atomic(fig, OUTPUT, dpi=PROTOTYPE_DPI)
        plt.close(fig)


def _write_caption_and_qc(centers: pd.DataFrame) -> None:
    caption_en = """# Fig. S4 caption (English)

**Figure S4. Supplementary structure of the k=4 sensitivity analysis.** (a) Standardized k=4 centers for the five clustering variables. MLLCL was transformed as log(1+MLLCL) before all five variables were standardized across the 790-event sample; each line is the mean standardized value within an accepted k=4 label. (b–f) Raw distributions of MLCAPE, MLLCL, 2-m dewpoint, 0–6-km bulk shear, and 0–1-km storm-relative helicity. Boxes show the interquartile range, center lines show medians, whiskers span the 10th–90th percentiles, and translucent points show all events. The k=4 groups retain their formal numerical labels only; no additional physical group names are assigned.
"""
    caption_zh = """# 图S4图注（中文）

**图S4. k=4敏感性分析的补充结构。**（a）五个聚类变量的k=4标准化中心。MLLCL先进行log(1+MLLCL)变换，随后五个变量在790个事件样本中统一标准化；每条线表示相应k=4正式标签内标准化变量的平均值。（b–f）依次展示MLCAPE、MLLCL、2 m露点、0–6 km整层风切变和0–1 km风暴相对螺旋度的原始分布。箱体表示四分位距，中线表示中位数，须线覆盖第10–90百分位，半透明点表示全部事件。k=4组别仅保留正式数字标签，不赋予额外物理名称。
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
        figure_id="Fig. S4",
        status_label="SUPPLEMENT_CANDIDATE",
        scientific_checks=[
            "The formal environment table and accepted k=4 labels both contain exactly 790 unique events.",
            "k=4 counts close exactly at K4_C0=125, K4_C1=206, K4_C2=249, and K4_C3=210.",
            "No clustering or label optimization was rerun.",
            "The established transformation and standardization path reproduces the accepted k=3 center table with maximum absolute difference "
            f"{centers.attrs['k3_reproduction_max_abs_difference']:.3g}.",
            "All raw events are retained in every distribution panel; no extreme values were removed.",
        ],
        layout_checks=[
            "The prescribed 2×3 layout contains one standardized-center panel and five raw-distribution panels.",
            "A single four-group legend carries sample sizes; sample sizes are not repeated in each panel.",
            "The formal k=4 group labels, palette, and line/marker encodings are consistent across all panels.",
            "The file is a 200 dpi RGB review image with no alpha channel.",
            "All publication-visible text and both captions passed the internal-term search.",
        ],
        interpretation_checks=[
            "The k=4 result is identified as a sensitivity analysis and is not described as a unique natural classification.",
            "Only formal numerical labels K4_C0 through K4_C3 are used; no physical names were inferred.",
            "Panel (a) distinguishes the transformed standardized input from the raw quantities shown in panels (b–f).",
        ],
        deviations=[],
        review_notes=[
            "The standardized k=4 centers are plotting summaries of accepted labels under the exactly reproduced preprocessing path, not a new clustering result.",
            "No publisher-specific font or line-width compliance is claimed because a target journal has not yet been designated.",
        ],
    )


def build() -> dict:
    ensure_output_dirs()
    joined, centers = _load_and_prepare()
    _write_snapshot(joined, centers)
    _draw(joined, centers)
    _write_caption_and_qc(centers)

    image_info = inspect_png(OUTPUT)
    assert_exact("FigS4 mode", image_info["mode"], "RGB")
    assert_exact("FigS4 alpha", image_info["alpha_present"], False)
    metadata = {
        "figure_id": "FigS4",
        "status_label": "SUPPLEMENT_CANDIDATE",
        "output_stage": "200_DPI_RGB_REVIEW_PROTOTYPE",
        "figure_file": relative_source(OUTPUT),
        "plotting_snapshot": relative_source(SNAPSHOT),
        "caption_en": relative_source(CAPTION_EN),
        "caption_zh": relative_source(CAPTION_ZH),
        "qc_report": relative_source(QC_REPORT),
        "script": relative_source(Path(__file__)),
        "inputs": [
            {"path": ENV_SOURCE, "sha256": sha256_file(PROJECT_ROOT / ENV_SOURCE)},
            {
                "path": K4_LABEL_SOURCE,
                "sha256": sha256_file(PROJECT_ROOT / K4_LABEL_SOURCE),
            },
            {
                "path": K4_MAPPING_SOURCE,
                "sha256": sha256_file(PROJECT_ROOT / K4_MAPPING_SOURCE),
            },
            {
                "path": K3_LABEL_SOURCE,
                "sha256": sha256_file(PROJECT_ROOT / K3_LABEL_SOURCE),
            },
            {
                "path": K3_CENTER_REFERENCE,
                "sha256": sha256_file(PROJECT_ROOT / K3_CENTER_REFERENCE),
            },
        ],
        "render": image_info,
        "scientific_result_changed": False,
        "clustering_rerun": False,
        "k3_preprocessing_reproduction_max_abs_difference": centers.attrs[
            "k3_reproduction_max_abs_difference"
        ],
    }
    write_json_atomic(METADATA, metadata)
    write_text_atomic(
        BUILD_LOG,
        "\n".join(
            [
                "FIGURE_ID=FigS4",
                "STATUS=BUILT_AND_QC_COMPLETE",
                f"OUTPUT={relative_source(OUTPUT)}",
                f"SHA256={image_info['sha256']}",
                "K4_COUNTS=K4_C0:125,K4_C1:206,K4_C2:249,K4_C3:210",
                "CLUSTERING_RERUN=FALSE",
                "PHYSICAL_NAMES_ASSIGNED=FALSE",
                "LAYOUT=2X3",
                "",
            ]
        ),
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
