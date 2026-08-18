"""Gate 1 prototype for Fig. 3: raw-unit distributions and overlap."""

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
    PROTOTYPE_DPI,
    REGIME_COLORS,
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


ENVIRONMENT_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "02_environment_table/06_environment_table_new790_v3.csv"
)
LABEL_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/04_clustering/"
    "12_clustering_results_v3/30_labels_k3_regime_ids_v3.csv"
)
JITTER_BASE_SEED = 20260804

VARIABLES = (
    ("MLCAPE", "MLCAPE_Jkg", r"MLCAPE (J kg$^{-1}$)", "J kg−1"),
    ("MLLCL", "MLLCL_m", "MLLCL (m)", "m"),
    ("2-m dew point", "ERA5_d2m_K", "2-m dew point (K)", "K"),
    (
        "0–6-km shear",
        "SHR6_ms",
        r"0–6-km bulk shear (m s$^{-1}$)",
        "m s−1",
    ),
    (
        "0–1-km SRH",
        "SRH1_m2s2",
        r"Signed 0–1-km SRH (m$^2$ s$^{-2}$)",
        "m2 s−2",
    ),
)


def build() -> dict[str, object]:
    ensure_output_dirs()
    environment = read_csv_checked(
        ENVIRONMENT_SOURCE,
        required_columns=("event_id", *(item[1] for item in VARIABLES)),
        dtype={"event_id": str},
    )
    labels = read_csv_checked(
        LABEL_SOURCE,
        required_columns=("event_id", "regime_id"),
        dtype={"event_id": str},
    )[["event_id", "regime_id"]]
    assert_exact("environment rows", len(environment), 790)
    assert_exact("environment unique event_id", environment["event_id"].nunique(), 790)
    assert_exact("label rows", len(labels), 790)
    assert_exact("label unique event_id", labels["event_id"].nunique(), 790)

    merged = environment[["event_id", *(item[1] for item in VARIABLES)]].merge(
        labels, on="event_id", how="inner", validate="one_to_one"
    )
    observed_counts = (
        merged["regime_id"].value_counts().reindex(REGIME_ORDER).astype(int).tolist()
    )
    assert_exact("k=3 counts", observed_counts, [131, 307, 352])
    if merged[[item[1] for item in VARIABLES]].isna().any().any():
        raise ValueError("Fig.3 raw clustering variables contain missing values.")

    sample_size_by_group = (
        merged.groupby("regime_id", observed=False)["event_id"].size().to_dict()
    )
    plotting_parts: list[pd.DataFrame] = []
    for variable, column, _, unit in VARIABLES:
        part = merged[["event_id", "regime_id", column]].rename(
            columns={"regime_id": "group_label", column: "value"}
        )
        part["variable"] = variable
        part["unit"] = unit
        part["statistic"] = "event observation"
        part["sample_size"] = part["group_label"].map(sample_size_by_group)
        part["source_file"] = relative_source(project_path(ENVIRONMENT_SOURCE))
        part["label_source"] = relative_source(project_path(LABEL_SOURCE))
        part["transformation_note"] = (
            "raw signed value; no clipping or axis transformation"
            if variable == "0–1-km SRH"
            else "raw value; no axis transformation"
        )
        plotting_parts.append(part)
    plotting = pd.concat(plotting_parts, ignore_index=True)
    plotting["group_label"] = pd.Categorical(
        plotting["group_label"], categories=REGIME_ORDER, ordered=True
    )
    assert_exact("plotting snapshot rows", len(plotting), 790 * 5)
    if not np.isfinite(plotting["value"]).all():
        raise ValueError("Fig.3 plotting snapshot contains non-finite values.")
    assert_exact(
        "negative signed SRH retained",
        int((plotting.loc[plotting["variable"] == "0–1-km SRH", "value"] < 0).sum()),
        int((merged["SRH1_m2s2"] < 0).sum()),
    )

    data_path = PLOTTING_DATA_DIR / "Fig3_plotting_data.csv"
    output_path = PNG_DIR / "Fig3_raw_distributions_GATE1_prototype.png"
    write_csv_atomic(data_path, plotting)

    with manuscript_style():
        mosaic = [
            ["a", "a", "b", "b", "c", "c"],
            [".", "d", "d", "e", "e", "."],
        ]
        fig, axes = plt.subplot_mosaic(
            mosaic,
            figsize=mm_to_inches(*FIGURE_SIZES_MM["Fig3"]),
            layout="constrained",
        )
        for panel_index, (variable, _, ylabel, _) in enumerate(VARIABLES):
            key = "abcde"[panel_index]
            ax = axes[key]
            group_values = [
                plotting.loc[
                    (plotting["variable"] == variable)
                    & (plotting["group_label"] == regime),
                    "value",
                ].to_numpy(dtype=float)
                for regime in REGIME_ORDER
            ]
            box = ax.boxplot(
                group_values,
                positions=np.arange(3),
                widths=0.42,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                medianprops={"color": "#202020", "linewidth": 1.0},
                whiskerprops={"color": "#555555", "linewidth": 0.75},
                capprops={"color": "#555555", "linewidth": 0.75},
                boxprops={"linewidth": 0.9},
                zorder=2,
            )
            for patch, regime in zip(box["boxes"], REGIME_ORDER):
                patch.set_facecolor(REGIME_COLORS[regime])
                patch.set_alpha(0.18)
                patch.set_edgecolor(REGIME_COLORS[regime])

            for x_position, (regime, values) in enumerate(
                zip(REGIME_ORDER, group_values)
            ):
                rng = np.random.default_rng(
                    stable_seed(JITTER_BASE_SEED, f"{variable}:{regime}")
                )
                jitter = rng.uniform(-0.16, 0.16, size=len(values))
                ax.scatter(
                    x_position + jitter,
                    values,
                    s=5.0,
                    marker=REGIME_MARKERS[regime],
                    color=REGIME_COLORS[regime],
                    alpha=0.18,
                    edgecolors="none",
                    rasterized=True,
                    zorder=1,
                )

            ax.set_xticks(
                np.arange(3),
                [
                    f"{regime}\nn={sample_size_by_group[regime]}"
                    for regime in REGIME_ORDER
                ],
            )
            ax.set_ylabel(ylabel)
            ax.set_title(variable, pad=3.0)
            ax.margins(x=0.14, y=0.06)
            style_axis(ax, grid_axis="y")
            panel_label(ax, f"({key})")
        save_figure_atomic(fig, output_path, dpi=PROTOTYPE_DPI)
        plt.close(fig)

    caption_en = """# Fig. 3 caption (English)

Raw-unit distributions of the five pre-specified clustering variables for C0,
C1, and C2 within the formal 790-event confirmed-tornado sample. Boxes show the
median and interquartile range, with whiskers extending to 1.5 times the
interquartile range. Every event is retained as a transparent, deterministically
jittered point; boxplot fliers are hidden only because the complete observations
are already displayed. Panels show (a) MLCAPE, (b) MLLCL, (c) ERA5 2-m dew
point, (d) 0–6-km bulk wind shear, and (e) signed 0–1-km storm-relative
helicity. Negative helicity values are retained. No axis transform, truncation,
outlier deletion, or display downsampling is applied. Group sizes are
C0=131, C1=307, and C2=352.
"""
    caption_zh = """# Fig. 3 图注（中文）

正式790例已确认龙卷样本中C0、C1和C2五个预先固定聚类变量的原始单位分布。
箱体表示中位数和四分位距，须线延伸至1.5倍四分位距。全部事件均以透明、
确定性抖动散点保留；箱线图自身不重复绘制离群点，是因为完整观测值已经显示。
各面板依次为：（a）MLCAPE，（b）MLLCL，（c）ERA5 2 m露点，
（d）0–6 km体积风切变和（e）带符号的0–1 km风暴相对螺旋度。
负SRH值被完整保留。图中未采用坐标变换、截断、异常值删除或展示下采样。
组样本量为C0=131、C1=307和C2=352。
"""
    prohibited = scan_forbidden_language(caption_en + caption_zh)
    if prohibited:
        raise AssertionError(f"Fig.3 caption contains prohibited text: {prohibited}")
    write_text_atomic(CAPTION_DIR / "Fig3_caption_en.md", caption_en)
    write_text_atomic(CAPTION_DIR / "Fig3_caption_zh.md", caption_zh)

    metadata = inspect_png(output_path)
    write_json_atomic(QC_DIR / "Fig3_file_metadata.json", metadata)
    write_qc_report(
        QC_DIR / "Fig3_qc_report.md",
        figure_id="Fig.3",
        status="PASS_WITH_NONBLOCKING_NOTES",
        data_checks=[
            "The formal environment table and formal k=3 regime labels each contain 790 unique event identifiers.",
            "One-to-one merging preserves C0/C1/C2 counts of 131/307/352.",
            "All five raw clustering variables are complete; signed SRH is not clipped.",
        ],
        number_checks=[
            "The plotting snapshot contains exactly 3,950 event-variable rows.",
            "All event observations are displayed; no statistical outlier is removed.",
        ],
        interpretation_checks=[
            "The panels show within-sample distributional differences and overlap.",
            "No group ranking, causal statement, or discrete-class claim is encoded.",
        ],
        visual_checks=[
            f"Prototype PNG is {metadata['width_px']}×{metadata['height_px']} px in RGB mode.",
            "Boxes and all observations share fixed group colors; group position and marker shape provide redundant identification.",
            "Each variable retains its own raw-unit linear scale.",
        ],
        nonblocking_notes=[
            "Dense points are raster content in the prototype; final vector containers will retain vector text and axes with selectively rasterized observations.",
            "Target-journal requirements and final-size typography remain pending Gate 4.",
        ],
    )

    write_text_atomic(
        LOG_DIR / "Fig3_build.log",
        f"""FIGURE=Fig3
STATUS=PASS_WITH_NONBLOCKING_NOTES
SOURCE_ENVIRONMENT={project_path(ENVIRONMENT_SOURCE)}
SOURCE_LABELS={project_path(LABEL_SOURCE)}
PLOTTING_DATA={data_path}
OUTPUT={output_path}
TRANSFORMATION=wide event table to 3950-row long plotting snapshot
JITTER=uniform[-0.16,0.16], deterministic stable seeds from base {JITTER_BASE_SEED}
DOWNSAMPLING=NONE
OUTLIER_DELETION=NONE
AXIS_TRANSFORMATION=NONE
SCIENTIFIC_REANALYSIS=FALSE
""",
    )
    return {
        "figure_id": "Fig3",
        "status": "PASS_WITH_NONBLOCKING_NOTES",
        "output": str(output_path),
        "plotting_data": str(data_path),
        "script": str(SCRIPT_PATH),
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(build())
