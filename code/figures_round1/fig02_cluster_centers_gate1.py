"""Gate 1 prototype for Fig. 2: standardized k=3 cluster centers."""

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
    assert_exact,
    inspect_png,
    scan_forbidden_language,
    write_qc_report,
)


CENTERS_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/06_figures_tables/"
    "21_figures_and_tables_v3/source_data/"
    "fig05_standardized_centers_source_v3.csv"
)
MAPPING_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "23_regime_interpretation_audit/01_k3_raw_to_formal_label_mapping.csv"
)

VARIABLES = (
    (
        "MLCAPE",
        "MLCAPE_Jkg_z_center",
        "identity → StandardScaler on formal 790",
    ),
    (
        "MLLCL",
        "MLLCL_m_z_center",
        "log1p → StandardScaler on formal 790",
    ),
    (
        "2-m dew point",
        "ERA5_d2m_K_z_center",
        "identity → StandardScaler on formal 790",
    ),
    (
        "0–6-km shear",
        "SHR6_ms_z_center",
        "identity → StandardScaler on formal 790",
    ),
    (
        "0–1-km SRH",
        "SRH1_m2s2_z_center",
        "signed identity → StandardScaler on formal 790",
    ),
)


def build() -> dict[str, object]:
    ensure_output_dirs()
    centers = read_csv_checked(
        CENTERS_SOURCE,
        required_columns=(
            "k3_cluster_raw",
            *(column for _, column, _ in VARIABLES),
        ),
    )
    mapping = read_csv_checked(
        MAPPING_SOURCE,
        required_columns=(
            "raw_kmeans_label",
            "formal_label",
            "event_count",
        ),
    )
    assert_exact("center row count", len(centers), 3)
    assert_exact(
        "mapping event counts",
        mapping.set_index("formal_label")["event_count"]
        .astype(int)
        .reindex(REGIME_ORDER)
        .tolist(),
        [131, 307, 352],
    )
    assert_exact("mapping count closure", int(mapping["event_count"].sum()), 790)

    mapped = centers.merge(
        mapping[
            ["raw_kmeans_label", "formal_label", "event_count"]
        ].rename(columns={"raw_kmeans_label": "k3_cluster_raw"}),
        on="k3_cluster_raw",
        how="left",
        validate="one_to_one",
    )
    if mapped["formal_label"].isna().any():
        raise ValueError("One or more k=3 centers lack an accepted label mapping.")

    plotting_rows: list[dict[str, object]] = []
    for _, row in mapped.iterrows():
        for variable, column, transformation in VARIABLES:
            plotting_rows.append(
                {
                    "event_id": "",
                    "aggregation_level": "k3_formal_regime",
                    "group_label": row["formal_label"],
                    "variable": variable,
                    "value": float(row[column]),
                    "unit": "standard deviations",
                    "statistic": "standardized cluster center",
                    "sample_size": int(row["event_count"]),
                    "source_file": relative_source(
                        project_path(CENTERS_SOURCE)
                    ),
                    "mapping_source": relative_source(
                        project_path(MAPPING_SOURCE)
                    ),
                    "transformation_note": transformation,
                }
            )
    plotting = pd.DataFrame(plotting_rows)
    plotting["group_label"] = pd.Categorical(
        plotting["group_label"], categories=REGIME_ORDER, ordered=True
    )
    plotting = plotting.sort_values(
        ["group_label", "variable"], kind="stable"
    )
    assert_exact("plotting snapshot rows", len(plotting), 15)
    if not np.isfinite(plotting["value"]).all():
        raise ValueError("Fig.2 plotting data contain non-finite centers.")

    data_path = PLOTTING_DATA_DIR / "Fig2_plotting_data.csv"
    output_path = PNG_DIR / "Fig2_cluster_centers_GATE1_prototype.png"
    write_csv_atomic(data_path, plotting)

    with manuscript_style():
        fig, ax = plt.subplots(
            figsize=mm_to_inches(*FIGURE_SIZES_MM["Fig2"]),
            layout="constrained",
        )
        y_positions = np.arange(len(VARIABLES))
        ax.axvline(
            0,
            color=REFERENCE_COLOR,
            linewidth=0.85,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        for regime in REGIME_ORDER:
            subset = plotting.loc[plotting["group_label"] == regime]
            values = [
                float(
                    subset.loc[subset["variable"] == variable, "value"].iloc[
                        0
                    ]
                )
                for variable, _, _ in VARIABLES
            ]
            n_events = int(subset["sample_size"].iloc[0])
            ax.plot(
                values,
                y_positions,
                color=REGIME_COLORS[regime],
                marker=REGIME_MARKERS[regime],
                linestyle=REGIME_LINESTYLES[regime],
                linewidth=1.45,
                markersize=4.8,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=f"{regime} (n={n_events})",
                zorder=3,
            )
        ax.set_yticks(y_positions)
        ax.set_yticklabels([item[0] for item in VARIABLES])
        ax.invert_yaxis()
        ax.set_xlabel("Standardized center (SD)")
        ax.set_xlim(-2.15, 1.75)
        ax.set_xticks([-2, -1, 0, 1])
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.54, 1.015),
            ncol=3,
            frameon=False,
            handlelength=1.8,
            columnspacing=0.9,
            borderaxespad=0.2,
        )
        style_axis(ax, grid_axis="x")
        panel_label(ax, "(a)")
        save_figure_atomic(fig, output_path, dpi=PROTOTYPE_DPI)
        plt.close(fig)

    caption_en = """# Fig. 2 caption (English)

Standardized centers of the three primary environmental regimes within the
formal 790-event confirmed-tornado sample. The five variables are shown in the
pre-specified clustering order. MLCAPE, 2-m dew point, 0–6-km bulk wind shear,
and signed 0–1-km storm-relative helicity retain their identity transforms;
MLLCL is transformed with log1p before all five variables are standardized
using parameters fitted on the formal 790 events. Lines connect variables only
to aid comparison of each multivariable profile and do not imply continuity or
a hazard ordering. Colors, markers, and line styles redundantly identify C0
(n=131), C1 (n=307), and C2 (n=352).
"""
    caption_zh = """# Fig. 2 图注（中文）

正式790例已确认龙卷样本中三个主要环境组的标准化中心。五个变量按预先固定的
聚类变量顺序排列。MLCAPE、2 m露点、0–6 km体积风切变和带符号的0–1 km
风暴相对螺旋度保持恒等变换；MLLCL先进行log1p变换，随后五个变量均使用在
正式790例上拟合的参数进行标准化。连线仅用于辅助比较同一环境组的多变量
组合，不表示变量连续性或危险性排序。颜色、标记和线型共同区分C0（n=131）、
C1（n=307）和C2（n=352）。
"""
    prohibited = scan_forbidden_language(caption_en + caption_zh)
    if prohibited:
        raise AssertionError(f"Fig.2 caption contains prohibited text: {prohibited}")
    write_text_atomic(CAPTION_DIR / "Fig2_caption_en.md", caption_en)
    write_text_atomic(CAPTION_DIR / "Fig2_caption_zh.md", caption_zh)

    metadata = inspect_png(output_path)
    write_json_atomic(QC_DIR / "Fig2_file_metadata.json", metadata)
    write_qc_report(
        QC_DIR / "Fig2_qc_report.md",
        figure_id="Fig.2",
        status="PASS_WITH_NONBLOCKING_NOTES",
        data_checks=[
            "The center table and accepted raw-to-formal label map are registered Gate 0 sources.",
            "Three centers and all five pre-specified variables are present; no extra variable is plotted.",
            "Mapping counts are C0/C1/C2 = 131/307/352 and close to 790.",
        ],
        number_checks=[
            "The plotting snapshot contains exactly 15 finite center values.",
            "Variable order and formal transformations match the frozen specification.",
        ],
        interpretation_checks=[
            "The display is a multivariable profile within the confirmed-tornado sample.",
            "Lines are described as visual connectors only; no hazard ordering or class exclusivity is implied.",
        ],
        visual_checks=[
            f"Prototype PNG is {metadata['width_px']}×{metadata['height_px']} px with an opaque background.",
            "C0/C1/C2 use fixed colors plus distinct markers and line styles.",
            "A restrained zero reference and x-grid support value lookup.",
        ],
        nonblocking_notes=[
            "Target-journal requirements are pending; vector export and final 600-dpi output belong to Gate 4.",
        ],
    )

    log_text = f"""FIGURE=Fig2
STATUS=PASS_WITH_NONBLOCKING_NOTES
SOURCE_CENTERS={project_path(CENTERS_SOURCE)}
SOURCE_MAPPING={project_path(MAPPING_SOURCE)}
PLOTTING_DATA={data_path}
OUTPUT={output_path}
TRANSFORMATION=wide standardized centers to ordered long plotting table
SCIENTIFIC_REANALYSIS=FALSE
RANDOMNESS=NONE
"""
    write_text_atomic(LOG_DIR / "Fig2_build.log", log_text)
    return {
        "figure_id": "Fig2",
        "status": "PASS_WITH_NONBLOCKING_NOTES",
        "output": str(output_path),
        "plotting_data": str(data_path),
        "script": str(SCRIPT_PATH),
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(build())
