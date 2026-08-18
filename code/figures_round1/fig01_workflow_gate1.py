"""Gate 1 prototype for Fig. 1: study design and sample closure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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
    assert_exact,
    inspect_png,
    scan_forbidden_language,
    write_qc_report,
)


CLOSURE_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "01_sample_audit/03_sample_closure_v3.csv"
)
SAMPLE_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/"
    "01_sample_audit/05_event_id_list_new790.csv"
)
CONCLUSION_SOURCE = (
    "paper_rebuild/00_governance/pre_manuscript_conclusion_library/"
    "01_master_core_conclusions.md"
)

PROCESS_COLORS = {
    "source": ("#F1F3F5", "#59636E"),
    "audit": ("#E8F0F7", "#4C78A8"),
    "analysis": ("#E7F2F0", "#4D8C87"),
    "primary": ("#FFF3D6", "#B07D16"),
    "context": ("#F2ECF5", "#80628B"),
    "boundary": ("#F7F7F7", "#777777"),
}


def add_box(
    ax,
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    style: str,
    title_size: float = 7.4,
    subtitle_size: float = 6.3,
) -> None:
    face, edge = PROCESS_COLORS[style]
    x = center[0] - width / 2
    y = center[1] - height / 2
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.05,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1] + (0.018 if subtitle else 0),
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#202020",
        zorder=3,
    )
    if subtitle:
        ax.text(
            center[0],
            center[1] - 0.035,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color="#4D4D4D",
            zorder=3,
        )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.85,
        color="#6E6E6E",
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            midpoint[0] + 0.035,
            midpoint[1],
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.0,
            color="#555555",
            zorder=3,
        )


def build() -> dict[str, object]:
    ensure_output_dirs()
    closure = read_csv_checked(
        CLOSURE_SOURCE,
        required_columns=(
            "step",
            "operation",
            "count_before",
            "change",
            "count_after",
            "status",
        ),
    )
    sample = read_csv_checked(
        SAMPLE_SOURCE,
        required_columns=("event_id", "date_utc", "sample_name", "status"),
        dtype={"event_id": str},
    )
    assert_exact("closure rows", len(closure), 5)
    assert_exact("initial records", int(closure.iloc[0]["count_after"]), 909)
    assert_exact("after 2025 exclusion", int(closure.iloc[1]["count_after"]), 795)
    duplicate_rows = closure.loc[
        closure["operation"].str.contains("duplicate", case=False, na=False)
    ]
    assert_exact("duplicate exclusions", int(-duplicate_rows["change"].astype(int).sum()), 2)
    assert_exact("post-duplicate count", int(duplicate_rows.iloc[-1]["count_after"]), 793)
    assert_exact("formal sample count", int(closure.iloc[-1]["count_after"]), 790)
    assert_exact("formal sample rows", len(sample), 790)
    assert_exact("formal unique event_id", sample["event_id"].nunique(), 790)
    dates = pd.to_datetime(sample["date_utc"], errors="raise")
    assert_exact("study year range", [int(dates.dt.year.min()), int(dates.dt.year.max())], [2006, 2024])
    assert_exact("season month range", [int(dates.dt.month.min()), int(dates.dt.month.max())], [3, 10])
    if not dates.dt.month.between(3, 10).all():
        raise AssertionError("Formal sample contains a month outside March–October.")

    closure_display = [
        ("sample_909", "Initial source records", 909, "source"),
        ("sample_795", "Exclude 2025 events", 795, "audit"),
        ("sample_793", "Exclude two duplicates", 793, "audit"),
        ("sample_790", "Formal March–October sample", 790, "primary"),
    ]
    workflow_nodes = [
        ("event_audit", "Event database + sample screening", "", "audit"),
        ("era5", "ERA5 environment extraction", "", "analysis"),
        (
            "five_variables",
            "Five-variable clustering",
            "MLCAPE · MLLCL · Td2m · SHR6 · SRH1",
            "analysis",
        ),
        ("stability", "Stability analysis", "", "analysis"),
        ("k3", "k=3 primary", "", "primary"),
        ("k4", "k=4 sensitivity", "", "context"),
    ]
    evidence_nodes = [
        (
            "scope",
            "Final tornado sample",
            "2006–2024, March–October (n=790)",
            "primary",
        ),
        (
            "regimes",
            "Multivariate environmental regimes\nand their stability",
            "",
            "audit",
        ),
        (
            "vertical",
            "Vertical thermodynamic and\nkinematic characteristics",
            "Temperature · humidity · environmental wind",
            "analysis",
        ),
        (
            "weather",
            "Post-hoc weather-type,\nseasonal, and spatial context",
            "",
            "context",
        ),
        (
            "boundary",
            "Interpretation restricted to\nthe confirmed-tornado sample",
            "",
            "boundary",
        ),
    ]

    source_closure = relative_source(project_path(CLOSURE_SOURCE))
    source_sample = relative_source(project_path(SAMPLE_SOURCE))
    source_conclusion = relative_source(project_path(CONCLUSION_SOURCE))
    records: list[dict[str, object]] = []
    for panel, node_list in (
        ("a", closure_display),
        ("b", workflow_nodes),
        ("c", evidence_nodes),
    ):
        for node in node_list:
            node_id, title, value_or_subtitle, style = node
            records.append(
                {
                    "record_type": "node",
                    "panel": panel,
                    "node_id": node_id,
                    "from_node": "",
                    "to_node": "",
                    "label": title,
                    "subtitle": (
                        str(value_or_subtitle)
                        if not isinstance(value_or_subtitle, int)
                        else ""
                    ),
                    "count": (
                        value_or_subtitle
                        if isinstance(value_or_subtitle, int)
                        else ""
                    ),
                    "style_role": style,
                    "source_file": (
                        source_closure
                        if panel == "a"
                        else f"{source_sample};{source_conclusion}"
                    ),
                    "transformation_note": (
                        "two one-record duplicate exclusions aggregated for display"
                        if node_id == "sample_793"
                        else "frozen workflow statement"
                    ),
                }
            )
    edge_definitions = [
        ("a", "sample_909", "sample_795", "−114 from 2025"),
        ("a", "sample_795", "sample_793", "−2 duplicates"),
        ("a", "sample_793", "sample_790", "−3 outside season"),
        ("b", "event_audit", "era5", ""),
        ("b", "era5", "five_variables", ""),
        ("b", "five_variables", "stability", ""),
        ("b", "stability", "k3", ""),
        ("b", "stability", "k4", ""),
        ("c", "scope", "direction_a", ""),
        ("c", "direction_a", "direction_b", ""),
        ("c", "direction_b", "weather", ""),
        ("c", "weather", "boundary", ""),
    ]
    for panel, from_node, to_node, label in edge_definitions:
        records.append(
            {
                "record_type": "edge",
                "panel": panel,
                "node_id": "",
                "from_node": from_node,
                "to_node": to_node,
                "label": label,
                "subtitle": "",
                "count": "",
                "style_role": "connector",
                "source_file": (
                    source_closure if panel == "a" else source_conclusion
                ),
                "transformation_note": "diagram connector; width carries no quantitative meaning",
            }
        )
    plotting = pd.DataFrame(records)
    data_path = PLOTTING_DATA_DIR / "Fig1_plotting_data.csv"
    output_path = PNG_DIR / "Fig1_workflow_GATE1_prototype.png"
    write_csv_atomic(data_path, plotting)

    with manuscript_style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=mm_to_inches(*FIGURE_SIZES_MM["Fig1"]),
            layout="constrained",
            gridspec_kw={"width_ratios": [0.88, 1.08, 1.12]},
        )
        for ax in axes:
            ax.set_axis_off()

        ax_a, ax_b, ax_c = axes
        panel_label(ax_a, "(a)")
        panel_label(ax_b, "(b)")
        panel_label(ax_c, "(c)")
        ax_a.set_title("Sample closure", pad=8)
        ax_b.set_title("Analysis workflow", pad=8)
        ax_c.set_title("Scientific evidence framework", pad=8)

        closure_y = [0.85, 0.63, 0.41, 0.19]
        closure_titles = [
            ("909", "initial source records", "source"),
            ("795", "2006–2024 candidates", "audit"),
            ("793", "unique in-window candidates", "audit"),
            ("790", "formal March–October sample", "primary"),
        ]
        for y_value, (number, subtitle, style) in zip(
            closure_y, closure_titles
        ):
            add_box(
                ax_a,
                center=(0.48, y_value),
                width=0.60,
                height=0.13,
                title=number,
                subtitle=subtitle,
                style=style,
                title_size=9.2,
                subtitle_size=6.1,
            )
        for index, label in enumerate(
            ("−114 from 2025", "−2 duplicates", "−3 outside Mar–Oct")
        ):
            add_arrow(
                ax_a,
                (0.48, closure_y[index] - 0.075),
                (0.48, closure_y[index + 1] + 0.075),
                label=label,
            )

        workflow_y = [0.88, 0.70, 0.50, 0.30]
        workflow_specs = [
            ("Event database + sample screening", "", "audit"),
            ("ERA5 environment extraction", "", "analysis"),
            (
                "Five-variable clustering",
                "MLCAPE · MLLCL · Td2m · SHR6 · SRH1",
                "analysis",
            ),
            ("Stability analysis", "", "analysis"),
        ]
        for y_value, (title, subtitle, style) in zip(
            workflow_y, workflow_specs
        ):
            add_box(
                ax_b,
                center=(0.50, y_value),
                width=0.76,
                height=0.12,
                title=title,
                subtitle=subtitle,
                style=style,
            )
        for start_y, end_y in zip(workflow_y[:-1], workflow_y[1:]):
            add_arrow(
                ax_b, (0.50, start_y - 0.065), (0.50, end_y + 0.065)
            )
        add_box(
            ax_b,
            center=(0.27, 0.10),
            width=0.37,
            height=0.11,
            title="k=3 primary",
            style="primary",
        )
        add_box(
            ax_b,
            center=(0.73, 0.10),
            width=0.37,
            height=0.11,
            title="k=4 sensitivity",
            style="context",
        )
        add_arrow(ax_b, (0.45, 0.235), (0.29, 0.16))
        add_arrow(ax_b, (0.55, 0.235), (0.71, 0.16))

        evidence_specs = [
            (
                0.89,
                "Final tornado sample",
                "2006–2024, March–October (n=790)",
                "primary",
            ),
            (
                0.69,
                "Multivariate environmental regimes\nand their stability",
                "",
                "audit",
            ),
            (
                0.48,
                "Vertical thermodynamic and\nkinematic characteristics",
                "Temperature · humidity · environmental wind",
                "analysis",
            ),
            (
                0.27,
                "Post-hoc weather-type,\nseasonal, and spatial context",
                "",
                "context",
            ),
            (
                0.08,
                "Interpretation restricted to\nthe confirmed-tornado sample",
                "",
                "boundary",
            ),
        ]
        for y_value, title, subtitle, style in evidence_specs:
            add_box(
                ax_c,
                center=(0.50, y_value),
                width=0.82,
                height=0.13,
                title=title,
                subtitle=subtitle,
                style=style,
                title_size=7.1,
                subtitle_size=6.1,
            )
        for first, second in zip(evidence_specs[:-1], evidence_specs[1:]):
            add_arrow(
                ax_c,
                (0.50, first[0] - 0.07),
                (0.50, second[0] + 0.07),
            )

        save_figure_atomic(fig, output_path, dpi=PROTOTYPE_DPI)
        plt.close(fig)

    caption_en = """# Fig. 1 caption (English)

Study design, sample closure, and scientific evidence framework. (a) The event
chain proceeds from 909 initial records to 795 records after excluding 2025,
to 793 after removing two duplicates, and to the formal 790-event sample after
excluding three events outside March–October. (b) The formal events are linked
to ERA5 environmental extraction, five-variable clustering, stability
analysis, the k=3 primary solution, and k=4 sensitivity analysis. (c) The
evidence framework connects multivariate environmental regimes and their
stability, vertical thermodynamic and kinematic characteristics, and post-hoc
weather-type, seasonal, and spatial context. All interpretation is restricted
to the confirmed-tornado sample from March–October 2006–2024. Arrow widths
carry no quantitative meaning.
"""
    caption_zh = """# Fig. 1 图注（中文）

研究设计、样本闭合和科学证据框架。（a）事件链由909条初始记录开始，排除2025年
记录后为795条，删除2条重复记录后为793条，再排除3条3–10月之外的事件，得到
正式790例样本。（b）正式事件依次进入ERA5环境提取、五变量聚类、稳定性分析、
k=3主要方案和k=4敏感性分析。（c）科学证据框架连接多变量环境组及其稳定性、
垂直热力和运动学特征，以及事后天气型、季节和空间背景。所有解释均限定于
2006–2024年3–10月的已确认龙卷样本。箭头宽度不承载定量含义。
"""
    prohibited = scan_forbidden_language(caption_en + caption_zh)
    if prohibited:
        raise AssertionError(f"Fig.1 caption contains prohibited text: {prohibited}")
    write_text_atomic(CAPTION_DIR / "Fig1_caption_en.md", caption_en)
    write_text_atomic(CAPTION_DIR / "Fig1_caption_zh.md", caption_zh)

    metadata = inspect_png(output_path)
    write_json_atomic(QC_DIR / "Fig1_file_metadata.json", metadata)
    write_qc_report(
        QC_DIR / "Fig1_qc_report.md",
        figure_id="Fig.1",
        status="PASS_WITH_NONBLOCKING_NOTES",
        data_checks=[
            "The V3 closure table and formal event list are authoritative Gate 0 sources.",
            "The formal list contains 790 unique event identifiers from 2006–2024, all in March–October.",
            "No non-tornado comparison or unfinished analysis is depicted as completed.",
        ],
        number_checks=[
            "The displayed closure chain reproduces 909→795→793→790.",
            "The two individual duplicate exclusions are aggregated only for diagram presentation.",
        ],
        interpretation_checks=[
            "k=3 is labeled primary and k=4 is labeled structural sensitivity.",
            "Weather type is described as post-hoc context and the scope is within the confirmed-tornado sample.",
            "No operational or hazard-oriented module is shown.",
        ],
        visual_checks=[
            f"Prototype PNG is {metadata['width_px']}×{metadata['height_px']} px in RGB mode.",
            "Process-stage colors are distinct from the fixed regime palette.",
            "Arrow widths are uniform and explicitly carry no quantitative meaning.",
        ],
        nonblocking_notes=[
            "Diagram spacing and typographic hierarchy will be reconsidered with the full eight-figure system in Gate 3.",
            "Publisher-specific rules remain pending.",
        ],
    )
    write_text_atomic(
        LOG_DIR / "Fig1_build.log",
        f"""FIGURE=Fig1
STATUS=PASS_WITH_NONBLOCKING_NOTES
SOURCE_CLOSURE={project_path(CLOSURE_SOURCE)}
SOURCE_SAMPLE={project_path(SAMPLE_SOURCE)}
SOURCE_CONCLUSIONS={project_path(CONCLUSION_SOURCE)}
PLOTTING_DATA={data_path}
OUTPUT={output_path}
TRANSFORMATION=aggregate two one-record duplicate steps into one displayed -2 transition
ARROW_WIDTH_ENCODING=NONE
SCIENTIFIC_REANALYSIS=FALSE
""",
    )
    return {
        "figure_id": "Fig1",
        "status": "PASS_WITH_NONBLOCKING_NOTES",
        "output": str(output_path),
        "plotting_data": str(data_path),
        "script": str(SCRIPT_PATH),
        "metadata": metadata,
    }


if __name__ == "__main__":
    print(build())
