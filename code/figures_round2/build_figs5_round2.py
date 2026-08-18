"""Build Fig. S5 for the accepted rated-event STP_mod comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import auc, roc_curve

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
AUC_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/07_stp_reassessment/"
    "22_stp_reassessment_v3/05_stp_mod_auc_summary_v3.csv"
)
HANDOFF_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/25_stpmod_supplementary_audit/"
    "CHATGPT_HANDOFF.yaml"
)

OUTPUT = SUPP_FIGURE_DIR / "FigS5_stpmod_rated_events_ROUND2_review.png"
SNAPSHOT = PLOTTING_DATA_DIR / "FigS5_plotting_snapshot.csv"
CAPTION_EN = CAPTION_DIR / "FigS5_caption_en.md"
CAPTION_ZH = CAPTION_DIR / "FigS5_caption_zh.md"
METADATA = QC_DIR / "FigS5_metadata.json"
QC_REPORT = QC_DIR / "FigS5_QC.md"
BUILD_LOG = LOG_DIR / "FigS5_build.log"

GROUP_ORDER = ("EF0–EF1", "EF2+")
GROUP_COLORS = {"EF0–EF1": "#6F7C85", "EF2+": "#D55E00"}
ACCEPTED_AUC = 0.6413
ACCEPTED_CI = (0.5521, 0.7267)


def _load_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    env = read_csv_checked(
        ENV_SOURCE,
        required_columns=("event_id", "f_scale", "STP_mod"),
    )[["event_id", "f_scale", "STP_mod"]].copy()
    auc_summary = read_csv_checked(
        AUC_SOURCE,
        required_columns=("rated_n", "EF2_plus", "EF0_EF1", "AUC"),
    )
    with (PROJECT_ROOT / HANDOFF_SOURCE).open("r", encoding="utf-8") as handle:
        handoff = yaml.safe_load(handle)

    assert_exact("FigS5 full environment rows", len(env), 790)
    assert_exact("FigS5 stored STP_mod missing values", int(env["STP_mod"].isna().sum()), 0)
    rated = env.loc[env["f_scale"].isin(("EF0", "EF1", "EF2", "EF3", "EF4"))].copy()
    rated["comparison_group"] = np.where(
        rated["f_scale"].isin(("EF2", "EF3", "EF4")),
        "EF2+",
        "EF0–EF1",
    )
    assert_exact("FigS5 rated n", len(rated), 181)
    assert_exact(
        "FigS5 comparison counts",
        rated["comparison_group"].value_counts().to_dict(),
        {"EF0–EF1": 108, "EF2+": 73},
    )

    result = handoff["STPMOD_RESULTS"]
    assert_exact("FigS5 handoff rated n", int(result["RATED_N"]), 181)
    assert_exact("FigS5 handoff EF0-EF1 n", int(result["EF0_EF1_N"]), 108)
    assert_exact("FigS5 handoff EF2+ n", int(result["EF2PLUS_N"]), 73)
    assert_close(
        "FigS5 accepted AUC",
        float(result["OVERALL_AUC"]),
        ACCEPTED_AUC,
        1e-12,
    )
    assert_close(
        "FigS5 accepted CI low",
        float(result["OVERALL_AUC_CI_LOW"]),
        ACCEPTED_CI[0],
        1e-12,
    )
    assert_close(
        "FigS5 accepted CI high",
        float(result["OVERALL_AUC_CI_HIGH"]),
        ACCEPTED_CI[1],
        1e-12,
    )
    assert_exact("FigS5 summary rated n", int(auc_summary.loc[0, "rated_n"]), 181)
    assert_exact("FigS5 summary EF0-EF1 n", int(auc_summary.loc[0, "EF0_EF1"]), 108)
    assert_exact("FigS5 summary EF2+ n", int(auc_summary.loc[0, "EF2_plus"]), 73)
    return rated, auc_summary, handoff


def _roc_coordinates(
    rated: pd.DataFrame, auc_summary: pd.DataFrame
) -> pd.DataFrame:
    outcome = rated["comparison_group"].eq("EF2+").astype(int).to_numpy()
    score = rated["STP_mod"].to_numpy(dtype=float)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        outcome, score, drop_intermediate=False
    )
    plotted_auc = float(auc(false_positive_rate, true_positive_rate))
    assert_close(
        "FigS5 event-level ROC AUC vs summary",
        plotted_auc,
        float(auc_summary.loc[0, "AUC"]),
        1e-12,
    )
    assert_close(
        "FigS5 event-level ROC AUC vs accepted display",
        plotted_auc,
        ACCEPTED_AUC,
        6.5e-5,
    )
    return pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "auc_from_curve": plotted_auc,
        }
    )


def _write_snapshot(rated: pd.DataFrame, roc: pd.DataFrame) -> None:
    event_snapshot = rated[
        ["event_id", "f_scale", "comparison_group", "STP_mod"]
    ].copy()
    event_snapshot.insert(0, "record_type", "rated_event")
    roc_snapshot = roc.copy()
    roc_snapshot.insert(0, "record_type", "roc_coordinate")
    write_csv_atomic(
        SNAPSHOT,
        pd.concat([event_snapshot, roc_snapshot], ignore_index=True, sort=False),
    )


def _draw(rated: pd.DataFrame, roc: pd.DataFrame) -> None:
    width, height = FIGURE_SIZES_MM["FigS5"]
    with manuscript_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=mm_to_inches(width, height),
            gridspec_kw={"width_ratios": [1.0, 1.0]},
            layout="constrained",
        )
        ax_distribution, ax_roc = axes

        rng = np.random.default_rng(181)
        distributions = [
            rated.loc[rated["comparison_group"] == group, "STP_mod"].to_numpy()
            for group in GROUP_ORDER
        ]
        box = ax_distribution.boxplot(
            distributions,
            positions=[0, 1],
            widths=0.48,
            whis=(10, 90),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#202020", "linewidth": 1.1},
            whiskerprops={"color": "#666666", "linewidth": 0.7},
            capprops={"color": "#666666", "linewidth": 0.7},
        )
        for patch, group in zip(box["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_edgecolor(GROUP_COLORS[group])
            patch.set_alpha(0.30)
            patch.set_linewidth(0.9)
        for position, group, values in zip([0, 1], GROUP_ORDER, distributions):
            jitter = rng.uniform(-0.16, 0.16, len(values))
            ax_distribution.scatter(
                position + jitter,
                values,
                s=8,
                color=GROUP_COLORS[group],
                alpha=0.28,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
        ax_distribution.set_xticks(
            [0, 1],
            [
                f"EF0–EF1\n(n={len(distributions[0])})",
                f"EF2+\n(n={len(distributions[1])})",
            ],
        )
        ax_distribution.set_ylabel("STP_mod")
        ax_distribution.set_title("Rated-event distributions", pad=4)
        style_axis(ax_distribution, grid_axis="y")
        panel_label(ax_distribution, "(a)", x=-0.16, y=1.04)

        ax_roc.plot(
            roc["false_positive_rate"],
            roc["true_positive_rate"],
            color="#0072B2",
            linewidth=1.8,
        )
        ax_roc.plot(
            [0, 1],
            [0, 1],
            color="#777777",
            linewidth=0.8,
            linestyle=":",
        )
        ax_roc.text(
            0.97,
            0.06,
            "AUC = 0.6413\n95% CI: 0.5521–0.7267",
            transform=ax_roc.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.4,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#B0B0B0",
                "linewidth": 0.55,
                "alpha": 0.92,
            },
        )
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_aspect("equal", adjustable="box")
        ax_roc.set_xlabel("False-positive rate")
        ax_roc.set_ylabel("True-positive rate")
        ax_roc.set_title("ROC curve for EF2+ discrimination", pad=4)
        style_axis(ax_roc, grid_axis="both")
        panel_label(ax_roc, "(b)", x=-0.16, y=1.04)

        save_figure_atomic(fig, OUTPUT, dpi=PROTOTYPE_DPI)
        plt.close(fig)


def _write_caption_and_qc() -> None:
    caption_en = """# Fig. S5 caption (English)

**Figure S5. STP_mod in the accepted rated-tornado comparison.** (a) Distribution of the stored STP_mod values for EF0–EF1 (n=108) and EF2+ (n=73) events. Boxes show the interquartile range, center lines show medians, whiskers span the 10th–90th percentiles, and translucent points show all 181 rated events. (b) Receiver operating characteristic curve for discrimination of EF2+ from EF0–EF1 events (AUC=0.6413; 95% confidence interval, 0.5521–0.7267). STP_mod is a modified STP-like index without a CIN term or cap. No decision cutoff is displayed, and the result is restricted to supplemental description of this rated-event sample.
"""
    caption_zh = """# 图S5图注（中文）

**图S5. 已接受的有等级龙卷比较中的STP_mod。**（a）EF0–EF1（n=108）与EF2+（n=73）事件的已存储STP_mod分布。箱体表示四分位距，中线表示中位数，须线覆盖第10–90百分位，半透明点表示全部181个有等级事件。（b）区分EF2+与EF0–EF1事件的受试者工作特征曲线（AUC=0.6413；95%置信区间为0.5521–0.7267）。STP_mod是一种不含CIN项且不设上限的修正STP类指标。图中不展示决策截点，结果仅限于对该有等级事件样本的补充性描述。
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
        figure_id="Fig. S5",
        status_label="DO_NOT_USE_WITHOUT_RESEARCHER_DECISION",
        scientific_checks=[
            "The exact accepted target contains 181 rated tornado events: 108 EF0–EF1 and 73 EF2+.",
            "Panel (a) uses the stored STP_mod column directly; the index formula was not recalculated.",
            "ROC coordinates were derived only to render the accepted event-level comparison and reproduce AUC=0.6413622527.",
            "The displayed AUC and confidence interval exactly follow the accepted reporting values 0.6413 and 0.5521–0.7267.",
            "No cutoff, regime-specific AUC, or additional discrimination statistic was calculated.",
        ],
        layout_checks=[
            "The prescribed 1×2 layout contains the exact-target distribution and ROC panels.",
            "All 181 rated events remain visible in panel (a).",
            "No point or annotation appears on the ROC curve as a selected cutoff.",
            "The file is a 200 dpi RGB review image with no alpha channel.",
            "All publication-visible text and both captions passed the internal-term search.",
        ],
        interpretation_checks=[
            "The caption defines STP_mod as a modified STP-like index without a CIN term or cap.",
            "The caption restricts the result to supplemental description of the rated-event sample.",
            "The figure and caption contain no forecast-use, event-likelihood, or selected-cutoff interpretation.",
        ],
        deviations=[],
        review_notes=[
            "Researcher approval remains mandatory before manuscript use, as encoded by the figure status label.",
            "The accepted handoff confidence interval is displayed; the separate later summary table contains a closely aligned bootstrap interval generated under a different recorded procedure.",
            "No publisher-specific font or line-width compliance is claimed because a target journal has not yet been designated.",
        ],
    )


def build() -> dict:
    ensure_output_dirs()
    rated, auc_summary, _ = _load_sources()
    roc = _roc_coordinates(rated, auc_summary)
    _write_snapshot(rated, roc)
    _draw(rated, roc)
    _write_caption_and_qc()

    image_info = inspect_png(OUTPUT)
    assert_exact("FigS5 mode", image_info["mode"], "RGB")
    assert_exact("FigS5 alpha", image_info["alpha_present"], False)
    metadata = {
        "figure_id": "FigS5",
        "status_label": "DO_NOT_USE_WITHOUT_RESEARCHER_DECISION",
        "output_stage": "200_DPI_RGB_REVIEW_PROTOTYPE",
        "figure_file": relative_source(OUTPUT),
        "plotting_snapshot": relative_source(SNAPSHOT),
        "caption_en": relative_source(CAPTION_EN),
        "caption_zh": relative_source(CAPTION_ZH),
        "qc_report": relative_source(QC_REPORT),
        "script": relative_source(Path(__file__)),
        "inputs": [
            {"path": ENV_SOURCE, "sha256": sha256_file(PROJECT_ROOT / ENV_SOURCE)},
            {"path": AUC_SOURCE, "sha256": sha256_file(PROJECT_ROOT / AUC_SOURCE)},
            {
                "path": HANDOFF_SOURCE,
                "sha256": sha256_file(PROJECT_ROOT / HANDOFF_SOURCE),
            },
        ],
        "render": image_info,
        "scientific_result_changed": False,
        "comparison_target": "EF2+ versus EF0–EF1 among 181 rated tornado events",
        "accepted_auc_display": ACCEPTED_AUC,
        "accepted_auc_ci_display": list(ACCEPTED_CI),
        "cutoff_displayed": False,
    }
    write_json_atomic(METADATA, metadata)
    write_text_atomic(
        BUILD_LOG,
        "\n".join(
            [
                "FIGURE_ID=FigS5",
                "STATUS=BUILT_AND_QC_COMPLETE",
                f"OUTPUT={relative_source(OUTPUT)}",
                f"SHA256={image_info['sha256']}",
                "RATED_N=181",
                "TARGET_COUNTS=EF0_EF1:108,EF2PLUS:73",
                "AUC_DISPLAY=0.6413",
                "AUC_CI_DISPLAY=0.5521-0.7267",
                "CUTOFF_DISPLAYED=FALSE",
                "STATUS_LABEL=DO_NOT_USE_WITHOUT_RESEARCHER_DECISION",
                "",
            ]
        ),
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
