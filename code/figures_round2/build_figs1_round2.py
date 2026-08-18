"""Build Fig. S1: event-centered pressure-level composite-mean fields."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from round2_io import (
    CAPTION_DIR,
    LOG_DIR,
    PLOTTING_DATA_DIR,
    PROJECT_ROOT,
    QC_DIR,
    SUPP_FIGURE_DIR,
    ensure_output_dirs,
    inspect_png,
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
    REGIME_ORDER,
    manuscript_style,
    mm_to_inches,
    panel_label,
)


GRID_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/39_direction_b_2d_resume_full790/"
    "07_full790_pressure_composite_grids.pkl"
)
CENTER_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/39_direction_b_2d_resume_full790/"
    "09_full790_composite_statistics.csv"
)
CLAIM_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/39_direction_b_2d_resume_full790/"
    "09_direction_b_final_claim_adjudication_v6.csv"
)
GRID_DEFINITION_SOURCE = (
    "paper_rebuild/17_core_sample_reopen_new790/38_direction_b_full790_2d_completion/"
    "02_composite_grid_definition.yaml"
)

OUTPUT = SUPP_FIGURE_DIR / "FigS1_pressure_composites_ROUND2_review.png"
SNAPSHOT = PLOTTING_DATA_DIR / "FigS1_plotting_snapshot.csv"
CAPTION_EN = CAPTION_DIR / "FigS1_caption_en.md"
CAPTION_ZH = CAPTION_DIR / "FigS1_caption_zh.md"
METADATA = QC_DIR / "FigS1_metadata.json"
QC_REPORT = QC_DIR / "FigS1_QC.md"
BUILD_LOG = LOG_DIR / "FigS1_build.log"

VARIABLES = (
    {
        "key": "Z500",
        "level": 500,
        "source_vars": ("z",),
        "title": "500-hPa geopotential height",
        "colorbar": "Composite mean (gpm)",
        "cmap": "cividis",
        "vmin": 5600.0,
        "vmax": 5900.0,
    },
    {
        "key": "RH500",
        "level": 500,
        "source_vars": ("r",),
        "title": "500-hPa relative humidity",
        "colorbar": "Composite mean (%)",
        "cmap": "YlGnBu",
        "vmin": 40.0,
        "vmax": 75.0,
    },
    {
        "key": "WS200",
        "level": 200,
        "source_vars": ("u", "v"),
        "title": "200-hPa wind-vector magnitude",
        "colorbar": "Magnitude of composite mean (m s$^{-1}$)",
        "cmap": "magma",
        "vmin": 0.0,
        "vmax": 32.0,
    },
)
REGIME_N = {"C0": 131, "C1": 307, "C2": 352}


def _load_grid() -> dict:
    path = PROJECT_ROOT / GRID_SOURCE
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        grid = pickle.load(handle)
    assert_exact("FigS1 n_events", int(grid["n_events"]), 790)
    assert_exact("FigS1 grid_size", int(grid["grid_size"]), 81)
    assert_close("FigS1 window degrees", float(grid["window_deg"]), 10.0, 1e-12)
    assert_close("FigS1 resolution", float(grid["resolution"]), 0.25, 1e-12)
    for regime in REGIME_ORDER:
        for spec in VARIABLES:
            for variable in spec["source_vars"]:
                key = (regime, spec["level"], variable)
                if key not in grid["accum"] or key not in grid["count"]:
                    raise AssertionError(f"FigS1 missing field: {key}")
                assert_exact(
                    f"FigS1 {key} shape",
                    tuple(grid["accum"][key][0].shape),
                    (81, 81),
                )
                assert_exact(
                    f"FigS1 {key} center n",
                    int(grid["count"][key][40, 40]),
                    REGIME_N[regime],
                )
    return grid


def _mean_field(grid: dict, regime: str, level: int, variable: str) -> np.ndarray:
    key = (regime, level, variable)
    total = np.asarray(grid["accum"][key][0], dtype=float)
    count = np.asarray(grid["count"][key], dtype=float)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _display_fields(grid: dict) -> dict[tuple[str, str], np.ndarray]:
    fields: dict[tuple[str, str], np.ndarray] = {}
    for regime in REGIME_ORDER:
        z500 = _mean_field(grid, regime, 500, "z") / 9.80665
        rh500 = _mean_field(grid, regime, 500, "r")
        u200 = _mean_field(grid, regime, 200, "u")
        v200 = _mean_field(grid, regime, 200, "v")
        fields[(regime, "Z500")] = z500
        fields[(regime, "RH500")] = rh500
        fields[(regime, "WS200")] = np.hypot(u200, v200)
    return fields


def _verify_centers(fields: dict[tuple[str, str], np.ndarray]) -> None:
    center = pd.read_csv(PROJECT_ROOT / CENTER_SOURCE, encoding="utf-8-sig")
    required = {"regime", "level", "var", "center_n", "center_value"}
    assert_exact("FigS1 center table fields", required.issubset(center.columns), True)
    for regime in REGIME_ORDER:
        assert_close(
            f"{regime} Z500 center",
            fields[(regime, "Z500")][40, 40],
            center.query("regime == @regime and level == 500 and var == 'z'")[
                "center_value"
            ].iloc[0],
            0.01,
        )
        assert_close(
            f"{regime} RH500 center",
            fields[(regime, "RH500")][40, 40],
            center.query("regime == @regime and level == 500 and var == 'r'")[
                "center_value"
            ].iloc[0],
            0.01,
        )
        u = center.query("regime == @regime and level == 200 and var == 'u'")[
            "center_value"
        ].iloc[0]
        v = center.query("regime == @regime and level == 200 and var == 'v'")[
            "center_value"
        ].iloc[0]
        assert_close(
            f"{regime} WS200 center",
            fields[(regime, "WS200")][40, 40],
            np.hypot(u, v),
            0.02,
        )


def _write_snapshot(
    grid: dict, fields: dict[tuple[str, str], np.ndarray]
) -> None:
    coordinates = np.linspace(
        -float(grid["window_deg"]),
        float(grid["window_deg"]),
        int(grid["grid_size"]),
    )
    relative_lon, relative_lat = np.meshgrid(coordinates, coordinates)
    frames = []
    for regime in REGIME_ORDER:
        for spec in VARIABLES:
            values = fields[(regime, spec["key"])]
            frames.append(
                pd.DataFrame(
                    {
                        "regime": regime,
                        "regime_n": REGIME_N[regime],
                        "field": spec["key"],
                        "relative_longitude_deg": relative_lon.ravel(),
                        "relative_latitude_deg": relative_lat.ravel(),
                        "composite_mean_value": values.ravel(),
                        "center_grid_cell": (
                            (relative_lon.ravel() == 0) & (relative_lat.ravel() == 0)
                        ),
                    }
                )
            )
    write_csv_atomic(SNAPSHOT, pd.concat(frames, ignore_index=True))


def _draw(grid: dict, fields: dict[tuple[str, str], np.ndarray]) -> None:
    width, height = FIGURE_SIZES_MM["FigS1"]
    extent = [
        -float(grid["window_deg"]),
        float(grid["window_deg"]),
        -float(grid["window_deg"]),
        float(grid["window_deg"]),
    ]
    with manuscript_style():
        fig, axes = plt.subplots(
            3,
            3,
            figsize=mm_to_inches(width, height),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        images = []
        labels = iter("abcdefghi")
        for row, regime in enumerate(REGIME_ORDER):
            for col, spec in enumerate(VARIABLES):
                ax = axes[row, col]
                image = ax.imshow(
                    fields[(regime, spec["key"])],
                    origin="lower",
                    extent=extent,
                    interpolation="nearest",
                    cmap=spec["cmap"],
                    norm=Normalize(spec["vmin"], spec["vmax"]),
                    aspect="equal",
                    rasterized=True,
                )
                images.append(image)
                ax.axhline(0, color="white", linewidth=0.35, alpha=0.55)
                ax.axvline(0, color="white", linewidth=0.35, alpha=0.55)
                ax.plot(
                    0,
                    0,
                    marker="+",
                    markersize=6,
                    markeredgewidth=1.0,
                    color="#111111",
                    zorder=3,
                )
                ax.set_xticks([-10, -5, 0, 5, 10])
                ax.set_yticks([-10, -5, 0, 5, 10])
                ax.tick_params(direction="out")
                panel_label(ax, f"({next(labels)})", x=-0.19, y=1.025)
                if row == 0:
                    ax.set_title(spec["title"], pad=4)
                if col == 0:
                    ax.set_ylabel(f"{regime} (n={REGIME_N[regime]})\nRelative latitude (°)")
                if row == 2:
                    ax.set_xlabel("Relative longitude (°)")

        for col, spec in enumerate(VARIABLES):
            colorbar = fig.colorbar(
                images[col],
                ax=axes[:, col],
                orientation="horizontal",
                location="bottom",
                fraction=0.050,
                pad=0.045,
                aspect=26,
            )
            colorbar.set_label(spec["colorbar"], labelpad=2)
            colorbar.outline.set_linewidth(0.55)
            colorbar.ax.tick_params(length=2.5, width=0.55)
        fig.suptitle(
            "Event-centered composite-mean fields",
            fontsize=9.2,
            fontweight="normal",
        )
        save_figure_atomic(fig, OUTPUT, dpi=PROTOTYPE_DPI)
        plt.close(fig)


def _write_caption_and_qc() -> None:
    caption_en = """# Fig. S1 caption (English)

**Figure S1. Event-centered pressure-level composite-mean fields for the three environmental regimes.** Rows show C0, C1, and C2; columns show 500-hPa geopotential height, 500-hPa relative humidity, and the magnitude of the 200-hPa composite-mean wind vector. Every panel uses the same ±10° event-relative domain and 0.25° grid, and each variable column uses one shared color scale. The black plus marks the event-centered grid cell; sample sizes are given in the row labels. These gridded composite means provide spatial context and are distinct from the event-point median statistics used for the primary vertical comparisons. The wind quantity is the magnitude formed from the composite-mean u and v components, not the mean of event-level wind speeds. No spatial smoothing or geographic regridding was applied.
"""
    caption_zh = """# 图S1图注（中文）

**图S1. 三类环境型的事件中心气压层合成平均场。** 行依次为C0、C1和C2，列依次为500 hPa位势高度、500 hPa相对湿度以及200 hPa合成平均风矢量的模。所有面板采用相同的事件相对±10°范围和0.25°网格，同一变量列共用一个色标。黑色加号表示事件中心网格点，样本量标于行标签。这些格点合成平均场仅提供空间背景，与主要垂直比较所采用的事件点中位数统计相互区分。风速量由合成平均u、v分量求模，并非事件风速的平均值。未实施空间平滑或地理重网格。
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
        figure_id="Fig. S1",
        status_label="SUPPLEMENT_CANDIDATE",
        scientific_checks=[
            "The source grid contains exactly 790 events on an 81×81, ±10°, 0.25° event-relative grid.",
            "Center-cell effective sample sizes close at C0=131, C1=307, and C2=352 for every displayed source variable.",
            "Displayed center-cell Z500 and RH500 values reproduce the accepted composite-center table within its displayed precision.",
            "Displayed WS200 is the vector magnitude of the accepted composite-mean u and v fields; it is not an event-level wind-speed statistic.",
            "No smoothing, interpolation, or anomaly transformation was added.",
        ],
        layout_checks=[
            "The scientific panel matrix is exactly three regimes by three variables.",
            "All nine panels use an identical coordinate domain, grid, aspect ratio, and origin marker.",
            "Each variable column has one shared normalization and one shared colorbar.",
            "The file is a 200 dpi RGB review image with no alpha channel.",
            "All publication-visible text and both captions passed the internal-term search.",
        ],
        interpretation_checks=[
            "The title and colorbars identify the displayed fields as composite means.",
            "The caption distinguishes gridded composite-center context from event-point median statistics.",
            "The fixed-level upper-tropospheric field is presented descriptively without tropopause-relative interpretation.",
        ],
        deviations=[
            "A geographic map projection is not used because the accepted grid is event-centered and relative, not a common geographic coordinate field. Identical Cartesian relative-coordinate axes preserve the actual data geometry without implying false coastlines or absolute locations.",
        ],
        review_notes=[
            "The 3×3 layout remains legible at the provisional 178-mm review width, so the wind column was not split into another figure.",
            "No publisher-specific font or line-width compliance is claimed because a target journal has not yet been designated.",
        ],
    )


def build() -> dict:
    ensure_output_dirs()
    grid = _load_grid()
    fields = _display_fields(grid)
    _verify_centers(fields)
    _write_snapshot(grid, fields)
    _draw(grid, fields)
    _write_caption_and_qc()

    image_info = inspect_png(OUTPUT)
    assert_exact("FigS1 mode", image_info["mode"], "RGB")
    assert_exact("FigS1 alpha", image_info["alpha_present"], False)
    metadata = {
        "figure_id": "FigS1",
        "status_label": "SUPPLEMENT_CANDIDATE",
        "output_stage": "200_DPI_RGB_REVIEW_PROTOTYPE",
        "figure_file": relative_source(OUTPUT),
        "plotting_snapshot": relative_source(SNAPSHOT),
        "caption_en": relative_source(CAPTION_EN),
        "caption_zh": relative_source(CAPTION_ZH),
        "qc_report": relative_source(QC_REPORT),
        "script": relative_source(Path(__file__)),
        "inputs": [
            {"path": GRID_SOURCE, "sha256": sha256_file(PROJECT_ROOT / GRID_SOURCE)},
            {"path": CENTER_SOURCE, "sha256": sha256_file(PROJECT_ROOT / CENTER_SOURCE)},
            {"path": CLAIM_SOURCE, "sha256": sha256_file(PROJECT_ROOT / CLAIM_SOURCE)},
            {
                "path": GRID_DEFINITION_SOURCE,
                "sha256": sha256_file(PROJECT_ROOT / GRID_DEFINITION_SOURCE),
            },
        ],
        "render": image_info,
        "scientific_result_changed": False,
        "display_field_definition": {
            "Z500": "sum(z)/count/9.80665",
            "RH500": "sum(r)/count",
            "WS200": "hypot(sum(u)/count, sum(v)/count)",
        },
    }
    write_json_atomic(METADATA, metadata)
    write_text_atomic(
        BUILD_LOG,
        "\n".join(
            [
                "FIGURE_ID=FigS1",
                "STATUS=BUILT_AND_QC_COMPLETE",
                f"OUTPUT={relative_source(OUTPUT)}",
                f"SHA256={image_info['sha256']}",
                "SOURCE_EVENTS=790",
                "PANEL_MATRIX=3_REGIMES_X_3_VARIABLES",
                "GRID=EVENT_CENTERED_RELATIVE_81X81",
                "SHARED_COLOR_SCALE=PER_VARIABLE_COLUMN",
                "SPATIAL_SMOOTHING=NONE",
                "",
            ]
        ),
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
