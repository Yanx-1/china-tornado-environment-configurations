"""Shared infrastructure for the final manuscript visual-asset package.

This module only reads accepted plotting snapshots and table sources.  It writes
to paper_rebuild/52_manuscript_visual_assets_final and never mutates upstream
scientific assets.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FINAL_ROOT.parents[1]
PAPER_ROOT = PROJECT_ROOT / "paper_rebuild"

ROUND1_ROOT = PAPER_ROOT / "50_manuscript_figures_round1_topjournal"
ROUND2_ROOT = PAPER_ROOT / "51_figure_expansion_round2"
TABLE_SOURCE_ROOT = (
    PAPER_ROOT
    / "17_core_sample_reopen_new790"
    / "27_direction_a_figure_table_regeneration"
)

MAIN_FIGURE_DIR = FINAL_ROOT / "01_main_figures"
SUPP_FIGURE_DIR = FINAL_ROOT / "02_supplementary_figures"
HOLD_DIR = FINAL_ROOT / "03_held_optional_figures"
MAIN_TABLE_DIR = FINAL_ROOT / "04_main_tables"
SUPP_TABLE_DIR = FINAL_ROOT / "05_supplementary_tables"
CAPTION_DIR = FINAL_ROOT / "06_captions"
GUIDE_DIR = FINAL_ROOT / "07_manuscript_call_guide"
CROSSWALK_DIR = FINAL_ROOT / "08_crosswalks"
MONTAGE_DIR = FINAL_ROOT / "09_montages"
QC_DIR = FINAL_ROOT / "10_qc"
LOG_DIR = FINAL_ROOT / "11_logs"
HANDOFF_DIR = FINAL_ROOT / "12_handoff"
README_DIR = FINAL_ROOT / "00_README"
SNAPSHOT_DIR = README_DIR / "plotting_data_snapshots"
SOURCE_REGISTRY_DIR = README_DIR / "source_registry"

REGIME_ORDER = ("C0", "C1", "C2")
REGIME_COLORS = {"C0": "#0072B2", "C1": "#E69F00", "C2": "#009E73"}
REGIME_MARKERS = {"C0": "o", "C1": "s", "C2": "^"}
REGIME_LINESTYLES = {"C0": "-", "C1": "--", "C2": "-."}
REGIME_N = {"C0": 131, "C1": 307, "C2": 352}

K4_ORDER = ("K4_C0", "K4_C1", "K4_C2", "K4_C3")
K4_DISPLAY = {
    "K4_C0": "K4-C0",
    "K4_C1": "K4-C1",
    "K4_C2": "K4-C2",
    "K4_C3": "K4-C3",
}
K4_COLORS = {
    "K4_C0": "#0072B2",
    "K4_C1": "#D55E00",
    "K4_C2": "#009E73",
    "K4_C3": "#8B5CF6",
}
K4_MARKERS = {"K4_C0": "o", "K4_C1": "s", "K4_C2": "^", "K4_C3": "D"}
K4_LINESTYLES = {
    "K4_C0": "-",
    "K4_C1": "--",
    "K4_C2": "-.",
    "K4_C3": ":",
}

FORMAL_FORBIDDEN_TERMS = (
    "Direction A",
    "Direction B",
    "Gate",
    "PASS",
    "frozen",
    "prototype",
    "audit status",
    "deprecated reporting number",
    "old793",
    "legacy790",
    "risk corridor",
    "tornado corridor",
    "operational threshold",
    "standard STP",
    "storm-relative hodograph",
)


def ensure_directories() -> None:
    directories = [
        README_DIR,
        SNAPSHOT_DIR,
        SOURCE_REGISTRY_DIR,
        CAPTION_DIR,
        GUIDE_DIR,
        CROSSWALK_DIR,
        MONTAGE_DIR,
        QC_DIR,
        QC_DIR / "visible_text",
        QC_DIR / "image_metadata",
        LOG_DIR,
        HANDOFF_DIR,
        SCRIPT_DIR,
        HOLD_DIR,
    ]
    for root in (MAIN_FIGURE_DIR, SUPP_FIGURE_DIR):
        directories.extend(
            root / child
            for child in ("png_600dpi", "pdf_vector", "svg_vector", "review_png")
        )
    for root in (MAIN_TABLE_DIR, SUPP_TABLE_DIR):
        directories.extend(
            root / child for child in ("csv", "xlsx", "markdown", "previews")
        )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def mm_to_in(value: float) -> float:
    return value / 25.4


def relative_to_final(path: Path) -> str:
    return path.resolve().relative_to(FINAL_ROOT.resolve()).as_posix()


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_target(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(fd)
    return Path(name)


def write_text(path: Path, text: str) -> None:
    temporary = _atomic_target(path)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv_rows(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    temporary = _atomic_target(path)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextlib.contextmanager
def publication_style():
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Microsoft YaHei",
            "DejaVu Sans",
        ],
        "font.size": 8.0,
        "axes.titlesize": 8.7,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "figure.titlesize": 9.2,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
    }
    with mpl.rc_context(settings):
        yield


def style_axis(ax, grid_axis: str | None = None) -> None:
    ax.tick_params(direction="out")
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color="#D6D6D6",
            linewidth=0.45,
            linestyle=":",
            zorder=0,
        )
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.11, y: float = 1.035) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
        clip_on=False,
    )


def regime_legend_handles(include_n: bool = True):
    from matplotlib.lines import Line2D

    handles = []
    for regime in REGIME_ORDER:
        label = f"{regime} (n={REGIME_N[regime]})" if include_n else regime
        handles.append(
            Line2D(
                [],
                [],
                color=REGIME_COLORS[regime],
                marker=REGIME_MARKERS[regime],
                linestyle=REGIME_LINESTYLES[regime],
                markerfacecolor="white",
                markeredgecolor=(
                    "#855A00" if regime == "C1" else REGIME_COLORS[regime]
                ),
                markeredgewidth=0.9,
                markersize=5,
                label=label,
            )
        )
    return handles


def _visible_text_inventory(fig) -> tuple[list[dict], int]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    inventory: list[dict] = []
    overflow = 0
    for artist in fig.findobj(match=lambda value: isinstance(value, Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        clipped = bool(
            bbox.x0 < -1.5
            or bbox.y0 < -1.5
            or bbox.x1 > width + 1.5
            or bbox.y1 > height + 1.5
        )
        overflow += int(clipped)
        inventory.append(
            {
                "text": artist.get_text(),
                "font_size_pt": float(artist.get_fontsize()),
                "font_family": list(artist.get_fontfamily()),
                "bbox_px": [
                    round(float(bbox.x0), 2),
                    round(float(bbox.y0), 2),
                    round(float(bbox.x1), 2),
                    round(float(bbox.y1), 2),
                ],
                "outside_canvas": clipped,
            }
        )
    return inventory, overflow


def _save_png_rgb(fig, destination: Path, dpi: int) -> None:
    temporary = _atomic_target(destination)
    try:
        fig.savefig(
            temporary,
            format="png",
            dpi=dpi,
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
        with Image.open(temporary) as image:
            rgb = image.convert("RGB")
            rgb.save(
                destination,
                format="PNG",
                dpi=(dpi, dpi),
                optimize=False,
                compress_level=6,
            )
    finally:
        if temporary.exists():
            temporary.unlink()


def export_formal_figure(fig, stem: str, scope: str) -> dict:
    """Export one figure to formal PNG, review PNG, PDF and SVG."""

    ensure_directories()
    root = MAIN_FIGURE_DIR if scope == "main" else SUPP_FIGURE_DIR
    targets = {
        "png_600dpi": root / "png_600dpi" / f"{stem}.png",
        "pdf_vector": root / "pdf_vector" / f"{stem}.pdf",
        "svg_vector": root / "svg_vector" / f"{stem}.svg",
        "review_png": root / "review_png" / f"{stem}_review.png",
    }
    inventory, overflow = _visible_text_inventory(fig)
    visible_text = "\n".join(item["text"] for item in inventory)
    term_counts = {
        term: visible_text.lower().count(term.lower()) for term in FORMAL_FORBIDDEN_TERMS
    }
    if any(term_counts.values()):
        raise AssertionError(f"Forbidden visible text in {stem}: {term_counts}")
    if "587.3" in visible_text:
        raise AssertionError(f"Deprecated statistic found in visible text for {stem}")

    _save_png_rgb(fig, targets["png_600dpi"], 600)
    _save_png_rgb(fig, targets["review_png"], 250)
    for key, format_name in (("pdf_vector", "pdf"), ("svg_vector", "svg")):
        temporary = _atomic_target(targets[key])
        try:
            with mpl.rc_context(
                {
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                    "svg.fonttype": "none",
                }
            ):
                fig.savefig(
                    temporary,
                    format=format_name,
                    facecolor="white",
                    edgecolor="white",
                    transparent=False,
                )
            os.replace(temporary, targets[key])
        finally:
            if temporary.exists():
                temporary.unlink()

    with Image.open(targets["png_600dpi"]) as image:
        image_meta = {
            "mode": image.mode,
            "pixel_width": image.width,
            "pixel_height": image.height,
            "dpi": [round(float(v), 3) for v in image.info.get("dpi", (0, 0))],
        }
    metadata = {
        "stem": stem,
        "scope": scope,
        "canvas_width_mm": round(fig.get_figwidth() * 25.4, 3),
        "canvas_height_mm": round(fig.get_figheight() * 25.4, 3),
        "visible_text_overflow_count": overflow,
        "forbidden_visible_text_counts": term_counts,
        "image": image_meta,
        "files": {
            key: {
                "path": relative_to_final(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for key, path in targets.items()
        },
    }
    write_json(QC_DIR / "visible_text" / f"{stem}.json", inventory)
    write_json(QC_DIR / "image_metadata" / f"{stem}.json", metadata)
    plt.close(fig)
    return metadata


def font_properties(bold: bool = False, size: float = 8.0) -> FontProperties:
    font_path = Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf")
    if font_path.is_file():
        return FontProperties(fname=str(font_path), size=size)
    return FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal", size=size)
