"""Atomic I/O and rendered-file inspection for Round 2."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROUND_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ROUND_ROOT.parents[1]

SOURCE_REGISTRY_DIR = ROUND_ROOT / "00_source_registry"
PLOTTING_DATA_DIR = ROUND_ROOT / "01_plotting_data"
CONFIG_DIR = ROUND_ROOT / "03_configs"
MAIN_FIGURE_DIR = ROUND_ROOT / "04_candidate_main_figures"
SUPP_FIGURE_DIR = ROUND_ROOT / "05_candidate_supplementary_figures"
CAPTION_DIR = ROUND_ROOT / "06_caption_drafts"
QC_DIR = ROUND_ROOT / "07_qc_reports"
MONTAGE_DIR = ROUND_ROOT / "08_montage"
LOG_DIR = ROUND_ROOT / "09_logs"
HANDOFF_DIR = ROUND_ROOT / "10_handoff"


def ensure_output_dirs() -> None:
    for path in (
        SOURCE_REGISTRY_DIR,
        PLOTTING_DATA_DIR,
        CONFIG_DIR,
        MAIN_FIGURE_DIR,
        SUPP_FIGURE_DIR,
        CAPTION_DIR,
        QC_DIR,
        MONTAGE_DIR,
        LOG_DIR,
        HANDOFF_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def project_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_csv_checked(
    relative_path: str | Path,
    *,
    required_columns: Iterable[str] = (),
    **kwargs: Any,
) -> pd.DataFrame:
    path = project_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return frame


def _temporary_path(destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )


def write_text_atomic(destination: Path, text: str) -> None:
    descriptor, temporary_name = _temporary_path(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json_atomic(destination: Path, payload: Any) -> None:
    write_text_atomic(
        destination,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def write_csv_atomic(destination: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = _temporary_path(destination)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            frame.to_csv(handle, index=False)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_figure_atomic(fig, destination: Path, *, dpi: int) -> None:
    from matplotlib.text import Text

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    visible_text = [
        str(artist.get_text()).strip()
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and str(artist.get_text()).strip()
    ]
    write_json_atomic(
        QC_DIR
        / "publication_visible_text"
        / f"{destination.stem}_visible_text.json",
        {
            "figure_file": relative_source(destination),
            "text_artist_count": len(visible_text),
            "texts": visible_text,
        },
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    try:
        fig.savefig(
            temporary_name,
            dpi=dpi,
            facecolor="white",
            transparent=False,
            bbox_inches=None,
        )
        with Image.open(temporary_name) as image:
            rgb = image.convert("RGB")
        rgb.save(temporary_name, dpi=(dpi, dpi))
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def inspect_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", ("", ""))
        return {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "alpha_present": "A" in image.getbands(),
            "dpi_x": dpi[0] if isinstance(dpi, tuple) and dpi else "",
            "dpi_y": dpi[1] if isinstance(dpi, tuple) and len(dpi) > 1 else "",
            "file_size_bytes": path.stat().st_size,
        }
