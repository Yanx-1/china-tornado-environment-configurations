"""Shared I/O and deterministic plotting transformations for Gate 1."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROUND_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ROUND_ROOT.parents[1]

PLOTTING_DATA_DIR = ROUND_ROOT / "01_plotting_data"
PNG_DIR = ROUND_ROOT / "04_main_figures_png"
CAPTION_DIR = ROUND_ROOT / "08_caption_drafts"
QC_DIR = ROUND_ROOT / "09_qc_reports"
MONTAGE_DIR = ROUND_ROOT / "10_montage"
LOG_DIR = ROUND_ROOT / "11_logs"
HANDOFF_DIR = ROUND_ROOT / "12_handoff"


def ensure_output_dirs() -> None:
    for path in (
        PLOTTING_DATA_DIR,
        PNG_DIR,
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


def read_csv_checked(
    relative_path: str | Path,
    *,
    required_columns: Iterable[str] = (),
    **kwargs: Any,
) -> pd.DataFrame:
    path = project_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required source does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def _temporary_path(destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )


def write_text_atomic(destination: Path, text: str) -> None:
    file_descriptor, temporary_name = _temporary_path(destination)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline=""
        ) as handle:
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
    file_descriptor, temporary_name = _temporary_path(destination)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            frame.to_csv(handle, index=False)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_figure_atomic(fig, destination: Path, *, dpi: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Capture only rendered Matplotlib Text artists. This sidecar is the
    # machine-readable publication-facing text layer used by the Gate 3
    # terminology audit; filenames, logs, and QC prose are deliberately
    # outside its scope.
    from matplotlib.text import Text

    fig.canvas.draw()
    visible_text = [
        str(artist.get_text()).strip()
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and str(artist.get_text()).strip()
    ]
    inventory_path = (
        QC_DIR
        / "publication_visible_text"
        / f"{destination.stem}_visible_text.json"
    )
    write_json_atomic(
        inventory_path,
        {
            "figure_file": relative_source(destination),
            "text_artist_count": len(visible_text),
            "texts": visible_text,
        },
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
    )
    os.close(file_descriptor)
    try:
        fig.savefig(
            temporary_name,
            dpi=dpi,
            facecolor="white",
            transparent=False,
            bbox_inches=None,
        )
        if destination.suffix.lower() == ".png":
            with Image.open(temporary_name) as source_image:
                rgb_image = source_image.convert("RGB")
            rgb_image.save(temporary_name, dpi=(dpi, dpi))
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


def stable_seed(base_seed: int, key: str) -> int:
    key_digest = hashlib.sha256(key.encode("utf-8")).digest()
    offset = int.from_bytes(key_digest[:4], "little")
    return (base_seed + offset) % (2**32 - 1)


def bootstrap_median_ci(
    values: np.ndarray | pd.Series,
    *,
    seed: int,
    n_boot: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty or fully missing array.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    estimates = np.median(array[indices], axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return float(np.median(array)), float(low), float(high), int(array.size)


def relative_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())
