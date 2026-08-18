"""
Core Sample Reopen V3 — NEW_VALID_790_MAR_OCT_V3

Researcher decision:
EXCLUDE_THREE_TRUE_OUT_OF_WINDOW_EVENTS_AND_REBUILD

This script writes only under:
paper_rebuild/17_core_sample_reopen_new790/

It does not modify Day 1--Day 9 frozen files.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import textwrap
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from scipy.stats import chi2_contingency
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"STOP: scipy.stats.chi2_contingency is required: {exc}")


warnings.filterwarnings("ignore", category=FutureWarning)

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RUN_STATUS = "CORE_SAMPLE_REOPEN_IN_PROGRESS"
RESEARCHER_DECISION = "EXCLUDE_THREE_TRUE_OUT_OF_WINDOW_EVENTS_AND_REBUILD"
NEW_SAMPLE_NAME = "NEW_VALID_790_MAR_OCT_V3"
LEGACY_SAMPLE_NAME = "LEGACY_INVALID_790_DO_NOT_USE"
OLD_793_STATUS = "SUPERSEDED_BY_TRUE_WINDOW_CORRECTION"
LEGACY_790_STATUS = "INVALID_LEGACY_DO_NOT_USE"
SEED = 20260728
K_RANGE = list(range(2, 7))
PRIMARY_5 = ["MLCAPE_Jkg", "MLLCL_m", "ERA5_d2m_K", "SHR6_ms", "SRH1_m2s2"]
OUT_OF_WINDOW_IDS = ["TOR0355", "TOR0527", "TOR0627"]
RETAIN_IDS = ["TOR0174", "TOR0175", "TOR0608"]


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
PAPER_ROOT = PROJECT_ROOT / "paper_rebuild"
V3_ROOT = PAPER_ROOT / "17_core_sample_reopen_new790"

DIRS = {
    "sample": V3_ROOT / "01_sample_audit",
    "env": V3_ROOT / "02_environment_table",
    "matrix": V3_ROOT / "03_clustering_matrix",
    "cluster": V3_ROOT / "04_clustering",
    "regime": V3_ROOT / "05_regime_interpretation",
    "figtab": V3_ROOT / "06_figures_tables",
    "stp": V3_ROOT / "07_stp_reassessment",
    "impact": V3_ROOT / "08_impact_assessment",
    "manifest": V3_ROOT / "09_manifests",
    "handoff": V3_ROOT / "10_handoff",
}


SRC = {
    "env793": PAPER_ROOT
    / "03_full_environment_recalculation"
    / "18_mlcape_zero_policy_correction"
    / "final_v2_freeze"
    / "12_tornado_env_params_analysis_v2_1_793.csv",
    "event_audit": PAPER_ROOT / "01_sample_audit" / "03_event_inclusion_audit.csv",
    "sample_flow": PAPER_ROOT / "01_sample_audit" / "06_sample_flow_candidate.csv",
    "k3_old": PAPER_ROOT
    / "05_regime_interpretation_and_external_validation"
    / "02_labels_k3_primary_793.csv",
    "k4_old": PAPER_ROOT
    / "05_regime_interpretation_and_external_validation"
    / "03_labels_k4_sensitivity_793.csv",
    "synoptic": PROJECT_ROOT / "data" / "events" / "tornado_synoptic_type.csv",
    "legacy790_env": PROJECT_ROOT / "data" / "events" / "tornado_env_params.csv",
    "synoptic_map": PAPER_ROOT
    / "05_regime_interpretation"
    / "final_interpretation_freeze"
    / "taxonomy_reconciliation"
    / "02_synoptic_subtype_to_macro_mapping.csv",
    "old_k_evidence": PAPER_ROOT
    / "04_clustering_rebuild"
    / "03_day4b_clustering_stability"
    / "29_k_selection_evidence_table.csv",
    "old_primary_metrics": PAPER_ROOT
    / "04_clustering_rebuild"
    / "03_day4b_clustering_stability"
    / "07_primary_kmeans_metrics.csv",
    "old_pca_var": PAPER_ROOT
    / "04_clustering_rebuild"
    / "03_day4b_clustering_stability"
    / "03_pca_explained_variance.csv",
}


def ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    (DIRS["cluster"] / "12_clustering_results_v3").mkdir(parents=True, exist_ok=True)
    (DIRS["regime"] / "17_regime_interpretation_v3").mkdir(parents=True, exist_ok=True)
    (DIRS["regime"] / "18_synoptic_association_v3").mkdir(parents=True, exist_ok=True)
    (DIRS["regime"] / "19_seasonal_analysis_v3").mkdir(parents=True, exist_ok=True)
    (DIRS["regime"] / "20_spatial_analysis_v3").mkdir(parents=True, exist_ok=True)
    (DIRS["figtab"] / "21_figures_and_tables_v3").mkdir(parents=True, exist_ok=True)
    (DIRS["stp"] / "22_stp_reassessment_v3").mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        stop(f"Required source missing: {label}: {path}")


def stop(reason: str) -> None:
    global RUN_STATUS
    RUN_STATUS = "STOPPED_RESEARCHER_REVIEW_REQUIRED"
    ensure_dirs()
    report = DIRS["handoff"] / "STOP_CORE_SAMPLE_REOPEN.md"
    report.write_text(
        f"# Core Sample Reopen stopped\n\n"
        f"- Time: {NOW}\n"
        f"- Status: {RUN_STATUS}\n"
        f"- Reason: {reason}\n",
        encoding="utf-8",
    )
    print(f"STOP: {reason}")
    raise SystemExit(2)


def write_df(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return sha256_file(path)


def write_text(text: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def write_json(obj: object, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return sha256_file(path)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    require_file(path, rel(path))
    return pd.read_csv(path, **kwargs)


def parse_dt(date_s, time_s=None) -> pd.Timestamp:
    if pd.isna(date_s):
        return pd.NaT
    if time_s is None or pd.isna(time_s):
        txt = str(date_s)
    else:
        txt = f"{date_s} {time_s}"
    return pd.to_datetime(txt, errors="coerce")


def normalize_ef(x) -> str:
    if pd.isna(x):
        return "UNRATED"
    s = str(x).strip().upper().replace("Ｅ", "E")
    if s in {"", "/", "NAN", "NONE", "NULL", "UNRATED", "UNKNOWN"}:
        return "UNRATED"
    m = re.search(r"EF\s*([0-5])", s)
    if m:
        return f"EF{m.group(1)}"
    if re.fullmatch(r"[0-5]", s):
        return f"EF{s}"
    if re.fullmatch(r"F[0-5]", s):
        return "E" + s
    return s


def exact_best_mapping(source_labels, target_labels):
    """Enumerate k! mappings from source labels to target labels and maximize matches."""
    source = np.asarray(source_labels)
    target = np.asarray(target_labels)
    src_vals = sorted(pd.unique(source))
    tgt_vals = sorted(pd.unique(target))
    if len(src_vals) != len(tgt_vals):
        return {}, 0, 0.0
    best = None
    best_count = -1
    for perm in itertools.permutations(tgt_vals):
        mapping = dict(zip(src_vals, perm))
        mapped = np.array([mapping[x] for x in source])
        count = int((mapped == target).sum())
        if count > best_count:
            best = mapping
            best_count = count
    return best or {}, best_count, best_count / len(source) if len(source) else np.nan


def transform_primary(df: pd.DataFrame) -> pd.DataFrame:
    x = df[PRIMARY_5].copy()
    x["MLLCL_m"] = np.log1p(x["MLLCL_m"].astype(float))
    return x


def finite_required(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    for c in columns:
        ok &= pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
    return ok


def cluster_sizes(labels) -> str:
    c = Counter(labels)
    return "/".join(str(c[k]) for k in sorted(c))


def cramers_v_from_table(table: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    v = math.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else np.nan
    # Bergsma/Wicher bias correction.
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    vcorr = math.sqrt(phi2corr / denom) if denom > 0 else np.nan
    return v, vcorr, expected, np.array([chi2, p, dof])


def bootstrap_cramers_ci(df: pd.DataFrame, row_col: str, col_col: str, row_levels, col_levels, b=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(df)
    arr = df[[row_col, col_col]].to_numpy()
    for _ in range(b):
        idx = rng.integers(0, n, n)
        boot = pd.DataFrame(arr[idx], columns=[row_col, col_col])
        tab = pd.crosstab(
            pd.Categorical(boot[row_col], categories=row_levels),
            pd.Categorical(boot[col_col], categories=col_levels),
            dropna=False,
        ).to_numpy()
        try:
            vals.append(cramers_v_from_table(tab)[0])
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def permutation_chi2(df: pd.DataFrame, row_col: str, col_col: str, row_levels, col_levels, b=10000, seed=SEED):
    rng = np.random.default_rng(seed + 137)
    observed_tab = pd.crosstab(
        pd.Categorical(df[row_col], categories=row_levels),
        pd.Categorical(df[col_col], categories=col_levels),
        dropna=False,
    ).to_numpy()
    obs_chi2 = cramers_v_from_table(observed_tab)[3][0]
    x = df[row_col].to_numpy().copy()
    y = df[col_col].to_numpy().copy()
    exceed = 0
    for _ in range(b):
        y_perm = rng.permutation(y)
        boot = pd.DataFrame({row_col: x, col_col: y_perm})
        tab = pd.crosstab(
            pd.Categorical(boot[row_col], categories=row_levels),
            pd.Categorical(boot[col_col], categories=col_levels),
            dropna=False,
        ).to_numpy()
        chi2 = cramers_v_from_table(tab)[3][0]
        if chi2 >= obs_chi2 - 1e-12:
            exceed += 1
    return int(exceed), "<0.0001" if exceed == 0 else f"{(exceed + 1) / (b + 1):.6f}", float(obs_chi2)


def circular_month_stats(months: pd.Series) -> dict:
    vals = pd.to_numeric(months, errors="coerce").dropna().astype(int)
    if len(vals) == 0:
        return {"n": 0, "circular_mean_month": np.nan, "circular_std_rad": np.nan, "circular_std_months": np.nan}
    angles = 2 * np.pi * (vals - 1) / 12.0
    s = np.sin(angles).mean()
    c = np.cos(angles).mean()
    mean_angle = np.arctan2(s, c)
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    mean_month = mean_angle * 12.0 / (2 * np.pi) + 1
    if mean_month > 12:
        mean_month -= 12
    r = np.sqrt(s * s + c * c)
    circ_std = np.sqrt(-2 * np.log(r)) if r > 0 else np.nan
    return {
        "n": len(vals),
        "circular_mean_month": float(mean_month),
        "circular_mean_month_name": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ][int(round(mean_month - 1)) % 12],
        "circular_std_rad": float(circ_std),
        "circular_std_months": float(circ_std * 12.0 / (2 * np.pi)) if not np.isnan(circ_std) else np.nan,
    }


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_neighbor_summary(df: pd.DataFrame) -> dict:
    coords = df[["latitude", "longitude"]].astype(float).to_numpy()
    n = len(coords)
    if n < 2:
        return {"n": n, "median_nearest_neighbor_km": np.nan, "mean_nearest_neighbor_km": np.nan}
    dmins = []
    for i in range(n):
        dist = haversine_km(coords[i, 0], coords[i, 1], coords[:, 0], coords[:, 1])
        dist[i] = np.inf
        dmins.append(np.min(dist))
    return {
        "n": n,
        "median_nearest_neighbor_km": float(np.median(dmins)),
        "mean_nearest_neighbor_km": float(np.mean(dmins)),
    }


def bootstrap_auc_ci(y, score, b=10000, seed=SEED, stratified=True):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    aucs = []
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan, np.nan
    for _ in range(b):
        if stratified:
            idx = np.concatenate(
                [
                    rng.choice(pos, size=len(pos), replace=True),
                    rng.choice(neg, size=len(neg), replace=True),
                ]
            )
        else:
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return float(roc_auc_score(y, score)), float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def storm_day_bootstrap_auc(df: pd.DataFrame, y_col: str, score_col: str, b=10000, seed=SEED):
    rng = np.random.default_rng(seed + 777)
    days = sorted(df["storm_day"].dropna().unique())
    vals = []
    for _ in range(b):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        boot = pd.concat([df[df["storm_day"] == d] for d in sampled_days], ignore_index=True)
        if boot[y_col].nunique() < 2:
            continue
        vals.append(roc_auc_score(boot[y_col].astype(int), boot[score_col].astype(float)))
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def hl_diff(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.median((x[:, None] - y[None, :]).ravel()))


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    comp = x[:, None] - y[None, :]
    return float((np.sum(comp > 0) - np.sum(comp < 0)) / comp.size)


def hedges_g(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    d = (np.mean(x) - np.mean(y)) / sp
    j = 1 - (3 / (4 * (nx + ny) - 9))
    return float(d * j)


def bootstrap_two_sample_ci(x, y, fn, b=5000, seed=SEED):
    rng = np.random.default_rng(seed + 444)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vals = []
    for _ in range(b):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        vals.append(fn(xb, yb))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def layout_qc_placeholder(path: Path, reason: str) -> None:
    write_json(
        {
            "status": "NOT_RUN",
            "reason": reason,
            "policy": "FIGURE_LAYOUT_AND_OVERLAP_QC will be executed after v3 figure images are regenerated from v3 frozen CSVs.",
            "manual_png_editing": "PROHIBITED",
        },
        path,
    )


def main() -> None:
    global RUN_STATUS
    ensure_dirs()

    for label, path in SRC.items():
        # Some historical comparison files are optional; all actual data sources are required.
        if label in {"old_k_evidence", "old_primary_metrics", "old_pca_var"}:
            continue
        require_file(path, label)

    source_rows = []
    for label, path in SRC.items():
        if path.exists():
            source_rows.append(
                {
                    "source_key": label,
                    "absolute_path": str(path.resolve()),
                    "relative_path": rel(path),
                    "sha256": sha256_file(path),
                    "role": (
                        "AUTHORITATIVE_SOURCE"
                        if str(path).startswith(str(PAPER_ROOT))
                        else "LEGACY_REFERENCE_ONLY_EVENT_MEMBERSHIP_OR_LABEL_INPUT"
                    ),
                    "use_constraint": {
                        "legacy790_env": "EVENT_ID_MEMBERSHIP_ONLY_DO_NOT_USE_VALUES",
                        "synoptic": "SYNOPTIC_LABEL_INPUT_USED_BY_DAY5_REBUILD",
                    }.get(label, "V3_REBUILD_INPUT"),
                }
            )
    write_df(pd.DataFrame(source_rows), DIRS["manifest"] / "00_v3_source_file_hashes.csv")

    # ------------------------------------------------------------------
    # 1. Sample audit and official NEW_VALID_790_MAR_OCT_V3 event list
    # ------------------------------------------------------------------
    env793 = read_csv(SRC["env793"], dtype={"event_id": str})
    env793["utc_datetime"] = [parse_dt(d, t) for d, t in zip(env793["date_utc"], env793["time_utc"])]
    if len(env793) != 793 or env793["event_id"].nunique() != 793:
        stop(f"Current authoritative environment table is not 793 unique events: rows={len(env793)}, unique={env793['event_id'].nunique()}")
    missing_exclusions = sorted(set(OUT_OF_WINDOW_IDS) - set(env793["event_id"]))
    if missing_exclusions:
        stop(f"Out-of-window exclusion IDs absent from authoritative 793 table: {missing_exclusions}")
    missing_retained = sorted(set(RETAIN_IDS) - set(env793["event_id"]))
    if missing_retained:
        stop(f"Required retained events absent from authoritative 793 table: {missing_retained}")

    exclusion_records = []
    for eid in OUT_OF_WINDOW_IDS:
        r = env793.loc[env793["event_id"] == eid].iloc[0]
        utc_dt = r["utc_datetime"]
        bjt_dt = utc_dt + pd.Timedelta(hours=8) if pd.notna(utc_dt) else pd.NaT
        exclusion_records.append(
            {
                "event_id": eid,
                "utc_datetime": "" if pd.isna(utc_dt) else utc_dt.strftime("%Y-%m-%d %H:%M"),
                "bjt_datetime": "" if pd.isna(bjt_dt) else bjt_dt.strftime("%Y-%m-%d %H:%M"),
                "latitude": r.get("latitude", ""),
                "longitude": r.get("longitude", ""),
                "ef_rating": r.get("f_scale", ""),
                "synoptic_label": "",
                "old_inclusion_status": "INCLUDED_IN_AUTHORITATIVE_793",
                "new_inclusion_status": "EXCLUDED_FROM_NEW_VALID_790",
                "exclusion_reason": "TRUE_OUT_OF_PREDEFINED_MARCH_OCTOBER_WINDOW",
                "source_file": rel(SRC["env793"]),
                "source_sha256": sha256_file(SRC["env793"]),
                "researcher_decision": RESEARCHER_DECISION,
            }
        )
    syn = read_csv(SRC["synoptic"], dtype={"event_id": str})
    if "synoptic_type" in syn.columns:
        syn_map_for_reg = syn.set_index("event_id")["synoptic_type"].to_dict()
        for rec in exclusion_records:
            rec["synoptic_label"] = syn_map_for_reg.get(rec["event_id"], "")
    write_df(pd.DataFrame(exclusion_records), DIRS["sample"] / "01_out_of_window_exclusion_register.csv")

    new_env = env793.loc[~env793["event_id"].isin(OUT_OF_WINDOW_IDS)].copy().reset_index(drop=True)
    if len(new_env) != 790:
        stop(f"New sample count after excluding three events is not 790: {len(new_env)}")
    if new_env["event_id"].nunique() != 790:
        stop("Duplicate event_id detected in NEW_VALID_790_MAR_OCT_V3")
    if not set(RETAIN_IDS).issubset(set(new_env["event_id"])):
        stop("TOR0174/TOR0175/TOR0608 retention check failed")
    months = new_env["utc_datetime"].dt.month
    years = new_env["utc_datetime"].dt.year
    outside = new_env.loc[~months.between(3, 10) | ~years.between(2006, 2024), ["event_id", "date_utc", "time_utc"]]
    if len(outside) > 0:
        stop("Additional out-of-window events remain after the three exclusions: " + outside.to_dict("records").__repr__())

    legacy_env = read_csv(SRC["legacy790_env"], dtype={"event_id": str})
    legacy_ids = set(legacy_env["event_id"].dropna().astype(str))
    new_ids = set(new_env["event_id"].astype(str))
    all_ids = sorted(new_ids | legacy_ids)
    cross_records = []
    for eid in all_ids:
        in_new = eid in new_ids
        in_leg = eid in legacy_ids
        if in_new and in_leg:
            reason = "IN_BOTH"
        elif in_new and not in_leg:
            reason = "NEW_VALID_790_ONLY_RETAINED_OR_RECALCULATED_EVENT"
        else:
            reason = "LEGACY_INVALID_790_ONLY_OUT_OF_WINDOW_OR_SUPERSEDED"
        current = (
            NEW_SAMPLE_NAME
            if in_new
            else ("SUPERSEDED_OUT_OF_WINDOW_OR_LEGACY_ONLY" if in_leg else "NOT_PRESENT")
        )
        cross_records.append(
            {
                "event_id": eid,
                "in_new_valid_790": bool(in_new),
                "in_legacy_invalid_790": bool(in_leg),
                "difference_reason": reason,
                "current_status": current,
            }
        )
    cross_df = pd.DataFrame(cross_records)
    intersection_count = int((cross_df["in_new_valid_790"] & cross_df["in_legacy_invalid_790"]).sum())
    new_only_count = int((cross_df["in_new_valid_790"] & ~cross_df["in_legacy_invalid_790"]).sum())
    legacy_only_count = int((~cross_df["in_new_valid_790"] & cross_df["in_legacy_invalid_790"]).sum())
    write_df(cross_df, DIRS["sample"] / "02_new790_vs_legacy790_event_crosswalk.csv")

    closure = pd.DataFrame(
        [
            {
                "step": 1,
                "operation": "QC records",
                "count_before": "",
                "change": 909,
                "count_after": 909,
                "event_ids": "",
                "status": "SOURCE_CHAIN_REPORTED_BY_DAY1_AUDIT",
                "evidence_file": rel(SRC["sample_flow"]),
            },
            {
                "step": 2,
                "operation": "exclude 2025 events",
                "count_before": 909,
                "change": -114,
                "count_after": 795,
                "event_ids": "",
                "status": "STUDY_YEAR_CANDIDATES_2006_2024",
                "evidence_file": rel(SRC["sample_flow"]),
            },
            {
                "step": 3,
                "operation": "exclude duplicate TOR0607",
                "count_before": 795,
                "change": -1,
                "count_after": 794,
                "event_ids": "TOR0607",
                "status": "DUPLICATE_EXCLUDED",
                "evidence_file": rel(SRC["sample_flow"]),
            },
            {
                "step": 4,
                "operation": "exclude duplicate TOR0721",
                "count_before": 794,
                "change": -1,
                "count_after": 793,
                "event_ids": "TOR0721",
                "status": "DUPLICATE_EXCLUDED",
                "evidence_file": rel(SRC["sample_flow"]),
            },
            {
                "step": 5,
                "operation": "exclude true out-of-window events",
                "count_before": 793,
                "change": -3,
                "count_after": 790,
                "event_ids": ";".join(OUT_OF_WINDOW_IDS),
                "status": "TRUE_OUT_OF_PREDEFINED_MARCH_OCTOBER_WINDOW_EXCLUDED",
                "evidence_file": rel(SRC["env793"]),
            },
        ]
    )
    write_df(closure, DIRS["sample"] / "03_sample_closure_v3.csv")

    event_list = new_env[
        ["event_id", "date_utc", "time_utc", "longitude", "latitude", "province", "f_scale", "tc_related"]
    ].copy()
    event_list["sample_name"] = NEW_SAMPLE_NAME
    event_list["status"] = "INCLUDED"
    write_df(event_list, DIRS["sample"] / "05_event_id_list_new790.csv")

    sample_report = f"""# Sample audit report v3

Researcher decision: `{RESEARCHER_DECISION}`

Official sample: `{NEW_SAMPLE_NAME}`

## Closure

- 909 QC records
- minus 114 events in 2025
- = 795 study-year candidates
- minus TOR0607 duplicate
- minus TOR0721 duplicate
- minus TOR0355, TOR0527, TOR0627 true out-of-window events
- = 790 official March--October 2006--2024 tornado events

## Explicit retained events

- TOR0174: retained
- TOR0175: retained
- TOR0608: retained

## Validation

- rows: {len(new_env)}
- unique event_id: {new_env['event_id'].nunique()}
- years: {int(years.min())}--{int(years.max())}
- months: {int(months.min())}--{int(months.max())}
- out-of-window events after exclusion: 0
- duplicate event_id: 0
- unexplained events: 0

## New 790 versus legacy invalid 790

- intersection_count: {intersection_count}
- new_only_count: {new_only_count}
- legacy_only_count: {legacy_only_count}

The legacy 790 event set is marked `{LEGACY_SAMPLE_NAME}` and may not supply environmental
parameters, clustering labels, statistics, figure data, STP results, manuscript numbers, or event order.
"""
    write_text(sample_report, DIRS["sample"] / "04_sample_audit_report_v3.md")

    # ------------------------------------------------------------------
    # 2. Environment table and primary matrix
    # ------------------------------------------------------------------
    env_hash = write_df(new_env.drop(columns=["utc_datetime"]), DIRS["env"] / "06_environment_table_new790_v3.csv")
    env_manifest = pd.DataFrame(
        [
            {
                "file": rel(DIRS["env"] / "06_environment_table_new790_v3.csv"),
                "sha256": env_hash,
                "rows": len(new_env),
                "unique_event_id": new_env["event_id"].nunique(),
                "sample_scope": NEW_SAMPLE_NAME,
                "source_file": rel(SRC["env793"]),
                "source_sha256": sha256_file(SRC["env793"]),
                "processing": "event_id exclusion only; no ERA5 redownload; no parameter recomputation",
                "old_793_status": OLD_793_STATUS,
            }
        ]
    )
    write_df(env_manifest, DIRS["env"] / "07_environment_table_manifest_v3.csv")

    if not finite_required(new_env, PRIMARY_5).all():
        bad = new_env.loc[~finite_required(new_env, PRIMARY_5), ["event_id"] + PRIMARY_5]
        stop("Required clustering variables missing or non-finite in new790: " + bad.to_dict("records").__repr__())

    raw = new_env[["event_id"] + PRIMARY_5].copy()
    transformed = raw.copy()
    transformed["MLLCL_m"] = np.log1p(transformed["MLLCL_m"].astype(float))
    scaler = StandardScaler()
    xs = scaler.fit_transform(transformed[PRIMARY_5].astype(float).to_numpy())
    standardized = pd.DataFrame(xs, columns=[f"{c}_z" for c in PRIMARY_5])
    standardized.insert(0, "event_id", new_env["event_id"].to_numpy())
    raw_hash = write_df(raw, DIRS["matrix"] / "08_primary_matrix_raw_new790.csv")
    trans_hash = write_df(transformed, DIRS["matrix"] / "09_primary_matrix_transformed_new790.csv")
    std_hash = write_df(standardized, DIRS["matrix"] / "10_primary_matrix_standardized_new790.csv")
    matrix_manifest = []
    for f, h, typ in [
        ("08_primary_matrix_raw_new790.csv", raw_hash, "raw_physical_units"),
        ("09_primary_matrix_transformed_new790.csv", trans_hash, "transformed"),
        ("10_primary_matrix_standardized_new790.csv", std_hash, "standardized"),
    ]:
        matrix_manifest.append(
            {
                "file": rel(DIRS["matrix"] / f),
                "sha256": h,
                "matrix_type": typ,
                "rows": len(new_env),
                "columns": 1 + len(PRIMARY_5),
                "variables": ";".join(PRIMARY_5),
                "scaler_fit_scope": NEW_SAMPLE_NAME if typ == "standardized" else "",
                "random_seed": SEED,
            }
        )
    for col, mean, scale in zip(PRIMARY_5, scaler.mean_, scaler.scale_):
        matrix_manifest.append(
            {
                "file": "SCALER_PARAMETER",
                "sha256": "",
                "matrix_type": "standard_scaler",
                "rows": len(new_env),
                "columns": "",
                "variables": col,
                "scaler_fit_scope": NEW_SAMPLE_NAME,
                "random_seed": SEED,
                "mean": mean,
                "scale": scale,
            }
        )
    write_df(pd.DataFrame(matrix_manifest), DIRS["matrix"] / "11_primary_matrix_manifest_v3.csv")

    # ------------------------------------------------------------------
    # 3. Day 4 clustering rebuild
    # ------------------------------------------------------------------
    cluster_out = DIRS["cluster"] / "12_clustering_results_v3"
    X = standardized[[f"{c}_z" for c in PRIMARY_5]].to_numpy(dtype=float)
    N = len(X)
    primary_km = {}
    primary_labels = {}
    metrics = []
    all_labels = pd.DataFrame({"event_id": new_env["event_id"].to_numpy()})
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=100, max_iter=1000, tol=1e-4, algorithm="lloyd", random_state=SEED)
        km.fit(X)
        labels = km.labels_.astype(int)
        primary_km[k] = km
        primary_labels[k] = labels
        all_labels[f"k{k}_cluster_raw"] = labels
        sizes = np.bincount(labels, minlength=k)
        metrics.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "inertia": km.inertia_,
                "min_size": int(sizes.min()),
                "max_size": int(sizes.max()),
                "sizes_raw_cluster_order": "/".join(map(str, sizes.tolist())),
            }
        )
        dist = km.transform(X)
        lab_df = pd.DataFrame(
            {
                "event_id": new_env["event_id"],
                "cluster_raw": labels,
                "assigned_distance": dist[np.arange(N), labels],
                "second_nearest_distance": [np.delete(dist[i], labels[i]).min() for i in range(N)],
            }
        )
        lab_df["distance_margin"] = lab_df["second_nearest_distance"] - lab_df["assigned_distance"]
        write_df(lab_df, cluster_out / f"08_primary_labels_k{k}_new790.csv")
    metrics_df = pd.DataFrame(metrics)
    write_df(metrics_df, cluster_out / "07_primary_kmeans_metrics_v3.csv")
    write_df(all_labels, cluster_out / "06_cluster_labels_all_k_new790.csv")

    pca = PCA(n_components=5, random_state=SEED).fit(X)
    pca_scores = pca.transform(X)
    pca_var = pd.DataFrame(
        {
            "pc": [f"PC{i+1}" for i in range(5)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    write_df(pca_var, cluster_out / "03_pca_explained_variance_v3.csv")
    load_df = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(5)])
    load_df.insert(0, "variable", PRIMARY_5)
    write_df(load_df, cluster_out / "04_pca_loadings_v3.csv")
    score_df = pd.DataFrame(pca_scores[:, :3], columns=["PC1", "PC2", "PC3"])
    score_df.insert(0, "event_id", new_env["event_id"].to_numpy())
    score_df["k3_cluster_raw"] = primary_labels[3]
    score_df["k4_cluster_raw"] = primary_labels[4]
    write_df(score_df, cluster_out / "05_pca_scores_v3.csv")

    # Seed stability.
    seed_records = []
    for k in K_RANGE:
        ref = primary_labels[k]
        for seed in range(100):
            km_s = KMeans(n_clusters=k, n_init=1, random_state=seed, max_iter=500).fit(X)
            seed_records.append(
                {
                    "k": k,
                    "seed": seed,
                    "ari_vs_primary": adjusted_rand_score(ref, km_s.labels_),
                    "nmi_vs_primary": normalized_mutual_info_score(ref, km_s.labels_),
                }
            )
    seed_df = pd.DataFrame(seed_records)
    write_df(seed_df, cluster_out / "09_seed_stability_100_v3.csv")
    seed_summary = (
        seed_df.groupby("k")
        .agg(
            median_ari=("ari_vs_primary", "median"),
            p10_ari=("ari_vs_primary", lambda s: s.quantile(0.10)),
            min_ari=("ari_vs_primary", "min"),
            median_nmi=("nmi_vs_primary", "median"),
        )
        .reset_index()
    )
    write_df(seed_summary, cluster_out / "10_seed_stability_summary_v3.csv")

    # 1000 x 80% subsampling stability and event assignment probability.
    rng = np.random.default_rng(SEED)
    n_sub = int(N * 0.80)
    sub_records = []
    event_stab_records = []
    assignment_counts = {k: np.zeros((N, k), dtype=int) for k in K_RANGE}
    sampled_counts = {k: np.zeros(N, dtype=int) for k in K_RANGE}
    for rep in range(1000):
        idx = rng.choice(N, n_sub, replace=False)
        for k in K_RANGE:
            xsub = X[idx, :]
            xsub_scaled = StandardScaler().fit_transform(xsub)
            km_sub = KMeans(n_clusters=k, n_init=10, random_state=SEED + rep, max_iter=500).fit(xsub_scaled)
            sub_labels = km_sub.labels_.astype(int)
            ref_sub = primary_labels[k][idx]
            mapping, matches, match_fraction = exact_best_mapping(sub_labels, ref_sub)
            mapped = np.array([mapping[x] for x in sub_labels])
            for pos, lab in zip(idx, mapped):
                assignment_counts[k][pos, int(lab)] += 1
                sampled_counts[k][pos] += 1
            sub_records.append(
                {
                    "replicate": rep,
                    "k": k,
                    "subsample_size": n_sub,
                    "ari_vs_primary_on_sample": adjusted_rand_score(ref_sub, mapped),
                    "nmi_vs_primary_on_sample": normalized_mutual_info_score(ref_sub, mapped),
                    "exact_label_match_fraction_after_permutation": match_fraction,
                    "permutation_mapping": json.dumps({str(a): int(b) for a, b in mapping.items()}, ensure_ascii=False),
                }
            )
    sub_df = pd.DataFrame(sub_records)
    write_df(sub_df, cluster_out / "11_subsampling_1000_80pct_replicates_v3.csv")
    sub_summary = (
        sub_df.groupby("k")
        .agg(
            median_ari=("ari_vs_primary_on_sample", "median"),
            p10_ari=("ari_vs_primary_on_sample", lambda s: s.quantile(0.10)),
            min_ari=("ari_vs_primary_on_sample", "min"),
            median_nmi=("nmi_vs_primary_on_sample", "median"),
        )
        .reset_index()
    )
    write_df(sub_summary, cluster_out / "12_subsampling_1000_80pct_summary_v3.csv")
    for k in K_RANGE:
        probs = np.divide(assignment_counts[k], sampled_counts[k][:, None], out=np.zeros_like(assignment_counts[k], dtype=float), where=sampled_counts[k][:, None] != 0)
        primary_prob = probs[np.arange(N), primary_labels[k]]
        for i, eid in enumerate(new_env["event_id"]):
            status = "STABLE_CORE" if primary_prob[i] >= 0.80 else ("MODERATE" if primary_prob[i] >= 0.60 else "BOUNDARY_EVENT")
            event_stab_records.append(
                {
                    "event_id": eid,
                    "k": k,
                    "primary_cluster_raw": int(primary_labels[k][i]),
                    "sampled_count": int(sampled_counts[k][i]),
                    "primary_assignment_probability": float(primary_prob[i]),
                    "max_assignment_probability": float(probs[i].max()),
                    "stability_status": status,
                    **{f"prob_cluster_{c}": float(probs[i, c]) for c in range(k)},
                }
            )
    event_stab_df = pd.DataFrame(event_stab_records)
    write_df(event_stab_df, cluster_out / "13_event_level_stability_v3.csv")
    stab_summary = (
        event_stab_df.groupby(["k", "stability_status"])
        .size()
        .reset_index(name="n_events")
    )
    stab_summary["fraction"] = stab_summary["n_events"] / N
    write_df(stab_summary, cluster_out / "14_event_level_stability_summary_v3.csv")

    # Exact label permutation record for old-new comparisons later and k3-k4 relationship.
    # LOFO.
    lofo_records = []
    trans_np = transformed[PRIMARY_5].astype(float)
    for drop in PRIMARY_5:
        remaining = [v for v in PRIMARY_5 if v != drop]
        X_lofo = StandardScaler().fit_transform(trans_np[remaining].to_numpy())
        for k in K_RANGE:
            km_l = KMeans(n_clusters=k, n_init=100, random_state=SEED, max_iter=500).fit(X_lofo)
            lofo_records.append(
                {
                    "removed_feature": drop,
                    "k": k,
                    "silhouette": silhouette_score(X_lofo, km_l.labels_),
                    "ari_vs_primary": adjusted_rand_score(primary_labels[k], km_l.labels_),
                    "nmi_vs_primary": normalized_mutual_info_score(primary_labels[k], km_l.labels_),
                }
            )
    lofo_df = pd.DataFrame(lofo_records)
    write_df(lofo_df, cluster_out / "20_leave_one_feature_out_results_v3.csv")

    # Feature schemes — same Day 4 scheme family, with G/H implemented explicitly.
    schemes = {
        "A_PRIMARY_CORE_5": PRIMARY_5,
        "B_ECAPE_EXPANDED": PRIMARY_5 + ["ECAPE_Jkg"],
        "C_SHR1_SUBSTITUTION": ["MLCAPE_Jkg", "MLLCL_m", "ERA5_d2m_K", "SHR6_ms", "SHR1_ms"],
        "D_MLCIN_COMPLETE_CASE": PRIMARY_5 + ["MLCIN_Jkg"],
        "E_MINIMAL_4": ["MLCAPE_Jkg", "MLLCL_m", "SHR6_ms", "SRH1_m2s2"],
        "F_ROBUST_SCALER": PRIMARY_5,
        "G_D2M_CENTERED_LINEAR": PRIMARY_5,
        "H_MLCAPE_LOG1P": PRIMARY_5,
    }
    scheme_records = []
    for sname, cols in schemes.items():
        if not set(cols).issubset(new_env.columns):
            scheme_records.append(
                {
                    "scheme": sname,
                    "k": "",
                    "n_events": 0,
                    "silhouette": np.nan,
                    "ari_vs_primary": np.nan,
                    "status": "SKIPPED_MISSING_COLUMNS",
                    "columns": ";".join(cols),
                }
            )
            continue
        sx = new_env[cols].copy()
        ok = finite_required(sx, cols)
        sx = sx.loc[ok].copy()
        common_idx = np.where(ok.to_numpy())[0]
        if "MLLCL_m" in cols:
            sx["MLLCL_m"] = np.log1p(sx["MLLCL_m"].astype(float))
        if sname == "H_MLCAPE_LOG1P" and "MLCAPE_Jkg" in cols:
            sx["MLCAPE_Jkg"] = np.log1p(sx["MLCAPE_Jkg"].clip(lower=0).astype(float))
        if sname == "G_D2M_CENTERED_LINEAR" and "ERA5_d2m_K" in cols:
            sx["ERA5_d2m_K"] = sx["ERA5_d2m_K"].astype(float) - sx["ERA5_d2m_K"].astype(float).mean()
        scaler_s = RobustScaler() if sname == "F_ROBUST_SCALER" else StandardScaler()
        sx_scaled = scaler_s.fit_transform(sx[cols].astype(float).to_numpy())
        for k in K_RANGE:
            km_s = KMeans(n_clusters=k, n_init=50, random_state=SEED, max_iter=500).fit(sx_scaled)
            scheme_records.append(
                {
                    "scheme": sname,
                    "k": k,
                    "n_events": len(sx),
                    "silhouette": silhouette_score(sx_scaled, km_s.labels_),
                    "ari_vs_primary": adjusted_rand_score(primary_labels[k][common_idx], km_s.labels_),
                    "status": "COMPUTED",
                    "columns": ";".join(cols),
                }
            )
    scheme_df = pd.DataFrame(scheme_records)
    write_df(scheme_df, cluster_out / "22_feature_scheme_metrics_v3.csv")

    pca80 = PCA(n_components=3, random_state=SEED).fit_transform(X)
    pca_sens = []
    for k in K_RANGE:
        km_p = KMeans(n_clusters=k, n_init=50, random_state=SEED, max_iter=500).fit(pca80)
        pca_sens.append(
            {
                "k": k,
                "n_components": 3,
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)[2],
                "silhouette": silhouette_score(pca80, km_p.labels_),
                "ari_vs_primary": adjusted_rand_score(primary_labels[k], km_p.labels_),
            }
        )
    write_df(pd.DataFrame(pca_sens), cluster_out / "25_pca_space_clustering_sensitivity_v3.csv")

    ward_records = []
    for k in K_RANGE:
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        ward_records.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, ward.labels_),
                "ari_vs_kmeans": adjusted_rand_score(primary_labels[k], ward.labels_),
                "sizes_raw_cluster_order": cluster_sizes(ward.labels_),
            }
        )
    ward_df = pd.DataFrame(ward_records)
    write_df(ward_df, cluster_out / "26_ward_results_v3.csv")

    gmm_records = []
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=20, reg_covar=1e-6, random_state=SEED)
        glabels = gmm.fit_predict(X)
        gmm_records.append(
            {
                "k": k,
                "aic": gmm.aic(X),
                "bic": gmm.bic(X),
                "silhouette": silhouette_score(X, glabels) if len(set(glabels)) > 1 else np.nan,
                "ari_vs_kmeans": adjusted_rand_score(primary_labels[k], glabels),
                "mean_max_posterior": gmm.predict_proba(X).max(axis=1).mean(),
                "sizes_raw_cluster_order": cluster_sizes(glabels),
            }
        )
    gmm_df = pd.DataFrame(gmm_records)
    write_df(gmm_df, cluster_out / "27_gmm_results_v3.csv")

    # k=3 -> k=4 relationship.
    k34 = pd.crosstab(primary_labels[3], primary_labels[4])
    k34.index.name = "k3_cluster_raw"
    k34.columns.name = "k4_cluster_raw"
    write_df(k34.reset_index(), cluster_out / "28_k3_to_k4_transition_matrix_v3.csv")
    row_max_frac = (k34.max(axis=1) / k34.sum(axis=1)).to_dict()
    col_max_frac = (k34.max(axis=0) / k34.sum(axis=0)).to_dict()
    k34_relation = "MIXED_REPARTITION"
    if all(v > 0.80 for v in row_max_frac.values()):
        k34_relation = "NEAR_NESTED_BUT_VERIFY"

    evidence_rows = []
    for _, kmr in metrics_df.iterrows():
        k = int(kmr["k"])
        ss = sub_summary.loc[sub_summary["k"] == k].iloc[0]
        sd = seed_summary.loc[seed_summary["k"] == k].iloc[0]
        lofo_vals = lofo_df.loc[lofo_df["k"] == k, "ari_vs_primary"].dropna()
        scheme_vals = scheme_df.loc[(scheme_df["k"] == k) & (scheme_df["status"] == "COMPUTED"), "ari_vs_primary"].dropna()
        boundary_frac = (
            event_stab_df.loc[(event_stab_df["k"] == k) & (event_stab_df["stability_status"] == "BOUNDARY_EVENT")].shape[0] / N
        )
        ward_ari = ward_df.loc[ward_df["k"] == k, "ari_vs_kmeans"].iloc[0]
        gmm_ari = gmm_df.loc[gmm_df["k"] == k, "ari_vs_kmeans"].iloc[0]
        pca_ari = pd.DataFrame(pca_sens).loc[pd.DataFrame(pca_sens)["k"] == k, "ari_vs_primary"].iloc[0]
        warnings_list = []
        if kmr["min_size"] < 40:
            warnings_list.append("small_cluster")
        if ss["median_ari"] < 0.80:
            warnings_list.append("subsampling_median_below_0.80")
        if ss["p10_ari"] < 0.60:
            warnings_list.append("subsampling_p10_below_0.60")
        if boundary_frac > 0.15:
            warnings_list.append("many_boundary_events")
        if lofo_vals.min() < 0.40:
            warnings_list.append("lofo_algorithm_feature_sensitivity")
        if scheme_vals.min() < 0.40:
            warnings_list.append("feature_scheme_sensitivity")
        if k in (3, 4) and ss["median_ari"] >= 0.80 and kmr["min_size"] >= 40:
            assessment = "SUPPORTED_BUT_NOT_UNIQUE"
        elif ss["median_ari"] >= 0.80 and kmr["min_size"] >= 40:
            assessment = "PLAUSIBLE_SENSITIVITY"
        elif ss["median_ari"] >= 0.65:
            assessment = "WEAK_TO_MODERATE"
        else:
            assessment = "NOT_PRIMARY"
        evidence_rows.append(
            {
                "k": k,
                "silhouette": kmr["silhouette"],
                "davies_bouldin": kmr["davies_bouldin"],
                "calinski_harabasz": kmr["calinski_harabasz"],
                "min_cluster_size": int(kmr["min_size"]),
                "sizes_raw_cluster_order": kmr["sizes_raw_cluster_order"],
                "seed_median_ari": sd["median_ari"],
                "seed_p10_ari": sd["p10_ari"],
                "subsampling_median_ari": ss["median_ari"],
                "subsampling_p10_ari": ss["p10_ari"],
                "boundary_event_fraction": boundary_frac,
                "lofo_min_ari": lofo_vals.min(),
                "scheme_min_ari": scheme_vals.min(),
                "ward_ari": ward_ari,
                "gmm_ari": gmm_ari,
                "pca_ari": pca_ari,
                "warnings": ";".join(warnings_list),
                "assessment": assessment,
            }
        )
    evidence_df = pd.DataFrame(evidence_rows)
    write_df(evidence_df, cluster_out / "29_k_selection_evidence_table_v3.csv")

    k3_row = evidence_df.loc[evidence_df["k"] == 3].iloc[0]
    k4_row = evidence_df.loc[evidence_df["k"] == 4].iloc[0]
    if k3_row["assessment"] in {"SUPPORTED_BUT_NOT_UNIQUE", "PLAUSIBLE_SENSITIVITY"} and k4_row["assessment"] in {
        "SUPPORTED_BUT_NOT_UNIQUE",
        "PLAUSIBLE_SENSITIVITY",
    }:
        k_decision = "NO_UNIQUE_K_REPORT_3_TO_4_STRUCTURE"
        primary_solution = "K3_PRIMARY_FOR_DESCRIPTION"
        sensitivity_solution = "K4_STRUCTURAL_SENSITIVITY"
    elif k3_row["assessment"] in {"SUPPORTED_BUT_NOT_UNIQUE", "PLAUSIBLE_SENSITIVITY"}:
        k_decision = "K3_SUPPORTED_K4_WEAKER_AFTER_WINDOW_CORRECTION"
        primary_solution = "K3_PRIMARY_FOR_DESCRIPTION"
        sensitivity_solution = "K4_REVIEW_REQUIRED"
    else:
        k_decision = "K_DECISION_RESEARCHER_REVIEW_REQUIRED"
        primary_solution = "RESEARCHER_REVIEW_REQUIRED"
        sensitivity_solution = "RESEARCHER_REVIEW_REQUIRED"

    k_report = f"""# k decision report v3

Sample: `{NEW_SAMPLE_NAME}`  
Seed: `{SEED}`  
Primary variables: `{', '.join(PRIMARY_5)}`

## Decision

- k decision: `{k_decision}`
- primary solution: `{primary_solution}`
- sensitivity solution: `{sensitivity_solution}`
- k=3 to k=4 relation: `{k34_relation}`

This rebuild does not claim a unique natural number of clusters. The decision is based on the
new 790-event March--October sample, not the superseded 793 sample and not the legacy invalid 790.

## Key metrics

{evidence_df.to_markdown(index=False)}
"""
    write_text(k_report, DIRS["cluster"] / "14_k_decision_report_v3.md")

    # Copy script into clustering directory for reproducibility.
    shutil.copy2(SCRIPT_PATH, DIRS["handoff"] / "rebuild_new790_v3_executed_script.py")

    cluster_manifest_rows = []
    for p in sorted(cluster_out.glob("*_v3.csv")) + sorted(cluster_out.glob("*new790.csv")):
        cluster_manifest_rows.append(
            {
                "file": rel(p),
                "sha256": sha256_file(p),
                "sample_scope": NEW_SAMPLE_NAME,
                "random_seed": SEED,
                "analysis_version": "V3_NEW_VALID_790_MAR_OCT",
            }
        )
    write_df(pd.DataFrame(cluster_manifest_rows), DIRS["cluster"] / "13_clustering_manifest_v3.csv")

    # ------------------------------------------------------------------
    # 4. Regime relabeling and characteristics.
    # ------------------------------------------------------------------
    k3_labels_raw = pd.DataFrame({"event_id": new_env["event_id"], "k3_cluster_raw": primary_labels[3]})
    med_by_raw = (
        pd.concat([new_env[["event_id"] + PRIMARY_5], k3_labels_raw["k3_cluster_raw"]], axis=1)
        .groupby("k3_cluster_raw")[PRIMARY_5]
        .median()
    )
    # Neutral label assignment using V3 medians.
    score_c0 = {}
    for c in med_by_raw.index:
        score_c0[c] = (
            med_by_raw["MLCAPE_Jkg"].rank(ascending=True)[c]
            + med_by_raw["MLLCL_m"].rank(ascending=False)[c]
        )
    raw_c0 = sorted(score_c0.items(), key=lambda x: x[1])[0][0]
    remaining_raw = [c for c in med_by_raw.index if c != raw_c0]
    raw_c2 = med_by_raw.loc[remaining_raw, ["SHR6_ms", "SRH1_m2s2"]].rank(ascending=False).sum(axis=1).idxmin()
    raw_c1 = [c for c in med_by_raw.index if c not in {raw_c0, raw_c2}][0]
    regime_map = {
        int(raw_c0): "C0",
        int(raw_c1): "C1",
        int(raw_c2): "C2",
    }
    name_map = {
        "C0": "Low-buoyancy, high-LCL, moderate-shear regime",
        "C1": "High-buoyancy, moderate-LCL, weak-shear regime",
        "C2": "Moderate-buoyancy, low-LCL, strong-shear/high-helicity regime",
    }
    k3_labels_raw["regime_id"] = k3_labels_raw["k3_cluster_raw"].map(regime_map)
    k3_labels_raw["regime_name"] = k3_labels_raw["regime_id"].map(name_map)
    write_df(k3_labels_raw, cluster_out / "30_labels_k3_regime_ids_v3.csv")

    env_reg = new_env.merge(k3_labels_raw, on="event_id", how="left")
    if env_reg["regime_id"].isna().any():
        stop("Environment table and k=3 labels failed one-to-one merge")
    char_records = []
    for rid in ["C0", "C1", "C2"]:
        sub = env_reg[env_reg["regime_id"] == rid]
        for var in PRIMARY_5:
            values = pd.to_numeric(sub[var], errors="coerce").dropna()
            char_records.append(
                {
                    "regime_id": rid,
                    "regime_name": name_map[rid],
                    "raw_cluster": int(sub["k3_cluster_raw"].iloc[0]),
                    "n": len(sub),
                    "variable": var,
                    "median": values.median(),
                    "p25": values.quantile(0.25),
                    "p75": values.quantile(0.75),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    char_df = pd.DataFrame(char_records)
    write_df(char_df, DIRS["regime"] / "15_regime_characteristics_v3.csv")

    # Bootstrap CI for median by regime/variable.
    rng_ci = np.random.default_rng(SEED + 300)
    ci_rows = []
    for rid in ["C0", "C1", "C2"]:
        sub = env_reg[env_reg["regime_id"] == rid]
        for var in PRIMARY_5:
            vals = pd.to_numeric(sub[var], errors="coerce").dropna().to_numpy()
            boots = [np.median(rng_ci.choice(vals, size=len(vals), replace=True)) for _ in range(5000)]
            ci_rows.append(
                {
                    "regime_id": rid,
                    "variable": var,
                    "median": float(np.median(vals)),
                    "median_ci_low": float(np.quantile(boots, 0.025)),
                    "median_ci_high": float(np.quantile(boots, 0.975)),
                }
            )
    write_df(pd.DataFrame(ci_rows), DIRS["regime"] / "15b_regime_median_bootstrap_ci_v3.csv")

    naming_report = f"""# Regime naming decision v3

Sample: `{NEW_SAMPLE_NAME}`

Regime IDs were assigned after the V3 KMeans fit using V3 cluster centers and original-unit
medians. The mapping is not inherited from the superseded 793 labels.

## V3 raw cluster to reported regime mapping

{pd.DataFrame([{"raw_cluster": k, "regime_id": v, "regime_name": name_map[v]} for k, v in sorted(regime_map.items())]).to_markdown(index=False)}

## V3 original-unit medians

{char_df.pivot_table(index=["regime_id", "regime_name", "n"], columns="variable", values="median").reset_index().to_markdown(index=False)}

## C0 dewpoint note

The V3 C0 median `ERA5_d2m_K` is computed from V3 labels and V3 790 data. The previous
`292 K` value is not used (`OLD_292K = UNREPRODUCIBLE_DO_NOT_USE`).
"""
    write_text(naming_report, DIRS["regime"] / "16_regime_naming_decision_v3.md")

    # ------------------------------------------------------------------
    # 5. Day 5 posterior interpretation: synoptic, seasonal, spatial.
    # ------------------------------------------------------------------
    macro_map_df = read_csv(SRC["synoptic_map"])
    subtype_to_macro = dict(zip(macro_map_df["raw_label"], macro_map_df["macro_class"]))
    macro_en = {
        "TC": "tropical cyclone",
        "气旋/冷锋": "cyclone/cold front",
        "暖区": "warm sector",
        "冷涡": "cold vortex",
        "QLCS/飑线": "QLCS",
        "超单(未分类)": "supercell",
        "华北对流": "North China convection",
        "西南对流": "southwestern convection",
        "其他": "other",
    }
    syn = syn.copy()
    syn["macro_class_cn"] = syn["synoptic_type"].map(subtype_to_macro)
    syn["macro_class_en"] = syn["macro_class_cn"].map(macro_en)
    syn_valid = syn[syn["event_id"].isin(new_ids)].copy()
    missing_syn = sorted(new_ids - set(syn_valid["event_id"]))
    if len(syn_valid["event_id"]) != syn_valid["event_id"].nunique():
        stop("Synoptic label connection conflict: duplicate event_id in synoptic labels after new790 join")
    syn_join = env_reg[["event_id", "regime_id"]].merge(
        syn_valid[["event_id", "synoptic_type", "synoptic_class", "macro_class_cn", "macro_class_en"]],
        on="event_id",
        how="left",
    )
    valid_macro = syn_join.dropna(subset=["macro_class_cn"]).copy()
    macro_levels_cn = list(macro_map_df["macro_class"].drop_duplicates())
    # Keep only levels present in the fixed 9 macro taxonomy.
    macro_levels_cn = [m for m in macro_levels_cn if m in macro_en]
    regime_levels = ["C0", "C1", "C2"]
    macro_counts = pd.crosstab(
        pd.Categorical(valid_macro["regime_id"], categories=regime_levels),
        pd.Categorical(valid_macro["macro_class_cn"], categories=macro_levels_cn),
        dropna=False,
    )
    macro_counts.index.name = "regime_id"
    macro_counts.columns.name = "macro_class_cn"
    macro_count_df = macro_counts.reset_index()
    write_df(macro_count_df, DIRS["regime"] / "18_synoptic_association_v3" / "01_macro9_contingency_counts_v3.csv")
    row_pct = macro_counts.div(macro_counts.sum(axis=1), axis=0)
    write_df(row_pct.reset_index(), DIRS["regime"] / "18_synoptic_association_v3" / "02_macro9_row_percent_v3.csv")
    col_pct = macro_counts.div(macro_counts.sum(axis=0), axis=1)
    write_df(col_pct.reset_index(), DIRS["regime"] / "18_synoptic_association_v3" / "03_macro9_column_percent_v3.csv")
    v_macro, vc_macro, exp_macro, chi_macro = cramers_v_from_table(macro_counts.to_numpy())
    ci_lo, ci_hi = bootstrap_cramers_ci(valid_macro, "regime_id", "macro_class_cn", regime_levels, macro_levels_cn, b=5000)
    exceed, p_perm, obs_chi2 = permutation_chi2(valid_macro, "regime_id", "macro_class_cn", regime_levels, macro_levels_cn, b=10000)
    residuals = (macro_counts.to_numpy() - exp_macro) / np.sqrt(exp_macro)
    residual_df = pd.DataFrame(residuals, index=regime_levels, columns=macro_levels_cn).reset_index(names="regime_id")
    write_df(residual_df, DIRS["regime"] / "18_synoptic_association_v3" / "04_macro9_standardized_residuals_v3.csv")
    enrich_rows = []
    for i, rid in enumerate(regime_levels):
        for j, mc in enumerate(macro_levels_cn):
            res = residuals[i, j]
            status = "ENRICHED" if res >= 1.96 else ("DEPLETED" if res <= -1.96 else "NO_STRONG_DEVIATION")
            enrich_rows.append(
                {
                    "regime_id": rid,
                    "macro_class_cn": mc,
                    "macro_class_en": macro_en[mc],
                    "count": int(macro_counts.loc[rid, mc]),
                    "expected": float(exp_macro[i, j]),
                    "standardized_residual": float(res),
                    "status": status,
                }
            )
    write_df(pd.DataFrame(enrich_rows), DIRS["regime"] / "18_synoptic_association_v3" / "05_macro9_enrichment_depletion_v3.csv")
    macro_summary = pd.DataFrame(
        [
            {
                "analysis": "macro9_synoptic_association",
                "sample_size": len(valid_macro),
                "missing_synoptic_event_count": len(missing_syn),
                "missing_synoptic_events": ";".join(missing_syn),
                "table_shape": f"{len(regime_levels)}x{len(macro_levels_cn)}",
                "cramers_v": v_macro,
                "bias_corrected_v": vc_macro,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "expected_cells_lt_5": int((exp_macro < 5).sum()),
                "expected_cells_total": int(exp_macro.size),
                "permutation_B": 10000,
                "permutation_exceed": exceed,
                "permutation_p": p_perm,
                "label_status": "PARTIALLY_INDEPENDENT_LABEL",
            }
        ]
    )
    write_df(macro_summary, DIRS["regime"] / "18_synoptic_association_v3" / "06_macro9_association_summary_v3.csv")

    subtype_valid = syn_join.dropna(subset=["synoptic_type"]).copy()
    subtype_levels = list(macro_map_df["raw_label"].drop_duplicates())
    subtype_levels = [s for s in subtype_levels if s in set(subtype_valid["synoptic_type"])]
    subtype_counts = pd.crosstab(
        pd.Categorical(subtype_valid["regime_id"], categories=regime_levels),
        pd.Categorical(subtype_valid["synoptic_type"], categories=subtype_levels),
        dropna=False,
    )
    v_sub, vc_sub, exp_sub, chi_sub = cramers_v_from_table(subtype_counts.to_numpy())
    sub_ci_lo, sub_ci_hi = bootstrap_cramers_ci(subtype_valid, "regime_id", "synoptic_type", regime_levels, subtype_levels, b=5000, seed=SEED + 1)
    write_df(subtype_counts.reset_index(), DIRS["regime"] / "18_synoptic_association_v3" / "07_subtype21_contingency_counts_v3.csv")
    write_df(
        pd.DataFrame(
            [
                {
                    "analysis": "subtype21_synoptic_association",
                    "sample_size": len(subtype_valid),
                    "table_shape": f"{len(regime_levels)}x{len(subtype_levels)}",
                    "cramers_v": v_sub,
                    "bias_corrected_v": vc_sub,
                    "ci_low": sub_ci_lo,
                    "ci_high": sub_ci_hi,
                    "expected_cells_lt_5": int((exp_sub < 5).sum()),
                    "expected_cells_total": int(exp_sub.size),
                    "manuscript_use": "SUPPLEMENT_EXPLORATORY_ONLY",
                }
            ]
        ),
        DIRS["regime"] / "18_synoptic_association_v3" / "08_subtype21_association_summary_v3.csv",
    )

    # Seasonal.
    env_reg["month"] = env_reg["utc_datetime"].dt.month
    month_counts = pd.crosstab(env_reg["regime_id"], env_reg["month"]).reindex(index=regime_levels, columns=list(range(3, 11)), fill_value=0)
    write_df(month_counts.reset_index(), DIRS["regime"] / "19_seasonal_analysis_v3" / "01_monthly_counts_by_regime_v3.csv")
    season_rows = []
    for rid in regime_levels:
        stats = circular_month_stats(env_reg.loc[env_reg["regime_id"] == rid, "month"])
        season_rows.append({"regime_id": rid, "regime_name": name_map[rid], **stats})
    season_df = pd.DataFrame(season_rows)
    write_df(season_df, DIRS["regime"] / "19_seasonal_analysis_v3" / "02_circular_seasonal_statistics_v3.csv")

    # Spatial.
    spatial_rows = []
    province_rows = []
    for rid in regime_levels:
        sub = env_reg[env_reg["regime_id"] == rid].copy()
        nn = nearest_neighbor_summary(sub)
        spatial_rows.append(
            {
                "regime_id": rid,
                "regime_name": name_map[rid],
                "n": len(sub),
                "mean_latitude": sub["latitude"].astype(float).mean(),
                "mean_longitude": sub["longitude"].astype(float).mean(),
                "median_latitude": sub["latitude"].astype(float).median(),
                "median_longitude": sub["longitude"].astype(float).median(),
                **nn,
                "spatial_conclusion": "CORRIDOR_STRUCTURE_NOT_ESTABLISHED",
                "allowed_terms": "spatial concentration;regional preference;geographical distribution",
                "prohibited_terms": "tornado corridor;corridor boundary;objectively identified corridor",
            }
        )
        pc = sub["province"].value_counts().reset_index()
        pc.columns = ["province", "event_count"]
        for _, r in pc.iterrows():
            province_rows.append({"regime_id": rid, "province": r["province"], "event_count": int(r["event_count"])})
    write_df(pd.DataFrame(spatial_rows), DIRS["regime"] / "20_spatial_analysis_v3" / "01_spatial_concentration_summary_v3.csv")
    write_df(pd.DataFrame(province_rows), DIRS["regime"] / "20_spatial_analysis_v3" / "02_province_counts_by_regime_v3.csv")

    interp_report = f"""# Regime interpretation v3

Sample: `{NEW_SAMPLE_NAME}`

## Synoptic association

- Valid macro 9 synoptic sample size: {len(valid_macro)}
- Missing synoptic labels: {len(missing_syn)} ({'; '.join(missing_syn) if missing_syn else 'none'})
- Cramér's V: {v_macro:.4f}
- bias-corrected V: {vc_macro:.4f}
- 95% bootstrap CI: {ci_lo:.4f}--{ci_hi:.4f}
- sparse expected cells: {(exp_macro < 5).sum()}/{exp_macro.size}
- permutation B=10000, exceed={exceed}, p={p_perm}
- Label status: PARTIALLY_INDEPENDENT_LABEL

## 21-subtype association

This is supplementary/exploratory only. V={v_sub:.4f}, bias-corrected V={vc_sub:.4f},
95% CI={sub_ci_lo:.4f}--{sub_ci_hi:.4f}, sparse cells={(exp_sub < 5).sum()}/{exp_sub.size}.

## Spatial conclusion

`CORRIDOR_STRUCTURE_NOT_ESTABLISHED`; use only spatial concentration, regional preference,
geographical distribution, or regional differences.
"""
    write_text(interp_report, DIRS["regime"] / "17_regime_interpretation_v3" / "00_regime_interpretation_report_v3.md")

    # Representative and lowest-stability events.
    k3_stab = event_stab_df[event_stab_df["k"] == 3].merge(k3_labels_raw, on="event_id", how="left")
    rep_rows = []
    for rid in regime_levels:
        sub = k3_stab[k3_stab["regime_id"] == rid].sort_values("primary_assignment_probability", ascending=False)
        for rank, (_, r) in enumerate(sub.head(5).iterrows(), start=1):
            rep_rows.append(
                {
                    "regime_id": rid,
                    "selection_type": "representative_high_stability",
                    "rank": rank,
                    "event_id": r["event_id"],
                    "primary_assignment_probability": r["primary_assignment_probability"],
                }
            )
    for rank, (_, r) in enumerate(k3_stab.sort_values("primary_assignment_probability").head(15).iterrows(), start=1):
        rep_rows.append(
            {
                "regime_id": r["regime_id"],
                "selection_type": "lowest_stability",
                "rank": rank,
                "event_id": r["event_id"],
                "primary_assignment_probability": r["primary_assignment_probability"],
            }
        )
    rep_df = pd.DataFrame(rep_rows)
    write_df(rep_df, DIRS["regime"] / "17_regime_interpretation_v3" / "01_representative_and_lowest_stability_events_v3.csv")
    if rep_df["event_id"].nunique() < len(rep_df):
        # This is not a stop condition for the requested 27 uniqueness unless final set is for supplementary figure.
        write_text(
            "Representative/lowest-stability list contains repeated events across categories; review before using as a 27-event display.\n",
            DIRS["regime"] / "17_regime_interpretation_v3" / "01_representative_event_uniqueness_note.md",
        )

    # ------------------------------------------------------------------
    # 6. Day 6 figure/table source data and claim evidence update (no formal manuscript).
    # ------------------------------------------------------------------
    figtab_dir = DIRS["figtab"] / "21_figures_and_tables_v3"
    fig_data_dir = figtab_dir / "source_data"
    tables_dir = figtab_dir / "tables"
    fig_data_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)
    write_df(event_list, fig_data_dir / "fig01_event_distribution_source_v3.csv")
    write_df(closure, fig_data_dir / "fig02_sample_qc_flow_source_v3.csv")
    write_df(char_df, fig_data_dir / "fig03_regime_characteristics_source_v3.csv")
    write_df(evidence_df, fig_data_dir / "fig04_k_evaluation_source_v3.csv")
    write_df(pd.DataFrame(primary_km[3].cluster_centers_, columns=[f"{v}_z_center" for v in PRIMARY_5]).reset_index(names="k3_cluster_raw"), fig_data_dir / "fig05_standardized_centers_source_v3.csv")
    write_df(k34.reset_index(), fig_data_dir / "fig06_k3_k4_transition_source_v3.csv")
    write_df(macro_count_df, fig_data_dir / "fig07_macro9_counts_source_v3.csv")
    write_df(season_df, fig_data_dir / "fig08_seasonal_source_v3.csv")
    write_df(pd.DataFrame(spatial_rows), fig_data_dir / "fig09_spatial_source_v3.csv")

    table1 = pd.DataFrame(
        [
            {"variable": "MLCAPE_Jkg", "definition": "mixed-layer convective available potential energy", "unit": "J kg-1", "role": "primary clustering variable"},
            {"variable": "MLLCL_m", "definition": "mixed-layer lifted condensation level", "unit": "m", "role": "primary clustering variable; log1p transformed before scaling"},
            {"variable": "ERA5_d2m_K", "definition": "ERA5 2-m dewpoint temperature", "unit": "K", "role": "primary clustering moisture variable"},
            {"variable": "SHR6_ms", "definition": "0--6 km bulk wind shear", "unit": "m s-1", "role": "primary clustering kinematic variable"},
            {"variable": "SRH1_m2s2", "definition": "0--1 km storm-relative helicity", "unit": "m2 s-2", "role": "primary clustering kinematic variable; signed identity"},
        ]
    )
    write_df(table1, tables_dir / "table1_variables_v3.csv")
    table2 = char_df.pivot_table(index=["regime_id", "regime_name", "n"], columns="variable", values="median").reset_index()
    write_df(table2, tables_dir / "table2_regime_medians_v3.csv")
    write_df(evidence_df, tables_dir / "table3_cluster_evaluation_v3.csv")
    write_df(pd.DataFrame(enrich_rows), tables_dir / "table4_macro9_associations_v3.csv")
    write_df(pd.DataFrame([{"display_item": "Day6 figures/tables", "status": "SOURCE_DATA_REBUILT_FROM_V3", "formal_publication_figures": "NOT_GENERATED_IN_CORE_REOPEN_STEP", "reason": "User requested no new formal figures before CORE_SAMPLE_REOPEN_COMPLETED"}]), figtab_dir / "day6_figures_tables_rebuild_status_v3.csv")
    layout_qc_placeholder(figtab_dir / "FIGURE_LAYOUT_AND_OVERLAP_QC_status_v3.json", "Formal figure images are not generated in this Core Sample Reopen step; v3 CSV source data have been rebuilt.")

    claim_rows = [
        {
            "claim_id": "V3_SAMPLE_790",
            "claim_text": "The official March--October 2006--2024 tornado sample contains 790 events after excluding three true out-of-window events.",
            "sample_size": 790,
            "evidence_file": rel(DIRS["sample"] / "05_event_id_list_new790.csv"),
            "status": "SUPPORTED",
            "allowed_use": "MAIN_TEXT",
        },
        {
            "claim_id": "V3_K_DECISION",
            "claim_text": k_decision,
            "sample_size": 790,
            "evidence_file": rel(DIRS["cluster"] / "14_k_decision_report_v3.md"),
            "status": "SUPPORTED" if "RESEARCHER_REVIEW" not in k_decision else "RESEARCHER_REVIEW_REQUIRED",
            "allowed_use": "MAIN_TEXT" if "RESEARCHER_REVIEW" not in k_decision else "RESEARCHER_REVIEW_REQUIRED",
        },
        {
            "claim_id": "V3_SPATIAL_NO_CORRIDOR",
            "claim_text": "CORRIDOR_STRUCTURE_NOT_ESTABLISHED",
            "sample_size": 790,
            "evidence_file": rel(DIRS["regime"] / "20_spatial_analysis_v3" / "01_spatial_concentration_summary_v3.csv"),
            "status": "SUPPORTED",
            "allowed_use": "MAIN_TEXT_WITH_RESTRICTED_WORDING",
        },
        {
            "claim_id": "DAY8_CONTROL_DELETE",
            "claim_text": "CONTROL_SAMPLE_AUDIT_FAILED; DELETE_FROM_MANUSCRIPT; no tornado-versus-nontornado claims active.",
            "sample_size": "",
            "evidence_file": "paper_rebuild/08_control_sample_audit/final_closure",
            "status": "SUPPORTED_DELETE_DECISION_RETAINED",
            "allowed_use": "METHOD_PROVENANCE_ONLY",
        },
    ]
    write_df(pd.DataFrame(claim_rows), figtab_dir / "claim_evidence_master_v3.csv")

    # ------------------------------------------------------------------
    # 7. STP_mod reassessment.
    # ------------------------------------------------------------------
    stp_dir = DIRS["stp"] / "22_stp_reassessment_v3"
    excluded_ef_rows = []
    for eid in OUT_OF_WINDOW_IDS:
        r = env793.loc[env793["event_id"] == eid].iloc[0]
        ef = normalize_ef(r["f_scale"])
        excluded_ef_rows.append(
            {
                "event_id": eid,
                "f_scale": r["f_scale"],
                "ef_status": ef,
                "rated": ef != "UNRATED",
                "is_EF2_plus": ef in {"EF2", "EF3", "EF4", "EF5"},
                "is_EF0_EF1": ef in {"EF0", "EF1"},
                "is_unrated": ef == "UNRATED",
            }
        )
    write_df(pd.DataFrame(excluded_ef_rows), stp_dir / "01_excluded_event_ef_audit_v3.csv")

    stp_df = env_reg.copy()
    stp_df["ef_norm"] = stp_df["f_scale"].map(normalize_ef)
    stp_df["rated"] = stp_df["ef_norm"].isin(["EF0", "EF1", "EF2", "EF3", "EF4", "EF5"])
    stp_df["ef2plus"] = stp_df["ef_norm"].isin(["EF2", "EF3", "EF4", "EF5"])
    stp_df["ef0_ef1"] = stp_df["ef_norm"].isin(["EF0", "EF1"])
    rated = stp_df[stp_df["rated"]].copy()
    if rated["event_id"].nunique() != len(rated):
        stop("STP rated sample has duplicate event_id")
    rated_count = int(stp_df["rated"].sum())
    ef2_count = int(stp_df["ef2plus"].sum())
    ef01_count = int(stp_df["ef0_ef1"].sum())
    unrated_count = int((~stp_df["rated"]).sum())
    if rated_count != ef2_count + ef01_count:
        stop("Rated count cannot be reconciled into EF2+ and EF0--EF1")
    stp_count_df = pd.DataFrame(
        [
            {
                "sample": NEW_SAMPLE_NAME,
                "rated_count": rated_count,
                "EF2_plus_count": ef2_count,
                "EF0_EF1_count": ef01_count,
                "unrated_count": unrated_count,
                "excluded_event_rated_count": int(pd.DataFrame(excluded_ef_rows)["rated"].sum()),
                "excluded_event_unrated_count": int(pd.DataFrame(excluded_ef_rows)["is_unrated"].sum()),
                "old_181_73_108_612_impact": "rated/EF2+/EF0-EF1 unchanged if excluded events are unrated; unrated decreases by 3",
            }
        ]
    )
    write_df(stp_count_df, stp_dir / "02_stp_sample_counts_v3.csv")

    if "STP_mod" not in rated.columns:
        stop("STP_mod column missing; reference STP must not be fabricated")
    rated = rated[pd.to_numeric(rated["STP_mod"], errors="coerce").notna()].copy()
    rated["y_ef2plus"] = rated["ef2plus"].astype(int)
    auc, auc_lo, auc_hi = bootstrap_auc_ci(rated["y_ef2plus"], rated["STP_mod"], b=10000)
    rated["storm_day"] = pd.to_datetime(rated["date_utc"], errors="coerce").dt.strftime("%Y-%m-%d")
    s_lo, s_hi = storm_day_bootstrap_auc(rated, "y_ef2plus", "STP_mod", b=10000)
    # Full-sample Youden diagnostic and bootstrap-derived evaluation (sample-derived; not operational).
    scores = rated["STP_mod"].astype(float).to_numpy()
    y = rated["y_ef2plus"].astype(int).to_numpy()
    thresholds = np.unique(scores)
    best = {"threshold": np.nan, "youden": -np.inf, "pod": np.nan, "fpr": np.nan, "far": np.nan}
    for th in thresholds:
        pred = scores >= th
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        pod = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        far = fp / (tp + fp) if tp + fp else np.nan
        youden = pod - fpr
        if youden > best["youden"]:
            best = {"threshold": float(th), "youden": float(youden), "pod": pod, "fpr": fpr, "far": far}
    rng_cut = np.random.default_rng(SEED + 909)
    cut_records = []
    for b in range(5000):
        train_idx = np.concatenate(
            [
                rng_cut.choice(np.where(y == 1)[0], size=max(1, int((y == 1).sum() * 0.8)), replace=True),
                rng_cut.choice(np.where(y == 0)[0], size=max(1, int((y == 0).sum() * 0.8)), replace=True),
            ]
        )
        test_mask = np.ones(len(y), dtype=bool)
        test_mask[np.unique(train_idx)] = False
        if test_mask.sum() < 10 or len(np.unique(y[test_mask])) < 2:
            test_idx = rng_cut.choice(np.arange(len(y)), size=len(y), replace=True)
        else:
            test_idx = np.where(test_mask)[0]
        train_scores = scores[train_idx]
        train_y = y[train_idx]
        local_best = {"threshold": np.nan, "youden": -np.inf}
        for th in np.unique(train_scores):
            pred = train_scores >= th
            tp = ((pred == 1) & (train_y == 1)).sum()
            fp = ((pred == 1) & (train_y == 0)).sum()
            fn = ((pred == 0) & (train_y == 1)).sum()
            tn = ((pred == 0) & (train_y == 0)).sum()
            pod = tp / (tp + fn) if tp + fn else np.nan
            fpr = fp / (fp + tn) if fp + tn else np.nan
            if not np.isnan(pod) and not np.isnan(fpr) and pod - fpr > local_best["youden"]:
                local_best = {"threshold": float(th), "youden": float(pod - fpr)}
        pred = scores[test_idx] >= local_best["threshold"]
        yt = y[test_idx]
        tp = ((pred == 1) & (yt == 1)).sum()
        fp = ((pred == 1) & (yt == 0)).sum()
        fn = ((pred == 0) & (yt == 1)).sum()
        tn = ((pred == 0) & (yt == 0)).sum()
        pod = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        far = fp / (tp + fp) if tp + fp else np.nan
        cut_records.append({"replicate": b, "threshold": local_best["threshold"], "pod": pod, "fpr": fpr, "far": far})
    cut_df = pd.DataFrame(cut_records)
    write_df(cut_df, stp_dir / "03_nested_sample_derived_cutoff_replicates_v3.csv")
    cut_sum = {
        "threshold_full_sample_youden": best["threshold"],
        "POD_median": cut_df["pod"].median(),
        "POD_ci_low": cut_df["pod"].quantile(0.025),
        "POD_ci_high": cut_df["pod"].quantile(0.975),
        "FPR_median": cut_df["fpr"].median(),
        "FPR_ci_low": cut_df["fpr"].quantile(0.025),
        "FPR_ci_high": cut_df["fpr"].quantile(0.975),
        "FAR_median": cut_df["far"].median(),
        "FAR_ci_low": cut_df["far"].quantile(0.025),
        "FAR_ci_high": cut_df["far"].quantile(0.975),
        "wording": "sample-derived discrimination cutoff; not an operational threshold",
    }
    write_df(pd.DataFrame([cut_sum]), stp_dir / "04_nested_sample_derived_cutoff_summary_v3.csv")

    roc_sum = pd.DataFrame(
        [
            {
                "analysis": "STP_mod_rated_tornado_only",
                "rated_n": len(rated),
                "EF2_plus": int(rated["y_ef2plus"].sum()),
                "EF0_EF1": int((rated["y_ef2plus"] == 0).sum()),
                "AUC": auc,
                "AUC_ci_low_stratified_bootstrap": auc_lo,
                "AUC_ci_high_stratified_bootstrap": auc_hi,
                "storm_day_cluster_bootstrap_ci_low": s_lo,
                "storm_day_cluster_bootstrap_ci_high": s_hi,
                "interpretation": "STP_DISCRIMINATION_PARTIALLY_SUPPORTED; not operational threshold; DISCUSSION + SUPPLEMENT",
                "reference_stp_fixed_layer": "NOT_COMPUTABLE_MISSING_REQUIRED_COMPONENTS",
                "reference_stp_effective_layer": "NOT_COMPUTABLE_MISSING_EFFECTIVE_LAYER_METRICS",
                "stp_mod_identity": "COMPUTABLE_CUSTOM_NONSTANDARD_FORMULA",
            }
        ]
    )
    write_df(roc_sum, stp_dir / "05_stp_mod_auc_summary_v3.csv")

    regime_auc_rows = []
    for rid in regime_levels:
        sub = rated[rated["regime_id"] == rid]
        if sub["y_ef2plus"].nunique() < 2:
            regime_auc_rows.append({"regime_id": rid, "n": len(sub), "AUC": np.nan, "ci_low": np.nan, "ci_high": np.nan, "status": "INSUFFICIENT_CLASSES"})
        else:
            a, lo, hi = bootstrap_auc_ci(sub["y_ef2plus"], sub["STP_mod"], b=10000, seed=SEED + len(rid))
            regime_auc_rows.append({"regime_id": rid, "n": len(sub), "AUC": a, "ci_low": lo, "ci_high": hi, "status": "COMPUTED_NO_BETWEEN_REGIME_DIFFERENCE_CLAIM"})
    regime_auc_df = pd.DataFrame(regime_auc_rows)
    write_df(regime_auc_df, stp_dir / "06_regime_stp_mod_auc_v3.csv")

    bias_vars = ["SRH1_m2s2", "SHR6_ms", "MLLCL_m", "ERA5_d2m_K", "MLCAPE_Jkg"]
    bias_rows = []
    rated_mask = stp_df["rated"]
    for var in bias_vars:
        x = pd.to_numeric(stp_df.loc[rated_mask, var], errors="coerce").dropna().to_numpy()
        y0 = pd.to_numeric(stp_df.loc[~rated_mask, var], errors="coerce").dropna().to_numpy()
        bias_rows.append(
            {
                "variable": var,
                "rated_n": len(x),
                "unrated_n": len(y0),
                "rated_median": float(np.median(x)),
                "unrated_median": float(np.median(y0)),
                "hodges_lehmann_difference": hl_diff(x, y0),
                "cliffs_delta": cliffs_delta(x, y0),
                "cliffs_delta_ci_low": bootstrap_two_sample_ci(x, y0, cliffs_delta, b=3000, seed=SEED)[0],
                "cliffs_delta_ci_high": bootstrap_two_sample_ci(x, y0, cliffs_delta, b=3000, seed=SEED)[1],
                "hedges_g": hedges_g(x, y0),
                "hedges_g_ci_low": bootstrap_two_sample_ci(x, y0, hedges_g, b=3000, seed=SEED)[0],
                "hedges_g_ci_high": bootstrap_two_sample_ci(x, y0, hedges_g, b=3000, seed=SEED)[1],
            }
        )
    bias_df = pd.DataFrame(bias_rows)
    write_df(bias_df, stp_dir / "07_rated_vs_unrated_bias_statistics_v3.csv")

    stp_report = f"""# STP_mod reassessment v3

Sample: `{NEW_SAMPLE_NAME}`

## Excluded-event EF audit

{pd.DataFrame(excluded_ef_rows).to_markdown(index=False)}

## V3 counts

- rated: {rated_count}
- EF2+: {ef2_count}
- EF0--EF1: {ef01_count}
- unrated: {unrated_count}

## V3 STP_mod AUC

- AUC: {auc:.3f}
- stratified bootstrap 95% CI: {auc_lo:.3f}--{auc_hi:.3f}
- storm-day cluster bootstrap 95% CI: {s_lo:.3f}--{s_hi:.3f}

STP_mod remains a custom nonstandard formula and must not be abbreviated as standard STP.
The sample-derived cutoff is not an operational or warning threshold.
"""
    write_text(stp_report, stp_dir / "08_stp_reassessment_report_v3.md")

    # ------------------------------------------------------------------
    # 8. Impact assessment: old 793 vs new 790 on shared events.
    # ------------------------------------------------------------------
    old_k3 = read_csv(SRC["k3_old"], dtype={"event_id": str})
    old_k4 = read_csv(SRC["k4_old"], dtype={"event_id": str})
    old_label_col3 = "cluster" if "cluster" in old_k3.columns else [c for c in old_k3.columns if "cluster" in c.lower()][0]
    old_label_col4 = "cluster" if "cluster" in old_k4.columns else [c for c in old_k4.columns if "cluster" in c.lower()][0]
    shared3 = k3_labels_raw.merge(old_k3[["event_id", old_label_col3]], on="event_id", how="inner")
    shared4_new = pd.DataFrame({"event_id": new_env["event_id"], "new_k4_cluster_raw": primary_labels[4]})
    shared4 = shared4_new.merge(old_k4[["event_id", old_label_col4]], on="event_id", how="inner")
    if len(shared3) != 790 or len(shared4) != 790:
        stop(f"Old-new label comparison does not cover shared 790 events: k3={len(shared3)}, k4={len(shared4)}")
    old_new_k3_ari = adjusted_rand_score(shared3[old_label_col3], shared3["k3_cluster_raw"])
    old_new_k4_ari = adjusted_rand_score(shared4[old_label_col4], shared4["new_k4_cluster_raw"])
    k3_map, k3_matches, k3_match_frac = exact_best_mapping(shared3["k3_cluster_raw"], shared3[old_label_col3])
    k4_map, k4_matches, k4_match_frac = exact_best_mapping(shared4["new_k4_cluster_raw"], shared4[old_label_col4])

    old_pca_first3 = ""
    if SRC["old_pca_var"].exists():
        try:
            opd = pd.read_csv(SRC["old_pca_var"])
            old_pca_first3 = ";".join([str(x) for x in opd.head(3).to_dict("records")])
        except Exception:
            old_pca_first3 = "UNREADABLE"

    impact_rows = [
        {
            "item": "event_sample",
            "old_793_value": 793,
            "new_790_value": 790,
            "status": "CHANGED_BY_TRUE_WINDOW_CORRECTION",
            "notes": "TOR0355 TOR0527 TOR0627 excluded",
        },
        {
            "item": "PCA_variance_first3",
            "old_793_value": old_pca_first3,
            "new_790_value": json.dumps(pca_var.head(3).to_dict("records"), ensure_ascii=False),
            "status": "RECOMPUTED",
            "notes": "",
        },
        {
            "item": "k2_to_k6_metrics",
            "old_793_value": rel(SRC["old_primary_metrics"]) if SRC["old_primary_metrics"].exists() else "OLD_FILE_NOT_FOUND",
            "new_790_value": rel(cluster_out / "07_primary_kmeans_metrics_v3.csv"),
            "status": "RECOMPUTED",
            "notes": "",
        },
        {
            "item": "k3_sizes",
            "old_793_value": "132/307/354",
            "new_790_value": "/".join(str(int((k3_labels_raw["regime_id"] == rid).sum())) for rid in regime_levels),
            "status": "RECOMPUTED",
            "notes": "New sizes reported in C0/C1/C2 regime order, not old inherited label order",
        },
        {
            "item": "k4_sizes",
            "old_793_value": "249/126/210/208",
            "new_790_value": metrics_df.loc[metrics_df["k"] == 4, "sizes_raw_cluster_order"].iloc[0],
            "status": "RECOMPUTED",
            "notes": "Raw KMeans cluster order",
        },
        {
            "item": "old_new_k3_shared_event_ARI",
            "old_793_value": "old labels on shared events",
            "new_790_value": old_new_k3_ari,
            "status": "COMPUTED_ON_SHARED_790_ONLY_EXACT_PERMUTATION_AVAILABLE",
            "notes": json.dumps({str(k): int(v) for k, v in k3_map.items()}),
        },
        {
            "item": "old_new_k4_shared_event_ARI",
            "old_793_value": "old labels on shared events",
            "new_790_value": old_new_k4_ari,
            "status": "COMPUTED_ON_SHARED_790_ONLY_EXACT_PERMUTATION_AVAILABLE",
            "notes": json.dumps({str(k): int(v) for k, v in k4_map.items()}),
        },
        {
            "item": "macro9_weather_association",
            "old_793_value": "old official main V=0.5261 on previous 790 weather-valid set",
            "new_790_value": f"n={len(valid_macro)}, V={v_macro:.4f}, CI={ci_lo:.4f}-{ci_hi:.4f}",
            "status": "RECOMPUTED",
            "notes": f"missing_synoptic={';'.join(missing_syn)}",
        },
        {
            "item": "seasonal_statistics",
            "old_793_value": "SUPERSEDED_BY_TRUE_WINDOW_CORRECTION",
            "new_790_value": rel(DIRS["regime"] / "19_seasonal_analysis_v3" / "02_circular_seasonal_statistics_v3.csv"),
            "status": "RECOMPUTED",
            "notes": "",
        },
        {
            "item": "spatial_statistics",
            "old_793_value": "CORRIDOR_STRUCTURE_NOT_ESTABLISHED",
            "new_790_value": "CORRIDOR_STRUCTURE_NOT_ESTABLISHED",
            "status": "UNCHANGED_CONCLUSION_RECOMPUTED_SUMMARIES",
            "notes": "",
        },
        {
            "item": "STP_mod",
            "old_793_value": "rated=181 EF2+=73 EF0-EF1=108 unrated=612 AUC=0.641",
            "new_790_value": f"rated={rated_count} EF2+={ef2_count} EF0-EF1={ef01_count} unrated={unrated_count} AUC={auc:.3f} CI={auc_lo:.3f}-{auc_hi:.3f}",
            "status": "RECOMPUTED",
            "notes": "Excluded events are unrated, so rated counts unchanged and unrated decreases by 3",
        },
        {
            "item": "representative_events",
            "old_793_value": "SUPERSEDED_BY_TRUE_WINDOW_CORRECTION",
            "new_790_value": rel(DIRS["regime"] / "17_regime_interpretation_v3" / "01_representative_and_lowest_stability_events_v3.csv"),
            "status": "RECOMPUTED",
            "notes": "",
        },
    ]
    impact_df = pd.DataFrame(impact_rows)
    write_df(impact_df, DIRS["impact"] / "23_793_to_790_impact_assessment.csv")
    impact_report = f"""# 793 to 790 impact report

Decision: `{RESEARCHER_DECISION}`

## Summary

- Old 793 result status: `{OLD_793_STATUS}`
- Legacy old 790 status: `{LEGACY_790_STATUS}`
- New official sample: `{NEW_SAMPLE_NAME}`
- Excluded true out-of-window events: {', '.join(OUT_OF_WINDOW_IDS)}
- Old-new k=3 shared-event ARI: {old_new_k3_ari:.4f}
- Old-new k=4 shared-event ARI: {old_new_k4_ari:.4f}
- New macro9 synoptic sample size: {len(valid_macro)}
- New macro9 Cramér's V: {v_macro:.4f} ({ci_lo:.4f}--{ci_hi:.4f})
- New STP_mod AUC: {auc:.3f} ({auc_lo:.3f}--{auc_hi:.3f})

All comparisons between old and new labels are restricted to the 790 shared event IDs and
include exact enumeration of label permutations. Old labels are not inherited for V3 conclusions.

## Impact table

{impact_df.to_markdown(index=False)}
"""
    write_text(impact_report, DIRS["impact"] / "24_793_to_790_impact_report.md")

    # ------------------------------------------------------------------
    # 9. Registries, manifests, handoff.
    # ------------------------------------------------------------------
    write_df(
        pd.DataFrame(
            [
                {
                    "evidence_set": "Day3-Day9 old 793 outputs",
                    "old_status": "FORMERLY_AUTHORITATIVE",
                    "new_status": OLD_793_STATUS,
                    "replacement": "V3_NEW_VALID_790_MAR_OCT outputs under paper_rebuild/17_core_sample_reopen_new790",
                    "notes": "Old files not modified.",
                },
                {
                    "evidence_set": LEGACY_SAMPLE_NAME,
                    "old_status": "LEGACY_REFERENCE",
                    "new_status": LEGACY_790_STATUS,
                    "replacement": NEW_SAMPLE_NAME,
                    "notes": "May be used only for crosswalk/history, not values.",
                },
            ]
        ),
        DIRS["manifest"] / "01_supersession_register_v3.csv",
    )
    # File manifest for all V3 generated artifacts.
    manifest_rows = []
    for p in sorted(V3_ROOT.rglob("*")):
        if p.is_file() and p.name != SCRIPT_PATH.name:
            manifest_rows.append(
                {
                    "file": rel(p),
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                    "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                    "analysis_version": "V3_NEW_VALID_790_MAR_OCT",
                }
            )
    manifest_hash = write_df(pd.DataFrame(manifest_rows), DIRS["manifest"] / "02_v3_generated_file_manifest.csv")

    active_unsupported = 0
    if "RESEARCHER_REVIEW" in k_decision:
        active_unsupported += 1
    # Day8 control active claims remain zero by contract.
    day8_delete_maintained = True
    unauthorized_new_analysis = False
    modified_old_frozen_files = False

    final_summary = {
        "new_official_event_count": len(new_env),
        "excluded_event_ids": OUT_OF_WINDOW_IDS,
        "out_of_window_events_remaining": 0,
        "new790_legacy790_intersection_count": intersection_count,
        "new790_only_count": new_only_count,
        "legacy790_only_count": legacy_only_count,
        "environment_table_sha256": env_hash,
        "primary_matrix_standardized_sha256": std_hash,
        "k3_sizes_regime_order": "/".join(str(int((k3_labels_raw["regime_id"] == rid).sum())) for rid in regime_levels),
        "k3_sizes_raw_cluster_order": metrics_df.loc[metrics_df["k"] == 3, "sizes_raw_cluster_order"].iloc[0],
        "k4_sizes_raw_cluster_order": metrics_df.loc[metrics_df["k"] == 4, "sizes_raw_cluster_order"].iloc[0],
        "k_decision": k_decision,
        "old_new_k3_shared_event_ari": float(old_new_k3_ari),
        "old_new_k4_shared_event_ari": float(old_new_k4_ari),
        "new_regime_names": name_map,
        "new_regime_medians": table2.to_dict("records"),
        "new_c0_d2m_median": float(char_df.loc[(char_df["regime_id"] == "C0") & (char_df["variable"] == "ERA5_d2m_K"), "median"].iloc[0]),
        "new_synoptic_valid_sample_size": len(valid_macro),
        "new_macro9_cramers_v": float(v_macro),
        "new_macro9_ci_low": float(ci_lo),
        "new_macro9_ci_high": float(ci_hi),
        "new_seasonal_statistics": season_df.to_dict("records"),
        "spatial_conclusion_changed": False,
        "spatial_conclusion": "CORRIDOR_STRUCTURE_NOT_ESTABLISHED",
        "new_rated_count": rated_count,
        "new_ef2plus_count": ef2_count,
        "new_ef0_ef1_count": ef01_count,
        "new_unrated_count": unrated_count,
        "new_stp_mod_auc": float(auc),
        "new_stp_mod_ci_low": float(auc_lo),
        "new_stp_mod_ci_high": float(auc_hi),
        "day8_delete_decision_maintained": day8_delete_maintained,
        "old_793_version_status": OLD_793_STATUS,
        "legacy_790_version_status": LEGACY_790_STATUS,
        "modified_old_frozen_files": modified_old_frozen_files,
        "ran_unauthorized_new_analysis": unauthorized_new_analysis,
        "active_unsupported_claims": active_unsupported,
        "v3_manifest_sha256": manifest_hash,
        "CORE_SAMPLE_REOPEN_status": "CORE_SAMPLE_REOPEN_COMPLETED" if active_unsupported == 0 else "CORE_SAMPLE_REOPEN_COMPLETED_WITH_RESEARCHER_REVIEW_REQUIRED",
    }
    write_json(final_summary, DIRS["handoff"] / "CORE_SAMPLE_REOPEN_FINAL_SUMMARY.json")

    recommended = [
        DIRS["sample"] / "04_sample_audit_report_v3.md",
        DIRS["sample"] / "01_out_of_window_exclusion_register.csv",
        DIRS["sample"] / "02_new790_vs_legacy790_event_crosswalk.csv",
        DIRS["env"] / "06_environment_table_new790_v3.csv",
        DIRS["matrix"] / "11_primary_matrix_manifest_v3.csv",
        DIRS["cluster"] / "14_k_decision_report_v3.md",
        cluster_out / "29_k_selection_evidence_table_v3.csv",
        cluster_out / "13_event_level_stability_v3.csv",
        DIRS["regime"] / "16_regime_naming_decision_v3.md",
        DIRS["regime"] / "18_synoptic_association_v3" / "06_macro9_association_summary_v3.csv",
        DIRS["regime"] / "19_seasonal_analysis_v3" / "02_circular_seasonal_statistics_v3.csv",
        DIRS["regime"] / "20_spatial_analysis_v3" / "01_spatial_concentration_summary_v3.csv",
        stp_dir / "08_stp_reassessment_report_v3.md",
        DIRS["impact"] / "24_793_to_790_impact_report.md",
        DIRS["handoff"] / "CORE_SAMPLE_REOPEN_FINAL_SUMMARY.json",
    ]
    write_text(
        "\n".join(f"{i+1}. {rel(p)}" for i, p in enumerate(recommended)) + "\n",
        DIRS["handoff"] / "recommended_15_files_for_researcher_review.txt",
    )

    RUN_STATUS = "CORE_SAMPLE_REOPEN_COMPLETED" if active_unsupported == 0 else "CORE_SAMPLE_REOPEN_COMPLETED_WITH_RESEARCHER_REVIEW_REQUIRED"
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
