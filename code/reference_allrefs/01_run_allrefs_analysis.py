from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


SEED = 20260814
N_BOOT = 10_000
EXPECTED_REFERENCE_SHA = "e6aa1422406ab73fc75d56eb577de1d558b0eea669fdbce6a1ce55c6c30c7ba9"
EXPECTED_CLUSTER_COUNTS = {"C0": 26, "C1": 12, "C2": 67}
VARIABLES = {
    "MLCAPE": ("MLCAPE_Jkg", "J kg-1"),
    "MLLCL": ("MLLCL_m", "m"),
    "d2m": ("ERA5_d2m_K", "K"),
    "SHR6": ("SHR6_ms", "m s-1"),
    "SRH1": ("SRH1_m2s2", "m2 s-2"),
}
REQUIRED_DIRS = [
    "00_input_validation", "01_pair_dataset", "02_allref_descriptive",
    "03_anchor_balanced_effects", "04_matched_set_inference",
    "05_cluster_stratified", "06_reference_rank_sensitivity",
    "07_reference_type_sensitivity", "08_old_new_consistency",
    "09_figures_tables", "10_independent_audit", "11_freeze", "logs",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_for(label: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Seventeen significant digits preserve IEEE-754 binary64 values on CSV round-trip.
    # This is required because near-tied absolute D_mean values can otherwise change
    # Wilcoxon ranks when an independent audit reloads the saved contrast table.
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    threshold = q * weights.sum()
    idx = int(np.searchsorted(np.cumsum(weights), threshold, side="left"))
    return float(values[min(idx, len(values) - 1)])


def rank_components(delta: np.ndarray) -> tuple[float, float, float, int]:
    nonzero = np.asarray(delta, dtype=float)
    nonzero = nonzero[np.isfinite(nonzero) & (nonzero != 0)]
    if len(nonzero) == 0:
        return 0.0, 0.0, 0.0, 0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    denom = w_plus + w_minus
    return w_plus, w_minus, float((w_plus - w_minus) / denom), len(nonzero)


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    order = valid.sort_values(kind="mergesort").index
    ranked = valid.loc[order].to_numpy()
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def direction(value: float) -> str:
    if value > 0:
        return "TORNADO_GT_REFERENCE"
    if value < 0:
        return "TORNADO_LT_REFERENCE"
    return "NO_DIRECTION"


def anchor_arrays(frame: pd.DataFrame, delta_col: str) -> tuple[list[str], list[np.ndarray]]:
    ids, arrays = [], []
    for anchor, group in frame.groupby("tornado_id", sort=True):
        values = group.sort_values(["reference_rank", "global_process_id"], kind="mergesort")[delta_col].to_numpy(float)
        ids.append(str(anchor))
        arrays.append(values)
    return ids, arrays


def effect_estimates(frame: pd.DataFrame, delta_col: str, label: str) -> dict[str, float]:
    ids, arrays = anchor_arrays(frame, delta_col)
    n_anchors = len(ids)
    if n_anchors == 0:
        raise ValueError(f"No anchors for {label}")
    means = np.array([a.mean() for a in arrays], dtype=float)
    ps_anchor = np.array([(np.sum(a > 0) + 0.5 * np.sum(a == 0)) / len(a) for a in arrays], dtype=float)
    values = np.concatenate(arrays)
    weights = np.concatenate([np.full(len(a), 1.0 / len(a)) for a in arrays])
    weighted_mean = float(np.average(values, weights=weights))
    weighted_median = weighted_quantile(values, weights, 0.5)
    ps = float(ps_anchor.mean())

    rng = np.random.default_rng(seed_for(label))
    sampled = rng.integers(0, n_anchors, size=(N_BOOT, n_anchors))
    boot_mean = means[sampled].mean(axis=1)
    boot_ps = ps_anchor[sampled].mean(axis=1)
    boot_median = np.empty(N_BOOT, dtype=float)
    for b, row in enumerate(sampled):
        b_values = np.concatenate([arrays[k] for k in row])
        b_weights = np.concatenate([np.full(len(arrays[k]), 1.0 / len(arrays[k])) for k in row])
        boot_median[b] = weighted_quantile(b_values, b_weights, 0.5)
    q = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))
    mean_ci, median_ci, ps_ci = q(boot_mean), q(boot_median), q(boot_ps)
    return {
        "n_anchors": n_anchors, "n_pairs": len(frame),
        "weighted_mean_delta": weighted_mean,
        "weighted_mean_CI_low": mean_ci[0], "weighted_mean_CI_high": mean_ci[1],
        "weighted_median_delta": weighted_median,
        "weighted_median_CI_low": median_ci[0], "weighted_median_CI_high": median_ci[1],
        "PS": ps, "PS_CI_low": ps_ci[0], "PS_CI_high": ps_ci[1],
        "bootstrap_repetitions": N_BOOT, "bootstrap_seed": SEED,
        "bootstrap_seed_derived": seed_for(label), "inference_cluster": "TORNADO_ANCHOR",
    }


def median_bootstrap_ci(values: np.ndarray, label: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed_for(label))
    sampled = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    estimates = np.median(values[sampled], axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def old_tree_snapshot(old_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((p for p in old_root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(old_root).as_posix().lower()):
        rows.append({"relative_path": path.relative_to(old_root).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def validate_and_load(auth: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, Path]]:
    old = auth / "reference_era5_final"
    core = auth / "core_tor0608_correction_refreeze_v1"
    paths = {
        "reference_environment_181": old / "02_reference_environment" / "reference_environment_181.csv",
        "formal_tornado_environment": core / "01_corrected_formal790" / "FORMAL790_ENVIRONMENT_CORRECTED_V1.csv",
        "frozen_k3_labels": core / "03_clustering" / "K3_CORRECTED_LABELS.csv",
        "old_freeze": old / "11_freeze" / "REFERENCE_ERA5_ANALYSIS_FREEZE.md",
        "old_manifest": old / "11_freeze" / "reference_analysis_manifest.json",
        "old_hash_registry": old / "11_freeze" / "reference_analysis_file_hashes.csv",
        "old_primary": old / "03_primary_rank1" / "primary_paired_statistics.csv",
        "old_secondary": old / "04_secondary_anchor_median" / "secondary_anchor_median_statistics.csv",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs: " + "; ".join(missing))
    manifest = json.loads(paths["old_manifest"].read_text(encoding="utf-8"))
    freeze_text = paths["old_freeze"].read_text(encoding="utf-8")
    if manifest.get("REFERENCE_ERA5_ANALYSIS_GATE") != "PASS" or manifest.get("READY_FOR_REFERENCE_MANUSCRIPT_INTEGRATION") != "YES":
        raise RuntimeError("Old frozen manifest gate is not PASS/YES")
    for token in ["REFERENCE_ERA5_ANALYSIS_STAGE_FROZEN", "REFERENCE_ERA5_ANALYSIS_GATE: **PASS**", "READY_FOR_REFERENCE_MANUSCRIPT_INTEGRATION: **YES**"]:
        if token not in freeze_text:
            raise RuntimeError(f"Old freeze marker missing: {token}")
    expected_hashes = {
        "reference_environment_181": EXPECTED_REFERENCE_SHA,
        "formal_tornado_environment": manifest["frozen_input_hashes"]["formal790_environment"],
        "frozen_k3_labels": manifest["frozen_input_hashes"]["formal_k3_labels"],
    }
    hash_rows = []
    for role in ["reference_environment_181", "formal_tornado_environment", "frozen_k3_labels", "old_freeze", "old_manifest"]:
        actual = sha256(paths[role])
        expected = expected_hashes.get(role, actual)
        hash_rows.append({"role": role, "path": str(paths[role]), "expected_sha256": expected, "actual_sha256": actual, "status": "PASS" if actual == expected else "FAIL"})
    hash_status = pd.DataFrame(hash_rows)
    if (hash_status.status != "PASS").any():
        raise RuntimeError("Frozen input SHA256 validation failed")

    registry = pd.read_csv(paths["old_hash_registry"], dtype=str)
    registry_failures = []
    for row in registry.itertuples(index=False):
        p = Path(row.absolute_path)
        status = "PASS" if p.is_file() and sha256(p) == str(row.sha256).lower() else "FAIL"
        if status == "FAIL":
            registry_failures.append(str(row.relative_path))
    if registry_failures:
        raise RuntimeError("Old frozen hash registry mismatch: " + "; ".join(registry_failures))

    refs = pd.read_csv(paths["reference_environment_181"], dtype={"tornado_id": str, "global_process_id": str, "reference_type": str})
    tor = pd.read_csv(paths["formal_tornado_environment"], dtype={"event_id": str})
    labels = pd.read_csv(paths["frozen_k3_labels"], dtype={"event_id": str, "new_formal_k3": str})
    ref_cols = [v[0] for v in VARIABLES.values()]
    checks = {
        "rows_181": len(refs) == 181,
        "complete_five_181": int(refs[ref_cols].notna().all(axis=1).sum()) == 181,
        "unique_anchors_105": refs.tornado_id.nunique() == 105,
        "unique_processes_181": refs.global_process_id.nunique() == 181,
    }
    counts = refs.groupby("tornado_id", sort=True).agg(
        n_references=("reference_rank", "size"), rank_min=("reference_rank", "min"), rank_max=("reference_rank", "max"),
        reference_types=("reference_type", lambda s: "|".join(sorted(set(map(str, s))))),
    ).reset_index()
    continuity = []
    for anchor, group in refs.groupby("tornado_id", sort=True):
        ranks = sorted(pd.to_numeric(group.reference_rank).astype(int).tolist())
        continuity.append(ranks == list(range(1, len(ranks) + 1)) and 1 <= len(ranks) <= 5)
    checks["rank_continuity_and_range"] = all(continuity)
    checks["sum_m_181"] = int(counts.n_references.sum()) == 181
    checks["formal_tornado_unique"] = not tor.event_id.duplicated().any()
    checks["all_anchor_tornado_rows_found"] = set(refs.tornado_id).issubset(set(tor.event_id))
    checks["all_anchor_labels_found"] = set(refs.tornado_id).issubset(set(labels.event_id))
    if not all(checks.values()):
        raise RuntimeError("INPUT_GATE failure: " + ", ".join(k for k, v in checks.items() if not v))

    output.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    write_csv(hash_status, output / "00_input_validation" / "input_hash_status.csv")
    write_csv(counts, output / "00_input_validation" / "reference_count_per_anchor.csv")
    write_csv(pd.DataFrame([{"check": k, "status": "PASS" if v else "FAIL"} for k, v in checks.items()]), output / "00_input_validation" / "input_structure_checks.csv")
    write_csv(old_tree_snapshot(old), output / "00_input_validation" / "old_reference_era5_full_tree_start_hashes.csv")
    (output / "00_input_validation" / "INPUT_VALIDATION_REPORT.md").write_text(
        "# Input validation\n\n"
        "INPUT_GATE = PASS\n\n"
        "All inputs were read from frozen local assets. ERA5_CTRL_ROOT (H: era5_ctrl) was not used. "
        "The 181 pair rows form 105 tornado-anchor matched sets and are not 181 independent inferential units.\n",
        encoding="utf-8", newline="\n",
    )
    return refs, tor, labels, manifest, paths


def build_pairs(refs: pd.DataFrame, tor: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    tor_cols = ["event_id"] + [v[0] for v in VARIABLES.values()]
    merged = refs.merge(tor[tor_cols], left_on="tornado_id", right_on="event_id", how="left", validate="many_to_one", suffixes=("_reference", "_tornado"))
    merged = merged.merge(labels[["event_id", "new_formal_k3"]], left_on="tornado_id", right_on="event_id", how="left", validate="many_to_one", suffixes=("", "_label"))
    n_refs = merged.groupby("tornado_id").reference_rank.transform("size").astype(int)
    out = pd.DataFrame({
        "tornado_id": merged.tornado_id.astype(str),
        "anchor_k3": merged.new_formal_k3.astype(str),
        "reference_rank": pd.to_numeric(merged.reference_rank).astype(int),
        "n_references_for_anchor": n_refs,
        "global_process_id": merged.global_process_id.astype(str),
        "reference_type": merged.reference_type.astype(str),
        "pair_weight": 1.0 / n_refs,
        "delta_definition": "TORNADO_MINUS_REFERENCE",
    })
    for key, (col, _unit) in VARIABLES.items():
        t = pd.to_numeric(merged[f"{col}_tornado"], errors="coerce")
        r = pd.to_numeric(merged[f"{col}_reference"], errors="coerce")
        out[f"tornado_{key}"] = t
        out[f"reference_{key}"] = r
        out[f"delta_{key}"] = t - r
    return out.sort_values(["tornado_id", "reference_rank", "global_process_id"], kind="mergesort").reset_index(drop=True)


def descriptive_tables(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    desc, fractions = [], []
    for key, (_col, unit) in VARIABLES.items():
        d = pairs[f"delta_{key}"].to_numpy(float)
        w = pairs.pair_weight.to_numpy(float)
        desc.append({
            "variable": key, "unit": unit, "n_pairs": len(d), "n_anchors": pairs.tornado_id.nunique(),
            "unweighted_median": float(np.median(d)), "unweighted_Q1": float(np.quantile(d, .25)),
            "unweighted_Q3": float(np.quantile(d, .75)), "unweighted_min": float(np.min(d)), "unweighted_max": float(np.max(d)),
            "unweighted_status": "PAIR_LEVEL_DESCRIPTIVE_ONLY",
            "weighted_mean": float(np.average(d, weights=w)), "weighted_Q1": weighted_quantile(d, w, .25),
            "weighted_median": weighted_quantile(d, w, .5), "weighted_Q3": weighted_quantile(d, w, .75),
            "weighting": "1/N_REFERENCES_FOR_ANCHOR",
        })
        for balance, ww in [("UNWEIGHTED_PAIR_DESCRIPTIVE_ONLY", np.ones(len(d))), ("ANCHOR_BALANCED", w)]:
            total = ww.sum()
            fractions.append({
                "variable": key, "balance": balance,
                "fraction_delta_gt_0": float(ww[d > 0].sum() / total),
                "fraction_delta_eq_0": float(ww[d == 0].sum() / total),
                "fraction_delta_lt_0": float(ww[d < 0].sum() / total),
            })
    return pd.DataFrame(desc), pd.DataFrame(fractions)


def overall_effects(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ps_rows = [], []
    for key in VARIABLES:
        estimate = effect_estimates(pairs, f"delta_{key}", f"overall|{key}")
        estimate.update({"variable": key, "direction": direction(estimate["weighted_median_delta"]), "analysis_status": "POST_HOC_ADDITIONAL_ROBUSTNESS"})
        rows.append(estimate)
        ps_rows.append({k: estimate[k] for k in ["variable", "n_anchors", "n_pairs", "PS", "PS_CI_low", "PS_CI_high", "bootstrap_repetitions", "bootstrap_seed", "inference_cluster"]})
    return pd.DataFrame(rows), pd.DataFrame(ps_rows)


def anchor_contrasts_and_inference(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrast_rows = []
    for anchor, group in pairs.groupby("tornado_id", sort=True):
        row = {"tornado_id": anchor, "anchor_k3": group.anchor_k3.iloc[0], "n_references": len(group)}
        for key in VARIABLES:
            row[f"D_mean_{key}"] = float(group[f"delta_{key}"].mean())
            row[f"D_median_{key}"] = float(group[f"delta_{key}"].median())
        contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)
    inference = []
    for key in VARIABLES:
        dm = contrasts[f"D_mean_{key}"].to_numpy(float)
        dmed = contrasts[f"D_median_{key}"].to_numpy(float)
        dm_ci = median_bootstrap_ci(dm, f"D_mean|{key}")
        dmed_ci = median_bootstrap_ci(dmed, f"D_median|{key}")
        w_plus, w_minus, rrb, nonzero_n = rank_components(dm)
        if nonzero_n == 0:
            w_stat, p, status = 0.0, 1.0, "ALL_ZERO_DIFFERENCES"
        else:
            test = stats.wilcoxon(dm, zero_method="wilcox", correction=False, alternative="two-sided", method="auto")
            w_stat, p, status = float(test.statistic), float(test.pvalue), "PERFORMED"
        inference.append({
            "variable": key, "N_anchors": len(dm), "D_mean_median": float(np.median(dm)),
            "D_mean_CI_low": dm_ci[0], "D_mean_CI_high": dm_ci[1],
            "D_median_median": float(np.median(dmed)), "D_median_CI_low": dmed_ci[0], "D_median_CI_high": dmed_ci[1],
            "W_statistic": w_stat, "W_plus": w_plus, "W_minus": w_minus,
            "rank_biserial": rrb, "raw_p": p, "zero_difference_n": int(np.sum(dm == 0)),
            "nonzero_difference_n": nonzero_n, "test_status": status,
            "bootstrap_repetitions": N_BOOT, "bootstrap_seed": SEED,
            "inference_unit": "TORNADO_ANCHOR", "primary_family": "FIVE_VARIABLE_D_MEAN_WILCOXON",
            "analysis_status": "POST_HOC_ADDITIONAL_ROBUSTNESS",
        })
    inf = pd.DataFrame(inference)
    inf["q_BH"] = bh_adjust(inf.raw_p)
    return contrasts, inf


def cluster_analysis(pairs: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cluster in ["C0", "C1", "C2"]:
        subset = pairs[pairs.anchor_k3 == cluster].copy()
        for key in VARIABLES:
            est = effect_estimates(subset, f"delta_{key}", f"cluster|{cluster}|{key}")
            dm = contrasts.loc[contrasts.anchor_k3 == cluster, f"D_mean_{key}"].to_numpy(float)
            dmed = contrasts.loc[contrasts.anchor_k3 == cluster, f"D_median_{key}"].to_numpy(float)
            est.update({
                "anchor_k3": cluster, "variable": key, "D_mean_median": float(np.median(dm)),
                "D_median_median": float(np.median(dmed)), "direction": direction(est["weighted_median_delta"]),
                "analysis_tier": "EXPLORATORY_POST_HOC",
                "uncertainty_note": "SMALL_N_HIGH_UNCERTAINTY" if cluster == "C1" else "EXPLORATORY",
                "reference_label_semantics": f"ANCHOR_{cluster}_REFERENCE",
            })
            rows.append(est)
    counts = pairs.groupby("anchor_k3", sort=True).agg(n_anchors=("tornado_id", "nunique"), n_pairs=("global_process_id", "size")).reset_index()
    return pd.DataFrame(rows), counts


def rank_analysis(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank, subset in pairs.groupby("reference_rank", sort=True):
        for key in VARIABLES:
            d = subset[f"delta_{key}"].to_numpy(float)
            ps = float((np.sum(d > 0) + .5 * np.sum(d == 0)) / len(d))
            rows.append({
                "reference_rank": int(rank), "variable": key, "n_pairs": len(subset), "n_anchors": subset.tornado_id.nunique(),
                "median_delta": float(np.median(d)), "Q1": float(np.quantile(d, .25)), "Q3": float(np.quantile(d, .75)),
                "PS": ps, "direction": direction(float(np.median(d))), "analysis_tier": "DESCRIPTIVE_SENSITIVITY",
            })
    return pd.DataFrame(rows)


def type_analysis(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ref_type in ["RAIN", "WIND", "RAIN+WIND"]:
        subset = pairs[pairs.reference_type == ref_type].copy()
        if subset.empty:
            raise RuntimeError(f"Expected frozen reference_type absent: {ref_type}")
        for key in VARIABLES:
            est = effect_estimates(subset, f"delta_{key}", f"type|{ref_type}|{key}")
            est.update({"reference_type": ref_type, "variable": key, "direction": direction(est["weighted_median_delta"]), "analysis_tier": "EXPLORATORY", "type_semantics": "FROZEN_REFERENCE_TYPE_ONLY"})
            rows.append(est)
    return pd.DataFrame(rows)


def old_new_consistency(paths: dict[str, Path], effects: pd.DataFrame, inference: pd.DataFrame) -> pd.DataFrame:
    primary = pd.read_csv(paths["old_primary"])
    secondary = pd.read_csv(paths["old_secondary"])
    rows = []
    for key, (old_var, _unit) in VARIABLES.items():
        p = primary.loc[primary.variable == old_var].iloc[0]
        s = secondary.loc[secondary.variable == old_var].iloc[0]
        n = effects.loc[effects.variable == key].iloc[0]
        ni = inference.loc[inference.variable == key].iloc[0]
        signs = np.sign([p.paired_median_difference, s.paired_median_difference, n.weighted_median_delta])
        if np.any(signs == 0):
            consistency = "UNCERTAIN"
        elif signs[0] == signs[1] == signs[2]:
            consistency = "SAME_DIRECTION_ALL_THREE"
        elif signs[0] == signs[2]:
            consistency = "PRIMARY_NEW_SAME_DIRECTION"
        else:
            consistency = "DIRECTION_REVERSAL"
        if float(p.paired_median_difference) == 0:
            magnitude_ratio = np.nan
            magnitude_class = "UNCERTAIN"
        else:
            magnitude_ratio = abs(float(n.weighted_median_delta) / float(p.paired_median_difference))
            magnitude_class = "MAGNITUDE_SHIFT" if magnitude_ratio < .5 or magnitude_ratio > 1.5 else "WITHIN_50_PERCENT_OF_OLD_PRIMARY"
        rows.append({
            "variable": key,
            "OLD_PRIMARY_median_delta": p.paired_median_difference, "OLD_PRIMARY_CI_low": p.bootstrap_CI_low,
            "OLD_PRIMARY_CI_high": p.bootstrap_CI_high, "OLD_PRIMARY_effect_rank_biserial": p.rank_biserial, "OLD_PRIMARY_q": p.q_BH,
            "OLD_SECONDARY_median_delta": s.paired_median_difference, "OLD_SECONDARY_CI_low": s.bootstrap_CI_low,
            "OLD_SECONDARY_CI_high": s.bootstrap_CI_high, "OLD_SECONDARY_effect_rank_biserial": s.rank_biserial, "OLD_SECONDARY_q": s.q_BH,
            "NEW_ALLREF_weighted_mean_delta": n.weighted_mean_delta, "NEW_ALLREF_weighted_median_delta": n.weighted_median_delta,
            "NEW_ALLREF_CI_low": n.weighted_median_CI_low, "NEW_ALLREF_CI_high": n.weighted_median_CI_high,
            "NEW_ALLREF_PS": n.PS, "NEW_ALLREF_PS_CI_low": n.PS_CI_low, "NEW_ALLREF_PS_CI_high": n.PS_CI_high,
            "NEW_ALLREF_D_mean_median": ni.D_mean_median, "NEW_ALLREF_q": ni.q_BH,
            "direction_consistency": consistency, "magnitude_ratio_abs_new_vs_old_primary": magnitude_ratio,
            "magnitude_change_class": magnitude_class,
            "magnitude_rule": "DESCRIPTIVE: MAGNITUDE_SHIFT IF ABS_RATIO OUTSIDE [0.5,1.5]",
            "old_primary_status": "UNCHANGED_AND_PRESERVED", "new_analysis_status": "POST_HOC_ADDITIONAL_ROBUSTNESS",
        })
    return pd.DataFrame(rows)


def make_figures(output: Path, pairs: pd.DataFrame, effects: pd.DataFrame, clusters: pd.DataFrame, ranks: pd.DataFrame, types: pd.DataFrame, consistency: pd.DataFrame) -> None:
    out = output / "09_figures_tables"
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"C0": "#0072B2", "C1": "#D55E00", "C2": "#009E73", "RAIN": "#0072B2", "WIND": "#D55E00", "RAIN+WIND": "#009E73"}

    write_csv(pairs[["tornado_id", "anchor_k3", "reference_rank", "global_process_id"] + [f"delta_{k}" for k in VARIABLES]], out / "FIG-AR1_plotting_data.csv")
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 9.5), layout="constrained")
    anchor_order = sorted(pairs.tornado_id.unique())
    xpos = {a: i for i, a in enumerate(anchor_order)}
    for ax, key in zip(axes, VARIABLES):
        for anchor, g in pairs.groupby("tornado_id", sort=True):
            vals = g[f"delta_{key}"].to_numpy(float)
            x = xpos[anchor]
            ax.vlines(x, vals.min(), vals.max(), color="#999999", lw=.45, alpha=.65)
            ax.scatter(np.full(len(vals), x), vals, s=3.5, color="#333333", alpha=.65, linewidths=0)
        ax.axhline(0, color="#B2182B", lw=.7, ls="--")
        ax.set_ylabel(f"Δ {key}\n({VARIABLES[key][1]})")
        ax.set_xlim(-1, len(anchor_order))
    axes[-1].set_xlabel("Tornado anchors (105; each vertical group is one matched set)")
    axes[-1].set_xticks([])
    fig.suptitle("FIG-AR1. Pairwise deltas grouped by tornado anchor (181 pairs; not iid)")
    save_figure(fig, out / "FIG-AR1_pairwise_delta_by_anchor")

    write_csv(effects, out / "FIG-AR2_plotting_data.csv")
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7), layout="constrained")
    for ax, key in zip(axes, VARIABLES):
        r = effects.loc[effects.variable == key].iloc[0]
        ax.errorbar(r.weighted_median_delta, 0, xerr=[[r.weighted_median_delta-r.weighted_median_CI_low], [r.weighted_median_CI_high-r.weighted_median_delta]], fmt="o", color="#0072B2", capsize=3)
        ax.axvline(0, color="#777777", ls="--", lw=.8)
        ax.set_title(key)
        ax.set_yticks([])
        ax.set_xlabel(f"Δ ({VARIABLES[key][1]})\nPS={r.PS:.2f} [{r.PS_CI_low:.2f}, {r.PS_CI_high:.2f}]")
    fig.suptitle("FIG-AR2. Anchor-balanced weighted median Δ (95% anchor-cluster bootstrap CI)")
    save_figure(fig, out / "FIG-AR2_anchor_balanced_effect_forest")

    write_csv(consistency, out / "FIG-AR3_plotting_data.csv")
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.2), layout="constrained")
    labels = ["Old rank1", "Old anchor median", "New all refs"]
    markers = ["o", "s", "^"]
    for ax, key in zip(axes, VARIABLES):
        r = consistency.loc[consistency.variable == key].iloc[0]
        vals = [r.OLD_PRIMARY_median_delta, r.OLD_SECONDARY_median_delta, r.NEW_ALLREF_weighted_median_delta]
        los = [r.OLD_PRIMARY_CI_low, r.OLD_SECONDARY_CI_low, r.NEW_ALLREF_CI_low]
        his = [r.OLD_PRIMARY_CI_high, r.OLD_SECONDARY_CI_high, r.NEW_ALLREF_CI_high]
        for y, (v, lo, hi, marker) in enumerate(zip(vals, los, his, markers)):
            ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt=marker, capsize=2.5, color=["#0072B2", "#E69F00", "#009E73"][y])
        ax.axvline(0, color="#777777", ls="--", lw=.8)
        ax.set_title(key); ax.set_yticks(range(3)); ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.set_xlabel(f"Δ ({VARIABLES[key][1]})")
    fig.suptitle("FIG-AR3. Frozen old analyses versus post-hoc all-reference analysis")
    save_figure(fig, out / "FIG-AR3_old_new_effect_comparison")

    write_csv(clusters, out / "FIG-AR4_plotting_data.csv")
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.2), layout="constrained")
    for ax, key in zip(axes, VARIABLES):
        for y, cluster in enumerate(["C0", "C1", "C2"]):
            r = clusters[(clusters.variable == key) & (clusters.anchor_k3 == cluster)].iloc[0]
            ax.errorbar(r.weighted_median_delta, y, xerr=[[r.weighted_median_delta-r.weighted_median_CI_low], [r.weighted_median_CI_high-r.weighted_median_delta]], fmt=["o", "s", "^"][y], color=colors[cluster], capsize=2.5)
        ax.axvline(0, color="#777777", ls="--", lw=.8); ax.set_title(key)
        ax.set_yticks(range(3)); ax.set_yticklabels(["C0 n=26", "C1 n=12*", "C2 n=67"] if ax is axes[0] else [])
        ax.set_xlabel(f"Δ ({VARIABLES[key][1]})")
    fig.suptitle("FIG-AR4. Exploratory anchor-stratified effects (*small N/high uncertainty)")
    save_figure(fig, out / "FIG-AR4_cluster_stratified_effects")

    write_csv(ranks, out / "FIG-AR5_plotting_data.csv")
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.0), layout="constrained")
    for ax, key in zip(axes, VARIABLES):
        g = ranks[ranks.variable == key].sort_values("reference_rank")
        ax.errorbar(g.reference_rank, g.median_delta, yerr=[g.median_delta-g.Q1, g.Q3-g.median_delta], marker="o", color="#0072B2", capsize=2.5)
        ax.axhline(0, color="#777777", ls="--", lw=.8); ax.set_title(key); ax.set_xticks(range(1, 6)); ax.set_xlabel("Reference rank")
        ax.set_ylabel(f"Median Δ ({VARIABLES[key][1]})" if ax is axes[0] else "")
    fig.suptitle("FIG-AR5. Descriptive reference-rank sensitivity (median and IQR)")
    save_figure(fig, out / "FIG-AR5_reference_rank_sensitivity")

    write_csv(types, out / "FIG-AR6_plotting_data.csv")
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.2), layout="constrained")
    for ax, key in zip(axes, VARIABLES):
        for y, ref_type in enumerate(["RAIN", "WIND", "RAIN+WIND"]):
            r = types[(types.variable == key) & (types.reference_type == ref_type)].iloc[0]
            ax.errorbar(r.weighted_median_delta, y, xerr=[[r.weighted_median_delta-r.weighted_median_CI_low], [r.weighted_median_CI_high-r.weighted_median_delta]], fmt=["o", "s", "^"][y], color=colors[ref_type], capsize=2.5)
        ax.axvline(0, color="#777777", ls="--", lw=.8); ax.set_title(key)
        ax.set_yticks(range(3)); ax.set_yticklabels(["RAIN", "WIND", "RAIN+WIND"] if ax is axes[0] else [])
        ax.set_xlabel(f"Δ ({VARIABLES[key][1]})")
    fig.suptitle("FIG-AR6. Exploratory frozen-reference-type effects")
    save_figure(fig, out / "FIG-AR6_reference_type_effects")

    metadata = []
    for p in sorted(out.glob("FIG-AR*.png")):
        try:
            from PIL import Image
            with Image.open(p) as im:
                metadata.append({"file": p.name, "width_px": im.width, "height_px": im.height, "format": im.format, "mode": im.mode, "size_bytes": p.stat().st_size})
        except Exception as exc:
            metadata.append({"file": p.name, "width_px": np.nan, "height_px": np.nan, "format": "UNINSPECTED", "mode": str(exc), "size_bytes": p.stat().st_size})
    write_csv(pd.DataFrame(metadata), out / "figure_export_metadata.csv")
    (out / "figure_alt_text.md").write_text(
        "# Figure accessibility descriptions\n\n"
        "- FIG-AR1: Five stacked panels show all 181 tornado-minus-reference deltas grouped into 105 vertical anchor sets; within-anchor ranges and points emphasize dependence.\n"
        "- FIG-AR2: Five variable-specific point-and-interval panels show anchor-balanced weighted medians with 95% anchor-cluster bootstrap intervals; probability-of-superiority values are printed below.\n"
        "- FIG-AR3: Five panels compare the frozen rank-1 primary, frozen anchor-median secondary, and post-hoc all-reference estimates with intervals.\n"
        "- FIG-AR4: Five panels compare exploratory C0, C1, and C2 anchor-stratified estimates; C1 is marked small-N/high-uncertainty.\n"
        "- FIG-AR5: Five panels show median and interquartile delta by frozen reference rank.\n"
        "- FIG-AR6: Five panels show exploratory anchor-balanced effects by frozen RAIN, WIND, and RAIN+WIND labels.\n",
        encoding="utf-8", newline="\n",
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def write_report(output: Path, counts: pd.DataFrame, effects: pd.DataFrame, consistency: pd.DataFrame) -> None:
    dist = counts.n_references.value_counts().sort_index().to_dict()
    result_lines = []
    for r in effects.itertuples(index=False):
        result_lines.append(
            f"- {r.variable}: weighted mean Δ={r.weighted_mean_delta:.6g} "
            f"[{r.weighted_mean_CI_low:.6g}, {r.weighted_mean_CI_high:.6g}]; "
            f"weighted median Δ={r.weighted_median_delta:.6g} "
            f"[{r.weighted_median_CI_low:.6g}, {r.weighted_median_CI_high:.6g}]; "
            f"PS={r.PS:.4f} [{r.PS_CI_low:.4f}, {r.PS_CI_high:.4f}]."
        )
    consistency_lines = [f"- {r.variable}: {r.direction_consistency}; {r.magnitude_change_class}." for r in consistency.itertuples(index=False)]
    text = (
        "# ALL-REFERENCE MATCHED-SET ANALYSIS REPORT\n\n"
        "ANALYSIS_STATUS = POST_HOC_ADDITIONAL_ROBUSTNESS\n\n"
        "The frozen old primary (105 × 105 rank-1) and old secondary analyses remain unchanged and preserved. "
        "This analysis retains all 181 frozen severe-convection references in 105 tornado-anchor matched sets. "
        "Each pair receives weight 1/m_i, so every anchor contributes total weight one; all bootstrap and formal inference use the tornado anchor as the resampling or testing unit.\n\n"
        "## Sample structure\n\n"
        f"Reference-count distribution: n=1: {dist.get(1,0)} anchors; n=2: {dist.get(2,0)}; n=3: {dist.get(3,0)}; n=4: {dist.get(4,0)}; n=5: {dist.get(5,0)}.\n\n"
        "## Overall post-hoc results\n\n" + "\n".join(result_lines) + "\n\n"
        "## Old versus new direction and magnitude\n\n" + "\n".join(consistency_lines) + "\n\n"
        "## Scientific boundary and same-system limitation\n\n"
        "These are environmental contrasts under closely matched spatiotemporal backgrounds. The frozen reference labels identify severe-convection references; they do not establish fully independent non-tornado storm climatologies, tornado probability, or causation. "
        "Some references and their tornado anchors may belong to the same larger convective system or share a mesoscale weather background. No storm-object or radar-identity analysis proves that all paired observations are independent storm systems. "
        "Spatiotemporal proximity reduces confounding from different broad weather backgrounds but can also make tornado and reference environments naturally similar. Results are therefore post-hoc robustness evidence, not confirmatory primary evidence.\n"
    )
    (output / "ALLREF_MATCHEDSET_ANALYSIS_REPORT.md").write_text(text, encoding="utf-8", newline="\n")


def run(output: Path, replay_mode: bool) -> None:
    script = Path(__file__).resolve()
    auth = script.parents[2] if script.parents[1].name == "reference_allrefs_matchedset" else Path(r"PROJECT_ROOT")
    if auth.name != "paper_rebuild":
        auth = Path(r"PROJECT_ROOT")
    refs, tor, labels, old_manifest, paths = validate_and_load(auth, output)
    counts = pd.read_csv(output / "00_input_validation" / "reference_count_per_anchor.csv")
    pairs = build_pairs(refs, tor, labels)
    delta_cols = [f"delta_{k}" for k in VARIABLES]
    if len(pairs) != 181 or pairs.tornado_id.nunique() != 105 or pairs.global_process_id.nunique() != 181 or pairs[delta_cols].isna().any().any():
        raise RuntimeError("PAIR_DATA_GATE or ALLREF_COMPLETENESS_GATE failed")
    weight_sums = pairs.groupby("tornado_id", sort=True).pair_weight.sum()
    weight_error = float(np.max(np.abs(weight_sums.to_numpy() - 1.0)))
    if weight_error > 1e-12 or abs(float(pairs.pair_weight.sum()) - 105.0) > 1e-12:
        raise RuntimeError("PAIR_WEIGHT_GATE failed")
    write_csv(pairs, output / "01_pair_dataset" / "all_reference_pairs_181.csv")
    write_csv(pd.DataFrame({"tornado_id": weight_sums.index, "pair_weight_sum": weight_sums.values, "absolute_error_from_1": np.abs(weight_sums.values - 1)}), output / "01_pair_dataset" / "pair_weight_by_anchor.csv")

    desc, fractions = descriptive_tables(pairs)
    write_csv(desc, output / "02_allref_descriptive" / "allref_pair_descriptive.csv")
    write_csv(fractions, output / "02_allref_descriptive" / "allref_direction_fractions.csv")
    effects, ps = overall_effects(pairs)
    write_csv(effects, output / "03_anchor_balanced_effects" / "anchor_balanced_pair_effects.csv")
    write_csv(ps, output / "03_anchor_balanced_effects" / "anchor_balanced_probability_superiority.csv")

    contrasts, inference = anchor_contrasts_and_inference(pairs)
    # Direction robustness is deliberately strict: the anchor-level D_mean median,
    # the cluster-bootstrap estimands (weighted mean and median), and PS relative to
    # 0.5 must all point the same way. Any disagreement is reported as caution.
    dmean_map = inference.set_index("variable")["D_mean_median"]
    effects["direction_9A_D_mean_median"] = effects.variable.map(lambda v: direction(float(dmean_map.loc[v])))
    effects["direction_9B_weighted_mean"] = effects.weighted_mean_delta.map(direction)
    effects["direction_9B_weighted_median"] = effects.weighted_median_delta.map(direction)
    effects["direction_9B_PS"] = effects.PS.map(lambda x: "TORNADO_GT_REFERENCE" if x > .5 else ("TORNADO_LT_REFERENCE" if x < .5 else "NO_DIRECTION"))
    effects["DIRECTION_ROBUST"] = effects.apply(
        lambda r: "YES" if len({r.direction_9A_D_mean_median, r.direction_9B_weighted_mean, r.direction_9B_weighted_median, r.direction_9B_PS}) == 1 else "INTERPRETATION_CAUTION",
        axis=1,
    )
    effects["direction_robustness_rule"] = "9A_D_MEAN_MEDIAN,9B_WEIGHTED_MEAN,9B_WEIGHTED_MEDIAN,AND_PS_MUST_AGREE"
    # Rewrite the formal effects table now that 9A/9B consistency is available.
    write_csv(effects, output / "03_anchor_balanced_effects" / "anchor_balanced_pair_effects.csv")
    write_csv(contrasts, output / "04_matched_set_inference" / "anchor_set_contrasts_105.csv")
    write_csv(inference, output / "04_matched_set_inference" / "anchor_level_matched_set_inference.csv")
    write_csv(pd.DataFrame([{
        "status": "NOT_RUN_ASSUMPTION_NOT_JUSTIFIED", "analysis_tier": "SUPPLEMENTARY_RANDOMIZATION_STYLE_INFERENCE",
        "reason": "Observational tornado designation and frozen reference construction do not justify within-set exchangeability under the sharp null.",
        "planned_permutations_if_justified": 20000, "seed": SEED,
    }]), output / "04_matched_set_inference" / "matched_set_permutation.csv")

    clusters, cluster_counts = cluster_analysis(pairs, contrasts)
    if dict(zip(cluster_counts.anchor_k3, cluster_counts.n_anchors)) != EXPECTED_CLUSTER_COUNTS:
        raise RuntimeError("CLUSTER_LABEL_GATE failed")
    write_csv(clusters, output / "05_cluster_stratified" / "cluster_stratified_allrefs.csv")
    write_csv(cluster_counts, output / "05_cluster_stratified" / "cluster_matched_counts.csv")
    write_csv(contrasts[["tornado_id", "anchor_k3", "n_references"] + [c for c in contrasts if c.startswith("D_")]], output / "05_cluster_stratified" / "cluster_anchor_contrasts.csv")

    ranks = rank_analysis(pairs)
    types = type_analysis(pairs)
    write_csv(ranks, output / "06_reference_rank_sensitivity" / "reference_rank_effect_summary.csv")
    write_csv(types, output / "07_reference_type_sensitivity" / "reference_type_effect_summary.csv")
    consistency = old_new_consistency(paths, effects, inference)
    write_csv(consistency, output / "08_old_new_consistency" / "old_new_analysis_consistency.csv")

    gates = pd.DataFrame([
        {"gate": "INPUT_GATE", "status": "PASS", "details": "frozen hashes, rows, completeness, ranks, anchors and processes verified"},
        {"gate": "PAIR_DATA_GATE", "status": "PASS", "details": "181 actual pairs; 105 anchors; 181 unique processes"},
        {"gate": "PAIR_WEIGHT_GATE", "status": "PASS", "details": f"max anchor error={weight_error:.3g}; total weight={pairs.pair_weight.sum():.15g}"},
        {"gate": "ALLREF_COMPLETENESS_GATE", "status": "PASS", "details": "all five deltas complete for 181 pairs"},
        {"gate": "ANCHOR_SET_GATE", "status": "PASS", "details": "D_mean and D_median computed for 105 anchors"},
        {"gate": "CLUSTER_BOOTSTRAP_GATE", "status": "PASS", "details": "10000 resamples of entire tornado-anchor matched sets"},
        {"gate": "MULTIPLE_TESTING_GATE", "status": "PASS", "details": "BH applied to five-variable D_mean Wilcoxon family"},
        {"gate": "OLD_ANALYSIS_PROTECTION_GATE", "status": "PASS", "details": "old tree baseline captured; no old writes"},
        {"gate": "NO_PSEUDOREPLICATION_GATE", "status": "PASS", "details": "formal inference unit is tornado anchor; no iid pair inference"},
        {"gate": "CLUSTER_LABEL_GATE", "status": "PASS", "details": "frozen corrected labels only; C0=26 C1=12 C2=67"},
    ])
    write_csv(gates, output / "00_input_validation" / "analysis_gate_pre_audit.csv")
    method = {
        "analysis_status": "POST_HOC_ADDITIONAL_ROBUSTNESS", "pair_rows": 181, "matched_anchors": 105,
        "inference_cluster": "TORNADO_ANCHOR", "pair_weight": "1 / NUMBER_OF_REFERENCES_FOR_ANCHOR",
        "delta": "TORNADO_MINUS_REFERENCE", "bootstrap_repetitions": N_BOOT, "bootstrap_seed": SEED,
        "weighted_quantile": "inverse weighted empirical CDF; first value with cumulative weight >= q*total weight",
        "old_primary_status": "UNCHANGED_AND_PRESERVED", "old_primary": "105 x 105 rank1",
        "new_analysis": "all 181 references retained in 105 matched sets", "H_ERA5_CTRL_used": False,
        "permutation": "NOT_RUN_ASSUMPTION_NOT_JUSTIFIED",
    }
    (output / "00_input_validation" / "analysis_method_manifest.json").write_text(json.dumps(method, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    if replay_mode:
        return
    make_figures(output, pairs, effects, clusters, ranks, types, consistency)
    table_map = {
        "TABLE-AR1_all_reference_overall_matched_set.csv": effects,
        "TABLE-AR2_anchor_level_Dmean_Dmedian.csv": inference,
        "TABLE-AR3_old_vs_new_consistency.csv": consistency,
        "TABLE-AR4_cluster_subgroups.csv": clusters,
        "TABLE-SAR1_rank_sensitivity.csv": ranks,
        "TABLE-SAR2_reference_type_analysis.csv": types,
    }
    for name, frame in table_map.items():
        write_csv(frame, output / "09_figures_tables" / name)
    provenance = {
        "source_pair_data": "01_pair_dataset/all_reference_pairs_181.csv", "analysis_script": str(script),
        "uncertainty": "95% percentile intervals; tornado-anchor cluster bootstrap; 10000 repetitions; seed 20260814",
        "figure_population": "181 pair rows grouped in 105 tornado-anchor matched sets", "missing_data": "none in five formal variables",
        "figure_status": "provisional general scientific figures; no journal-specific compliance claim",
    }
    (output / "09_figures_tables" / "figure_table_provenance_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_report(output, counts, effects, consistency)
    (output / "logs" / "analysis_run.json").write_text(json.dumps({
        "status": "ANALYSIS_COMPLETE_PENDING_INDEPENDENT_AUDIT", "seed": SEED, "bootstrap_repetitions": N_BOOT,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": mpl.__version__,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replay-mode", action="store_true")
    args = parser.parse_args()
    run(args.output_root.resolve(), args.replay_mode)
    print(f"ANALYSIS_RUN=PASS output={args.output_root.resolve()} replay_mode={args.replay_mode}")


if __name__ == "__main__":
    main()
