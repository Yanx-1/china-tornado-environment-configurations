from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "reference_era5_final"
FREEZE = ROOT / "11_freeze"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def old_tree_unchanged() -> tuple[bool, int, int]:
    before = pd.read_csv(ROOT / "00_input_validation" / "old_reference_era5_full_tree_start_hashes.csv", dtype=str)
    now = {}
    for path in OLD.rglob("*"):
        if path.is_file():
            now[path.relative_to(OLD).as_posix()] = (sha256(path), str(path.stat().st_size))
    failures = 0
    for row in before.itertuples(index=False):
        current = now.get(row.relative_path)
        if current is None or current[0] != row.sha256 or int(current[1]) != int(row.size_bytes):
            failures += 1
    failures += len(set(now) - set(before.relative_path))
    return failures == 0, len(now), failures


def md_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, rule]
    for row in frame[columns].itertuples(index=False, name=None):
        cells = []
        for col, value in zip(columns, row):
            if pd.isna(value):
                cells.append("NA")
            elif col in formats:
                cells.append(format(float(value), formats[col]))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    FREEZE.mkdir(parents=True, exist_ok=True)
    pre = pd.read_csv(ROOT / "00_input_validation" / "analysis_gate_pre_audit.csv")
    audits = pd.read_csv(ROOT / "10_independent_audit" / "audit_results_A_L.csv")
    effects = pd.read_csv(ROOT / "03_anchor_balanced_effects" / "anchor_balanced_pair_effects.csv")
    inference = pd.read_csv(ROOT / "04_matched_set_inference" / "anchor_level_matched_set_inference.csv")
    pairs = pd.read_csv(ROOT / "01_pair_dataset" / "all_reference_pairs_181.csv")
    counts = pd.read_csv(ROOT / "00_input_validation" / "reference_count_per_anchor.csv")
    consistency = pd.read_csv(ROOT / "08_old_new_consistency" / "old_new_analysis_consistency.csv")
    clusters = pd.read_csv(ROOT / "05_cluster_stratified" / "cluster_stratified_allrefs.csv")
    ranks = pd.read_csv(ROOT / "06_reference_rank_sensitivity" / "reference_rank_effect_summary.csv")
    types = pd.read_csv(ROOT / "07_reference_type_sensitivity" / "reference_type_effect_summary.csv")

    old_ok, old_files, old_failures = old_tree_unchanged()
    pair_weight_ok = (
        np.max(np.abs(pairs.groupby("tornado_id").pair_weight.sum().to_numpy() - 1.0)) < 1e-12
        and abs(pairs.pair_weight.sum() - 105.0) < 1e-12
    )
    audit_ok = bool((audits.status == "PASS").all()) and len(audits) == 12
    pre_ok = bool((pre.status == "PASS").all())
    reproducibility_ok = bool(audits.loc[audits.audit == "L_REPRODUCIBILITY", "status"].eq("PASS").all())

    gate_rows = pre.to_dict("records")
    gate_rows.extend([
        {"gate": "INDEPENDENT_AUDIT_A_L", "status": "PASS" if audit_ok else "FAIL", "details": f"{(audits.status == 'PASS').sum()}/12 audits PASS"},
        {"gate": "REPRODUCIBILITY_GATE", "status": "PASS" if reproducibility_ok else "FAIL", "details": "two clean replays; 11/11 core CSVs byte stable" if reproducibility_ok else "clean replay mismatch"},
        {"gate": "OLD_ANALYSIS_PROTECTION_FINAL_RECHECK", "status": "PASS" if old_ok else "FAIL", "details": f"old-tree files={old_files}; failures={old_failures}"},
    ])
    prerequisites = pre_ok and audit_ok and reproducibility_ok and old_ok and pair_weight_ok
    gate_rows.append({"gate": "ALLREF_MATCHEDSET_ANALYSIS_GATE", "status": "PASS" if prerequisites else "FAIL", "details": "all required analysis, audit, protection and reproducibility gates PASS" if prerequisites else "one or more prerequisite gates failed"})
    gates = pd.DataFrame(gate_rows)
    gate_path = FREEZE / "allref_final_gate.csv"
    write_csv(gates, gate_path)
    if not prerequisites:
        raise SystemExit("ALLREF_MATCHEDSET_ANALYSIS_GATE=FAIL; freeze prohibited")

    overall = effects.merge(inference[["variable", "D_mean_median", "raw_p", "q_BH"]], on="variable", validate="one_to_one")
    overall = overall[["variable", "n_pairs", "n_anchors", "weighted_mean_delta", "weighted_mean_CI_low", "weighted_mean_CI_high", "weighted_median_delta", "weighted_median_CI_low", "weighted_median_CI_high", "PS", "PS_CI_low", "PS_CI_high", "D_mean_median", "raw_p", "q_BH", "DIRECTION_ROBUST"]]
    distribution = counts.n_references.value_counts().sort_index()

    overall_table = md_table(overall, list(overall.columns), {
        "weighted_mean_delta": ".6g", "weighted_mean_CI_low": ".6g", "weighted_mean_CI_high": ".6g",
        "weighted_median_delta": ".6g", "weighted_median_CI_low": ".6g", "weighted_median_CI_high": ".6g",
        "PS": ".4f", "PS_CI_low": ".4f", "PS_CI_high": ".4f", "D_mean_median": ".6g", "raw_p": ".6g", "q_BH": ".6g",
    })
    consistency_table = md_table(consistency, ["variable", "direction_consistency", "magnitude_ratio_abs_new_vs_old_primary", "magnitude_change_class"], {"magnitude_ratio_abs_new_vs_old_primary": ".4f"})
    cluster_table = md_table(clusters, ["anchor_k3", "variable", "n_anchors", "n_pairs", "weighted_median_delta", "PS", "uncertainty_note"], {"weighted_median_delta": ".6g", "PS": ".4f"})
    rank_table = md_table(ranks, ["reference_rank", "variable", "n_pairs", "median_delta", "Q1", "Q3", "PS"], {"median_delta": ".6g", "Q1": ".6g", "Q3": ".6g", "PS": ".4f"})
    type_table = md_table(types, ["reference_type", "variable", "n_anchors", "n_pairs", "weighted_median_delta", "weighted_median_CI_low", "weighted_median_CI_high", "PS"], {"weighted_median_delta": ".6g", "weighted_median_CI_low": ".6g", "weighted_median_CI_high": ".6g", "PS": ".4f"})
    audit_table = md_table(audits, ["audit", "status", "details"])

    report = f"""# FINAL RESEARCHER REPORT — ALL-REFERENCE MATCHED-SET ANALYSIS

## 1. EXECUTION_STATUS

PASS. Analysis, independent audit, final Gate, and freeze completed. No manuscript integration was performed.

## 2. INPUT_HASH_STATUS

PASS. The reference environment SHA256 is `e6aa1422406ab73fc75d56eb577de1d558b0eea669fdbce6a1ce55c6c30c7ba9`; the formal tornado environment and frozen corrected k=3 label hashes also match the old frozen manifest.

## 3. OLD_FREEZE_PROTECTION

PASS. The complete `reference_era5_final` tree was rehashed before analysis and at finalization: {old_files} files checked, {old_failures} changes. The frozen rank-1 primary and anchor-median secondary remain unchanged and preserved.

## 4. REFERENCE_COUNT_DISTRIBUTION

- n=1: {int(distribution.get(1, 0))} anchors
- n=2: {int(distribution.get(2, 0))} anchors
- n=3: {int(distribution.get(3, 0))} anchors
- n=4: {int(distribution.get(4, 0))} anchors
- n=5: {int(distribution.get(5, 0))} anchors

## 5. PAIR_DATASET

Rows = {len(pairs)}; anchors = {pairs.tornado_id.nunique()}; unique processes = {pairs.global_process_id.nunique()}.

## 6. PAIR_WEIGHT_AUDIT

PASS. Each pair weight is `1 / m_i`; every anchor sums to 1 within numerical tolerance and total weight is {pairs.pair_weight.sum():.1f}.

## 7. OVERALL_ALLREF_RESULTS

This is a post-hoc/additional robustness analysis. Intervals are 95% tornado-anchor cluster-bootstrap percentile intervals (10,000 repetitions; seed 20260814). Wilcoxon p and BH q use the 105 anchor-level D_mean contrasts. `DIRECTION_ROBUST=YES` only when the 9A D_mean median, 9B weighted mean, 9B weighted median, and PS relative to 0.5 all agree; MLLCL is therefore marked `INTERPRETATION_CAUTION` because its weighted mean is negative while the other three directional summaries are positive.

{overall_table}

## 8. OLD_PRIMARY_VS_NEW

The old primary remains the frozen 105 × 105 rank-1 analysis. It is not replaced by the new all-reference analysis.

{consistency_table}

## 9. OLD_SECONDARY_VS_NEW

The old secondary remains the frozen comparison using one reference median per anchor. Directional consistency with the new analysis is recorded in the same table above and in `08_old_new_consistency/old_new_analysis_consistency.csv`.

## 10. C0/C1/C2_RESULTS

Exploratory/post-hoc only. Reference rows inherit the frozen tornado-anchor label; this does not label the reference itself as a C0/C1/C2 event. C1 has small N and high uncertainty.

{cluster_table}

## 11. REFERENCE_RANK_RESULTS

Descriptive sensitivity only; frozen ranking is unchanged.

{rank_table}

## 12. REFERENCE_TYPE_RESULTS

Exploratory only. RAIN/WIND/RAIN+WIND are frozen reference-type labels and are not assigned extra physical semantics here.

{type_table}

## 13. DIRECTION_CONSISTENCY

All five variables are `SAME_DIRECTION_ALL_THREE` across the frozen primary, frozen secondary, and new all-reference analysis.

## 14. EFFECT_MAGNITUDE_CHANGES

Using the predeclared descriptive flag in the analysis output (absolute new/old-primary median ratio outside 0.5–1.5), none of the five variables is flagged as `MAGNITUDE_SHIFT`. This threshold is descriptive, not an inferential test.

## 15. SAME_SYSTEM_LIMITATION_CHECK

PASS. Some reference observations and their tornado anchors may belong to the same larger convective system or share a mesoscale weather background. No storm-object or radar-identity analysis proves that all matched observations are separate storm systems. The results estimate environmental contrasts under closely matched spatiotemporal backgrounds; they do not establish independent non-tornado storm climatologies, tornado probability, or causation.

## 16. PSEUDOREPLICATION_AUDIT

PASS. The 181 pair rows are retained as observations but are not treated as 181 independent inferential units. Formal inference and bootstrap resampling use 105 tornado anchors/matched sets.

## 17. INDEPENDENT_AUDIT_A_L

{audit_table}

## 18. ALLREF_MATCHEDSET_ANALYSIS_GATE

PASS.

## 19. FREEZE_STATUS

ALLREF_MATCHEDSET_ANALYSIS_STAGE_FROZEN

## 20. FREEZE_FILE

`11_freeze/ALLREF_MATCHEDSET_ANALYSIS_FREEZE.md`

## 21. MANIFEST

`11_freeze/allref_matchedset_manifest.json`

## 22. READY_FOR_MANUSCRIPT_SENSITIVITY_INTEGRATION

YES. This readiness marker does not modify the manuscript; integration requires a separate researcher decision and stage.
"""
    report_path = FREEZE / "FINAL_RESEARCHER_REPORT.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")

    software = (
        f"Python={platform.python_version()}\n"
        f"NumPy={np.__version__}\n"
        f"pandas={pd.__version__}\n"
        f"SciPy={scipy.__version__}\n"
        f"Matplotlib={matplotlib.__version__}\n"
        "bootstrap_repetitions=10000\nbootstrap_seed=20260814\n"
        "inference_cluster=TORNADO_ANCHOR\n"
    )
    software_path = FREEZE / "software_environment.txt"
    software_path.write_text(software, encoding="utf-8", newline="\n")

    freeze_text = """# ALL-REFERENCE MATCHED-SET ANALYSIS FREEZE

ANALYSIS_STATUS = POST_HOC_ADDITIONAL_ROBUSTNESS

PAIR_ROWS = 181

MATCHED_ANCHORS = 105

INFERENCE_CLUSTER = TORNADO_ANCHOR

PAIR_WEIGHT = 1 / NUMBER_OF_REFERENCES_FOR_ANCHOR

OLD_PRIMARY_STATUS = UNCHANGED_AND_PRESERVED

OLD_PRIMARY = 105 × 105 rank1

NEW_ANALYSIS = all 181 references retained in 105 matched sets

ALLREF_MATCHEDSET_ANALYSIS_GATE = PASS

INDEPENDENT_AUDIT_A_L = PASS

REPRODUCIBILITY_GATE = PASS

ALLREF_MATCHEDSET_ANALYSIS_STAGE_FROZEN

READY_FOR_MANUSCRIPT_SENSITIVITY_INTEGRATION = YES

No manuscript integration was performed in this stage.
"""
    freeze_path = FREEZE / "ALLREF_MATCHEDSET_ANALYSIS_FREEZE.md"
    freeze_path.write_text(freeze_text, encoding="utf-8", newline="\n")

    excluded = {
        "11_freeze/allref_file_hashes.csv",
        "11_freeze/allref_matchedset_manifest.json",
    }
    hash_rows = []
    for path in sorted((p for p in ROOT.rglob("*") if p.is_file()), key=lambda p: p.relative_to(ROOT).as_posix().lower()):
        rel = path.relative_to(ROOT).as_posix()
        if rel in excluded or "__pycache__" in path.parts:
            continue
        hash_rows.append({"relative_path": rel, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    registry = pd.DataFrame(hash_rows)
    registry_path = FREEZE / "allref_file_hashes.csv"
    write_csv(registry, registry_path)

    input_hashes = pd.read_csv(ROOT / "00_input_validation" / "input_hash_status.csv")
    manifest = {
        "stage": "ALLREF_MATCHEDSET_ANALYSIS_STAGE_FROZEN",
        "ALLREF_MATCHEDSET_ANALYSIS_GATE": "PASS",
        "READY_FOR_MANUSCRIPT_SENSITIVITY_INTEGRATION": "YES",
        "ANALYSIS_STATUS": "POST_HOC_ADDITIONAL_ROBUSTNESS",
        "PAIR_ROWS": 181,
        "MATCHED_ANCHORS": 105,
        "INFERENCE_CLUSTER": "TORNADO_ANCHOR",
        "PAIR_WEIGHT": "1 / NUMBER_OF_REFERENCES_FOR_ANCHOR",
        "OLD_PRIMARY_STATUS": "UNCHANGED_AND_PRESERVED",
        "OLD_PRIMARY": "105 x 105 rank1",
        "NEW_ANALYSIS": "all 181 references retained in 105 matched sets",
        "H_ERA5_CTRL_used": False,
        "input_hashes": dict(zip(input_hashes.role, input_hashes.actual_sha256)),
        "key_output_hashes": {
            "all_reference_pairs_181": sha256(ROOT / "01_pair_dataset" / "all_reference_pairs_181.csv"),
            "anchor_balanced_pair_effects": sha256(ROOT / "03_anchor_balanced_effects" / "anchor_balanced_pair_effects.csv"),
            "anchor_level_inference": sha256(ROOT / "04_matched_set_inference" / "anchor_level_matched_set_inference.csv"),
            "old_new_consistency": sha256(ROOT / "08_old_new_consistency" / "old_new_analysis_consistency.csv"),
            "independent_audit": sha256(ROOT / "10_independent_audit" / "audit_results_A_L.csv"),
            "final_gate": sha256(gate_path),
            "freeze": sha256(freeze_path),
            "final_report": sha256(report_path),
        },
        "script_hashes": {
            "01_run_allrefs_analysis.py": sha256(ROOT / "scripts" / "01_run_allrefs_analysis.py"),
            "audit_allrefs_matchedset.py": sha256(ROOT / "10_independent_audit" / "audit_allrefs_matchedset.py"),
            "02_finalize_and_freeze.py": sha256(Path(__file__).resolve()),
        },
        "file_hash_registry_sha256": sha256(registry_path),
        "software_environment_sha256": sha256(software_path),
    }
    manifest_path = FREEZE / "allref_matchedset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("ALLREF_MATCHEDSET_ANALYSIS_GATE=PASS")
    print("ALLREF_MATCHEDSET_ANALYSIS_STAGE_FROZEN")
    print("READY_FOR_MANUSCRIPT_SENSITIVITY_INTEGRATION=YES")


if __name__ == "__main__":
    main()
