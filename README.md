# 中国龙卷的稳定但重叠热力—动力环境配置及其多层背景

Reproducibility package for the manuscript on the **stable-but-overlapping thermodynamic–kinematic environmental configurations of Chinese tornadoes**.

> **Status: local final public-release candidate. Nothing has been uploaded or published.**
> GitHub: `[GITHUB REPOSITORY URL PENDING]` · Zenodo DOI: `[ZENODO DOI PENDING]`

---

## Overview

This repository supports a study of the joint thermodynamic–kinematic environment of Chinese
tornadoes using a single national sample and a five-variable environmental space, situating
the result against multi-level moisture/environmental-wind context and a spatiotemporally
matched severe-convection reference comparison. The science is frozen (Science Freeze V3;
Chinese scientific content frozen; the English manuscript is prepared separately by the
authors and is NOT included here); this repository changes no scientific content.

## Associated manuscript

- Title (frozen): **中国龙卷的稳定但重叠热力—动力环境配置及其多层背景**
- Status: manuscript-submission reproducibility package.

## Scientific scope

The study characterises **thermodynamic–kinematic environmental heterogeneity** on the
ERA5 **environmental scale**. It identifies **stable but overlapping environmental
configurations** (not discrete natural tornado types). The matched comparison is a
**conditional environmental contrast** against a **matched severe-convection reference**;
regime-defining characteristics are distinguished from tornado-discriminating
characteristics. This repository makes **no** tornado-mechanism, probability, prediction, or
causal claims.

## Repository contents

| Path | Contents |
|---|---|
| `code/` | Analysis scripts (clustering, reference matched-set, figure/table generation, weather-type classification). MIT. |
| `data/derived/` | Project-generated **aggregate** derived data (scalar constants, k-selection metrics, stability summaries) — CC BY 4.0. |
| `figure_source_data/` | Figure source data + final figure PNGs (aggregated; event-level source data excluded). |
| `table_source_data/` | Frozen table-source CSVs (aggregated), incl. V3 audit tables S6 (Spearman), S7 (Ward/GMM stable-core), S8 (negative signed SRH1). |
| `supplement/` | Supplementary figure PNGs (Fig S1–S3). |
| `docs/` | Frozen manuscript + supplement (Chinese V3), freeze manifests, number ledger, claim-evidence matrix, public freeze-provenance summary. |
| `manifest/` | `package_manifest.csv`, `package_hashes.csv`, `frozen_science_crosswalk.csv`. |
| `environment/` | `requirements.txt` + recorded software environment. |

## Reproducibility level

**Level C (aggregate figure/table reproduction) + archival support.** Full raw-to-paper
reproduction is **not** claimed: it requires raw ERA5 and MICAPS archives that are not
redistributed, and event-level data is excluded pending a data-rights decision.

## Quick start

```bash
# relative paths only; no machine-specific absolute paths in this repository
pip install -r environment/requirements.txt
# figure/table scripts read source data under figure_source_data/ and table_source_data/
python code/figures_final/build_final_figures.py
python code/figures_final/build_final_tables.py
```

## Software environment

Python 3.11.4 · NumPy 2.3.5 · pandas 2.3.3 · SciPy 1.16.3 · scikit-learn 1.8.0 ·
xarray 2025.10.1 · MetPy 1.7.1 · Matplotlib 3.10.9. See `environment/requirements.txt`.

## Analysis workflow

1. Five-variable environment characterisation (MLCAPE, MLLCL, 2-m dew-point `d2m`, SHR6,
   signed SRH1).
2. Standardised k-means clustering; k=3 as the main descriptive layer, k=4 as structural
   sensitivity (no unique natural k is claimed).
3. Stability and algorithm-sensitivity evaluation.
4. Multi-level moisture / environmental-wind context.
5. Spatiotemporally matched severe-convection reference comparison (rank-1 primary,
   anchor-median secondary, anchor-balanced post-hoc).

## Data sources

- **ERA5** (ECMWF/Copernicus) — environment fields. Obtain original data from the Copernicus
  Climate Data Store. This repository redistributes only project-computed parameters, not
  original ERA5 files.
- **MICAPS** (CMA) RAIN01_NATIONAL and MAX_WIND — reference construction. **Original MICAPS
  observations are not redistributed in this repository.**
- **中国龙卷风个例库** (https://www.fs121.com/tornado/#/list?code=11100) — tornado event
  source. The original source snapshot is not redistributed.

## Included derived data

Project-generated, aggregated derived data (no source-derived tornado identifiers or
coordinates) is included under CC BY 4.0. See `DATA_LICENSE.md`.

## Data not redistributed

- Raw ERA5 bulk archives.
- Raw MICAPS (RAIN01_NATIONAL, MAX_WIND).
- Original 909-event tornado source snapshot, and the source-derived tornado identifiers,
  coordinates, times, and metadata.

Source data are available from the original providers subject to their access terms.

## Licensing

- **Software/code**: MIT License (`LICENSE`).
- **Project-generated derived data**: CC BY 4.0 (`DATA_LICENSE.md`).
- **Third-party data**: not redistributed; governed by provider terms.

## Citation

Cite the associated manuscript. Persistent identifiers are pending:
GitHub `[GITHUB REPOSITORY URL PENDING]`, Zenodo DOI `[ZENODO DOI PENDING]`.

## Version

`1.0.0-submission` (see `VERSION`).

## Authors

1. Xu Yan (闫旭) — sole first author · ORCID 0009-0007-2799-2512
2. Zhengyang Zhang (张正阳) — co-second author · ORCID 0009-0006-1893-6562
3. Shuchang Liu (刘舒畅) — co-second author · ORCID 0009-0007-4262-8049
4. 韩艳 — co-corresponding author
5. 孔海江 — lead / primary corresponding author

Zhengyang Zhang and Shuchang Liu are co-second authors.

## Contact / corresponding author

孔海江 (lead contact) — hjkong@foxmail.com

## Known limitations

ERA5 environmental-scale analysis only; environment-group boundaries are sensitive to k and
algorithm; the matched reference subset is enriched in C2 and a stronger-kinematics
background and cannot represent all 790 events; weather-type labels are descriptive post-hoc
context, not independent validation.
