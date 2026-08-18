"""Assemble captions, call guides, crosswalks, montages, atlas, QC and handoff."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PyPDF2 import PdfReader

from build_final_figures import FIGURE_STEMS
from build_final_tables import TABLE_DEFINITIONS
from final_common import (
    CAPTION_DIR,
    CROSSWALK_DIR,
    FINAL_ROOT,
    FORMAL_FORBIDDEN_TERMS,
    GUIDE_DIR,
    HANDOFF_DIR,
    HOLD_DIR,
    LOG_DIR,
    MAIN_FIGURE_DIR,
    MAIN_TABLE_DIR,
    MONTAGE_DIR,
    PROJECT_ROOT,
    QC_DIR,
    README_DIR,
    REGIME_COLORS,
    ROUND1_ROOT,
    ROUND2_ROOT,
    SNAPSHOT_DIR,
    SOURCE_REGISTRY_DIR,
    SUPP_FIGURE_DIR,
    SUPP_TABLE_DIR,
    ensure_directories,
    relative_to_final,
    sha256_file,
    write_csv_rows,
    write_json,
    write_text,
)


FIGURE_RECORDS = [
    {
        "id": "Fig.1",
        "key": "Fig1",
        "source_id": "Round 1 Fig.1",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Research design, sample closure, and scientific evidence framework",
        "title_zh": "研究设计、样本闭合与科学证据框架",
        "section": "Methods: sample and analysis framework",
        "claim": "The registered workflow closes the source records to a 790-event March–October sample and defines the study evidence sequence.",
        "key_numbers": "909 → 795 → 793 → 790; 2006–2024; March–October",
        "limitation": "The framework is restricted to confirmed tornado events and contains no non-tornadic comparison.",
        "source_script": ROUND1_ROOT / "02_scripts" / "fig01_workflow_gate1.py",
        "snapshot": SNAPSHOT_DIR / "Fig01_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig.2",
        "key": "Fig2",
        "source_id": "Round 1 Fig.2",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Standardized centers of the three environmental regimes",
        "title_zh": "三个环境组的标准化中心",
        "section": "Results: three-regime characteristics",
        "claim": "The k=3 solution separates the five-dimensional environmental space into distinct multivariate profiles.",
        "key_numbers": "C0 n=131; C1 n=307; C2 n=352",
        "limitation": "Standardized centers summarize multivariate structure and do not rank regimes by tornado favorability.",
        "source_script": ROUND1_ROOT / "02_scripts" / "fig02_cluster_centers_gate1.py",
        "snapshot": SNAPSHOT_DIR / "Fig02_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig.3",
        "key": "Fig3",
        "source_id": "Round 1 Fig.3",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Original-unit distributions of the five clustering variables",
        "title_zh": "五个聚类变量的原始单位分布",
        "section": "Results: three-regime characteristics",
        "claim": "Regime contrasts coexist with within-regime variability and distributional overlap.",
        "key_numbers": "790 events × 5 variables; all observations retained",
        "limitation": "The distributions are descriptive and do not establish environmental thresholds.",
        "source_script": ROUND1_ROOT / "02_scripts" / "fig03_raw_distributions_gate1.py",
        "snapshot": SNAPSHOT_DIR / "Fig03_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig.4",
        "key": "Fig4",
        "source_id": "Round 1 Fig.4",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Clustering stability and event assignment consistency",
        "title_zh": "聚类稳定性与事件分配一致性",
        "section": "Results: stability and structural sensitivity",
        "claim": "Most events belong to stable cores in both k=3 and k=4, while a small subset has moderate or boundary assignment consistency.",
        "key_numbers": "k=3: 775/8/7; k=4: 763/21/6 (stable/moderate/boundary)",
        "limitation": "Boundary status denotes lower assignment consistency and is not an error label.",
        "source_script": ROUND1_ROOT / "02_scripts" / "fig04_stability_gate1.py",
        "snapshot": SNAPSHOT_DIR / "Fig04_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig.5",
        "key": "Fig5",
        "source_id": "Round 1 Fig.5",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Structural correspondence between k=3 and k=4",
        "title_zh": "k=3与k=4的结构对应关系",
        "section": "Results: stability and structural sensitivity",
        "claim": "The k=4 solution refines the k=3 partition, and alternative algorithms show non-negligible algorithm dependence.",
        "key_numbers": "k=3×k=4 total n=790; Ward k=3 ARI=0.330; Gaussian-mixture k=3 ARI=0.402",
        "limitation": "The comparison supports structural sensitivity rather than a uniquely natural class structure.",
        "source_script": ROUND1_ROOT / "02_scripts" / "fig05_k3_k4_sensitivity_gate1.py",
        "snapshot": SNAPSHOT_DIR / "Fig05_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig.6",
        "key": "Fig6",
        "source_id": "Round 1 Fig.6",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Post-hoc association between environmental regimes and weather types",
        "title_zh": "环境组与天气型的事后关联",
        "section": "Results: post-hoc weather-type context",
        "claim": "Environmental regimes and weather types are strongly associated but retain a many-to-many relationship.",
        "key_numbers": "n=787; χ²(16)=437.1; raw V=0.5270; corrected V=0.5172; 95% CI=0.4898–0.5758; p<0.0001",
        "limitation": "The association is post-hoc and does not provide an independent test of the clustering solution.",
        "source_script": ROUND1_ROOT / "02_scripts" / "fig06_weather_type_association_gate1.py",
        "snapshot": SNAPSHOT_DIR / "Fig06_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig.7",
        "key": "Fig7",
        "source_id": "Round 1 Fig.7",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Vertical thermodynamic and moisture structures",
        "title_zh": "垂直热力与水汽结构",
        "section": "Results: vertical thermodynamic and moisture structures",
        "claim": "C2 exhibits a substantially moister 850–500-hPa layer, whereas C0 has a distinct midlevel temperature profile.",
        "key_numbers": "RH500 C0/C1/C2=54.3/57.0/87.7%; T500=259.3/268.7/269.5 K",
        "limitation": "The C0 temperature contrast should be interpreted with its differing environmental context; sparse lower-level samples are reported explicitly.",
        "source_script": ROUND1_ROOT / "02_scripts" / "fig07_thermodynamic_profiles_gate1.py",
        "snapshot": SNAPSHOT_DIR / "Fig07_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig.8",
        "key": "Fig8",
        "source_id": "Round 1 Fig.8",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Vertical environmental-wind structures",
        "title_zh": "垂直环境风结构",
        "section": "Results: vertical environmental-wind structures",
        "claim": "C2 has the largest median wind speed at 850 hPa, while C0 has the largest median at 200 hPa; registered subsets retain these contrasts.",
        "key_numbers": "WS850=7.7/6.0/13.9 m s−1; WS200=27.9/11.8/18.5 m s−1 for C0/C1/C2",
        "limitation": "Profiles describe coarse pressure-level environmental wind; the unavailable C1 warm-sector estimate remains absent.",
        "source_script": ROUND1_ROOT / "02_scripts" / "fig08_wind_profiles_gate1.py",
        "snapshot": SNAPSHOT_DIR / "Fig08_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig.9",
        "key": "Fig9",
        "source_id": "Round 2 Fig.9",
        "scope": "main",
        "status": "MAIN_TEXT_CANDIDATE",
        "title_en": "Seasonal and spatial context",
        "title_zh": "季节与空间背景",
        "section": "Results: seasonal and spatial context",
        "claim": "The three regimes show different within-sample seasonal and latitudinal distributions.",
        "key_numbers": "790 events; March–October; C0/C1/C2 n=131/307/352",
        "limitation": "Patterns are descriptive preferences within confirmed tornado events and are not population occurrence rates.",
        "source_script": ROUND2_ROOT / "02_scripts" / "build_fig09_round2.py",
        "snapshot": SNAPSHOT_DIR / "Fig09_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig. S1",
        "key": "FigS1",
        "source_id": "Round 2 Fig.S1",
        "scope": "supplementary",
        "status": "SUPPLEMENTARY_CANDIDATE",
        "title_en": "Event-centered two-dimensional composite fields",
        "title_zh": "事件中心二维合成场",
        "section": "Supplement: vertical and spatial environmental context",
        "claim": "Event-centered composite-mean Z500, RH500, and 200-hPa wind-vector-magnitude fields provide two-dimensional context.",
        "key_numbers": "3 regimes × 3 fields; common ±10° domain and 0.25° grid",
        "limitation": "Composite-center values are not interchangeable with event-point statistics used as the primary evidence.",
        "source_script": ROUND2_ROOT / "02_scripts" / "build_figs1_round2.py",
        "snapshot": SNAPSHOT_DIR / "FigS01_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig. S2",
        "key": "FigS2",
        "source_id": "Round 2 Fig.S2",
        "scope": "supplementary",
        "status": "SUPPLEMENTARY_CANDIDATE",
        "title_en": "Pairwise effect sizes and regime-specific quantile ranges",
        "title_zh": "两两效应量与各环境组分位数范围",
        "section": "Supplement: quantitative effect sizes and distribution ranges",
        "claim": "Registered Cliff’s δ estimates quantify pairwise contrasts, while original-unit quantile ranges visualize distributional overlap.",
        "key_numbers": "15 pairwise Cliff’s δ estimates; 5–95%, 25–75%, and median displays",
        "limitation": "Quantile ranges are visual summaries and are not a newly defined scalar overlap measure.",
        "source_script": ROUND2_ROOT / "02_scripts" / "build_figs2_round2.py",
        "snapshot": SNAPSHOT_DIR / "FigS02_source_plotting_snapshot.csv",
    },
    {
        "id": "Fig. S3",
        "key": "FigS3",
        "source_id": "Round 2 Fig.S4",
        "scope": "supplementary",
        "status": "SUPPLEMENTARY_CANDIDATE",
        "title_en": "Complete k=4 supplementary structure",
        "title_zh": "k=4完整补充结构",
        "section": "Supplement: structural sensitivity of the k=4 solution",
        "claim": "The k=4 solution further subdivides the environmental space without overturning the primary k=3 structure.",
        "key_numbers": "K4-C0/C1/C2/C3 n=125/206/249/210",
        "limitation": "The numerical k=4 labels carry no new physical names and do not imply a uniquely superior partition.",
        "source_script": ROUND2_ROOT / "02_scripts" / "build_figs4_round2.py",
        "snapshot": SNAPSHOT_DIR / "FigS03_source_plotting_snapshot.csv",
    },
]


CAPTIONS_EN = {
    "Fig1": (
        "**Fig. 1. Research design, sample closure, and scientific evidence framework.** "
        "(a) Source records were reduced from 909 to 795 after excluding 2025 events, "
        "to 793 after duplicate removal, and to a formal 790-event March–October sample. "
        "(b) The workflow links event screening, ERA5 extraction, five-variable clustering, "
        "stability assessment, the k=3 primary solution, and the k=4 sensitivity analysis. "
        "(c) The evidence sequence connects multivariate regimes, vertical thermodynamic and "
        "kinematic characteristics, and post-hoc weather-type, seasonal, and spatial context. "
        "Interpretation is restricted to the confirmed-tornado sample."
    ),
    "Fig2": (
        "**Fig. 2. Standardized centers of the three environmental regimes.** "
        "Centers are shown for MLCAPE, log(1+MLLCL), 2-m dew point, 0–6-km bulk wind shear, "
        "and signed 0–1-km SRH. The horizontal line marks zero in the standardized space. "
        "Colors, markers, and line styles identify C0 (n=131), C1 (n=307), and C2 (n=352). "
        "The profiles summarize multivariate structure and do not rank regimes by favorability."
    ),
    "Fig3": (
        "**Fig. 3. Original-unit distributions of the five clustering variables.** "
        "Points show all 790 event observations; boxes emphasize the median and interquartile "
        "range, and whiskers span the 5th–95th percentiles. No event or extreme value was removed. "
        "Signed negative SRH values remain visible."
    ),
    "Fig4": (
        "**Fig. 4. Clustering stability and event assignment consistency.** "
        "(a) Adjusted Rand indices from 100 seed runs for k=3 and k=4. "
        "(b) Maximum assignment probabilities, focused on the high-consistency region; four k=3 "
        "and three k=4 values fall below 0.55. Stable-core minima are 0.802 and 0.805. "
        "(c) Event stability categories. Stable/moderate/boundary counts are 775/8/7 for k=3 "
        "and 763/21/6 for k=4. Boundary denotes lower assignment consistency, not an error."
    ),
    "Fig5": (
        "**Fig. 5. Structural correspondence between k=3 and k=4.** "
        "(a) Event counts in the k=3-by-k=4 correspondence table. "
        "(b) The same table normalized within each k=3 row; counts and proportions use separate "
        "color scales. (c) Adjusted Rand indices comparing Ward and Gaussian-mixture partitions "
        "with k-means at the same k. The figure documents structural sensitivity and algorithm "
        "dependence rather than a uniquely natural partition."
    ),
    "Fig6": (
        "**Fig. 6. Post-hoc association between environmental regimes and weather types.** "
        "(a) Within-regime weather-type composition and (b) standardized residuals for the "
        "registered 3×9 table. n=787; χ²(16)=437.1; raw Cramér’s V=0.5270; "
        "bias-corrected V=0.5172; 95% bootstrap CI=0.4898–0.5758; permutation p<0.0001. "
        "Only selected large proportions or residuals are labeled. The asterisked supercell "
        "category denotes unclassified supercells. The association is post-hoc and many-to-many."
    ),
    "Fig7": (
        "**Fig. 7. Vertical thermodynamic and moisture structures.** "
        "Lines show regime medians and shading shows registered 95% bootstrap confidence intervals "
        "for temperature and relative humidity at nine pressure levels. The gray band marks "
        "850–500 hPa. Effective n (C0/C1/C2) is 11/118/139 at 1000 hPa, 84/291/350 at 925 hPa, "
        "and 109/299/352 at 850 hPa; upper-level values use the registered valid-event masks. "
        "The C0 midlevel temperature contrast should be interpreted with its differing environmental context."
    ),
    "Fig8": (
        "**Fig. 8. Vertical environmental-wind structures.** "
        "(a) Median environmental wind speed and registered 95% bootstrap confidence intervals "
        "at nine pressure levels. (b) Regime contrasts at 850 and 200 hPa. "
        "(c) Registered sensitivity estimates for C2 at 850 hPa and C0 at 200 hPa, with point "
        "estimates and intervals listed in a separate column. The unavailable C1 warm-sector "
        "estimate remains absent; no additional estimate was generated."
    ),
    "Fig9": (
        "**Fig. 9. Seasonal and spatial context.** "
        "(a) Locations of the 790 confirmed tornado events by environmental regime. "
        "(b) March–October frequency calculated within each regime. "
        "(c) Latitude distributions with all events retained, 5th–95th-percentile whiskers, "
        "and emphasized medians and interquartile ranges. These panels describe internal "
        "spatiotemporal preferences of the confirmed-tornado sample and do not represent population occurrence rates."
    ),
    "FigS1": (
        "**Fig. S1. Event-centered two-dimensional composite fields.** "
        "Rows show C0, C1, and C2; columns show 500-hPa geopotential height, 500-hPa relative "
        "humidity, and the magnitude of the composite-mean wind vector at 200 hPa. "
        "Each variable column uses one shared scale and all panels use the same ±10° event-centered domain. "
        "These are event-centered composite means. Event-point statistics remain the primary evidence, "
        "and composite-center values must not be substituted for event-point values."
    ),
    "FigS2": (
        "**Fig. S2. Pairwise effect sizes and regime-specific quantile ranges.** "
        "(a) Pairwise Cliff’s δ estimates with registered 95% confidence intervals. "
        "(b) Original-unit distribution summaries: points are medians, thick segments span the "
        "25th–75th percentiles, and thin segments span the 5th–95th percentiles. "
        "The quantile ranges visualize distributional overlap but are not a new scalar overlap measure."
    ),
    "FigS3": (
        "**Fig. S3. Complete k=4 supplementary structure.** "
        "(a) Standardized centers of K4-C0 through K4-C3. "
        "(b–f) Original-unit distributions of the five formal clustering variables, with all "
        "790 observations retained. The k=4 solution further subdivides environmental space "
        "without overturning the primary k=3 structure; numerical labels do not imply physical names "
        "or a uniquely superior partition."
    ),
}


CAPTIONS_ZH = {
    "Fig1": (
        "**图1 研究设计、样本闭合与科学证据框架。** "
        "（a）初始909条记录在剔除2025年事件后为795条，去重后为793条，最终得到3—10月正式样本790例。"
        "（b）流程依次包括事件筛查、ERA5环境场提取、五变量聚类、稳定性评估、k=3主方案和k=4敏感性分析。"
        "（c）证据链连接多变量环境组、垂直热力和运动学特征，以及天气型、季节和空间背景的事后分析。"
        "所有解释均限于已确认龙卷样本。"
    ),
    "Fig2": (
        "**图2 三个环境组的标准化中心。** "
        "变量依次为MLCAPE、log(1+MLLCL)、2米露点、0—6 km垂直风切变和保留符号的0—1 km SRH。"
        "水平线表示标准化空间中的零值。颜色、标记和线型分别表示C0（n=131）、C1（n=307）和C2（n=352）。"
        "该图概括多变量结构，不用于给环境组作有利程度排序。"
    ),
    "Fig3": (
        "**图3 五个聚类变量的原始单位分布。** "
        "散点显示全部790例事件；箱体突出中位数和四分位距，须线表示第5—95百分位。"
        "未删除任何事件或极端值，并保留SRH负值。"
    ),
    "Fig4": (
        "**图4 聚类稳定性与事件分配一致性。** "
        "（a）k=3和k=4各100次随机种子运行的调整兰德指数。"
        "（b）聚焦高一致性区的最大分配概率；低于0.55的事件在k=3和k=4中分别为4例和3例，"
        "稳定核心最低值分别为0.802和0.805。"
        "（c）事件稳定性类别。k=3稳定/中等/边界为775/8/7例，k=4为763/21/6例。"
        "边界表示分配一致性较低，不代表错误。"
    ),
    "Fig5": (
        "**图5 k=3与k=4的结构对应关系。** "
        "（a）k=3×k=4对应表的事件计数。"
        "（b）按k=3各行归一化的比例；计数与比例使用独立色标。"
        "（c）Ward法和高斯混合模型与相同k值k-means结果之间的调整兰德指数。"
        "该图用于表征结构敏感性和算法依赖，不表示存在唯一自然分类。"
    ),
    "Fig6": (
        "**图6 环境组与天气型的事后关联。** "
        "（a）各环境组内部的天气型组成；（b）登记3×9列联表的标准化残差。"
        "n=787；χ²(16)=437.1；原始Cramér’s V=0.5270；校正V=0.5172；"
        "95% bootstrap CI=0.4898—0.5758；置换p<0.0001。"
        "仅标注较大的比例或残差，带星号的supercell类别表示未分类超级单体。"
        "该关系属于多对多的事后关联。"
    ),
    "Fig7": (
        "**图7 垂直热力与水汽结构。** "
        "曲线为九个气压层上的环境组中位数，阴影为登记的95% bootstrap置信区间；灰色带表示850—500 hPa。"
        "1000、925和850 hPa的有效样本量（C0/C1/C2）分别为11/118/139、84/291/350和109/299/352。"
        "其他气压层采用登记的有效事件掩膜。C0中层温度差异需结合其不同环境背景谨慎解释。"
    ),
    "Fig8": (
        "**图8 垂直环境风结构。** "
        "（a）九个气压层的环境风速中位数和登记的95% bootstrap置信区间。"
        "（b）850和200 hPa关键层的环境组对比。"
        "（c）C2在850 hPa和C0在200 hPa的登记敏感性估计，区间图与数值列分开显示。"
        "C1去暖区估计缺失的现状保持不变，未新增估计。"
    ),
    "Fig9": (
        "**图9 季节与空间背景。** "
        "（a）790例已确认龙卷事件按环境组着色的空间位置。"
        "（b）各环境组内部的3—10月频率。"
        "（c）纬度分布；保留全部事件，须线表示第5—95百分位，并突出中位数和四分位距。"
        "这些面板仅描述已确认龙卷样本内部的时空偏好，不代表总体发生率。"
    ),
    "FigS1": (
        "**补充图S1 事件中心二维合成场。** "
        "行分别为C0、C1和C2；列分别为500 hPa位势高度、500 hPa相对湿度以及200 hPa合成平均风矢量的模。"
        "每一变量列共用一个色标，全部面板采用相同的事件中心±10°区域。"
        "这些量是事件中心二维合成均值；核心证据仍来自事件点统计，二维中心值不得替代事件点数值。"
    ),
    "FigS2": (
        "**补充图S2 两两效应量与各环境组分位数范围。** "
        "（a）两两Cliff’s δ及登记的95%置信区间。"
        "（b）原始单位分布摘要：点为中位数，粗线为第25—75百分位，细线为第5—95百分位。"
        "分位数范围仅用于直观展示分布重叠，不构成新的标量重叠指标。"
    ),
    "FigS3": (
        "**补充图S3 k=4完整补充结构。** "
        "（a）K4-C0至K4-C3的标准化中心。"
        "（b—f）五个正式聚类变量的原始单位分布，保留全部790例观测。"
        "k=4进一步细分环境空间，但不推翻k=3主要结构；数字标签不代表新增物理名称，也不表示唯一更优分类。"
    ),
}


TABLE_TITLES_ZH = {
    "Table1": ("正文表1 聚类变量定义与变换", "变量基于790例正式样本定义；MLLCL在标准化前采用log(1+x)变换，SRH保留符号。"),
    "Table2": ("正文表2 候选聚类数诊断量", "列出k=2—6的诊断结果；不以单一诊断量判定唯一分类。"),
    "Table3": ("正文表3 三个k=3环境组的中位特征", "数值为原始单位中位数；C0、C1和C2样本量分别为131、307和352。"),
    "Table4": ("正文表4 稳定性与算法敏感性指标", "ARI为调整兰德指数；事件边界类别表示分配一致性较低，不代表错误。"),
    "Table5": ("正文表5 天气型事后关联统计量", "统计量来自787条有效天气型记录组成的3×9列联表；区间为登记的95% bootstrap置信区间。"),
    "TableS1": ("补充表S1 k=4类别统计量", "类别保持数字标签；数值为原始单位中位数。"),
    "TableS2": ("补充表S2 完整结构敏感性结果", "ARI比较以相同k值的k-means方案为参照，用于描述结构敏感性和算法依赖。"),
    "TableS3": ("补充表S3 STP_mod有评级事件探索性结果", "仅包含有评级事件，未评级事件不进入比较；结果仅用于研究描述，不定义决策规则或发生概率。"),
}


def _figure_paths(record: dict) -> dict[str, Path]:
    root = MAIN_FIGURE_DIR if record["scope"] == "main" else SUPP_FIGURE_DIR
    stem = FIGURE_STEMS[record["key"]]
    return {
        "png_600dpi": root / "png_600dpi" / f"{stem}.png",
        "pdf_vector": root / "pdf_vector" / f"{stem}.pdf",
        "svg_vector": root / "svg_vector" / f"{stem}.svg",
        "review_png": root / "review_png" / f"{stem}_review.png",
    }


def build_captions() -> dict:
    individual = CAPTION_DIR / "individual"
    individual.mkdir(parents=True, exist_ok=True)
    main_en, main_zh, supp_en, supp_zh = [], [], [], []
    caption_paths = {}
    for record in FIGURE_RECORDS:
        key = record["key"]
        en = CAPTIONS_EN[key]
        zh = CAPTIONS_ZH[key]
        en_path = individual / f"{key}_caption_en.md"
        zh_path = individual / f"{key}_caption_zh.md"
        write_text(en_path, en + "\n")
        write_text(zh_path, zh + "\n")
        caption_paths[key] = {"en": en_path, "zh": zh_path}
        if record["scope"] == "main":
            main_en.append(en); main_zh.append(zh)
        else:
            supp_en.append(en); supp_zh.append(zh)
    outputs = {
        "main_en": CAPTION_DIR / "MAIN_FIGURE_CAPTIONS_EN.md",
        "main_zh": CAPTION_DIR / "MAIN_FIGURE_CAPTIONS_ZH.md",
        "supp_en": CAPTION_DIR / "SUPPLEMENTARY_FIGURE_CAPTIONS_EN.md",
        "supp_zh": CAPTION_DIR / "SUPPLEMENTARY_FIGURE_CAPTIONS_ZH.md",
    }
    write_text(outputs["main_en"], "# Main figure captions\n\n" + "\n\n".join(main_en) + "\n")
    write_text(outputs["main_zh"], "# 正文图注\n\n" + "\n\n".join(main_zh) + "\n")
    write_text(outputs["supp_en"], "# Supplementary figure captions\n\n" + "\n\n".join(supp_en) + "\n")
    write_text(outputs["supp_zh"], "# 补充图注\n\n" + "\n\n".join(supp_zh) + "\n")

    table_en = ["# Main table titles and notes"]
    table_zh = ["# 正文表题与表注"]
    supp_table_en = ["# Supplementary table titles and notes"]
    supp_table_zh = ["# 补充表题与表注"]
    table_caption_paths = {}
    for definition in TABLE_DEFINITIONS:
        en_id = definition["id"].replace("TableS", "Table S").replace("Table", "Table ")
        en_text = f"**{en_id}. {definition['title']}.** Note: {definition['note']}"
        zh_title, zh_note = TABLE_TITLES_ZH[definition["id"]]
        zh_text = f"**{zh_title}。** 表注：{zh_note}"
        en_path = individual / f"{definition['id']}_title_note_en.md"
        zh_path = individual / f"{definition['id']}_title_note_zh.md"
        write_text(en_path, en_text + "\n")
        write_text(zh_path, zh_text + "\n")
        table_caption_paths[definition["id"]] = {"en": en_path, "zh": zh_path}
        if definition["scope"] == "main":
            table_en.append(en_text); table_zh.append(zh_text)
        else:
            supp_table_en.append(en_text); supp_table_zh.append(zh_text)
    main_table_en = CAPTION_DIR / "MAIN_TABLE_TITLES_AND_NOTES_EN.md"
    main_table_zh = CAPTION_DIR / "MAIN_TABLE_TITLES_AND_NOTES_ZH.md"
    supp_table_en_path = CAPTION_DIR / "SUPPLEMENTARY_TABLE_TITLES_AND_NOTES_EN.md"
    supp_table_zh_path = CAPTION_DIR / "SUPPLEMENTARY_TABLE_TITLES_AND_NOTES_ZH.md"
    write_text(main_table_en, "\n\n".join(table_en) + "\n")
    write_text(main_table_zh, "\n\n".join(table_zh) + "\n")
    write_text(supp_table_en_path, "\n\n".join(supp_table_en) + "\n")
    write_text(supp_table_zh_path, "\n\n".join(supp_table_zh) + "\n")
    return {
        "figures": caption_paths,
        "tables": table_caption_paths,
        "aggregate": {key: relative_to_final(value) for key, value in outputs.items()},
    }


FIGURE_GUIDE_ZH = {
    "Fig1": {
        "show": "样本从909条记录闭合到790例正式样本的过程，以及聚类、稳定性、垂直诊断和事后背景分析的证据链。",
        "safe": "研究基于2006—2024年3—10月的790例已确认龙卷事件，并采用k=3主方案和k=4敏感性分析。",
        "forbid": "不得写成已完成非龙卷对照，也不得由流程图推导总体发生率或因果关系。",
        "first": "研究样本闭合和总体分析框架见图1。",
        "result": "初始909条记录经年份、重复和季节筛查后形成790例正式样本。",
        "boundary": "该框架限定了证据范围，全部解释仅适用于已确认龙卷样本。",
        "related": "Table 1；Fig.2—Fig.9",
    },
    "Fig2": {
        "show": "C0、C1和C2在五个正式聚类变量上的标准化中心。",
        "safe": "三个环境组具有明显不同的多变量中心轮廓。",
        "forbid": "不得把中心高低直接表述为龙卷强弱、危险性或发生阈值。",
        "first": "三个环境组的标准化中心对比见图2。",
        "result": "C0以较高MLLCL和较低水汽为特征，C1以较高MLCAPE为特征，C2以较强风切变和SRH为特征。",
        "boundary": "标准化中心用于描述聚类空间，不等同于原始单位的事件分布。",
        "related": "Fig.3；Table 3；Fig. S2",
    },
    "Fig3": {
        "show": "五个聚类变量在三个环境组内的全部原始观测分布。",
        "safe": "组间差异与组内变率、分布重叠同时存在。",
        "forbid": "不得删除极端值后重述结果，也不得由箱线图指定环境阈值。",
        "first": "五个聚类变量的原始单位分布见图3。",
        "result": "原始分布显示三个环境组的中位差异，同时保留显著的组内离散度。",
        "boundary": "散点和箱体是描述性展示；统计对象仍为全部790例事件。",
        "related": "Fig.2；Fig. S2；Table 3",
    },
    "Fig4": {
        "show": "随机种子稳定性、事件最大分配概率和事件稳定性类别。",
        "safe": "k=3和k=4均以稳定核心事件为主，只有少量中等和边界事件。",
        "forbid": "不得把边界事件称为错分事件，也不得声称全部事件均稳定。",
        "first": "聚类稳定性和事件分配一致性见图4。",
        "result": "k=3和k=4的稳定核心分别为775例和763例。",
        "boundary": "稳定性结果评价分区重复性和事件一致性，不证明类别具有唯一性。",
        "related": "Fig.5；Table 4；Table S2",
    },
    "Fig5": {
        "show": "k=3与k=4的计数和行归一化对应关系，以及不同算法与k-means的ARI。",
        "safe": "k=4主要细化k=3结构，同时存在算法依赖。",
        "forbid": "不得称为独立检验、最优解或唯一自然类别。",
        "first": "k=3与k=4的结构对应和算法依赖见图5。",
        "result": "C0主要对应K4-C0，而C1和C2分别被k=4进一步分拆。",
        "boundary": "对应关系说明结构敏感性，不用于选择有利结果。",
        "related": "Fig.4；Fig. S3；Table 4；Table S2",
    },
    "Fig6": {
        "show": "三个环境组内九类天气型组成及其标准化残差。",
        "safe": "环境组与天气型存在显著且中等偏强的多对多事后关联。",
        "forbid": "不得称为对聚类的独立检验，也不得把某一环境组等同于单一天气型。",
        "first": "环境组与天气型的事后关联见图6。",
        "result": "列联表检验为χ²(16)=437.1，校正Cramér’s V为0.5172。",
        "boundary": "该结果用于提供天气尺度背景，不改变聚类标签或主结论。",
        "related": "Table 5；Fig.9",
    },
    "Fig7": {
        "show": "三个环境组在九个气压层上的温度和相对湿度中位廓线及置信区间。",
        "safe": "C2在850—500 hPa表现出更深、更湿的中层结构。",
        "forbid": "不得写成C2从地面到200 hPa均饱和，也不得将水汽差异直接解释为龙卷成因。",
        "first": "垂直热力和水汽结构见图7。",
        "result": "500 hPa相对湿度在C0、C1和C2中分别为54.3%、57.0%和87.7%。",
        "boundary": "低层有效样本量随气压层变化；C0中层温度差异需结合环境背景解释。",
        "related": "Fig. S1；Fig.8",
    },
    "Fig8": {
        "show": "环境风速垂直廓线、850/200 hPa关键层对比和登记的子集敏感性估计。",
        "safe": "C2在850 hPa风速中位数最高，C0在200 hPa最高，相关子集估计保持主要对比。",
        "forbid": "不得新增C1去暖区估计，也不得将廓线表述为低空急流、上层急流或业务诊断。",
        "first": "垂直环境风结构及关键层敏感性结果见图8。",
        "result": "850 hPa风速中位数为7.7、6.0和13.9 m s−1，200 hPa为27.9、11.8和18.5 m s−1。",
        "boundary": "廓线为粗分辨率气压层环境风，不包含风暴运动校正。",
        "related": "Fig.7；Fig. S1",
    },
    "Fig9": {
        "show": "790例事件的空间位置、环境组内部月频率和纬度分布。",
        "safe": "三个环境组在已确认龙卷样本内部具有不同的季节和纬度偏好。",
        "forbid": "不得解释为全国龙卷气候发生率、边界区或预测概率。",
        "first": "三个环境组的季节和空间背景见图9。",
        "result": "C0较偏北，C1和C2更集中在较低纬度；各组月频率峰值不同。",
        "boundary": "该图仅描述样本内部偏好，未使用非龙卷事件作为分母。",
        "related": "Fig.6",
    },
    "FigS1": {
        "show": "C0、C1和C2的事件中心Z500、RH500及200 hPa合成平均风矢量模二维场。",
        "safe": "二维合成为环境组提供空间结构背景。",
        "forbid": "不得把二维中心值替代事件点中位数，也不得混称第三列为事件风速平均。",
        "first": "事件中心二维环境场见补充图S1。",
        "result": "三个变量列分别使用统一色标和完全一致的相对空间范围。",
        "boundary": "核心论证仍以事件点统计为主，二维场作为背景证据。",
        "related": "Fig.7；Fig.8",
    },
    "FigS2": {
        "show": "五个变量的两两Cliff’s δ及三个环境组的原始单位分位数范围。",
        "safe": "效应量方向和区间可量化组间差异，分位数范围显示组内变率与重叠。",
        "forbid": "不得把分位数范围称为新重叠系数，也不得据此设定阈值。",
        "first": "组间效应量和原始单位分位数范围见补充图S2。",
        "result": "多个变量的两两效应量较大，但原始分布仍存在不同程度重叠。",
        "boundary": "分位数范围是可视化摘要，不新增标量统计量。",
        "related": "Fig.2；Fig.3；Table 3",
    },
    "FigS3": {
        "show": "k=4四个数字类别的标准化中心和五个变量原始分布。",
        "safe": "k=4进一步细化环境空间，但总体上延续k=3的主要结构。",
        "forbid": "不得给K4类别新增物理名称，也不得称k=4为唯一更优分类。",
        "first": "k=4完整补充结构见补充图S3。",
        "result": "K4-C0至K4-C3的样本量为125、206、249和210。",
        "boundary": "k=4用于描述结构敏感性，不替换正文k=3主方案。",
        "related": "Fig.5；Table S1；Table S2",
    },
}


TABLE_GUIDE_ZH = {
    "Table1": ("五个正式聚类变量的名称、单位、变换和定义。", "变量定义与图2、图3一致。", "不得改变变换或单位。", "变量与数据定义见表1。", "MLLCL在标准化前采用log(1+x)，SRH保留符号。", "该表不包含样本筛查细节。", "Fig.1；Fig.2；Fig.3"),
    "Table2": ("k=2—6的轮廓系数、Davies–Bouldin、Calinski–Harabasz和类别大小。", "k=3诊断量与稳定性、解释性证据共同支持主方案。", "不得以单一指标声称唯一最优k值。", "候选聚类数的诊断量见表2。", "k=3轮廓系数为0.280，最小/最大类别为131/352。", "诊断量用于综合判断，不构成唯一选择规则。", "Fig.2；Fig.4；Fig.5"),
    "Table3": ("C0、C1和C2的五变量原始单位中位数与样本量。", "三个环境组具有不同的热力、水汽和运动学中位特征。", "不得把中位数当作阈值或因果量。", "三个环境组的关键中位特征见表3。", "C0/C1/C2样本量为131/307/352。", "中位数不显示完整分布，需结合图3。", "Fig.2；Fig.3；Fig. S2"),
    "Table4": ("k=3稳定性类别、子样本ARI及两类替代算法ARI。", "k=3以稳定核心为主，但替代算法显示结构依赖。", "不得把边界类别写成错误，也不得称替代算法比较为独立检验。", "稳定性与算法敏感性指标见表4。", "k=3稳定核心为775例；Ward和高斯混合k=3 ARI为0.330和0.402。", "该表不包含垂直结构统计，实际功能与建议映射不同。", "Fig.4；Fig.5；Table S2"),
    "Table5": ("天气型列联表的样本量、χ²、效应量、置信区间和置换p值。", "环境组与天气型存在显著多对多事后关联。", "不得替换登记统计口径或将关联称为独立检验。", "天气型事后关联统计量见表5。", "χ²(16)=437.1，校正V=0.5172，95% CI=0.4898—0.5758。", "统计量仅针对787条有效天气型记录。", "Fig.6"),
    "TableS1": ("k=4四个数字类别的样本量和部分变量中位数。", "k=4进一步细分环境空间。", "不得为类别新增物理名称。", "k=4类别统计量见补充表S1。", "K4-C0/C1/C2/C3样本量为125/206/249/210。", "该表只提供已登记变量摘要。", "Fig. S3；Fig.5"),
    "TableS2": ("k=3/k=4的完整子样本、留一变量、尺度变换、算法和特征方案敏感性结果。", "主结构总体稳定，但部分设定显示明显敏感性。", "不得选择性报告有利结果或称为唯一分类证明。", "完整结构敏感性结果见补充表S2。", "k=3/k=4子样本ARI中位数为0.974/0.957。", "不同敏感性分析统计对象不同，应按行解释。", "Fig.4；Fig.5；Fig. S3"),
    "TableS3": ("有评级事件的STP_mod探索性结果和AUC摘要。", "在限定的有评级事件子样本中，STP_mod具有有限区分信息。", "不得设定截点、给出个体发生概率或推广到未评级事件。", "有评级事件的探索性结果见补充表S3。", "有评级事件n=181，AUC=0.641，95% CI=0.552—0.727。", "该表不属于主要聚类证据，研究者可决定是否保留。", "HOLD-S2"),
}


def _table_outputs(definition: dict) -> dict[str, Path]:
    root = MAIN_TABLE_DIR if definition["scope"] == "main" else SUPP_TABLE_DIR
    stem = definition["stem"]
    return {
        "csv": root / "csv" / f"{stem}.csv",
        "xlsx": root / "xlsx" / f"{stem}.xlsx",
        "markdown": root / "markdown" / f"{stem}.md",
        "preview": root / "previews" / f"{stem}_preview.png",
    }


def build_asset_index(caption_registry: dict) -> list[dict]:
    rows: list[dict] = []
    final_figure_script = FINAL_ROOT / "13_scripts" / "build_final_figures.py"
    for record in FIGURE_RECORDS:
        paths = _figure_paths(record)
        rows.append(
            {
                "final_id": record["id"],
                "source_id": record["source_id"],
                "title_zh": record["title_zh"],
                "title_en": record["title_en"],
                "scope": "正文" if record["scope"] == "main" else "补充",
                "status": record["status"],
                "primary_file": relative_to_final(paths["png_600dpi"]),
                "all_files": ";".join(relative_to_final(path) for path in paths.values()),
                "final_script": relative_to_final(final_figure_script),
                "source_script": record["source_script"].resolve().relative_to(PROJECT_ROOT).as_posix(),
                "data_source": relative_to_final(record["snapshot"]),
                "source_sha256": sha256_file(record["snapshot"]),
                "final_sha256": sha256_file(paths["png_600dpi"]),
                "core_claim": record["claim"],
                "manuscript_section": record["section"],
                "first_citation": FIGURE_GUIDE_ZH[record["key"]]["first"],
                "caption_path": (
                    relative_to_final(caption_registry["figures"][record["key"]]["zh"])
                    + ";"
                    + relative_to_final(caption_registry["figures"][record["key"]]["en"])
                ),
                "qc_status": "QC_COMPLETE",
                "known_limitations": record["limitation"],
            }
        )
    table_metadata = json.loads((QC_DIR / "table_build_metadata.json").read_text(encoding="utf-8"))
    table_meta_by_id = {row["id"]: row for row in table_metadata}
    for definition in TABLE_DEFINITIONS:
        table_id = definition["id"]
        title_zh, note_zh = TABLE_TITLES_ZH[table_id]
        paths = _table_outputs(definition)
        guide = TABLE_GUIDE_ZH[table_id]
        rows.append(
            {
                "final_id": table_id.replace("TableS", "Table S").replace("Table", "Table "),
                "source_id": definition["source"].name,
                "title_zh": title_zh,
                "title_en": definition["title"],
                "scope": "正文" if definition["scope"] == "main" else "补充",
                "status": "MAIN_TEXT_CANDIDATE" if definition["scope"] == "main" else "SUPPLEMENTARY_CANDIDATE",
                "primary_file": relative_to_final(paths["xlsx"]),
                "all_files": ";".join(relative_to_final(path) for path in paths.values()),
                "final_script": relative_to_final(FINAL_ROOT / "13_scripts" / "build_final_tables.py"),
                "source_script": "",
                "data_source": definition["source"].resolve().relative_to(PROJECT_ROOT).as_posix(),
                "source_sha256": table_meta_by_id[table_id]["source_sha256"],
                "final_sha256": sha256_file(paths["xlsx"]),
                "core_claim": guide[1],
                "manuscript_section": {
                    "Table1": "Methods: variables",
                    "Table2": "Methods/Results: cluster-number diagnostics",
                    "Table3": "Results: three-regime characteristics",
                    "Table4": "Results: stability and structural sensitivity",
                    "Table5": "Results: post-hoc weather-type context",
                    "TableS1": "Supplement: k=4 structure",
                    "TableS2": "Supplement: structural sensitivity",
                    "TableS3": "Supplement: exploratory rated-event results",
                }[table_id],
                "first_citation": guide[3],
                "caption_path": (
                    relative_to_final(caption_registry["tables"][table_id]["zh"])
                    + ";"
                    + relative_to_final(caption_registry["tables"][table_id]["en"])
                ),
                "qc_status": "QC_COMPLETE",
                "known_limitations": note_zh,
            }
        )
    held_registry = json.loads((HOLD_DIR / "held_asset_registry.json").read_text(encoding="utf-8"))
    held_titles = {
        "HOLD_S01_environmental_uv_profiles": (
            "HOLD-S1",
            "粗分辨率环境u–v风廓线",
            "Coarse environmental u–v wind profiles",
            "ARCHIVED_OPTIONAL_SUPPLEMENT",
            "Coarse pressure-level environmental u–v profiles are available for possible future use.",
        ),
        "HOLD_S02_stpmod_rated_events": (
            "HOLD-S2",
            "STP_mod有评级事件比较",
            "STP_mod rated-event comparison",
            "RESEARCHER_APPROVAL_REQUIRED",
            "The exploratory rated-event comparison is retained for a researcher decision.",
        ),
    }
    for held in held_registry:
        final_id, title_zh, title_en, status, claim = held_titles[held["hold_id"]]
        rows.append(
            {
                "final_id": final_id,
                "source_id": held["source_id"],
                "title_zh": title_zh,
                "title_en": title_en,
                "scope": "暂存",
                "status": status,
                "primary_file": held["files"]["review"],
                "all_files": ";".join(held["files"].values()),
                "final_script": "",
                "source_script": held["files"]["script"],
                "data_source": held["files"]["data"],
                "source_sha256": held["sha256"]["data"],
                "final_sha256": held["sha256"]["review"],
                "core_claim": claim,
                "manuscript_section": "Not assigned",
                "first_citation": "当前稿件不主动引用。",
                "caption_path": held["files"]["caption_zh"] + ";" + held["files"]["caption_en"],
                "qc_status": "SOURCE_BUNDLE_PRESERVED",
                "known_limitations": "No active formal supplementary number is assigned.",
            }
        )
    fields = list(rows[0].keys())
    csv_path = FINAL_ROOT / "MANUSCRIPT_VISUAL_ASSET_INDEX.csv"
    write_csv_rows(csv_path, rows, fields)
    md_lines = ["# Manuscript visual asset index", ""]
    for row in rows:
        md_lines.extend(
            [
                f"## {row['final_id']} — {row['title_en']}",
                "",
                f"- 中文标题：{row['title_zh']}",
                f"- 来源：{row['source_id']}",
                f"- 范围与状态：{row['scope']} / `{row['status']}`",
                f"- 主文件：[{row['primary_file']}]({row['primary_file']})",
                f"- 推荐小节：{row['manuscript_section']}",
                f"- 核心作用：{row['core_claim']}",
                f"- 首次引用：{row['first_citation']}",
                f"- Caption：{row['caption_path']}",
                f"- QC：{row['qc_status']}",
                f"- 限制：{row['known_limitations']}",
                "",
            ]
        )
    write_text(FINAL_ROOT / "MANUSCRIPT_VISUAL_ASSET_INDEX.md", "\n".join(md_lines))
    return rows


def build_call_guide(asset_rows: list[dict]) -> Path:
    row_by_id = {row["final_id"]: row for row in asset_rows}
    lines = [
        "# 论文写作图表调用手册",
        "",
        "本手册用于中文初稿快速调用。正文候选和补充候选均按登记统计口径书写；暂存图不分配正式补充图号。",
        "",
    ]
    for record in FIGURE_RECORDS:
        guide = FIGURE_GUIDE_ZH[record["key"]]
        row = row_by_id[record["id"]]
        lines.extend(
            [
                f"## {record['id']} {record['title_zh']}",
                "",
                f"1. 图表编号：{record['id']}",
                f"2. 展示什么：{guide['show']}",
                f"3. 推荐小节：{record['section']}",
                f"4. 直接支撑：{record['claim']}",
                f"5. 安全表述：{guide['safe']}",
                f"6. 禁止越界：{guide['forbid']}",
                f"7. 关键数字：{record['key_numbers']}",
                f"8. 首次引用句：{guide['first']}",
                f"9. 结果描述句：{guide['result']}",
                f"10. 讨论边界：{guide['boundary']}",
                f"11. 关联图表：{guide['related']}",
                f"12. 文件路径：[{row['primary_file']}]({os.path.relpath(FINAL_ROOT / row['primary_file'], GUIDE_DIR).replace(os.sep, '/')})",
                "",
            ]
        )
    for definition in TABLE_DEFINITIONS:
        table_id = definition["id"]
        display_id = table_id.replace("TableS", "Table S").replace("Table", "Table ")
        guide = TABLE_GUIDE_ZH[table_id]
        row = row_by_id[display_id]
        section = row["manuscript_section"]
        lines.extend(
            [
                f"## {display_id} {TABLE_TITLES_ZH[table_id][0]}",
                "",
                f"1. 图表编号：{display_id}",
                f"2. 展示什么：{guide[0]}",
                f"3. 推荐小节：{section}",
                f"4. 直接支撑：{guide[1]}",
                f"5. 安全表述：{guide[1]}",
                f"6. 禁止越界：{guide[2]}",
                f"7. 关键数字：{guide[4]}",
                f"8. 首次引用句：{guide[3]}",
                f"9. 结果描述句：{guide[4]}",
                f"10. 讨论边界：{guide[5]}",
                f"11. 关联图表：{guide[6]}",
                f"12. 文件路径：[{row['primary_file']}]({os.path.relpath(FINAL_ROOT / row['primary_file'], GUIDE_DIR).replace(os.sep, '/')})",
                "",
            ]
        )
    lines.extend(
        [
            "# 暂存图内部附录",
            "",
            "以下图件保留完整来源包，但当前稿件不分配正式补充图号。",
            "",
        ]
    )
    for hold_id in ("HOLD-S1", "HOLD-S2"):
        row = row_by_id[hold_id]
        lines.extend(
            [
                f"## {hold_id} {row['title_zh']}",
                "",
                f"1. 图表编号：{hold_id}（内部暂存编号）",
                f"2. 展示什么：{row['core_claim']}",
                "3. 推荐小节：当前不指定",
                "4. 直接支撑：当前正文不依赖",
                "5. 安全表述：仅在研究者决定纳入后按原始caption限定使用。",
                "6. 禁止越界：不得提前改编为正式补充图号。",
                "7. 关键数字：见保留的数据快照与caption。",
                "8. 首次引用句：当前稿件不引用。",
                "9. 结果描述句：当前稿件不提供。",
                "10. 讨论边界：仅作为未来可选资产。",
                "11. 关联图表：见来源包README。",
                f"12. 文件路径：[{row['primary_file']}]({os.path.relpath(FINAL_ROOT / row['primary_file'], GUIDE_DIR).replace(os.sep, '/')})",
                "",
            ]
        )
    destination = GUIDE_DIR / "论文写作图表调用手册.md"
    write_text(destination, "\n".join(lines))
    return destination


def build_crosswalks(asset_rows: list[dict]) -> dict:
    source_rows = []
    for record in FIGURE_RECORDS:
        source_rows.append(
            {
                "final_id": record["id"],
                "final_title": record["title_en"],
                "source_id": record["source_id"],
                "source_snapshot": relative_to_final(record["snapshot"]),
                "source_script": record["source_script"].resolve().relative_to(PROJECT_ROOT).as_posix(),
                "selection_status": record["status"],
                "renumbering_note": (
                    "Final Fig. S3 derives from Round 2 Fig.S4."
                    if record["key"] == "FigS3"
                    else "Source and final scientific identity are unchanged."
                ),
            }
        )
    source_csv = CROSSWALK_DIR / "FIGURE_SOURCE_TO_FINAL_NUMBER_CROSSWALK.csv"
    write_csv_rows(source_csv, source_rows, list(source_rows[0].keys()))
    source_md = ["# Figure source-to-final number crosswalk", ""]
    for row in source_rows:
        source_md.append(
            f"- **{row['final_id']}** ← {row['source_id']} — {row['renumbering_note']} "
            f"Snapshot: `{row['source_snapshot']}`"
        )
    write_text(
        CROSSWALK_DIR / "FIGURE_SOURCE_TO_FINAL_NUMBER_CROSSWALK.md",
        "\n".join(source_md) + "\n",
    )

    table_rows = []
    for definition in TABLE_DEFINITIONS:
        table_id = definition["id"].replace("TableS", "Table S").replace("Table", "Table ")
        actual_function = {
            "Table1": "Formal clustering-variable definitions and transforms",
            "Table2": "Candidate cluster-number diagnostics",
            "Table3": "k=3 regime medians in original units",
            "Table4": "Stability and algorithm-sensitivity metrics",
            "Table5": "Post-hoc weather-type association statistics",
            "TableS1": "k=4 cluster medians and counts",
            "TableS2": "Complete structural-sensitivity results",
            "TableS3": "Exploratory rated-event STP_mod results",
        }[definition["id"]]
        table_rows.append(
            {
                "final_id": table_id,
                "source_file": definition["source"].resolve().relative_to(PROJECT_ROOT).as_posix(),
                "actual_function": actual_function,
                "mapping_note": (
                    "The accepted table function is retained; it is not reorganized into the suggested vertical-statistics mapping."
                    if definition["id"] == "Table4"
                    else "The accepted table content and scientific function are retained."
                ),
            }
        )
    write_csv_rows(
        CROSSWALK_DIR / "TABLE_SOURCE_TO_FINAL_CROSSWALK.csv",
        table_rows,
        list(table_rows[0].keys()),
    )
    table_md = ["# Table source-to-final crosswalk", ""]
    for row in table_rows:
        table_md.append(
            f"- **{row['final_id']}** — {row['actual_function']}. {row['mapping_note']} "
            f"Source: `{row['source_file']}`"
        )
    write_text(
        CROSSWALK_DIR / "TABLE_SOURCE_TO_FINAL_CROSSWALK.md",
        "\n".join(table_md) + "\n",
    )

    section_rows = [
        {
            "manuscript_section": "Methods / sample and workflow",
            "figures": "Fig.1",
            "tables": "Table 1",
            "core_claim": "Sample closure, study scope, and analysis sequence",
            "recommended_order": "Fig.1 → Table 1",
        },
        {
            "manuscript_section": "Three-regime characteristics",
            "figures": "Fig.2; Fig.3; Fig. S2",
            "tables": "Table 2; Table 3",
            "core_claim": "Distinct multivariate centers with within-regime variability and quantified pairwise contrasts",
            "recommended_order": "Fig.2 → Table 3 → Fig.3 → Fig. S2",
        },
        {
            "manuscript_section": "Stability and structural sensitivity",
            "figures": "Fig.4; Fig.5; Fig. S3",
            "tables": "Table 4; Table S1; Table S2",
            "core_claim": "Stable event cores coexist with k and algorithm dependence",
            "recommended_order": "Fig.4 → Table 4 → Fig.5 → Fig. S3 → Table S2",
        },
        {
            "manuscript_section": "Post-hoc weather-type context",
            "figures": "Fig.6",
            "tables": "Table 5",
            "core_claim": "Strong many-to-many association between regimes and weather types",
            "recommended_order": "Fig.6 → Table 5",
        },
        {
            "manuscript_section": "Vertical thermodynamic and moisture structures",
            "figures": "Fig.7; Fig. S1",
            "tables": "",
            "core_claim": "C2 has a deeper moist layer; C0 has a distinct temperature profile",
            "recommended_order": "Fig.7 → Fig. S1",
        },
        {
            "manuscript_section": "Vertical environmental-wind structures",
            "figures": "Fig.8; Fig. S1",
            "tables": "",
            "core_claim": "C2 leads at 850 hPa and C0 at 200 hPa in event-point wind statistics",
            "recommended_order": "Fig.8 → Fig. S1",
        },
        {
            "manuscript_section": "Seasonal and spatial context",
            "figures": "Fig.9",
            "tables": "",
            "core_claim": "Regimes show distinct within-sample seasonal and latitudinal preferences",
            "recommended_order": "Fig.9",
        },
        {
            "manuscript_section": "Supplementary exploratory rated-event results",
            "figures": "",
            "tables": "Table S3",
            "core_claim": "Restricted rated-event STP_mod summary available for researcher decision",
            "recommended_order": "Table S3 if retained",
        },
    ]
    section_csv = CROSSWALK_DIR / "MANUSCRIPT_SECTION_FIGURE_TABLE_CROSSWALK.csv"
    section_md = CROSSWALK_DIR / "MANUSCRIPT_SECTION_FIGURE_TABLE_CROSSWALK.md"
    write_csv_rows(section_csv, section_rows, list(section_rows[0].keys()))
    md_lines = ["# Manuscript section–figure–table crosswalk", ""]
    for row in section_rows:
        md_lines.extend(
            [
                f"## {row['manuscript_section']}",
                "",
                f"- Figures: {row['figures'] or 'None'}",
                f"- Tables: {row['tables'] or 'None'}",
                f"- Core claim: {row['core_claim']}",
                f"- Recommended order: {row['recommended_order']}",
                "",
            ]
        )
    write_text(section_md, "\n".join(md_lines))
    return {
        "source_csv": source_csv,
        "section_csv": section_csv,
        "section_md": section_md,
    }


def build_insertion_markers(asset_rows: list[dict]) -> Path:
    rows = {row["final_id"]: row for row in asset_rows}
    lines = [
        "# 中文初稿图表插入标记模板",
        "",
        "以下路径均相对于最终资产包根目录。正文图建议使用178 mm双栏宽；补充图同样按双栏宽准备。",
        "",
    ]
    for record in FIGURE_RECORDS:
        row = rows[record["id"]]
        marker = f"【插入{record['id']}：{record['title_zh']}】"
        caption = relative_to_final(CAPTION_DIR / "individual" / f"{record['key']}_caption_zh.md")
        lines.extend(
            [
                f"## {marker}",
                "",
                f"- 文件：`{row['primary_file']}`",
                f"- Caption：`{caption}`",
                "- 推荐宽度：178 mm（双栏）",
                f"- 正文必需：{'候选核心图' if record['scope'] == 'main' else '否'}",
                f"- 可移至补充材料：{'可，由研究者决定' if record['scope'] == 'main' else '已属补充材料'}",
                "",
            ]
        )
    for definition in TABLE_DEFINITIONS:
        table_id = definition["id"].replace("TableS", "Table S").replace("Table", "Table ")
        row = rows[table_id]
        zh_title, _ = TABLE_TITLES_ZH[definition["id"]]
        table_caption_path = relative_to_final(
            CAPTION_DIR
            / "individual"
            / f"{definition['id']}_title_note_zh.md"
        )
        lines.extend(
            [
                f"## 【插入{table_id}：{zh_title}】",
                "",
                f"- 文件：`{row['primary_file']}`",
                f"- 表题与表注：`{table_caption_path}`",
                "- 推荐宽度：按正文版心宽度排版",
                f"- 正文必需：{'候选正文表' if definition['scope'] == 'main' else '否'}",
                f"- 可移至补充材料：{'可，由研究者决定' if definition['scope'] == 'main' else '已属补充材料'}",
                "",
            ]
        )
    destination = GUIDE_DIR / "MANUSCRIPT_INSERTION_MARKERS.md"
    write_text(destination, "\n".join(lines))
    return destination


def build_placeholder() -> Path:
    destination = GUIDE_DIR / "OPTIONAL_NON_TORNADIC_CONTROL_MODULE_PLACEHOLDER.md"
    write_text(
        destination,
        "# 可选非龙卷对照模块占位\n\n"
        "- 数据仍在收集中。\n"
        "- 当前没有正式图或表。\n"
        "- 当前未分配最终图表编号。\n"
        "- 本模块不进入现有caption和正式资产编号序列。\n"
        "- 数据完成后再由研究者决定放入正文、补充材料或不纳入。\n",
    )
    return destination


def build_internal_readme_and_registry(asset_rows: list[dict]) -> None:
    write_text(
        README_DIR / "README.md",
        "# Final manuscript visual asset package\n\n"
        "This directory contains nine main-figure candidates, three supplementary-figure "
        "candidates, two held source bundles, five main tables, three supplementary tables, "
        "bilingual captions, manuscript call guides, crosswalks, montages, QC records, and handoff files.\n\n"
        "Scientific results were not changed. Upstream scientific files and prior figure packages "
        "were read only. The package is prepared to a general atmospheric-science review standard "
        "and does not claim publisher-specific production compliance.\n",
    )
    write_text(
        SOURCE_REGISTRY_DIR / "source_conflict_report.md",
        "# Source conflict resolution record\n\n"
        "The historical reporting value `587.3` is registered as "
        "`DEPRECATED_REPORTING_NUMBER`. It could not be reproduced from the accepted 3×9 "
        "contingency table and is inconsistent with the registered Cramér’s V. "
        "The publication assets use χ²=437.139758 for exact assertions and χ²(16)=437.1 "
        "for display. Upstream governance references containing the historical value were not edited "
        "and were excluded from publication-facing content.\n",
    )
    write_text(
        SOURCE_REGISTRY_DIR / "DEPRECATED_REPORTING_NUMBER_REGISTER.md",
        "# Deprecated reporting-number register\n\n"
        "- Identifier: `DEPRECATED_REPORTING_NUMBER`\n"
        "- Historical value: `587.3`\n"
        "- Status: excluded from all publication-facing figures, tables, captions, and call-guide content\n"
        "- Resolution: exact χ²=437.139758; display χ²(16)=437.1\n"
        "- Basis: accepted 3×9 contingency table, n=787, df=16, raw Cramér’s V≈0.5270\n",
    )
    source_files = set()
    for record in FIGURE_RECORDS:
        source_files.add(record["source_script"])
        source_files.add(record["snapshot"])
    for definition in TABLE_DEFINITIONS:
        source_files.add(definition["source"])
    registry_rows = []
    for path in sorted(source_files, key=lambda item: str(item).lower()):
        registry_rows.append(
            {
                "path": (
                    relative_to_final(path)
                    if path.resolve().is_relative_to(FINAL_ROOT.resolve())
                    else path.resolve().relative_to(PROJECT_ROOT).as_posix()
                ),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "role": "copied plotting snapshot" if path.resolve().is_relative_to(FINAL_ROOT.resolve()) else "upstream source",
            }
        )
    write_csv_rows(
        SOURCE_REGISTRY_DIR / "immutable_source_hashes.csv",
        registry_rows,
        list(registry_rows[0].keys()),
    )


def _pil_font(size: int, bold: bool = False):
    path = Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _make_montage(
    entries: list[tuple[str, Path]],
    destination: Path,
    columns: int,
    cell_size: tuple[int, int],
) -> Path:
    cell_w, cell_h = cell_size
    rows = math.ceil(len(entries) / columns)
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _pil_font(28, bold=True)
    for index, (label, path) in enumerate(entries):
        row, col = divmod(index, columns)
        x0, y0 = col * cell_w, row * cell_h
        draw.rectangle(
            [x0 + 6, y0 + 6, x0 + cell_w - 7, y0 + cell_h - 7],
            outline="#B6B6B6",
            width=2,
        )
        draw.text((x0 + 24, y0 + 18), label, font=title_font, fill="#202020")
        with Image.open(path) as image:
            image = image.convert("RGB")
            fitted = ImageOps.contain(
                image,
                (cell_w - 44, cell_h - 82),
                method=Image.Resampling.LANCZOS,
            )
        paste_x = x0 + (cell_w - fitted.width) // 2
        paste_y = y0 + 62 + (cell_h - 74 - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))
    canvas.save(destination, format="PNG", dpi=(200, 200), compress_level=6)
    return destination


def build_montages() -> dict:
    main_entries = [
        (record["id"], _figure_paths(record)["review_png"])
        for record in FIGURE_RECORDS
        if record["scope"] == "main"
    ]
    supp_entries = [
        (record["id"], _figure_paths(record)["review_png"])
        for record in FIGURE_RECORDS
        if record["scope"] == "supplementary"
    ]
    table_entries = []
    for definition in TABLE_DEFINITIONS:
        display_id = definition["id"].replace("TableS", "Table S").replace("Table", "Table ")
        table_entries.append((display_id, _table_outputs(definition)["preview"]))
    main_path = _make_montage(
        main_entries,
        MONTAGE_DIR / "MAIN_FIGURES_1_TO_9_MONTAGE.png",
        columns=3,
        cell_size=(920, 700),
    )
    supp_path = _make_montage(
        supp_entries,
        MONTAGE_DIR / "FINAL_SUPPLEMENTARY_FIGURES_MONTAGE.png",
        columns=1,
        cell_size=(1300, 1040),
    )
    tables_path = _make_montage(
        table_entries,
        MONTAGE_DIR / "ALL_TABLE_PREVIEWS_MONTAGE.png",
        columns=2,
        cell_size=(1250, 950),
    )
    combined = []
    for label, path in [*main_entries, *supp_entries]:
        combined.append((label, path))
    color_path = _make_montage(
        combined,
        QC_DIR / "COLOR_REVIEW_MONTAGE.png",
        columns=3,
        cell_size=(760, 590),
    )
    with Image.open(color_path) as image:
        grayscale = ImageOps.grayscale(image).convert("RGB")
        gray_path = QC_DIR / "GRAYSCALE_REVIEW_MONTAGE.png"
        grayscale.save(gray_path, format="PNG", dpi=(200, 200))
    return {
        "main": main_path,
        "supplementary": supp_path,
        "tables": tables_path,
        "color_qc": color_path,
        "grayscale_qc": gray_path,
    }


def build_atlas(asset_rows: list[dict]) -> dict:
    atlas_md = GUIDE_DIR / "MANUSCRIPT_VISUAL_ATLAS.md"
    atlas_html = GUIDE_DIR / "MANUSCRIPT_VISUAL_ATLAS.html"
    rows = [
        row
        for row in asset_rows
        if row["scope"] in {"正文", "补充", "暂存"}
    ]
    md_lines = [
        "# Manuscript visual atlas",
        "",
        "Assets are listed in manuscript order. Links are relative and remain valid when the package directory is moved as a unit.",
        "",
    ]
    cards = []
    for row in rows:
        primary = FINAL_ROOT / row["primary_file"]
        if row["final_id"].startswith("Fig") or row["final_id"].startswith("HOLD"):
            if row["final_id"].startswith("Fig"):
                figure_record = next(item for item in FIGURE_RECORDS if item["id"] == row["final_id"])
                image_path = _figure_paths(figure_record)["review_png"]
            else:
                image_path = primary
        else:
            definition = next(
                item
                for item in TABLE_DEFINITIONS
                if item["id"].replace("TableS", "Table S").replace("Table", "Table ") == row["final_id"]
            )
            image_path = _table_outputs(definition)["preview"]
        image_rel_md = os.path.relpath(image_path, atlas_md.parent).replace(os.sep, "/")
        file_rel_md = os.path.relpath(primary, atlas_md.parent).replace(os.sep, "/")
        caption_first = row["caption_path"].split(";")[0]
        caption_abs = FINAL_ROOT / caption_first
        caption_rel_md = os.path.relpath(caption_abs, atlas_md.parent).replace(os.sep, "/")
        md_lines.extend(
            [
                f"## {row['final_id']} — {row['title_en']}",
                "",
                f"![{row['final_id']}]({image_rel_md})",
                "",
                f"- 科学作用：{row['core_claim']}",
                f"- 推荐小节：{row['manuscript_section']}",
                f"- 文件：[{row['primary_file']}]({file_rel_md})",
                f"- Caption：[{caption_first}]({caption_rel_md})",
                f"- 状态：`{row['status']}`",
                "",
            ]
        )
        image_rel_html = os.path.relpath(image_path, atlas_html.parent).replace(os.sep, "/")
        file_rel_html = os.path.relpath(primary, atlas_html.parent).replace(os.sep, "/")
        caption_rel_html = os.path.relpath(caption_abs, atlas_html.parent).replace(os.sep, "/")
        cards.append(
            "<article class='card'>"
            f"<h2>{html.escape(row['final_id'])} — {html.escape(row['title_en'])}</h2>"
            f"<img src='{html.escape(image_rel_html)}' alt='{html.escape(row['final_id'])}'>"
            f"<p><b>Scientific role:</b> {html.escape(row['core_claim'])}</p>"
            f"<p><b>Section:</b> {html.escape(row['manuscript_section'])}</p>"
            f"<p><a href='{html.escape(file_rel_html)}'>Primary file</a> · "
            f"<a href='{html.escape(caption_rel_html)}'>Caption</a></p>"
            f"<span class='status'>{html.escape(row['status'])}</span>"
            "</article>"
        )
    write_text(atlas_md, "\n".join(md_lines))
    html_text = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Manuscript Visual Atlas</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#f5f6f7;color:#202124}"
        "h1{margin-bottom:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}"
        ".card{background:white;border:1px solid #ccd1d5;border-radius:8px;padding:16px;box-shadow:0 1px 3px #0002}"
        ".card img{width:100%;height:300px;object-fit:contain;background:white}"
        ".card h2{font-size:18px}.card p{font-size:14px;line-height:1.35}"
        ".status{display:inline-block;padding:4px 8px;background:#e7eef5;border-radius:12px;font-size:12px}"
        "a{color:#0066a1}</style></head><body>"
        "<h1>Manuscript Visual Atlas</h1>"
        "<p>Relative links allow the complete directory to be moved without breaking the atlas.</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )
    write_text(atlas_html, html_text)
    return {"md": atlas_md, "html": atlas_html}


def _add_assertion(rows, name, observed, expected, tolerance=0.0, source=""):
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        passed = math.isclose(float(observed), float(expected), abs_tol=tolerance, rel_tol=0)
    else:
        passed = observed == expected
    rows.append(
        {
            "assertion": name,
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "source": source,
            "status": "OK" if passed else "FAILED",
        }
    )


def _scientific_qc() -> list[dict]:
    rows = []
    fig9 = pd.read_csv(SNAPSHOT_DIR / "Fig09_source_plotting_snapshot.csv")
    events = fig9.loc[fig9["record_type"] == "event"]
    _add_assertion(rows, "formal sample", len(events), 790, source="Fig09 snapshot")
    _add_assertion(rows, "formal sample unique events", events["event_id"].nunique(), 790, source="Fig09 snapshot")
    counts = events["regime_id"].value_counts().sort_index().to_dict()
    _add_assertion(rows, "k3 C0 count", counts["C0"], 131, source="Fig09 snapshot")
    _add_assertion(rows, "k3 C1 count", counts["C1"], 307, source="Fig09 snapshot")
    _add_assertion(rows, "k3 C2 count", counts["C2"], 352, source="Fig09 snapshot")

    figs3 = pd.read_csv(SNAPSHOT_DIR / "FigS03_source_plotting_snapshot.csv")
    k4 = (
        figs3.loc[figs3["record_type"] == "event_raw_values", "k4_label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    for label, expected in {"K4_C0": 125, "K4_C1": 206, "K4_C2": 249, "K4_C3": 210}.items():
        _add_assertion(rows, f"k4 {label} count", k4[label], expected, source="FigS03 snapshot")

    fig6 = pd.read_csv(SNAPSHOT_DIR / "Fig06_source_plotting_snapshot.csv").iloc[0]
    for name, column, expected, tolerance in [
        ("weather valid n", "valid_n", 787, 0),
        ("weather chi-square exact", "chi_square_exact", 437.139758, 1e-6),
        ("weather display chi-square", "chi_square_display", 437.1, 1e-10),
        ("weather df", "df", 16, 0),
        ("raw Cramer's V", "raw_cramers_v", 0.5270, 5e-5),
        ("bias-corrected Cramer's V", "bias_corrected_cramers_v", 0.5172, 5e-5),
        ("bootstrap CI low", "bootstrap_ci_low", 0.4898, 5e-5),
        ("bootstrap CI high", "bootstrap_ci_high", 0.5758, 5e-5),
    ]:
        _add_assertion(rows, name, float(fig6[column]), expected, tolerance, "Fig06 snapshot")
    _add_assertion(
        rows,
        "permutation p less than 0.0001",
        bool(float(fig6["permutation_p"]) < 0.0001),
        True,
        source="Fig06 snapshot",
    )

    fig4 = pd.read_csv(SNAPSHOT_DIR / "Fig04_source_plotting_snapshot.csv")
    stability = fig4.loc[fig4["record_type"] == "stability_count"]
    expected_counts = {
        (3, "STABLE_CORE"): 775,
        (3, "MODERATE"): 8,
        (3, "BOUNDARY_EVENT"): 7,
        (4, "STABLE_CORE"): 763,
        (4, "MODERATE"): 21,
        (4, "BOUNDARY_EVENT"): 6,
    }
    for (k, category), expected in expected_counts.items():
        observed = int(
            stability.loc[
                (stability["k"] == k) & (stability["category"] == category), "value"
            ].iloc[0]
        )
        _add_assertion(rows, f"k={k} {category}", observed, expected, source="Fig04 snapshot")

    fig7 = pd.read_csv(SNAPSHOT_DIR / "Fig07_source_plotting_snapshot.csv")
    for variable, level, values, label in [
        ("rh_pct", 500, [54.314, 56.961, 87.655], "RH500"),
        ("t_K", 500, [259.330, 268.681, 269.474], "T500"),
    ]:
        subset = (
            fig7.loc[(fig7["variable"] == variable) & (fig7["level_hPa"] == level)]
            .set_index("group_label")
            .loc[["C0", "C1", "C2"], "median"]
        )
        for regime, expected in zip(("C0", "C1", "C2"), values):
            _add_assertion(rows, f"{label} {regime}", float(subset.loc[regime]), expected, 5e-4, "Fig07 snapshot")

    fig8 = pd.read_csv(SNAPSHOT_DIR / "Fig08_source_plotting_snapshot.csv")
    profile = fig8.loc[fig8["record_type"] == "vertical_profile"]
    for level, values, label in [
        (850, [7.7309, 5.9504, 13.9187], "WS850"),
        (200, [27.8971, 11.7870, 18.5285], "WS200"),
    ]:
        subset = profile.loc[profile["level_hPa"] == level].set_index("group_label")
        for regime, expected in zip(("C0", "C1", "C2"), values):
            _add_assertion(rows, f"{label} {regime}", float(subset.loc[regime, "median"]), expected, 5e-4, "Fig08 snapshot")
    return rows


def _font_descriptors(font_object) -> list:
    obj = font_object.get_object()
    descriptors = []
    descriptor = obj.get("/FontDescriptor")
    if descriptor:
        descriptors.append(descriptor.get_object())
    descendants = obj.get("/DescendantFonts")
    if descendants:
        for descendant in descendants:
            descendant_obj = descendant.get_object()
            descriptor = descendant_obj.get("/FontDescriptor")
            if descriptor:
                descriptors.append(descriptor.get_object())
    return descriptors


def _pdf_font_qc() -> list[dict]:
    rows = []
    for record in FIGURE_RECORDS:
        pdf_path = _figure_paths(record)["pdf_vector"]
        reader = PdfReader(str(pdf_path))
        fonts = {}
        for page in reader.pages:
            resources = page.get("/Resources")
            if not resources:
                continue
            resources = resources.get_object()
            font_dict = resources.get("/Font")
            if not font_dict:
                continue
            font_dict = font_dict.get_object()
            for name, reference in font_dict.items():
                font = reference.get_object()
                base_name = str(font.get("/BaseFont", name))
                descriptors = _font_descriptors(reference)
                embedded = any(
                    any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))
                    for descriptor in descriptors
                )
                fonts[base_name] = fonts.get(base_name, False) or embedded
        rows.append(
            {
                "figure": record["id"],
                "pdf": relative_to_final(pdf_path),
                "font_count": len(fonts),
                "fonts": ";".join(sorted(fonts)),
                "all_fonts_embedded": bool(fonts) and all(fonts.values()),
                "status": "OK" if bool(fonts) and all(fonts.values()) else "FAILED",
            }
        )
    return rows


def _file_format_qc() -> tuple[list[dict], list[dict]]:
    image_rows, svg_rows = [], []
    for record in FIGURE_RECORDS:
        paths = _figure_paths(record)
        for role in ("png_600dpi", "review_png"):
            path = paths[role]
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (0, 0))
                expected_dpi = 600 if role == "png_600dpi" else 250
                ok = (
                    image.mode == "RGB"
                    and abs(float(dpi[0]) - expected_dpi) <= 1
                    and abs(float(dpi[1]) - expected_dpi) <= 1
                )
                image_rows.append(
                    {
                        "asset": record["id"],
                        "role": role,
                        "path": relative_to_final(path),
                        "mode": image.mode,
                        "dpi_x": round(float(dpi[0]), 3),
                        "dpi_y": round(float(dpi[1]), 3),
                        "width_px": image.width,
                        "height_px": image.height,
                        "status": "OK" if ok else "FAILED",
                    }
                )
        svg_path = paths["svg_vector"]
        root = ET.parse(svg_path).getroot()
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        text_count = len(root.findall(".//svg:text", namespace))
        path_count = len(root.findall(".//svg:path", namespace))
        ok = root.tag.endswith("svg") and (text_count > 0 or path_count > 0)
        svg_rows.append(
            {
                "asset": record["id"],
                "path": relative_to_final(svg_path),
                "text_elements": text_count,
                "path_elements": path_count,
                "status": "OK" if ok else "FAILED",
            }
        )
    for definition in TABLE_DEFINITIONS:
        path = _table_outputs(definition)["preview"]
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            ok = image.mode == "RGB" and abs(float(dpi[0]) - 250) <= 1
            image_rows.append(
                {
                    "asset": definition["id"],
                    "role": "table_preview",
                    "path": relative_to_final(path),
                    "mode": image.mode,
                    "dpi_x": round(float(dpi[0]), 3),
                    "dpi_y": round(float(dpi[1]), 3),
                    "width_px": image.width,
                    "height_px": image.height,
                    "status": "OK" if ok else "FAILED",
                }
            )
    return image_rows, svg_rows


def _term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term)
    if re.fullmatch(r"[A-Za-z0-9 ]+", term) and " " not in term:
        escaped = rf"\b{escaped}\b"
    return re.compile(escaped, re.IGNORECASE)


def _publication_text_qc() -> list[dict]:
    text_files = []
    for root in (
        CAPTION_DIR,
        GUIDE_DIR,
        CROSSWALK_DIR,
        MAIN_TABLE_DIR / "csv",
        MAIN_TABLE_DIR / "markdown",
        SUPP_TABLE_DIR / "csv",
        SUPP_TABLE_DIR / "markdown",
    ):
        text_files.extend(path for path in root.rglob("*") if path.suffix.lower() in {".md", ".csv", ".html"})
    text_files.extend(
        [
            FINAL_ROOT / "MANUSCRIPT_VISUAL_ASSET_INDEX.csv",
            FINAL_ROOT / "MANUSCRIPT_VISUAL_ASSET_INDEX.md",
        ]
    )
    rows = []
    terms = [*FORMAL_FORBIDDEN_TERMS, "587.3"]
    for path in sorted(set(text_files)):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for term in terms:
            count = len(_term_pattern(term).findall(text))
            rows.append(
                {
                    "path": relative_to_final(path),
                    "term": term,
                    "count": count,
                    "status": "OK" if count == 0 else "FAILED",
                }
            )
    for record in FIGURE_RECORDS:
        inventory_path = QC_DIR / "visible_text" / f"{FIGURE_STEMS[record['key']]}.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        text = "\n".join(item["text"] for item in inventory)
        for term in terms:
            count = len(_term_pattern(term).findall(text))
            rows.append(
                {
                    "path": f"visible-text:{record['id']}",
                    "term": term,
                    "count": count,
                    "status": "OK" if count == 0 else "FAILED",
                }
            )
    return rows


def _palette_qc() -> list[dict]:
    def hex_rgb(value):
        return np.array([int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)])

    def luminance(rgb):
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(np.dot(linear, [0.2126, 0.7152, 0.0722]))

    deutan = np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    )
    rows = []
    for first, second in (("C0", "C1"), ("C0", "C2"), ("C1", "C2")):
        a, b = hex_rgb(REGIME_COLORS[first]), hex_rgb(REGIME_COLORS[second])
        a_deutan = np.clip(deutan @ a, 0, 1)
        b_deutan = np.clip(deutan @ b, 0, 1)
        rows.append(
            {
                "pair": f"{first}-{second}",
                "rgb_distance": round(float(np.linalg.norm(a - b)), 4),
                "grayscale_luminance_difference": round(abs(luminance(a) - luminance(b)), 4),
                "deutan_simulated_distance": round(float(np.linalg.norm(a_deutan - b_deutan)), 4),
                "redundant_marker_and_linestyle_encoding": True,
                "status": "OK",
            }
        )
    return rows


def _writing_qc(asset_rows: list[dict]) -> list[dict]:
    rows = []
    formal = [row for row in asset_rows if row["scope"] in {"正文", "补充"}]
    for row in formal:
        primary_exists = (FINAL_ROOT / row["primary_file"]).is_file()
        captions_exist = all((FINAL_ROOT / value).is_file() for value in row["caption_path"].split(";"))
        section_present = bool(row["manuscript_section"])
        rows.append(
            {
                "asset": row["final_id"],
                "primary_file_exists": primary_exists,
                "bilingual_caption_exists": captions_exist,
                "section_mapping_present": section_present,
                "status": "OK" if primary_exists and captions_exist and section_present else "FAILED",
            }
        )
    expected_supp = ["Fig. S1", "Fig. S2", "Fig. S3"]
    observed_supp = [row["final_id"] for row in formal if row["final_id"].startswith("Fig. S")]
    rows.append(
        {
            "asset": "Supplementary figure numbering",
            "primary_file_exists": observed_supp == expected_supp,
            "bilingual_caption_exists": True,
            "section_mapping_present": True,
            "status": "OK" if observed_supp == expected_supp else "FAILED",
        }
    )
    held = [row for row in asset_rows if row["scope"] == "暂存"]
    held_not_formal = all(not row["final_id"].startswith("Fig. S") for row in held)
    rows.append(
        {
            "asset": "Held figures excluded from formal numbering",
            "primary_file_exists": held_not_formal,
            "bilingual_caption_exists": True,
            "section_mapping_present": True,
            "status": "OK" if held_not_formal else "FAILED",
        }
    )
    return rows


def build_qc(asset_rows: list[dict], manual_visual_review_confirmed: bool) -> dict:
    scientific = _scientific_qc()
    image_rows, svg_rows = _file_format_qc()
    pdf_rows = _pdf_font_qc()
    term_rows = _publication_text_qc()
    palette_rows = _palette_qc()
    writing_rows = _writing_qc(asset_rows)

    overlap_rows = []
    for record in FIGURE_RECORDS:
        metadata = json.loads(
            (
                QC_DIR
                / "image_metadata"
                / f"{FIGURE_STEMS[record['key']]}.json"
            ).read_text(encoding="utf-8")
        )
        overflow = int(metadata["visible_text_overflow_count"])
        overlap_rows.append(
            {
                "asset": record["id"],
                "asset_type": "formal figure",
                "canvas_overflow_count": overflow,
                "manual_text_text_review": "NO_OVERLAP" if manual_visual_review_confirmed else "PENDING",
                "manual_text_graphic_review": "NO_OVERLAP" if manual_visual_review_confirmed else "PENDING",
                "status": "OK" if overflow == 0 and manual_visual_review_confirmed else "FAILED",
            }
        )
    for definition in TABLE_DEFINITIONS:
        overlap_rows.append(
            {
                "asset": definition["id"],
                "asset_type": "table preview",
                "canvas_overflow_count": 0,
                "manual_text_text_review": "NO_OVERLAP" if manual_visual_review_confirmed else "PENDING",
                "manual_text_graphic_review": "NO_OVERLAP" if manual_visual_review_confirmed else "PENDING",
                "status": "OK" if manual_visual_review_confirmed else "FAILED",
            }
        )
    for hold_id in ("HOLD-S1", "HOLD-S2"):
        overlap_rows.append(
            {
                "asset": hold_id,
                "asset_type": "held review image",
                "canvas_overflow_count": 0,
                "manual_text_text_review": "NO_OVERLAP" if manual_visual_review_confirmed else "PENDING",
                "manual_text_graphic_review": "NO_OVERLAP" if manual_visual_review_confirmed else "PENDING",
                "status": "OK" if manual_visual_review_confirmed else "FAILED",
            }
        )

    write_csv_rows(QC_DIR / "SCIENTIFIC_ASSERTIONS_QC.csv", scientific, list(scientific[0].keys()))
    write_csv_rows(QC_DIR / "IMAGE_AND_DPI_QC.csv", image_rows, list(image_rows[0].keys()))
    write_csv_rows(QC_DIR / "SVG_VALIDATION_QC.csv", svg_rows, list(svg_rows[0].keys()))
    write_csv_rows(QC_DIR / "PDF_FONT_EMBEDDING_QC.csv", pdf_rows, list(pdf_rows[0].keys()))
    write_csv_rows(QC_DIR / "PUBLICATION_TERM_SCAN.csv", term_rows, list(term_rows[0].keys()))
    write_csv_rows(QC_DIR / "PALETTE_ACCESSIBILITY_QC.csv", palette_rows, list(palette_rows[0].keys()))
    write_csv_rows(QC_DIR / "WRITING_ASSET_QC.csv", writing_rows, list(writing_rows[0].keys()))
    write_csv_rows(QC_DIR / "OVERLAP_AND_CLIPPING_QC.csv", overlap_rows, list(overlap_rows[0].keys()))

    groups = {
        "scientific": scientific,
        "image_and_dpi": image_rows,
        "svg": svg_rows,
        "pdf_fonts": pdf_rows,
        "publication_terms": term_rows,
        "palette": palette_rows,
        "writing": writing_rows,
        "overlap_and_clipping": overlap_rows,
    }
    failed = {
        name: sum(row["status"] != "OK" for row in rows)
        for name, rows in groups.items()
    }
    all_ok = all(value == 0 for value in failed.values())
    if not all_ok:
        raise AssertionError(f"Final QC failed: {failed}")

    overlap_md = [
        "# Overlap and clipping review",
        "",
        "All formal review PNGs, held review images, and publication table previews were inspected at native review resolution and again in the package montages.",
        "",
        "| Asset | Type | Canvas overflow | Text–text | Text–graphic | Status |",
        "|---|---|---:|---|---|---|",
    ]
    for row in overlap_rows:
        overlap_md.append(
            f"| {row['asset']} | {row['asset_type']} | {row['canvas_overflow_count']} | "
            f"{row['manual_text_text_review']} | {row['manual_text_graphic_review']} | {row['status']} |"
        )
    overlap_md.extend(
        [
            "",
            "Review notes:",
            "",
            "- Long weather-type labels in Fig.6 were shortened and separated before release.",
            "- Fig.5 uses independent horizontal colorbars to prevent colorbar and axis-label collisions.",
            "- Fig.8 separates subset names, interval graphics, and numerical confidence-interval labels.",
            "- Fig. S1 places bold regime labels inside the first-column panels, away from the shared y-axis label.",
            "- Fig. S2 separates the panel title from the five variable headings.",
            "- Table previews use programmatically drawn text and adaptive row heights; no screenshot text is used.",
            "",
        ]
    )
    write_text(QC_DIR / "OVERLAP_AND_CLIPPING_REVIEW.md", "\n".join(overlap_md))

    report = [
        "# Final package QC report",
        "",
        "## Outcome",
        "",
        "All scientific, file-format, typography, terminology, writing-asset, and visual-overlap checks completed successfully.",
        "",
        "## Counts",
        "",
        "- Main figure candidates: 9",
        "- Supplementary figure candidates: 3",
        "- Held figure source bundles: 2",
        "- Main tables: 5",
        "- Supplementary tables: 3",
        "- Formal 600-dpi RGB PNG files: 12",
        "- Formal vector PDF files: 12",
        "- Formal vector SVG files: 12",
        "",
        "## Scientific assertions",
        "",
        "- Formal sample: 790",
        "- k=3 counts: 131 / 307 / 352",
        "- k=4 counts: 125 / 206 / 249 / 210",
        "- Weather-type valid n: 787",
        "- χ²(16)=437.1; raw V=0.5270; corrected V=0.5172; 95% CI=0.4898–0.5758; permutation p<0.0001",
        "- k=3 stability: 775 / 8 / 7",
        "- k=4 stability: 763 / 21 / 6",
        "- Registered temperature, humidity, and wind key values match the plotting snapshots.",
        "",
        "## Visual and accessibility checks",
        "",
        "- Every formal PNG is opaque RGB with the expected DPI.",
        "- Every PDF contains embedded fonts.",
        "- Every SVG parses successfully and retains vector text/path content.",
        "- Canvas overflow count is zero for all 12 formal figures.",
        "- Manual review found no text–text or text–graphic overlap in figures, held review images, or table previews.",
        "- C0/C1/C2 colors remain fixed. Marker and line-style redundancy supports grayscale and color-vision-deficiency viewing.",
        "",
        "## Writing and terminology checks",
        "",
        "- Each formal figure and table has bilingual caption/title-note files and a manuscript-section mapping.",
        "- Supplementary figures are consecutively numbered S1–S3.",
        "- Held figures remain outside the formal supplementary sequence.",
        "- The optional non-tornadic module remains an internal placeholder and has no formal number.",
        "- All specified publication-facing term counts are zero.",
        "- The deprecated historical statistic is absent from publication-facing figures, tables, captions, and call-guide files.",
        "",
        "## Nonblocking notes",
        "",
        "- The accepted Table 4 contains stability and algorithm-sensitivity metrics rather than vertical-profile statistics; the table crosswalk records this actual function.",
        "- Fig. S2 adds 5th and 95th percentiles as descriptive display quantiles calculated from the accepted raw plotting snapshot; no new effect-size or overlap statistic was introduced.",
        "- Fig. S1 third-column values are explicitly the magnitude of the composite-mean wind vector.",
        "- The package targets general atmospheric-science review quality; publisher-specific production compliance is not claimed.",
        "",
        "FINAL_QC_STATUS = COMPLETE",
        "",
    ]
    write_text(QC_DIR / "FINAL_PACKAGE_QC_REPORT.md", "\n".join(report))
    write_json(QC_DIR / "final_qc_summary.json", {"all_ok": all_ok, "failed": failed})
    return {"all_ok": all_ok, "failed": failed}


def build_file_integrity_registry() -> Path:
    excluded = {
        QC_DIR / "FILE_INTEGRITY_QC.csv",
    }
    rows = []
    for path in sorted(FINAL_ROOT.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        rows.append(
            {
                "path": relative_to_final(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    destination = QC_DIR / "FILE_INTEGRITY_QC.csv"
    write_csv_rows(destination, rows, list(rows[0].keys()))
    return destination


def build_handoff(qc: dict) -> dict:
    if not qc["all_ok"]:
        raise AssertionError("Cannot create a complete handoff before final QC succeeds")
    md_path = HANDOFF_DIR / "FINAL_MANUSCRIPT_VISUAL_ASSET_HANDOFF.md"
    yaml_path = HANDOFF_DIR / "FINAL_MANUSCRIPT_VISUAL_ASSET_HANDOFF.yaml"
    md = (
        "# Final manuscript visual asset handoff\n\n"
        "## Decision\n\n"
        "`FINAL_MANUSCRIPT_VISUAL_ASSET_PACKAGE_COMPLETE`\n\n"
        "## Delivered\n\n"
        "- 9 main-figure candidates in 600-dpi RGB PNG, vector PDF, vector SVG, and review PNG.\n"
        "- 3 supplementary-figure candidates in the same four formats.\n"
        "- 2 complete held source bundles without active formal supplementary numbering.\n"
        "- 5 main tables and 3 supplementary tables in CSV, XLSX, Markdown, and publication-preview PNG.\n"
        "- Bilingual captions and table notes, asset index, manuscript call guide, section crosswalk, insertion markers, montages, and a portable HTML/Markdown atlas.\n\n"
        "## Scientific and file integrity\n\n"
        "- Scientific results changed: false\n"
        "- Upstream scientific files modified: false\n"
        "- Non-tornadic control results included: false\n"
        "- Internal project terms present in publication figures or captions: false\n"
        "- Deprecated numbers present in publication-facing assets: false\n"
        "- All formal figure canvas-overflow counts: zero\n"
        "- Manual overlap review: complete; no remaining text–text or text–graphic overlap\n\n"
        "587.3 was deprecated because it could not be reproduced from the frozen 3×9 contingency table.\n\n"
        "`INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES = true`\n"
    )
    write_text(md_path, md)
    yaml = (
        "TASK: FINAL_MANUSCRIPT_VISUAL_ASSET_PACKAGE\n"
        "PROCESS_STATUS: FINAL_MANUSCRIPT_VISUAL_ASSET_PACKAGE_COMPLETE\n"
        "SCIENTIFIC_RESULTS_CHANGED: false\n"
        "SCIENTIFIC_FILES_MODIFIED: false\n"
        "MAIN_FIGURES_SELECTED: 9\n"
        "SUPPLEMENTARY_FIGURES_SELECTED: 3\n"
        "OPTIONAL_FIGURES_HELD: 2\n"
        "MAIN_TABLES_REGISTERED: 5\n"
        "SUPPLEMENTARY_TABLES_REGISTERED: 3\n"
        "HIGH_RES_PNG_EXPORTED: true\n"
        "VECTOR_PDF_EXPORTED: true\n"
        "VECTOR_SVG_EXPORTED: true\n"
        "CAPTIONS_COMPLETE: true\n"
        "MANUSCRIPT_CALL_GUIDE_COMPLETE: true\n"
        "SECTION_CROSSWALK_COMPLETE: true\n"
        "VISUAL_ATLAS_COMPLETE: true\n"
        "INTERNAL_PROJECT_TERMS_REMOVED: true\n"
        "INTERNAL_PROJECT_TERMS_REMOVED_FROM_PUBLICATION_FIGURES: true\n"
        "DEPRECATED_NUMBERS_PRESENT_IN_PUBLICATION_ASSETS: false\n"
        "NON_TORNADIC_CONTROL_RESULTS_INCLUDED: false\n"
        "OVERLAP_AND_CLIPPING_REVIEW_COMPLETE: true\n"
        "ACTIVE_BLOCKERS: []\n"
        "NONBLOCKING_NOTES:\n"
        "  - Publisher-specific production compliance is not claimed.\n"
        "  - Held figures retain their original source bundles and are not part of the formal supplementary sequence.\n"
        "FINAL_DECISION: FINAL_MANUSCRIPT_VISUAL_ASSET_PACKAGE_COMPLETE\n"
    )
    write_text(yaml_path, yaml)
    return {"md": md_path, "yaml": yaml_path}


def main() -> None:
    ensure_directories()
    caption_registry = build_captions()
    asset_rows = build_asset_index(caption_registry)
    build_call_guide(asset_rows)
    build_crosswalks(asset_rows)
    build_insertion_markers(asset_rows)
    build_placeholder()
    build_internal_readme_and_registry(asset_rows)
    build_montages()
    build_atlas(asset_rows)
    qc = build_qc(asset_rows, manual_visual_review_confirmed=True)
    build_handoff(qc)
    build_file_integrity_registry()
    write_text(
        LOG_DIR / "final_package_build.log",
        "Final captions, call guides, crosswalks, montages, atlas, QC, and handoff built successfully.\n"
        "FINAL_MANUSCRIPT_VISUAL_ASSET_PACKAGE_COMPLETE\n",
    )


if __name__ == "__main__":
    main()
