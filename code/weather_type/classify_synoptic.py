"""
3.2 天气学分型脚本
基于环境参数 + 地理位置 + TC 关联标志，对每个龙卷事件标注天气系统类型

分类体系（按大纲要求）:
  - TC 龙卷 (TC)               : 热带气旋外雨带龙卷
  - 冷涡龙卷 (CV)              : 东北冷涡背景下龙卷
  - 气旋/冷锋龙卷 (CF)         : 江淮气旋或冷锋系统龙卷
  - 飑线/QLCS 龙卷 (QLCS)      : 飑线或准线性对流系统龙卷
  - 暖区龙卷 (WS)              : 暖区/干线龙卷（无明确强迫）
  - 其他 (OTHER)               : 无法归入以上类型的龙卷

输出: tornado_synoptic_type.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ── 路径配置 ──────────────────────────────────────────
ENV_PARAMS_PATH = r'ERA5_ROOT\tornado_env_params_v2.csv'
OUTPUT = r'ERA5_ROOT\tornado_synoptic_type.csv'

# ── 地理区域定义 ──────────────────────────────────────
# 东北地区（冷涡影响区）
NE_PROVINCES = {'辽宁', '吉林', '黑龙江', '内蒙古'}
# 江淮/黄淮地区（气旋-冷锋影响区）
JH_PROVINCES = {'江苏', '安徽', '河南', '山东', '湖北'}
# 华南沿海（TC 影响区，用于非 TC 标签的暖区识别）
SC_PROVINCES = {'广东', '广西', '海南', '福建', '江西', '湖南', '浙江'}
# 华北地区
NC_PROVINCES = {'河北', '山西', '陕西', '北京', '天津'}
# 西南地区
SW_PROVINCES = {'四川', '重庆', '贵州', '云南'}

# ── 读取环境参数 ──────────────────────────────────────
print("Loading environmental parameters...")
df = pd.read_csv(ENV_PARAMS_PATH)
n_total = len(df)
print(f"Total events: {n_total}")

# 解析月份
df['month'] = pd.to_datetime(df['date_utc'], format='%Y/%m/%d').dt.month

# ── 分类函数 ──────────────────────────────────────────

def classify_synoptic(row):
    """
    基于规则的分层分类决策树。
    优先级: TC 标志 > 地理+环境综合判断 > 默认其他
    """
    tc = row['tc_related']
    prov = row['province']
    month = row['month']
    lat = row['latitude']
    lon = row['longitude']

    # 环境参数（处理缺失值）
    cape = row['MLCAPE_Jkg'] if not pd.isna(row['MLCAPE_Jkg']) else row['ERA5_cape_Jkg']
    shr6 = row['SHR6_ms'] if not pd.isna(row['SHR6_ms']) else np.nan
    srh1 = row['SRH1_m2s2'] if not pd.isna(row['SRH1_m2s2']) else np.nan
    llcl = row['MLLCL_m'] if not pd.isna(row['MLLCL_m']) else np.nan
    ttd850 = row['T_Td_850_K'] if not pd.isna(row['T_Td_850_K']) else np.nan
    thse500 = row['ThetaE_500_K'] if not pd.isna(row['ThetaE_500_K']) else np.nan
    shr1 = row['SHR1_ms'] if not pd.isna(row['SHR1_ms']) else np.nan

    # ═══════════════════════════════════════════════════════
    # 第一层: TC 龙卷识别 (依据数据工程师的 tc_related 标志)
    # ═══════════════════════════════════════════════════════
    if tc == True:
        # TC 龙卷亚型区分
        if not pd.isna(srh1):
            if srh1 >= 150:
                subtype = 'TC-高螺旋度型'
            elif srh1 >= 80:
                subtype = 'TC-中等螺旋度型'
            else:
                subtype = 'TC-低螺旋度型'
        else:
            subtype = 'TC'

        # 进一步区分偏南/偏北急流型（基于纬度）
        if lat <= 25:
            subtype += '/偏南急流型'
        else:
            subtype += '/偏北急流型'

        return subtype

    # ═══════════════════════════════════════════════════════
    # 第二层: 非 TC 龙卷分类
    # ═══════════════════════════════════════════════════════

    # --- 2a. 东北冷涡型 ---
    # 特征: 东北地区, 5-9月, 冷核 (低 ThetaE_500), 较强深层切变
    if prov in NE_PROVINCES and month in [5, 6, 7, 8, 9]:
        if not pd.isna(shr6) and shr6 >= 10:
            if not pd.isna(thse500) and thse500 < 335:
                return '冷涡-冷核型'
            else:
                return '冷涡-普通型'
        # 东北非冷涡季节 → 继续往下判断

    # --- 2b. 江淮气旋 / 冷锋型 ---
    # 特征: 江淮/黄淮, 春夏季 (3-8月), 中等 CAPE + 较强切变
    if prov in JH_PROVINCES and month in [3, 4, 5, 6, 7, 8]:
        if not pd.isna(shr6) and shr6 >= 12:
            if not pd.isna(cape) and cape >= 800:
                return '气旋/冷锋-高能型'
            else:
                return '气旋/冷锋-低能型'
        # 江淮但条件不满足 → 继续判断

    # --- 2c. 飑线 / QLCS 型 ---
    # 特征: 强深层切变 + 低 CAPE + 低 SRH1 (线性强迫为主)
    if not pd.isna(shr6) and shr6 >= 18:
        if not pd.isna(cape) and cape < 1000:
            if not pd.isna(srh1) and srh1 < 100:
                return 'QLCS/飑线型'
        # 强切变但 CAPE 不低 → 可能是超级单体环境, 标记为 QLCS 可能

    # --- 2d. 暖区龙卷 ---
    # 特征: 华南/华东, 高 CAPE, 弱-中等切变, 无 TC 关联
    if prov in SC_PROVINCES.union(JH_PROVINCES):
        if not pd.isna(cape) and cape >= 1500:
            if not pd.isna(shr6) and shr6 < 15:
                return '暖区-高能低切型'
            elif not pd.isna(shr6) and shr6 >= 15:
                return '暖区-高能高切型'
        # 中等 CAPE 暖区
        if not pd.isna(cape) and cape >= 800:
            if not pd.isna(shr6) and shr6 < 15:
                return '暖区-中能低切型'

    # --- 2e. 补充判断: 基于环境参数剖面 ---
    # 低 LCL + 高 SRH1 → 可能超单 (任何地区)
    if not pd.isna(llcl) and llcl < 800 and not pd.isna(srh1) and srh1 > 100:
        if prov in NE_PROVINCES:
            return '冷涡-高超单潜势型'
        else:
            return '超单型(未分类)'

    # 华北夏季型（河北/山西/北京/天津）
    if prov in NC_PROVINCES and month in [6, 7, 8]:
        if not pd.isna(cape) and cape >= 500:
            return '华北夏季对流型'
        else:
            return '华北弱对流型'

    # 西南地区
    if prov in SW_PROVINCES:
        if not pd.isna(cape) and cape >= 1000:
            return '西南高能型'
        else:
            return '西南普通型'

    # --- 默认: 其他 ---
    # 尝试用风切变和环境做最后分类
    if not pd.isna(shr6) and shr6 >= 20:
        return '其他-强切变型'
    elif not pd.isna(cape) and cape >= 2000:
        return '其他-高能型'
    else:
        return '其他'


# ── 执行分类 ──────────────────────────────────────────
print("Running synoptic classification...")
df['synoptic_type'] = df.apply(classify_synoptic, axis=1)

# ── 提取大类（用于后续聚类分析） ──────────────────────
def simplify_type(st):
    if st.startswith('TC'):
        return 'TC'
    elif st.startswith('冷涡'):
        return '冷涡'
    elif st.startswith('气旋'):
        return '气旋/冷锋'
    elif st.startswith('QLCS'):
        return 'QLCS/飑线'
    elif st.startswith('暖区'):
        return '暖区'
    elif st.startswith('超单'):
        return '超单(未分类)'
    elif st.startswith('华北'):
        return '华北对流'
    elif st.startswith('西南'):
        return '西南对流'
    else:
        return '其他'

df['synoptic_class'] = df['synoptic_type'].apply(simplify_type)

# ── 导出 CSV ──────────────────────────────────────────
out_cols = [
    'event_id', 'date_utc', 'time_utc', 'longitude', 'latitude',
    'province', 'f_scale', 'tc_related', 'distance_to_tc_km',
    'synoptic_type', 'synoptic_class',
]
out_df = df[out_cols].copy()

out_df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f"\nOutput written to: {OUTPUT}")
print(f"Shape: {out_df.shape}")

# ── 统计摘要 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("天气学分型统计")
print("=" * 60)
print(f"\n大类分布 (synoptic_class):")
class_counts = df['synoptic_class'].value_counts()
for cls, cnt in class_counts.items():
    pct = cnt / n_total * 100
    print(f"  {cls:20s}: {cnt:4d} ({pct:5.1f}%)")

print(f"\n细分型分布 (synoptic_type, 前20):")
type_counts = df['synoptic_type'].value_counts()
for tp, cnt in type_counts.items():
    pct = cnt / n_total * 100
    print(f"  {tp:30s}: {cnt:4d} ({pct:5.1f}%)")

# ── 分型环境参数对比 ──────────────────────────────────
print(f"\n各类型环境参数中位数对比:")
env_cols = ['MLCAPE_Jkg', 'SHR6_ms', 'SHR1_ms', 'SRH1_m2s2', 'SRH3_m2s2',
            'MLLCL_m', 'T_Td_850_K', 'SCP', 'STP']
for cls in class_counts.index[:6]:
    sub = df[df['synoptic_class'] == cls]
    if len(sub) < 3:
        continue
    print(f"\n  [{cls}] (N={len(sub)})")
    for col in env_cols:
        if col in sub.columns:
            vals = sub[col].dropna()
            if len(vals) > 0:
                print(f"    {col:18s}: median={vals.median():.2f}, mean={vals.mean():.2f}")

print("\nDone.")
