# -*- coding: utf-8 -*-
"""
==============================================================================
模块名称：data_mock.py
所属系统：《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》
功能描述：模拟数据生成模块
          —— 基于长三角典型湖库的水文节律与生态学规律，生成符合自然常理
             的模拟监测数据，涵盖平水期、藻类生长期、爆发期三个水文时段，
             覆盖千岛湖、太湖、长荡湖、巢湖、淀山湖五大湖泊。
==============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================================
# 全局常量定义
# ============================================================================

# 五大湖泊名称列表
LAKE_NAMES: list = [
    "千岛湖",
    "太湖",
    "长荡湖",
    "巢湖",
    "淀山湖"
]

# 各湖泊采样点位配置
# 每个湖泊设3个典型点位：入湖口、湖心、出湖口
SAMPLING_POINTS: dict = {
    "千岛湖": ["千岛湖-入湖口", "千岛湖-湖心", "千岛湖-出湖口"],
    "太湖":   ["太湖-入湖口",   "太湖-湖心",   "太湖-出湖口"],
    "长荡湖": ["长荡湖-入湖口", "长荡湖-湖心", "长荡湖-出湖口"],
    "巢湖":   ["巢湖-入湖口",   "巢湖-湖心",   "巢湖-出湖口"],
    "淀山湖": ["淀山湖-入湖口", "淀山湖-湖心", "淀山湖-出湖口"],
}

# 三个水文期定义
HYDROLOGICAL_PERIODS: list = [
    "平水期（3-5月）",
    "藻类生长期（6-8月）",
    "爆发期（9-11月）"
]

# 各水文期的月份映射（用于生成符合季节特征的数据）
PERIOD_MONTHS: dict = {
    "平水期（3-5月）":    [3, 4, 5],
    "藻类生长期（6-8月）": [6, 7, 8],
    "爆发期（9-11月）":   [9, 10, 11],
}

# 各湖泊的营养状态分级（用于设定基础参数范围）
# 千岛湖 → 贫-中营养，太湖 → 富营养，长荡湖 → 中-富营养，巢湖 → 富营养，淀山湖 → 中营养
LAKE_TROPHIC_LEVEL: dict = {
    "千岛湖": "贫-中营养",
    "太湖":   "富营养",
    "长荡湖": "中-富营养",
    "巢湖":   "富营养",
    "淀山湖": "中营养",
}


def validate_lake_name(lake_name: str) -> bool:
    """
    校验湖泊名称是否在预定义列表中。

    参数
    ----
    lake_name : str
        待校验的湖泊名称字符串。

    返回
    ----
    bool
        若湖泊名称有效则返回 True，否则返回 False。

    示例
    ----
    >>> validate_lake_name("千岛湖")
    True
    >>> validate_lake_name("西湖")
    False
    """
    # 检查输入是否为有效字符串
    if not isinstance(lake_name, str):
        raise TypeError(f"湖泊名称必须为字符串类型，当前传入类型为：{type(lake_name)}")
    # 去除首尾空白后进行比较
    cleaned_name = lake_name.strip()
    if cleaned_name == "":
        raise ValueError("湖泊名称不能为空字符串。")
    return cleaned_name in LAKE_NAMES


def validate_hydrological_period(period: str) -> bool:
    """
    校验水文期名称是否在预定义列表中。

    参数
    ----
    period : str
        待校验的水文期名称字符串。

    返回
    ----
    bool
        若水文期名称有效则返回 True，否则返回 False。
    """
    if not isinstance(period, str):
        raise TypeError(f"水文期名称必须为字符串类型，当前传入类型为：{type(period)}")
    cleaned_period = period.strip()
    if cleaned_period == "":
        raise ValueError("水文期名称不能为空字符串。")
    return cleaned_period in HYDROLOGICAL_PERIODS


def validate_sample_count(sample_count: int) -> bool:
    """
    校验采样数量是否在合理范围内。

    参数
    ----
    sample_count : int
        每个点位每个水文期的采样数量。

    返回
    ----
    bool
        若采样数量在 [1, 1000] 范围内则返回 True，否则抛出异常。
    """
    if not isinstance(sample_count, int):
        raise TypeError(f"采样数量必须为整数类型，当前传入类型为：{type(sample_count)}")
    if sample_count < 1:
        raise ValueError(f"采样数量必须大于等于1，当前传入值为：{sample_count}")
    if sample_count > 1000:
        raise ValueError(f"采样数量不能超过1000，当前传入值为：{sample_count}")
    return True


# ============================================================================
# 核心数据生成函数
# ============================================================================

def generate_water_temperature(period: str, lake_name: str, size: int = 10) -> np.ndarray:
    """
    根据水文期和湖泊名称生成符合季节性规律的水温模拟数据（单位：℃）。

    水温的季节性规律设定：
    - 平水期（3-5月）：气温回升，水温在 12~22℃ 之间波动，均值为 17℃。
    - 藻类生长期（6-8月）：夏季高温，水温在 22~32℃ 之间波动，均值为 28℃。
    - 爆发期（9-11月）：秋季降温，水温在 10~22℃ 之间波动，均值为 16℃。

    湖泊差异设定：
    - 千岛湖（深水湖）：水温整体偏低约 1~2℃。
    - 太湖/巢湖（浅水湖）：水温整体偏高约 1~2℃。

    参数
    ----
    period : str
        水文期名称。
    lake_name : str
        湖泊名称。
    size : int
        生成的数据样本数量，默认值为 10。

    返回
    ----
    np.ndarray
        一维浮点数数组，包含生成的水温数据。
    """
    # --- 参数校验 ---
    validate_hydrological_period(period)
    validate_lake_name(lake_name)
    validate_sample_count(size)

    # --- 根据水文期设定基础均值和标准差 ---
    if period == "平水期（3-5月）":
        base_mean: float = 17.0
        base_std: float = 3.0
    elif period == "藻类生长期（6-8月）":
        base_mean = 28.0
        base_std = 3.5
    else:  # 爆发期（9-11月）
        base_mean = 16.0
        base_std = 3.5

    # --- 根据湖泊营养状态微调水温 ---
    trophic = LAKE_TROPHIC_LEVEL.get(lake_name, "中营养")
    if "贫" in trophic:
        # 贫营养湖泊（千岛湖）：水温偏低 1.5℃
        lake_offset: float = -1.5
    elif "富" in trophic:
        # 富营养湖泊（太湖、巢湖）：水温偏高 1.5℃（浅水湖升温快）
        lake_offset = 1.5
    else:
        # 中营养湖泊：不偏移
        lake_offset = 0.0

    adjusted_mean: float = base_mean + lake_offset

    # --- 生成正态分布随机数据 ---
    np.random.seed(abs(hash(f"{period}_{lake_name}")) % (2**31))
    temperature_data: np.ndarray = np.random.normal(
        loc=adjusted_mean,
        scale=base_std,
        size=size
    )

    # --- 边界裁剪：水温不应低于 0℃ 或高于 40℃ ---
    temperature_data = np.clip(temperature_data, 0.0, 40.0)

    # --- 四舍五入保留一位小数 ---
    temperature_data = np.round(temperature_data, 1)

    return temperature_data


def generate_dissolved_oxygen(period: str, lake_name: str, size: int = 10) -> np.ndarray:
    """
    根据水文期和湖泊名称生成溶解氧（DO）模拟数据（单位：mg/L）。

    溶解氧的季节性规律设定：
    - 平水期（3-5月）：春季浮游植物开始生长，DO 较高，均值约 9.0 mg/L。
    - 藻类生长期（6-8月）：水温升高，氧气溶解度下降，但藻类光合产氧增加，
      整体 DO 波动较大，均值约 6.5 mg/L。
    - 爆发期（9-11月）：藻类大量分解耗氧，DO 均值降低至 5.5 mg/L。

    湖泊差异设定：
    - 千岛湖（贫营养）：水质好，DO 整体偏高 1.5~2.0 mg/L。
    - 太湖/巢湖（富营养）：有机质多、耗氧大，DO 偏低 1.0~1.5 mg/L。

    参数
    ----
    period : str
        水文期名称。
    lake_name : str
        湖泊名称。
    size : int
        生成的数据样本数量，默认值为 10。

    返回
    ----
    np.ndarray
        一维浮点数数组，包含生成的 DO 浓度数据。
    """
    # --- 参数校验 ---
    validate_hydrological_period(period)
    validate_lake_name(lake_name)
    validate_sample_count(size)

    # --- 根据水文期设定基础均值和标准差 ---
    if period == "平水期（3-5月）":
        base_mean: float = 9.0
        base_std: float = 1.5
    elif period == "藻类生长期（6-8月）":
        base_mean = 6.5
        base_std = 2.0
    else:  # 爆发期
        base_mean = 5.5
        base_std = 2.0

    # --- 根据湖泊营养状态微调 ---
    trophic = LAKE_TROPHIC_LEVEL.get(lake_name, "中营养")
    if "贫" in trophic:
        lake_offset: float = 2.0
    elif "富" in trophic:
        lake_offset = -1.5
    else:
        lake_offset = 0.0

    adjusted_mean: float = base_mean + lake_offset

    # --- 生成正态分布随机数据 ---
    np.random.seed(abs(hash(f"DO_{period}_{lake_name}")) % (2**31))
    do_data: np.ndarray = np.random.normal(
        loc=adjusted_mean,
        scale=base_std,
        size=size
    )

    # --- 边界裁剪：DO 在天然水体中通常介于 0~18 mg/L ---
    do_data = np.clip(do_data, 0.1, 18.0)

    # --- 四舍五入保留一位小数 ---
    do_data = np.round(do_data, 1)

    return do_data


def generate_ph(period: str, lake_name: str, size: int = 10) -> np.ndarray:
    """
    根据水文期和湖泊名称生成 pH 值模拟数据。

    pH 值的季节性规律设定：
    - 平水期（3-5月）：pH 较为中性，均值约 7.2。
    - 藻类生长期（6-8月）：藻类光合作用消耗 CO₂，pH 升高，均值约 8.0。
    - 爆发期（9-11月）：藻类衰亡分解产生 CO₂，pH 回落，均值约 7.3。

    湖泊差异设定：
    - 富营养湖泊（太湖、巢湖）：藻类多，夏季 pH 可飙升至 8.5~9.0。
    - 贫营养湖泊（千岛湖）：pH 波动小，全年偏中性。

    参数
    ----
    period : str
        水文期名称。
    lake_name : str
        湖泊名称。
    size : int
        生成的数据样本数量。

    返回
    ----
    np.ndarray
        一维浮点数数组，包含生成的 pH 值数据。
    """
    # --- 参数校验 ---
    validate_hydrological_period(period)
    validate_lake_name(lake_name)
    validate_sample_count(size)

    # --- 根据水文期设定基础均值和标准差 ---
    if period == "平水期（3-5月）":
        base_mean: float = 7.2
        base_std: float = 0.4
    elif period == "藻类生长期（6-8月）":
        base_mean = 8.0
        base_std = 0.6
    else:  # 爆发期
        base_mean = 7.3
        base_std = 0.5

    # --- 根据湖泊营养状态微调 ---
    trophic = LAKE_TROPHIC_LEVEL.get(lake_name, "中营养")
    if "贫" in trophic:
        lake_offset: float = -0.3
    elif "富" in trophic:
        lake_offset = 0.5
    else:
        lake_offset = 0.0

    adjusted_mean: float = base_mean + lake_offset

    # --- 生成正态分布随机数据 ---
    np.random.seed(abs(hash(f"pH_{period}_{lake_name}")) % (2**31))
    ph_data: np.ndarray = np.random.normal(
        loc=adjusted_mean,
        scale=base_std,
        size=size
    )

    # --- 边界裁剪：天然水体 pH 通常介于 4~10 ---
    ph_data = np.clip(ph_data, 4.0, 10.0)
    ph_data = np.round(ph_data, 2)

    return ph_data


def generate_turbidity(period: str, lake_name: str, size: int = 10) -> np.ndarray:
    """
    根据水文期和湖泊名称生成浊度模拟数据（单位：NTU）。

    浊度的季节性规律设定：
    - 平水期（3-5月）：春季降雨适中，浊度中等，均值约 10 NTU。
    - 藻类生长期（6-8月）：夏季暴雨和藻类增殖，浊度升高，均值约 20 NTU。
    - 爆发期（9-11月）：藻类大量聚集，浊度可达最高，均值约 25 NTU。

    湖泊差异设定：
    - 千岛湖（深水贫营养）：浊度低，均值在 3~8 NTU。
    - 太湖（浅水富营养，风浪搅动）：浊度最高，均值可达 30 NTU。

    参数
    ----
    period : str
        水文期名称。
    lake_name : str
        湖泊名称。
    size : int
        生成的数据样本数量。

    返回
    ----
    np.ndarray
        一维浮点数数组，包含生成的浊度数据。
    """
    # --- 参数校验 ---
    validate_hydrological_period(period)
    validate_lake_name(lake_name)
    validate_sample_count(size)

    # --- 根据水文期设定基础均值和标准差 ---
    if period == "平水期（3-5月）":
        base_mean: float = 10.0
        base_std: float = 5.0
    elif period == "藻类生长期（6-8月）":
        base_mean = 20.0
        base_std = 8.0
    else:  # 爆发期
        base_mean = 25.0
        base_std = 10.0

    # --- 根据湖泊特征微调 ---
    trophic = LAKE_TROPHIC_LEVEL.get(lake_name, "中营养")
    if "贫" in trophic:
        # 千岛湖浊度极低
        lake_offset: float = -15.0
    elif "富" in trophic and lake_name == "太湖":
        # 太湖浊度偏高
        lake_offset = 8.0
    elif "富" in trophic:
        lake_offset = 3.0
    else:
        lake_offset = 0.0

    adjusted_mean: float = base_mean + lake_offset

    # --- 生成对数正态分布（浊度通常呈右偏分布） ---
    np.random.seed(abs(hash(f"Turb_{period}_{lake_name}")) % (2**31))
    turbidity_data: np.ndarray = np.random.lognormal(
        mean=np.log(max(adjusted_mean, 0.5)),
        sigma=0.5,
        size=size
    )

    # --- 边界裁剪 ---
    turbidity_data = np.clip(turbidity_data, 0.1, 200.0)
    turbidity_data = np.round(turbidity_data, 1)

    return turbidity_data


def generate_nutrients(
    period: str, lake_name: str, nutrient_type: str, size: int = 10
) -> np.ndarray:
    """
    生成营养盐（TN、TP、NH3-N、CODMn）的模拟浓度数据。

    营养盐的季节性规律设定：
    - TN（总氮）：
      * 平水期均值约 1.0 mg/L
      * 藻类生长期均值约 1.5 mg/L（农业面源输入增加）
      * 爆发期均值约 2.0 mg/L（藻类分解释放氮）
    - TP（总磷）：
      * 平水期均值约 0.05 mg/L
      * 藻类生长期均值约 0.10 mg/L
      * 爆发期均值约 0.15 mg/L
    - NH3-N（氨氮）：
      * 平水期均值约 0.15 mg/L
      * 藻类生长期均值约 0.25 mg/L
      * 爆发期均值约 0.35 mg/L
    - CODMn（高锰酸盐指数）：
      * 平水期均值约 3.0 mg/L
      * 藻类生长期均值约 4.5 mg/L
      * 爆发期均值约 6.0 mg/L

    湖泊差异设定：
    - 千岛湖（贫营养）：所有营养盐浓度偏低约 40%~60%。
    - 太湖（富营养）：所有营养盐浓度偏高约 50%~100%。

    参数
    ----
    period : str
        水文期名称。
    lake_name : str
        湖泊名称。
    nutrient_type : str
        营养盐类型，可选值：'TN', 'TP', 'NH3-N', 'CODMn'。
    size : int
        生成的数据样本数量。

    返回
    ----
    np.ndarray
        一维浮点数数组，包含生成的营养盐浓度数据。
    """
    # --- 参数校验 ---
    validate_hydrological_period(period)
    validate_lake_name(lake_name)
    validate_sample_count(size)

    valid_nutrients: list = ["TN", "TP", "NH3-N", "CODMn"]
    if nutrient_type not in valid_nutrients:
        raise ValueError(
            f"营养盐类型 '{nutrient_type}' 无效。"
            f"可选值：{valid_nutrients}"
        )

    # --- 各营养盐在各水文期的基础参数设定 ---
    nutrient_params: dict = {
        "TN": {
            "平水期（3-5月）":    (1.0, 0.4),
            "藻类生长期（6-8月）": (1.5, 0.6),
            "爆发期（9-11月）":   (2.0, 0.8),
        },
        "TP": {
            "平水期（3-5月）":    (0.05, 0.02),
            "藻类生长期（6-8月）": (0.10, 0.04),
            "爆发期（9-11月）":   (0.15, 0.06),
        },
        "NH3-N": {
            "平水期（3-5月）":    (0.15, 0.06),
            "藻类生长期（6-8月）": (0.25, 0.10),
            "爆发期（9-11月）":   (0.35, 0.15),
        },
        "CODMn": {
            "平水期（3-5月）":    (3.0, 1.0),
            "藻类生长期（6-8月）": (4.5, 1.5),
            "爆发期（9-11月）":   (6.0, 2.0),
        },
    }

    base_mean, base_std = nutrient_params[nutrient_type][period]

    # --- 根据湖泊营养状态进行缩放 ---
    trophic = LAKE_TROPHIC_LEVEL.get(lake_name, "中营养")
    if "贫" in trophic:
        # 千岛湖：营养盐浓度约为基线的 40%~50%
        scale_factor: float = 0.45
    elif "富" in trophic:
        if lake_name == "太湖":
            # 太湖营养盐浓度极高
            scale_factor = 2.0
        else:
            scale_factor = 1.5
    else:
        scale_factor = 1.0

    adjusted_mean: float = base_mean * scale_factor
    adjusted_std: float = base_std * scale_factor

    # --- 生成对数正态分布随机数据（浓度数据通常右偏） ---
    np.random.seed(abs(hash(f"{nutrient_type}_{period}_{lake_name}")) % (2**31))
    nutrient_data: np.ndarray = np.random.lognormal(
        mean=np.log(max(adjusted_mean, 0.001)),
        sigma=max(adjusted_std / max(adjusted_mean, 0.001), 0.1),
        size=size
    )

    # --- 边界裁剪 ---
    lower_bounds: dict = {"TN": 0.01, "TP": 0.001, "NH3-N": 0.005, "CODMn": 0.5}
    upper_bounds: dict = {"TN": 10.0, "TP": 2.0, "NH3-N": 5.0, "CODMn": 30.0}
    nutrient_data = np.clip(
        nutrient_data,
        lower_bounds.get(nutrient_type, 0.001),
        upper_bounds.get(nutrient_type, 50.0)
    )

    # --- 四舍五入 ---
    decimal_places: dict = {"TN": 2, "TP": 3, "NH3-N": 3, "CODMn": 2}
    nutrient_data = np.round(nutrient_data, decimal_places.get(nutrient_type, 2))

    return nutrient_data


def generate_chlorophyll_a(period: str, lake_name: str, size: int = 10) -> np.ndarray:
    """
    根据水文期和湖泊名称生成叶绿素a（Chl-a）模拟数据（单位：μg/L）。

    叶绿素a的季节性规律设定：
    - 平水期（3-5月）：藻类开始复苏，Chl-a 均值约 5 μg/L。
    - 藻类生长期（6-8月）：藻类大量增殖，Chl-a 均值约 25 μg/L。
    - 爆发期（9-11月）：水华高峰期，Chl-a 均值约 45 μg/L，部分点位可达 100+。

    湖泊差异设定：
    - 千岛湖（贫营养）：Chl-a 极低，通常在 1~5 μg/L。
    - 太湖（富营养）：Chl-a 极高，夏季可达 80~150 μg/L。

    参数
    ----
    period : str
        水文期名称。
    lake_name : str
        湖泊名称。
    size : int
        生成的数据样本数量。

    返回
    ----
    np.ndarray
        一维浮点数数组，包含生成的叶绿素a浓度数据。
    """
    # --- 参数校验 ---
    validate_hydrological_period(period)
    validate_lake_name(lake_name)
    validate_sample_count(size)

    # --- 根据水文期设定基础参数 ---
    if period == "平水期（3-5月）":
        base_mean: float = 5.0
        base_std: float = 3.0
    elif period == "藻类生长期（6-8月）":
        base_mean = 25.0
        base_std = 15.0
    else:  # 爆发期
        base_mean = 45.0
        base_std = 25.0

    # --- 根据湖泊营养状态缩放 ---
    trophic = LAKE_TROPHIC_LEVEL.get(lake_name, "中营养")
    if "贫" in trophic:
        scale_factor: float = 0.1
    elif "富" in trophic:
        if lake_name == "太湖":
            scale_factor = 2.5
        else:
            scale_factor = 1.8
    else:
        scale_factor = 0.5

    adjusted_mean: float = base_mean * scale_factor
    adjusted_std: float = base_std * scale_factor

    # --- 生成对数正态分布 ---
    np.random.seed(abs(hash(f"Chla_{period}_{lake_name}")) % (2**31))
    chla_data: np.ndarray = np.random.lognormal(
        mean=np.log(max(adjusted_mean, 0.1)),
        sigma=0.8,
        size=size
    )

    # --- 边界裁剪 ---
    chla_data = np.clip(chla_data, 0.1, 300.0)
    chla_data = np.round(chla_data, 1)

    return chla_data


def generate_odorants(period: str, lake_name: str, size: int = 10) -> pd.DataFrame:
    """
    生成嗅味物质（GSM 和 2-MIB）的模拟浓度数据（单位：ng/L）。

    嗅味物质的季节性规律设定：
    - GSM（土臭素）：
      * 平水期均值约 2 ng/L（低水平）
      * 藻类生长期均值约 8 ng/L
      * 爆发期均值约 18 ng/L（部分点位超标）
    - 2-MIB（2-甲基异莰醇）：
      * 平水期均值约 1.5 ng/L
      * 藻类生长期均值约 6 ng/L
      * 爆发期均值约 15 ng/L

    湖泊差异设定：
    - 千岛湖：嗅味物质浓度极低，GSM 通常 < 2 ng/L。
    - 太湖/巢湖：藻源嗅味严重，GSM 和 2-MIB 可超过 20 ng/L。

    与 Chl-a 的关联设定：
    - 嗅味物质浓度与叶绿素a呈正相关（产嗅藻类也是叶绿素的贡献者）。
    - 通过叠加相关系数模拟这种生态学关联。

    参数
    ----
    period : str
        水文期名称。
    lake_name : str
        湖泊名称。
    size : int
        生成的数据样本数量。

    返回
    ----
    pd.DataFrame
        包含 'GSM' 和 '2-MIB' 两列的 DataFrame。
    """
    # --- 参数校验 ---
    validate_hydrological_period(period)
    validate_lake_name(lake_name)
    validate_sample_count(size)

    # --- GSM 基础参数 ---
    gsm_params: dict = {
        "平水期（3-5月）":    (2.0, 1.0),
        "藻类生长期（6-8月）": (8.0, 4.0),
        "爆发期（9-11月）":   (18.0, 8.0),
    }

    # --- 2-MIB 基础参数 ---
    mib_params: dict = {
        "平水期（3-5月）":    (1.5, 0.8),
        "藻类生长期（6-8月）": (6.0, 3.0),
        "爆发期（9-11月）":   (15.0, 7.0),
    }

    gsm_mean, gsm_std = gsm_params[period]
    mib_mean, mib_std = mib_params[period]

    # --- 根据湖泊营养状态缩放 ---
    trophic = LAKE_TROPHIC_LEVEL.get(lake_name, "中营养")
    if "贫" in trophic:
        scale_factor: float = 0.2
    elif "富" in trophic:
        scale_factor = 2.0 if lake_name == "太湖" else 1.5
    else:
        scale_factor = 0.6

    gsm_mean *= scale_factor
    gsm_std *= scale_factor
    mib_mean *= scale_factor
    mib_std *= scale_factor

    # --- 生成 GSM 和 2-MIB 数据 ---
    np.random.seed(abs(hash(f"GSM_{period}_{lake_name}")) % (2**31))
    gsm_data: np.ndarray = np.random.lognormal(
        mean=np.log(max(gsm_mean, 0.01)),
        sigma=0.7,
        size=size
    )
    gsm_data = np.clip(gsm_data, 0.01, 100.0)
    gsm_data = np.round(gsm_data, 2)

    np.random.seed(abs(hash(f"MIB_{period}_{lake_name}")) % (2**31))
    mib_data: np.ndarray = np.random.lognormal(
        mean=np.log(max(mib_mean, 0.01)),
        sigma=0.7,
        size=size
    )
    mib_data = np.clip(mib_data, 0.01, 100.0)
    mib_data = np.round(mib_data, 2)

    # --- 组装为 DataFrame ---
    odorant_df: pd.DataFrame = pd.DataFrame({
        "GSM_ng_L": gsm_data,
        "2-MIB_ng_L": mib_data,
    })

    return odorant_df


# ============================================================================
# 主数据生成函数：组装完整模拟数据集
# ============================================================================

def generate_full_mock_dataset(
    samples_per_period: int = 20,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    生成完整的模拟监测数据集，覆盖五大湖泊、三个水文期、各三个点位。

    数据集包含以下字段：
    - 基本信息：湖泊名称、采样点位、水文期、采样日期
    - 常规理化指标：水温(℃)、pH、DO(mg/L)、浊度(NTU)
    - 营养盐指标：TN(mg/L)、TP(mg/L)、NH3-N(mg/L)、CODMn(mg/L)
    - 生物学指标：叶绿素a(μg/L)
    - 嗅味物质：GSM(ng/L)、2-MIB(ng/L)

    参数
    ----
    samples_per_period : int
        每个点位每个水文期的采样数量，默认值为 20。
        设 20 时，总数据量 = 5湖泊 × 3点位 × 3水文期 × 20 = 900 条记录。
    random_seed : int
        全局随机种子，用于保证结果可复现，默认值为 42。

    返回
    ----
    pd.DataFrame
        包含完整模拟监测数据的 DataFrame。

    示例
    ----
    >>> df = generate_full_mock_dataset(samples_per_period=5)
    >>> print(df.shape)
    (225, 14)
    >>> print(df.columns.tolist())
    ['湖泊名称', '采样点位', '水文期', '采样日期', '水温', 'pH', 'DO', '浊度',
     'TN', 'TP', 'NH3-N', 'CODMn', '叶绿素a', 'GSM', '2-MIB']
    """
    # --- 参数校验 ---
    if not isinstance(samples_per_period, int):
        raise TypeError(
            f"samples_per_period 必须为整数类型，"
            f"当前传入类型为：{type(samples_per_period)}"
        )
    if samples_per_period < 1:
        raise ValueError(
            f"samples_per_period 必须 >= 1，当前传入值为：{samples_per_period}"
        )
    if samples_per_period > 500:
        raise ValueError(
            f"samples_per_period 不得 > 500（避免生成过多数据），"
            f"当前传入值为：{samples_per_period}"
        )
    if not isinstance(random_seed, int):
        raise TypeError(
            f"random_seed 必须为整数类型，当前传入类型为：{type(random_seed)}"
        )

    # --- 设置全局随机种子 ---
    np.random.seed(random_seed)

    # --- 初始化结果列表 ---
    all_records: list = []

    # --- 三重循环：湖泊 × 点位 × 水文期 ---
    for lake_name in LAKE_NAMES:
        for point in SAMPLING_POINTS[lake_name]:
            for period in HYDROLOGICAL_PERIODS:

                # 获取该水文期对应的月份列表
                months = PERIOD_MONTHS[period]

                # 批量生成各指标数据
                water_temp: np.ndarray = generate_water_temperature(
                    period, lake_name, samples_per_period
                )
                ph_values: np.ndarray = generate_ph(
                    period, lake_name, samples_per_period
                )
                do_values: np.ndarray = generate_dissolved_oxygen(
                    period, lake_name, samples_per_period
                )
                turbidity: np.ndarray = generate_turbidity(
                    period, lake_name, samples_per_period
                )
                tn_values: np.ndarray = generate_nutrients(
                    period, lake_name, "TN", samples_per_period
                )
                tp_values: np.ndarray = generate_nutrients(
                    period, lake_name, "TP", samples_per_period
                )
                nh3n_values: np.ndarray = generate_nutrients(
                    period, lake_name, "NH3-N", samples_per_period
                )
                codmn_values: np.ndarray = generate_nutrients(
                    period, lake_name, "CODMn", samples_per_period
                )
                chla_values: np.ndarray = generate_chlorophyll_a(
                    period, lake_name, samples_per_period
                )
                odorant_df: pd.DataFrame = generate_odorants(
                    period, lake_name, samples_per_period
                )

                # 逐条组装记录
                for i in range(samples_per_period):
                    # 随机分配月份和日期
                    month: int = int(np.random.choice(months))
                    max_day: int = 28 if month == 2 else 30
                    day: int = int(np.random.randint(1, max_day + 1))

                    # 构造采样日期（统一使用2025年）
                    sample_date: str = f"2025-{month:02d}-{day:02d}"

                    record: dict = {
                        "湖泊名称": lake_name,
                        "采样点位": point,
                        "水文期":  period,
                        "采样日期": sample_date,
                        "水温":    water_temp[i],
                        "pH":      ph_values[i],
                        "DO":      do_values[i],
                        "浊度":    turbidity[i],
                        "TN":      tn_values[i],
                        "TP":      tp_values[i],
                        "NH3-N":   nh3n_values[i],
                        "CODMn":   codmn_values[i],
                        "叶绿素a": chla_values[i],
                        "GSM":     odorant_df.loc[i, "GSM_ng_L"],
                        "2-MIB":   odorant_df.loc[i, "2-MIB_ng_L"],
                    }
                    all_records.append(record)

    # --- 转换为 DataFrame ---
    full_dataset: pd.DataFrame = pd.DataFrame(all_records)

    # --- 打乱行顺序（模拟真实采样的无序性） ---
    full_dataset = full_dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return full_dataset


# ============================================================================
# 便捷导出函数
# ============================================================================

def export_dataset_to_csv(
    dataset: pd.DataFrame,
    file_path: str = "mock_water_quality_data.csv"
) -> str:
    """
    将模拟数据集导出为 CSV 文件，便于后续查看和分享。

    参数
    ----
    dataset : pd.DataFrame
        待导出的数据集。
    file_path : str
        目标 CSV 文件路径，默认为 'mock_water_quality_data.csv'。

    返回
    ----
    str
        成功导出的文件路径。

    示例
    ----
    >>> df = generate_full_mock_dataset(samples_per_period=5)
    >>> path = export_dataset_to_csv(df, "test_data.csv")
    >>> print(path)
    'test_data.csv'
    """
    # --- 参数校验 ---
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            f"dataset 必须为 pandas DataFrame，当前传入类型为：{type(dataset)}"
        )
    if dataset.empty:
        raise ValueError("dataset 不能为空 DataFrame。")
    if not isinstance(file_path, str):
        raise TypeError(
            f"file_path 必须为字符串类型，当前传入类型为：{type(file_path)}"
        )
    if file_path.strip() == "":
        raise ValueError("file_path 不能为空字符串。")

    # --- 确保文件扩展名为 .csv ---
    if not file_path.lower().endswith(".csv"):
        file_path += ".csv"

    # --- 写入 CSV ---
    dataset.to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path


# ============================================================================
# 数据集描述统计函数
# ============================================================================

def get_dataset_summary(dataset: pd.DataFrame) -> dict:
    """
    获取模拟数据集的描述性统计摘要。

    参数
    ----
    dataset : pd.DataFrame
        模拟监测数据集。

    返回
    ----
    dict
        包含各数值指标的均值、标准差、最小值、最大值等统计量的字典。
    """
    # --- 参数校验 ---
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            f"dataset 必须为 pandas DataFrame，当前传入类型为：{type(dataset)}"
        )
    if dataset.empty:
        raise ValueError("dataset 不能为空 DataFrame。")

    # --- 筛选数值列 ---
    numeric_cols: list = dataset.select_dtypes(include=[np.number]).columns.tolist()

    # --- 计算描述统计 ---
    summary: dict = {}
    for col in numeric_cols:
        col_data = dataset[col]
        summary[col] = {
            "样本量": int(col_data.count()),
            "均值":   round(float(col_data.mean()), 3),
            "标准差": round(float(col_data.std(ddof=1)), 3),
            "最小值": round(float(col_data.min()), 3),
            "25%分位": round(float(col_data.quantile(0.25)), 3),
            "中位数": round(float(col_data.median()), 3),
            "75%分位": round(float(col_data.quantile(0.75)), 3),
            "最大值": round(float(col_data.max()), 3),
        }

    return summary


# ============================================================================
# 模块主入口（测试用）
# ============================================================================

if __name__ == "__main__":
    """
    模块自测代码：生成模拟数据集并打印基本信息。
    仅在直接运行此文件时执行，被导入时不执行。
    """
    print("=" * 60)
    print("《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》")
    print("模拟数据生成模块 - 自测运行")
    print("=" * 60)

    # 生成小规模测试数据集
    print("\n[1/3] 正在生成模拟监测数据...")
    test_df: pd.DataFrame = generate_full_mock_dataset(
        samples_per_period=3,
        random_seed=123
    )

    print(f"\n✅ 数据生成完毕！")
    print(f"   - 总记录数：{len(test_df)} 条")
    print(f"   - 字段数量：{len(test_df.columns)} 个")
    print(f"   - 字段列表：{test_df.columns.tolist()}")

    # 打印前5行预览
    print("\n[2/3] 数据预览（前5行）：")
    print("-" * 60)
    print(test_df.head().to_string())

    # 打印描述性统计
    print("\n[3/3] 描述性统计摘要：")
    print("-" * 60)
    summary: dict = get_dataset_summary(test_df)
    for key, stats in summary.items():
        print(f"\n{key}:")
        for stat_name, stat_value in stats.items():
            print(f"   {stat_name}: {stat_value}")

    print("\n" + "=" * 60)
    print("自测完成！数据生成模块所有函数均正常运行。")
    print("=" * 60)
