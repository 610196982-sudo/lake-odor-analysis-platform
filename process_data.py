# -*- coding: utf-8 -*-
"""
==============================================================================
模块名称：process_data.py
所属系统：《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》
功能描述：数据处理与清洗模块
          —— 提供数据导入、异常值检测与剔除、缺失值插补、数据标准化、
             数据筛选与聚合等功能，为后续分析和可视化提供干净的数据基础。
==============================================================================
"""

import numpy as np
import pandas as pd
import os
import datetime
from typing import Optional, Tuple, List, Union
import warnings

# 忽略 pandas 的 FutureWarning（保持输出清洁）
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# 全局常量与配置
# ============================================================================

# 各理化指标在天然水体中的合理范围
# 用于异常值检测：超出此范围的值将被标记为可疑
REASONABLE_RANGES: dict = {
    "水温":    (0.0, 40.0),      # 水温（℃）：冰点至温泉级高温
    "pH":      (4.0, 10.5),      # pH：极端酸雨至强碱性湖泊
    "DO":      (0.1, 18.0),      # 溶解氧（mg/L）：缺氧至过饱和
    "浊度":    (0.0, 200.0),     # 浊度（NTU）：清澈至极端浑浊
    "TN":      (0.01, 10.0),     # 总氮（mg/L）
    "TP":      (0.001, 2.0),     # 总磷（mg/L）
    "NH3-N":   (0.005, 5.0),     # 氨氮（mg/L）
    "CODMn":   (0.5, 30.0),      # 高锰酸盐指数（mg/L）
    "叶绿素a": (0.1, 300.0),     # 叶绿素a（μg/L）
    "GSM":     (0.01, 100.0),    # 土臭素（ng/L）
    "2-MIB":   (0.01, 100.0),    # 2-甲基异莰醇（ng/L）
}

# 各指标的数据类型映射
COLUMN_DTYPES: dict = {
    "水温":    np.float64,
    "pH":      np.float64,
    "DO":      np.float64,
    "浊度":    np.float64,
    "TN":      np.float64,
    "TP":      np.float64,
    "NH3-N":   np.float64,
    "CODMn":   np.float64,
    "叶绿素a": np.float64,
    "GSM":     np.float64,
    "2-MIB":   np.float64,
}

# 环境驱动因子白名单：仅这些水质/生物指标参与「驱动因子」相关分析。
# 排除坐标（经度/纬度）与物理量（DEP/ALT/压力/盐度/DO饱和度等），
# 避免把这些非驱动项当作嗅味物质的环境驱动因子。
DRIVER_COLUMNS: list = [
    "水温", "DO", "pH", "浊度", "电导率", "TN", "TP", "NH3-N", "CODMn",
    "叶绿素a", "藻密度", "fDOM", "PC",
]


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    校验输入是否为有效的非空 pandas DataFrame。

    参数
    ----
    df : pd.DataFrame
        待校验的数据框。

    返回
    ----
    bool
        若数据框有效则返回 True。

    抛出
    ----
    TypeError
        若输入不是 pandas DataFrame。
    ValueError
        若 DataFrame 为空。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"输入必须为 pandas DataFrame，当前传入类型为：{type(df)}"
        )
    if df.empty:
        raise ValueError("输入的 DataFrame 不能为空。")
    return True


def validate_columns_exist(
    df: pd.DataFrame,
    required_cols: List[str]
) -> bool:
    """
    校验 DataFrame 中是否包含指定的列名。

    参数
    ----
    df : pd.DataFrame
        待校验的数据框。
    required_cols : List[str]
        必须存在的列名列表。

    返回
    ----
    bool
        若所有必需列都存在则返回 True。

    抛出
    ----
    ValueError
        若有缺失的列名。
    """
    missing_cols: list = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"以下必需列在 DataFrame 中不存在：{missing_cols}"
        )
    return True


# ============================================================================
# 数据导入函数
# ============================================================================

def load_dataset_from_csv(file_path: str) -> pd.DataFrame:
    """
    从 CSV 文件导入监测数据集。

    支持 UTF-8 和 GBK 两种编码自动尝试，确保中文 Windows 环境下的兼容性。

    参数
    ----
    file_path : str
        CSV 文件的完整路径。

    返回
    ----
    pd.DataFrame
        导入的数据框。

    示例
    ----
    >>> df = load_dataset_from_csv("mock_water_quality_data.csv")
    >>> print(df.shape)
    """
    # --- 参数校验 ---
    if not isinstance(file_path, str):
        raise TypeError(
            f"file_path 必须为字符串类型，当前传入类型为：{type(file_path)}"
        )
    if file_path.strip() == "":
        raise ValueError("文件路径不能为空字符串。")

    # --- 尝试多种编码读取 ---
    encodings_to_try: list = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]
    last_error: Optional[Exception] = None

    for encoding in encodings_to_try:
        try:
            df: pd.DataFrame = pd.read_csv(file_path, encoding=encoding)
            if df.empty:
                continue
            # 如果读到的行数 > 0，说明编码有效
            print(f"  [信息] 成功使用 {encoding} 编码读取文件，共 {len(df)} 条记录。")
            return df
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到指定的文件：{file_path}")
        except Exception as e:
            last_error = e
            continue

    # 若所有编码都失败
    raise ValueError(
        f"无法使用任何已知编码读取文件 {file_path}。"
        f"最后错误：{last_error}"
    )


def load_dataset_from_excel(file_path: str, sheet_name: str = 0) -> pd.DataFrame:
    """
    从 Excel 文件（.xlsx / .xls）导入监测数据集。

    参数
    ----
    file_path : str
        Excel 文件的完整路径。
    sheet_name : str or int
        工作表名称或索引，默认读取第一个工作表。

    返回
    ----
    pd.DataFrame
        导入的数据框。
    """
    # --- 参数校验 ---
    if not isinstance(file_path, str):
        raise TypeError(
            f"file_path 必须为字符串类型，当前传入类型为：{type(file_path)}"
        )
    if file_path.strip() == "":
        raise ValueError("文件路径不能为空字符串。")
    if not (file_path.lower().endswith(".xlsx") or file_path.lower().endswith(".xls")):
        raise ValueError(
            f"文件扩展名必须为 .xlsx 或 .xls，当前路径为：{file_path}"
        )

    try:
        df: pd.DataFrame = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"  [信息] 成功从 Excel 读取数据，共 {len(df)} 条记录。")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到指定的文件：{file_path}")
    except Exception as e:
        raise RuntimeError(f"读取 Excel 文件时发生错误：{e}")


# ============================================================================
# 数据清洗函数
# ============================================================================

def detect_outliers_iqr(
    data: np.ndarray,
    multiplier: float = 1.5
) -> np.ndarray:
    """
    使用四分位距法（IQR）检测异常值。

    异常值判定标准：
    - 下界 = Q1 - multiplier × IQR
    - 上界 = Q3 + multiplier × IQR
    - 超出此范围的数据点视为异常值。

    参数
    ----
    data : np.ndarray
        一维数值数组。
    multiplier : float
        IQR 的倍数系数，默认 1.5（标准箱线图标准）。
        设为 3.0 时只检测极端异常值。

    返回
    ----
    np.ndarray
        布尔型数组，True 表示该位置为异常值。

    示例
    ----
    >>> data = np.array([1, 2, 3, 4, 100])
    >>> detect_outliers_iqr(data)
    array([False, False, False, False,  True])
    """
    # --- 参数校验 ---
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"data 必须为一维数组，当前维度为：{data.ndim}")
    if len(data) < 4:
        raise ValueError(
            f"数据量不足以进行 IQR 计算，至少需要 4 个数据点，"
            f"当前为：{len(data)}"
        )
    if multiplier <= 0:
        raise ValueError(
            f"multiplier 必须为正数，当前传入值为：{multiplier}"
        )

    # --- 计算四分位数 ---
    q1: float = float(np.percentile(data, 25))
    q3: float = float(np.percentile(data, 75))
    iqr: float = q3 - q1

    # --- 计算上下界 ---
    lower_bound: float = q1 - multiplier * iqr
    upper_bound: float = q3 + multiplier * iqr

    # --- 标记异常值 ---
    outliers: np.ndarray = (data < lower_bound) | (data > upper_bound)

    return outliers


def detect_outliers_range(
    data: np.ndarray,
    lower: float,
    upper: float
) -> np.ndarray:
    """
    使用固定合理范围检测异常值。

    根据水环境科学常识设定的指标合理范围来标记异常值。
    适用于有明显物理/化学边界的指标（如 pH 不能 < 0，水温不能 > 50℃）。

    参数
    ----
    data : np.ndarray
        一维数值数组。
    lower : float
        合理范围下限。
    upper : float
        合理范围上限。

    返回
    ----
    np.ndarray
        布尔型数组，True 表示该位置为异常值。
    """
    # --- 参数校验 ---
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"data 必须为一维数组，当前维度为：{data.ndim}")
    if lower >= upper:
        raise ValueError(
            f"下限 (lower={lower}) 必须小于上限 (upper={upper})。"
        )

    outliers: np.ndarray = (data < lower) | (data > upper)
    return outliers


def clean_dataset(
    df: pd.DataFrame,
    method: str = "iqr",
    iqr_multiplier: float = 3.0,
    remove_outliers: bool = False
) -> pd.DataFrame:
    """
    对监测数据集进行全面清洗：
    1. 检测并报告各数值列的异常值
    2. 检测并报告缺失值
    3. 根据参数选择剔除或保留异常值

    参数
    ----
    df : pd.DataFrame
        待清洗的监测数据集。
    method : str
        异常值检测方法，可选 'iqr'（四分位距法）或 'range'（合理范围法）。
    iqr_multiplier : float
        IQR 倍数系数（仅在 method='iqr' 时有效），默认为 3.0（只检极端异常）。
    remove_outliers : bool
        是否直接剔除含有异常值的行。默认为 False，仅标记不剔除。

    返回
    ----
    pd.DataFrame
        清洗后的 DataFrame。
    """
    # --- 参数校验 ---
    validate_dataframe(df)
    if method not in ["iqr", "range"]:
        raise ValueError(
            f"method 必须为 'iqr' 或 'range'，当前传入值为：{method}"
        )
    if iqr_multiplier <= 0:
        raise ValueError(
            f"iqr_multiplier 必须为正数，当前传入值为：{iqr_multiplier}"
        )

    # --- 复制数据以免修改原始数据 ---
    cleaned_df: pd.DataFrame = df.copy()

    # --- 确定需要进行清洗的数值列 ---
    numeric_cols: list = cleaned_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # --- 统计信息 ---
    total_outliers: int = 0
    outlier_report: dict = {}

    for col in numeric_cols:
        col_data: np.ndarray = cleaned_df[col].dropna().values

        if len(col_data) < 4:
            # 数据太少，跳过 IQR 检测
            continue

        # --- 根据方法检测异常值 ---
        if method == "iqr":
            outlier_mask: np.ndarray = detect_outliers_iqr(
                col_data, multiplier=iqr_multiplier
            )
        else:  # method == "range"
            if col in REASONABLE_RANGES:
                lower, upper = REASONABLE_RANGES[col]
                outlier_mask = detect_outliers_range(col_data, lower, upper)
            else:
                continue  # 没有预定义范围则跳过

        n_outliers: int = int(outlier_mask.sum())
        if n_outliers > 0:
            outlier_report[col] = {
                "异常值数量": n_outliers,
                "占比": f"{n_outliers / len(col_data) * 100:.1f}%",
            }
            total_outliers += n_outliers

            # --- 若选择剔除异常值 ---
            if remove_outliers:
                abnormal_values = col_data[outlier_mask]
                # 将这些异常值替换为 NaN
                for val in abnormal_values:
                    cleaned_df.loc[
                        cleaned_df[col] == val, col
                    ] = np.nan

    # --- 缺失值统计与处理 ---
    missing_report: dict = {}
    for col in df.columns:
        n_missing: int = int(cleaned_df[col].isna().sum())
        if n_missing > 0:
            missing_report[col] = {
                "缺失值数量": n_missing,
                "占比": f"{n_missing / len(cleaned_df) * 100:.1f}%",
            }

    # --- 打印清洗报告（可选） ---
    if outlier_report or missing_report:
        print("\n" + "=" * 50)
        print("[数据清洗报告]")
        print("=" * 50)
        if outlier_report:
            print(f"\n  异常值检测结果（方法：{method}）：")
            for col, info in outlier_report.items():
                print(f"   - {col}: {info['异常值数量']} 个异常值 ({info['占比']})")
        if missing_report:
            print(f"\n  缺失值统计：")
            for col, info in missing_report.items():
                print(f"   - {col}: {info['缺失值数量']} 个缺失值 ({info['占比']})")
        if not outlier_report and not missing_report:
            print("   数据质量良好，未发现异常值或缺失值。")
        print("=" * 50 + "\n")

    return cleaned_df


# ============================================================================
# 缺失值处理函数
# ============================================================================

def fill_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    对数据集中的缺失值进行智能填补。

    支持的填补策略：
    - 'mean'：用该列的均值填补。
    - 'median'：用该列的中位数填补。
    - 'ffill'：用前一个有效值向前填充。
    - 'bfill'：用后一个有效值向后填充。
    - 'interpolate'：线性插值。
    - 'group_mean'：按分组变量（如'监测时段'）计算组内均值填补。

    参数
    ----
    df : pd.DataFrame
        包含缺失值的数据框。
    strategy : str
        缺失值填补策略，默认 'mean'。
    group_by : Optional[str]
        分组变量名（仅在 strategy='group_mean' 时需要）。

    返回
    ----
    pd.DataFrame
        填补后的数据框。
    """
    # --- 参数校验 ---
    validate_dataframe(df)
    valid_strategies: list = [
        "mean", "median", "ffill", "bfill", "interpolate", "group_mean"
    ]
    if strategy not in valid_strategies:
        raise ValueError(
            f"strategy 必须为 {valid_strategies} 之一，"
            f"当前传入值为：{strategy}"
        )
    if strategy == "group_mean" and group_by is None:
        raise ValueError("使用 'group_mean' 策略时必须指定 group_by 参数。")
    if group_by is not None and group_by not in df.columns:
        raise ValueError(
            f"分组变量 '{group_by}' 在 DataFrame 中不存在。"
        )

    # --- 复制数据 ---
    filled_df: pd.DataFrame = df.copy()

    # --- 获取数值列 ---
    numeric_cols: list = filled_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for col in numeric_cols:
        if filled_df[col].isna().sum() == 0:
            continue  # 无缺失值，跳过

        if strategy == "mean":
            fill_value: float = float(filled_df[col].mean())
            filled_df[col].fillna(fill_value, inplace=True)

        elif strategy == "median":
            fill_value = float(filled_df[col].median())
            filled_df[col].fillna(fill_value, inplace=True)

        elif strategy == "ffill":
            filled_df[col].fillna(method="ffill", inplace=True)

        elif strategy == "bfill":
            filled_df[col].fillna(method="bfill", inplace=True)

        elif strategy == "interpolate":
            filled_df[col] = filled_df[col].interpolate(
                method="linear", limit_direction="both"
            )

        elif strategy == "group_mean":
            # 按分组计算均值并填补
            group_means = filled_df.groupby(group_by)[col].transform("mean")
            filled_df[col].fillna(group_means, inplace=True)

    # --- 处理策略未覆盖的残余缺失值 ---
    remaining_missing: int = int(filled_df[numeric_cols].isna().sum().sum())
    if remaining_missing > 0:
        # 用全局均值兜底
        for col in numeric_cols:
            if filled_df[col].isna().sum() > 0:
                global_mean: float = float(filled_df[col].mean())
                filled_df[col].fillna(global_mean, inplace=True)

    print(f"  [信息] 缺失值填补完成，策略：{strategy}。")

    return filled_df


# ============================================================================
# 数据筛选与聚合函数
# ============================================================================

def filter_by_lake(
    df: pd.DataFrame,
    lake_names: Union[str, List[str]]
) -> pd.DataFrame:
    """
    按湖泊名称筛选数据子集。

    参数
    ----
    df : pd.DataFrame
        完整数据集。
    lake_names : str or List[str]
        单个湖泊名称或湖泊名称列表。

    返回
    ----
    pd.DataFrame
        筛选后的数据子集。
    """
    validate_dataframe(df)
    validate_columns_exist(df, ["湖泊名称"])

    # --- 参数标准化 ---
    if isinstance(lake_names, str):
        lake_names = [lake_names]

    # --- 校验传入的湖泊名称 ---
    valid_lakes: set = set(df["湖泊名称"].unique())
    for name in lake_names:
        if name not in valid_lakes:
            print(f"  [警告] 湖泊名称 '{name}' 在数据集中不存在，已跳过。")

    valid_names: list = [n for n in lake_names if n in valid_lakes]
    if not valid_names:
        raise ValueError("所有传入的湖泊名称在数据集中都不存在。")

    filtered: pd.DataFrame = df[df["湖泊名称"].isin(valid_names)].copy()
    print(f"  [信息] 按湖泊筛选后保留 {len(filtered)}/{len(df)} 条记录。")
    return filtered


def filter_by_period(
    df: pd.DataFrame,
    periods: Union[str, List[str]]
) -> pd.DataFrame:
    """
    按监测时段筛选数据子集。

    参数
    ----
    df : pd.DataFrame
        完整数据集。
    periods : str or List[str]
        单个监测时段名称或名称列表。

    返回
    ----
    pd.DataFrame
        筛选后的数据子集。
    """
    validate_dataframe(df)
    validate_columns_exist(df, ["监测时段"])

    # --- 参数标准化 ---
    if isinstance(periods, str):
        periods = [periods]

    # --- 校验 ---
    valid_periods: set = set(df["监测时段"].unique())
    valid_input: list = [p for p in periods if p in valid_periods]
    if not valid_input:
        raise ValueError("所有传入的监测时段名称在数据集中都不存在。")

    filtered: pd.DataFrame = df[df["监测时段"].isin(valid_input)].copy()
    print(f"  [信息] 按监测时段筛选后保留 {len(filtered)}/{len(df)} 条记录。")
    return filtered


def aggregate_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_cols: Optional[List[str]] = None,
    agg_func: str = "mean"
) -> pd.DataFrame:
    """
    按指定分组变量对数值指标进行聚合统计。

    参数
    ----
    df : pd.DataFrame
        待聚合的数据集。
    group_cols : List[str]
        分组列名列表（如 ['湖泊名称', '监测时段']）。
    agg_cols : Optional[List[str]]
        需要聚合的数值列名列表。若为 None，则对所有数值列进行聚合。
    agg_func : str
        聚合函数，可选 'mean', 'median', 'std', 'min', 'max', 'count'。

    返回
    ----
    pd.DataFrame
        聚合后的数据框。
    """
    # --- 参数校验 ---
    validate_dataframe(df)
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"分组列 '{col}' 在 DataFrame 中不存在。")

    valid_funcs: list = ["mean", "median", "std", "min", "max", "count"]
    if agg_func not in valid_funcs:
        raise ValueError(
            f"agg_func 必须为 {valid_funcs} 之一，"
            f"当前传入值为：{agg_func}"
        )

    # --- 确定聚合的目标列 ---
    if agg_cols is None:
        agg_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        for col in agg_cols:
            if col not in df.columns:
                raise ValueError(f"聚合列 '{col}' 在 DataFrame 中不存在。")

    # --- 执行分组聚合 ---
    agg_map: dict = {col: agg_func for col in agg_cols}
    aggregated: pd.DataFrame = df.groupby(group_cols, as_index=False).agg(agg_map)

    # --- 四舍五入 ---
    for col in agg_cols:
        if col in aggregated.columns:
            aggregated[col] = aggregated[col].round(3)

    print(
        f"  [信息] 按 {group_cols} 分组，"
        f"使用 {agg_func} 聚合，结果包含 {len(aggregated)} 行。"
    )
    return aggregated


# ============================================================================
# 数据标准化函数
# ============================================================================

def normalize_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "zscore"
) -> pd.DataFrame:
    """
    对指定数值列进行数据标准化（归一化）。

    支持的标准化方法：
    - 'zscore'：Z-score 标准化（均值0，标准差1）。
    - 'minmax'：Min-Max 归一化（缩放至 [0, 1] 区间）。
    - 'robust'：Robust 标准化（使用中位数和 IQR，对异常值不敏感）。

    参数
    ----
    df : pd.DataFrame
        待标准化的数据框。
    columns : Optional[List[str]]
        需要标准化的列名列表。若为 None，则对所有数值列标准化。
    method : str
        标准化方法，默认 'zscore'。

    返回
    ----
    pd.DataFrame
        标准化后的数据框（保留非数值列原样）。
    """
    # --- 参数校验 ---
    validate_dataframe(df)
    valid_methods: list = ["zscore", "minmax", "robust"]
    if method not in valid_methods:
        raise ValueError(
            f"method 必须为 {valid_methods} 之一，当前传入值为：{method}"
        )

    # --- 确定需要标准化的列 ---
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 在 DataFrame 中不存在。")
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  [警告] 列 '{col}' 不是数值类型，已跳过标准化。")
            continue

    # --- 执行标准化 ---
    normalized_df: pd.DataFrame = df.copy()

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_values: np.ndarray = normalized_df[col].values.astype(np.float64)

        if method == "zscore":
            mean_val: float = float(np.mean(col_values))
            std_val: float = float(np.std(col_values, ddof=1))
            if std_val > 0:
                normalized_df[col] = (col_values - mean_val) / std_val
            else:
                normalized_df[col] = 0.0  # 常量列全设为 0

        elif method == "minmax":
            min_val: float = float(np.min(col_values))
            max_val: float = float(np.max(col_values))
            if max_val - min_val > 0:
                normalized_df[col] = (
                    (col_values - min_val) / (max_val - min_val)
                )
            else:
                normalized_df[col] = 0.0

        elif method == "robust":
            median_val: float = float(np.median(col_values))
            q1: float = float(np.percentile(col_values, 25))
            q3: float = float(np.percentile(col_values, 75))
            iqr: float = q3 - q1
            if iqr > 0:
                normalized_df[col] = (col_values - median_val) / iqr
            else:
                normalized_df[col] = 0.0

    return normalized_df


# ============================================================================
# 智能列名映射与导入
# ============================================================================

# 列名映射表：将各种可能的原始列名映射到平台标准列名
COLUMN_NAME_MAP: dict = {
    # 溶解氧（含括号内英文缩写变体）
    "溶解氧（mg/L）": "DO",
    "溶解氧(mg/L)": "DO",
    "溶解氧（mg/L)": "DO",
    "溶解氧(mg/L)": "DO",
    "溶解氧（DO）（mg/L）": "DO",
    "溶解氧（DO）（mg/L)": "DO",
    "溶解氧(DO)(mg/L)": "DO",
    "溶解氧": "DO",
    "DO(mg/L)": "DO",
    "DO（mg/L）": "DO",
    "DO（mg/L)": "DO",
    "DO": "DO",
    # pH
    "pH": "pH",
    "PH": "pH",
    "ph": "pH",
    "Ph": "pH",
    # pH — 半角括号变体
    "pH（mg/L）": "pH",
    "pH(mg/L)": "pH",
    # 浊度
    "浊度（NTU）": "浊度",
    "浊度(NTU)": "浊度",
    "浊度NTU": "浊度",
    "浊度": "浊度",
    "NTU": "浊度",
    # 总氮
    "总氮（mg/L）": "TN",
    "总氮(mg/L)": "TN",
    "总氮TN": "TN",
    "总氮": "TN",
    "TN(mg/L)": "TN",
    "TN": "TN",
    # 总磷
    "总磷（mg/L）": "TP",
    "总磷(mg/L)": "TP",
    "总磷TP": "TP",
    "总磷": "TP",
    "TP(mg/L)": "TP",
    "TP": "TP",
    # 氨氮
    "氨氮（mg/L）": "NH3-N",
    "氨氮(mg/L)": "NH3-N",
    "氨氮NH3-N": "NH3-N",
    "氨氮": "NH3-N",
    "NH3-N(mg/L)": "NH3-N",
    "NH3-N": "NH3-N",
    "NH4-N": "NH3-N",
    "NH3N": "NH3-N",
    # CODMn（含括号内英文缩写变体）
    "COD锰（mg/L）": "CODMn",
    "COD锰(mg/L)": "CODMn",
    "CODMn（mg/L）": "CODMn",
    "高锰酸盐指数（mg/L）": "CODMn",
    "高锰酸盐指数(mg/L)": "CODMn",
    "高锰酸盐指数（CODMn）（mg/L）": "CODMn",
    "高锰酸盐指数(CODMn)(mg/L)": "CODMn",
    "高锰酸盐指数": "CODMn",
    "CODMn": "CODMn",
    "COD_Mn": "CODMn",
    # CODmn 半角括号 & 大小写变体
    "CODmn（mg/L）": "CODMn",
    "CODmn(mg/L)": "CODMn",
    "CODmn": "CODMn",
    "CODMN": "CODMn",
    # 叶绿素（扩展映射）
    "叶绿素（mg/L）": "叶绿素a_raw_mgL",
    "叶绿素(mg/L)": "叶绿素a_raw_mgL",
    "叶绿素a（mg/L）": "叶绿素a_raw_mgL",
    "叶绿素a(mg/L)": "叶绿素a_raw_mgL",
    "叶绿素a（μg/L）": "叶绿素a",
    "叶绿素a(μg/L)": "叶绿素a",
    "叶绿素a（ug/L）": "叶绿素a",
    "叶绿素a(ug/L)": "叶绿素a",
    "叶绿素（μg/L）": "叶绿素a",
    "叶绿素(μg/L)": "叶绿素a",
    "叶绿素（ug/L）": "叶绿素a",
    "叶绿素(ug/L)": "叶绿素a",
    "叶绿素a": "叶绿素a",
    "chl（现场）（ug/L）": "叶绿素a",
    "chl(现场)(ug/L)": "叶绿素a",
    "叶绿素（送检）（µg/L）": "叶绿素a_送检",
    "叶绿素(送检)(µg/L)": "叶绿素a_送检",
    "叶绿素（送检）（μg/L）": "叶绿素a_送检",
    "叶绿素(送检)(μg/L)": "叶绿素a_送检",
    "Chl-a(μg/L)": "叶绿素a",
    "Chl-a(ug/L)": "叶绿素a",
    "Chla": "叶绿素a",
    "叶绿素": "叶绿素a_raw_mgL",
    # 水温
    "水温（℃）": "水温",
    "水温(℃)": "水温",
    "水温": "水温",
    "温度（℃）": "水温",
    "温度(℃)": "水温",
    "温度": "水温",
    "WT": "水温",
    "WT（℃）": "水温",
    "WT(℃)": "水温",
    "Temp": "水温",
    # GSM / 2-MIB（扩展映射，覆盖更多写法）
    "GSM（ng/L）": "GSM",
    "GSM(ng/L)": "GSM",
    "GSM": "GSM",
    "土臭素（ng/L）": "GSM",
    "土臭素(ng/L)": "GSM",
    "土臭素": "GSM",
    "土臭素（ng/L)": "GSM",
    "土臭素(ng/L)": "GSM",
    "geosmin": "GSM",
    "Geosmin": "GSM",
    "2-MIB（ng/L）": "2-MIB",
    "2-MIB(ng/L)": "2-MIB",
    "2-MIB": "2-MIB",
    "2MIB": "2-MIB",
    "二甲基异莰醇": "2-MIB",
    "二甲基异莰醇（ng/L）": "2-MIB",
    "二甲基异莰醇(ng/L)": "2-MIB",
    "2-甲基异莰醇": "2-MIB",
    "2-甲基异莰醇（ng/L）": "2-MIB",
    "2-甲基异莰醇(ng/L)": "2-MIB",
    "2-甲基异莰醇（ng/L)": "2-MIB",
    "MIB": "2-MIB",
    "MIB（ng/L）": "2-MIB",
    "MIB(ng/L)": "2-MIB",
    # 点位/站点（含月份标注变体）
    "点位": "采样点位",
    "站点": "采样点位",
    "采样点": "采样点位",
    "监测点位": "采样点位",
    "采样点位": "采样点位",
    "监测点": "采样点位",
    "站点名称": "采样点位",
    "点位名称": "采样点位",
    "点位（5月）": "采样点位",
    "点位（6月）": "采样点位",
    "点位（3月）": "采样点位",
    "点位（4月）": "采样点位",
    "点位（7月）": "采样点位",
    "点位（8月）": "采样点位",
    "点位（9月）": "采样点位",
    "点位（10月）": "采样点位",
    "点位（11月）": "采样点位",
    "点位(5月)": "采样点位",
    "点位(6月)": "采样点位",
    "点位(3月)": "采样点位",
    "点位(4月)": "采样点位",
    # 经度/纬度
    "经度": "经度",
    "lon": "经度",
    "longitude": "经度",
    "纬度": "纬度",
    "lat": "纬度",
    "latitude": "纬度",
    # 电导率
    "电导率（ms/cm）": "电导率",
    "电导率(ms/cm)": "电导率",
    "电导率（μS/cm）": "电导率",
    "电导率(μS/cm)": "电导率",
    "电导率（μS/cm)" : "电导率",
    "电导率（SPC）（ms/cm）": "电导率",
    "电导率(SPC)(ms/cm)": "电导率",
    "电导率（SPC）": "电导率",
    "SPC（ms/cm）": "电导率",
    "SPC(ms/cm)": "电导率",
    "电导率": "电导率",
    "Cond": "电导率",
    "电导率（μs/cm）": "电导率",
    "电导率(μs/cm)": "电导率",
    "电导率（SPC）（us/cm）": "电导率",
    "电导率(SPC)(us/cm)": "电导率",
    "电导率（us/cm）": "电导率",
    "电导率(us/cm)": "电导率",
    # 氧化还原电位
    "氧化还原电位（mV）": "氧化还原电位",
    "氧化还原电位(mV)": "氧化还原电位",
    "氧化还原电位（ORF）（mV）": "氧化还原电位",
    "氧化还原电位(ORF)(mV)": "氧化还原电位",
    "氧化还原电位（ORP）（mV）": "氧化还原电位",
    "氧化还原电位(ORP)(mV)": "氧化还原电位",
    "氧化还原电位": "氧化还原电位",
    "ORP": "氧化还原电位",
    "ORF": "氧化还原电位",
    # 藻密度
    "藻密度（万个/L）": "藻密度",
    "藻密度(万个/L)": "藻密度",
    "藻密度": "藻密度",
    "藻细胞密度": "藻密度",
    "藻细胞密度（万个/L）": "藻密度",
    "藻细胞密度(万个/L)": "藻密度",
    # 水温（更多变体）
    "WT（℃）": "水温",
    "WT(℃)": "水温",
    "WT": "水温",
    "Temp": "水温",
    "温度（℃）": "水温",
    "温度": "水温",
    # 盐度
    "盐度（SAL）": "盐度",
    "盐度(SAL)": "盐度",
    "SAL": "盐度",
    "盐度": "盐度",
    # 总悬浮固体
    "总悬浮固体（TSS）（mg/L）": "TSS",
    "总悬浮固体(TSS)(mg/L)": "TSS",
    "TSS（mg/L）": "TSS",
    "TSS(mg/L)": "TSS",
    "TSS": "TSS",
    # DO 饱和度
    "DO（%）": "DO饱和度",
    "DO(%)": "DO饱和度",
    "DO（%RTB）": "DO饱和度RTB",
    "DO(%RTB)": "DO饱和度RTB",
    # 压力
    "压力（Kpa）": "压力",
    "压力(Kpa)": "压力",
    "压力（kPa）": "压力",
    "压力(kPa)": "压力",
    # PC
    "PC（ug/L）": "PC",
    "PC(ug/L)": "PC",
    "PC（μg/L）": "PC",
    "PC(μg/L)": "PC",
    # fDOM
    "fDOM（QSU）": "fDOM",
    "fDOM(QSU)": "fDOM",
    "fDOM ppd": "fDOM_ppd",
    # 水深 / 海拔
    "DEP（m）": "DEP",
    "DEP(m)": "DEP",
    "ALT（m）": "ALT",
    "ALT(m)": "ALT",
    # 月份标注列（如 "点位（5月）" → 用于推断监测时段）
    # 这些列在 smart_import 内部会被特殊处理，此处仅保留占位
}


# ============================================================================
# 转置格式参数映射表
# ============================================================================

# 用于 transposed 格式（参数在行、采样点在列）的参数名→标准名映射
TRANSPOSED_PARAM_MAP: dict = {
    "温度":      "水温",
    "Kpa":       "压力",
    "DO %":      "DO饱和度",
    "DO mg/L":   "DO",
    "DO %RTB":   "DO饱和度RTB",
    "SPC":       "电导率",
    "SAL":       "盐度",
    "pH":        "pH",
    "ORF  mV":   "氧化还原电位",
    "NTU":       "浊度",
    "TSS mg/L":  "TSS",
    "PC ug/L":   "PC",
    "chl ug/L":  "叶绿素a",
    "fDOM QSU":  "fDOM",
    "fDOM ppd":  "fDOM_ppd",
    "DEP m":     "DEP",
    "ALT m":     "ALT",
}


# ============================================================================
# 转置格式检测与转换
# ============================================================================

def _detect_transposed_format(df: pd.DataFrame) -> bool:
    """
    检测 DataFrame 是否为「转置格式」：参数名在行首列，采样点/湖泊在列标题。

    特征：
    - 首列为 Unnamed / 空名
    - 其他列标题含"湖"/"水库"等水体名
    - 首列值含 ≥3 个已知参数关键词（温度、DO、pH、SPC、NTU 等）
    - 列数 ≤ 4 且行数 > 列数（窄高形状）

    参数
    ----
    df : pd.DataFrame
        原始读取的 DataFrame。

    返回
    ----
    bool
        True 表示检测到转置格式。
    """
    if df.empty or len(df.columns) < 2 or len(df.columns) > 4:
        return False
    if len(df) <= len(df.columns):
        return False  # 行少列多 → 不是转置

    # 首列检查
    col0 = str(df.columns[0])
    if not (col0.startswith("Unnamed") or col0 == "" or col0 == "nan"):
        return False

    # 其他列标题是否包含水体标识
    other_headers = [str(c).strip() for c in df.columns[1:]]
    has_lake = any("湖" in h or "水库" in h for h in other_headers)
    if not has_lake:
        return False

    # 首列值是否包含参数关键词
    param_indicators = [
        "温度", "DO", "pH", "Kpa", "SPC", "SAL", "NTU",
        "TSS", "PC", "chl", "fDOM", "DEP", "ALT", "ORF",
        "°C", "℃",
    ]
    first_col_vals = [str(v).strip() for v in df.iloc[:, 0] if pd.notna(v)]
    matches = sum(
        1 for v in first_col_vals
        if any(indicator in v for indicator in param_indicators)
    )
    return matches >= 3


def _transform_transposed(
    df: pd.DataFrame,
    file_path: str = "",
) -> pd.DataFrame:
    """
    将转置格式的 DataFrame 转换为标准格式（每行=一个采样点）。

    转换逻辑：
    1. 首列值作为参数名，列标题（第 2..N 列）作为湖泊/点位名
    2. 转置后每个湖泊一行，各参数为列
    3. 末尾 2 行可能含坐标（度分秒格式），自动提取并转换为十进制度
    4. 参数名通过 TRANSPOSED_PARAM_MAP 映射到标准名

    参数
    ----
    df : pd.DataFrame
        转置格式的原始 DataFrame。
    file_path : str
        文件路径，用于日志输出。

    返回
    ----
    pd.DataFrame
        标准化后的 DataFrame。
    """
    import re

    n_cols = len(df.columns)
    lake_names = [str(c).strip() for c in df.columns[1:]]

    # --- 提取坐标行 ---
    lat_values: dict = {}   # lake_name → decimal latitude
    lon_values: dict = {}   # lake_name → decimal longitude

    # 扫描最后几行，查找坐标行
    # 坐标行的特征：行标签为 E/N/NTU，且值包含度分秒格式（如 "119度33分"）
    coord_candidates = {}  # row_idx → (coord_type, {lake: value}) or None if excluded
    coord_rows_to_skip = set()  # 确定为坐标行，从数据行中排除

    for i in range(max(0, len(df) - 5), len(df)):
        first_val = str(df.iloc[i, 0]).strip() if pd.notna(df.iloc[i, 0]) else ""

        # 检查该行的值是否像坐标（度分秒格式）
        looks_like_coords = False
        for j in range(1, n_cols):
            v = df.iloc[i, j]
            if pd.notna(v):
                vs = str(v).strip()
                if re.search(r'\d+度\d+分', vs):
                    looks_like_coords = True
                    break

        if not looks_like_coords:
            continue  # 不是坐标行，保留在数据中

        coord_rows_to_skip.add(i)

        if first_val in ("NTU", "N", "纬度", "Latitude"):
            # 纬度行
            vals = {}
            for j, lake in enumerate(lake_names):
                v = df.iloc[i, j + 1]
                vals[lake] = v if pd.notna(v) else None
            coord_candidates[i] = ("lat", vals)
        elif first_val in ("E", "经度", "Longitude"):
            # 经度行
            vals = {}
            for j, lake in enumerate(lake_names):
                v = df.iloc[i, j + 1]
                vals[lake] = v if pd.notna(v) else None
            coord_candidates[i] = ("lon", vals)

    # 解析坐标值
    for idx, (c_type, vals) in coord_candidates.items():
        for lake, raw_val in vals.items():
            if raw_val is not None:
                decimal = _parse_dms_to_decimal(str(raw_val))
                if decimal is not None:
                    if c_type == "lat":
                        lat_values[lake] = decimal
                    else:
                        lon_values[lake] = decimal

    # --- 构建数据区域（排除坐标行）---
    data_rows = []
    for i in range(len(df)):
        if i in coord_rows_to_skip:
            continue  # 坐标行，跳过
        first_val = str(df.iloc[i, 0]).strip() if pd.notna(df.iloc[i, 0]) else ""
        # 跳过空行
        if first_val == "" or first_val == "nan":
            continue
        param_name = first_val
        row_data = {"param": param_name}
        for j, lake in enumerate(lake_names):
            row_data[lake] = df.iloc[i, j + 1]
        data_rows.append(row_data)

    if not data_rows:
        raise ValueError("转置格式转换后无有效数据行。")

    # --- 转置：湖泊为行，参数为列 ---
    result_rows = []
    for lake in lake_names:
        row = {"湖泊名称": lake, "采样点位": lake}
        for dr in data_rows:
            param_raw = dr["param"]
            std_name = TRANSPOSED_PARAM_MAP.get(param_raw, param_raw)
            val = dr.get(lake)
            row[std_name] = val
        # 添加坐标
        if lake in lat_values:
            row["纬度"] = lat_values[lake]
        if lake in lon_values:
            row["经度"] = lon_values[lake]
        result_rows.append(row)

    result = pd.DataFrame(result_rows)

    # --- 过滤：移除几乎所有参数为空的湖泊行 ---
    param_cols = [
        c for c in result.columns
        if c not in ("湖泊名称", "采样点位", "经度", "纬度")
    ]
    if param_cols:
        result = result.dropna(subset=param_cols, how="all")

    print(f"  [信息] 检测到转置格式（{len(lake_names)} 个采样点），已自动转换为标准格式。"
          f" 结果：{len(result)} 行 × {len(result.columns)} 列。")

    return result


# ============================================================================
# Excel 日期序列号转换
# ============================================================================

def _excel_serial_to_date(serial) -> Optional[str]:
    """
    将 Excel 日期序列号（如 46183）转换为日期字符串 'YYYY-MM-DD'。

    参数
    ----
    serial : any
        Excel 日期序列号或其字符串表示。

    返回
    ----
    Optional[str]
        日期字符串，无法转换则返回 None。
    """
    if serial is None:
        return None
    try:
        s = float(serial)
        # Excel 日期序列号的有效范围（约 1900-01-01 到 2100-12-31）
        if s < 1 or s > 80000:
            return None
        base = datetime.datetime(1899, 12, 30)
        dt = base + datetime.timedelta(days=int(s))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        return None


# ============================================================================
# 点位→湖泊映射表
# ============================================================================

# 各监测点位所属湖泊/水库的映射关系
# 用于智能导入时自动识别点位归属，无需用户手动指定
POINT_LAKE_MAP: dict = {
    # === 太湖 ===
    "平台山":       "太湖",
    "泽山":         "太湖",
    "十四号灯标":   "太湖",
    "太湖平台山":   "太湖",
    "太湖泽山":     "太湖",
    "太湖十四号灯标": "太湖",
    "漫山":         "太湖",
    "拖山":         "太湖",
    "椒山":         "太湖",
    "沙渚":         "太湖",
    "沙墩港":       "太湖",
    # === 淀山湖 ===
    "淀山湖中":     "淀山湖",
    "淀山湖":       "淀山湖",
    # === 玄武湖 ===
    "玄武湖":       "玄武湖",
    # === 长荡湖 ===
    "长荡湖":       "长荡湖",
    # === 南湖 ===
    "南湖中心":     "南湖",
    "南湖":         "南湖",
    # === 潜明水库 ===
    "潜明水库":     "潜明水库",
    # === 千岛湖 ===
    "千岛湖":       "千岛湖",
    "千岛湖-入湖口": "千岛湖",
    "千岛湖-湖心":   "千岛湖",
    "千岛湖-出湖口": "千岛湖",
    # === 巢湖 ===
    "巢湖":         "巢湖",
    "巢湖-入湖口":   "巢湖",
    "巢湖-湖心":     "巢湖",
    "巢湖-出湖口":   "巢湖",
    # === 太湖（兼容旧格式）===
    "太湖-入湖口":   "太湖",
    "太湖-湖心":     "太湖",
    "太湖-出湖口":   "太湖",
    # === 长荡湖（兼容旧格式）===
    "长荡湖-入湖口": "长荡湖",
    "长荡湖-湖心":   "长荡湖",
    "长荡湖-出湖口": "长荡湖",
    # === 淀山湖（兼容旧格式）===
    "淀山湖-入湖口": "淀山湖",
    "淀山湖-湖心":   "淀山湖",
    "淀山湖-出湖口": "淀山湖",
}

# 8 个实测采样断面的基准坐标（十进制，(纬度, 经度)）。
# 用途：① GIS 地图无实测坐标时的兜底定位；② 源数据坐标填错时的纠偏基准。
POINT_COORDS: dict = {
    "平台山":     (31.223943, 120.108188),   # 太湖（苏州水域）
    "泽山":       (31.011452, 120.272031),   # 太湖
    "十四号灯标": (31.060807, 120.155375),   # 太湖
    "淀山湖中":   (31.117805, 120.964237),   # 淀山湖（上海青浦）
    "玄武湖":     (32.066667, 118.783333),   # 玄武湖（南京）
    "长荡湖":     (31.550000, 119.550000),   # 长荡湖（常州金坛）
    "南湖中心":   (30.760000, 120.760000),   # 南湖（嘉兴）
    "潜明水库":   (28.844975, 120.297794),   # 潜明水库（浙江缙云）
}

# 所有已知水体名称列表（动态从映射表提取 + 手动补充）
ALL_KNOWN_LAKES: list = sorted(set(
    list(POINT_LAKE_MAP.values()) + ["千岛湖", "太湖", "长荡湖", "巢湖", "淀山湖",
                                       "玄武湖", "南湖", "潜明水库"]
))


def resolve_lake_from_point(point_name: str) -> str:
    """
    根据采样点位名称推断所属湖泊/水库。

    优先查 POINT_LAKE_MAP 精确匹配，其次尝试模糊匹配
    （如点位名以湖泊名开头），最后返回原名称。

    参数
    ----
    point_name : str
        采样点位名称。

    返回
    ----
    str
        推断的湖泊名称。
    """
    if not isinstance(point_name, str) or point_name.strip() == "":
        return "未知水体"

    pn = point_name.strip()

    # 精确匹配
    if pn in POINT_LAKE_MAP:
        return POINT_LAKE_MAP[pn]

    # 模糊匹配：点位名以已知湖泊名开头
    for lake in ALL_KNOWN_LAKES:
        if pn.startswith(lake):
            return lake

    return pn  # 无法匹配则返回原点位名


def _parse_below_detection(value) -> Optional[float]:
    """
    解析低于检出限的表达（如 "＜2.2"、"<0.5"、"ND"）。

    策略：将低于检出限的值替换为检出限的一半（MDL/2），
    这是环境科学中常用的替代方法。

    参数
    ----
    value : any
        原始值，可能是字符串、数值或 None。

    返回
    ----
    Optional[float]
        解析后的数值，无法解析则返回 None。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if np.isnan(value) if isinstance(value, float) else False:
            return None
        return float(value)

    s = str(value).strip()
    if s == "" or s in ("/", "-", "—", "NA", "N/A", "ND", "nd", "null", "NULL"):
        return None

    # 匹配 "＜X.X" 或 "<X.X" 或 "≤X.X" 格式
    import re
    m = re.match(r'[＜<≤]\s*([\d.]+)', s)
    if m:
        detection_limit = float(m.group(1))
        # 返回检出限的一半（MDL/2 替代法）
        return round(detection_limit / 2.0, 4)

    # 尝试直接转数值
    try:
        return float(s)
    except ValueError:
        return None


def _parse_scientific_notation(value) -> Optional[float]:
    """
    解析文本形式的科学计数法，还原为十进制浮点数。

    Excel 中上标常被拍平为普通字符，导致出现如下写法：
    - "1.09×103"  → 实际为 1.09×10³ = 1090.0
    - "1.09×10^3" → 1090.0
    - "1.09E3" / "1.09e+3" → 1090.0

    参数
    ----
    value : any
        原始值，可能是字符串、数值或 None。

    返回
    ----
    Optional[float]
        解析后的浮点数，无法解析则返回 None。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)

    s = str(value).strip()
    if s == "":
        return None

    import re

    # 1) 标准科学计数法：1.09E3 / 1.09e+3 / 1.09e-3
    m = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)', s)
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))

    # 2) 乘号形式：1.09×10^3 / 1.09x10³ / 1.09×103（上标被拍平为普通数字）
    # 支持普通数字指数与 Unicode 上标数字（⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺）
    _superscript_trans = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    m = re.fullmatch(
        r'([+-]?\d+(?:\.\d+)?)\s*[×xX*]\s*10\s*(?:\^?\s*)([0-9⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺+-]+)',
        s,
    )
    if m:
        exp_str = m.group(2).translate(_superscript_trans)
        try:
            return float(m.group(1)) * (10.0 ** int(exp_str))
        except ValueError:
            pass

    return None


def _parse_numeric_cell(value) -> Optional[float]:
    """
    统一解析数值单元格：低于检出限（＜）、科学计数法、普通数值。

    优先尝试检出限解析（含普通数值），失败后尝试科学计数法解析。

    参数
    ----
    value : any
        原始单元格值。

    返回
    ----
    Optional[float]
        解析后的浮点数，无法解析则返回 None。
    """
    parsed = _parse_below_detection(value)
    if parsed is not None:
        return parsed
    return _parse_scientific_notation(value)


def _parse_dms_to_decimal(dms_str: str) -> Optional[float]:
    """
    将度分秒格式的坐标转换为十进制度。

    支持格式：
    - "120.1081885"（已是十进制）
    - "32度04分"（度分格式）
    - "118度47分"（度分格式）

    参数
    ----
    dms_str : str
        坐标字符串。

    返回
    ----
    Optional[float]
        十进制度数，无法解析则返回 None。
    """
    if dms_str is None:
        return None
    s = str(dms_str).strip()
    if s == "" or s in ("/", "-", "—"):
        return None

    import re

    # 尝试直接解析为浮点数
    try:
        return float(s)
    except ValueError:
        pass

    # 度分秒格式：XX度XX分 或 XX度XX分XX秒
    m = re.match(r'(\d+)\s*度\s*(\d+)\s*分(?:\s*(\d+(?:\.\d+)?)\s*秒)?', s)
    if m:
        deg = float(m.group(1))
        min_val = float(m.group(2))
        sec = float(m.group(3)) if m.group(3) else 0.0
        return round(deg + min_val / 60.0 + sec / 3600.0, 6)

    return None


# --- 编码检测 ---
def detect_encoding(file_path: str) -> str:
    """
    自动检测文本文件的编码格式。

    按优先级尝试常见的中文和通用编码，返回首先成功的编码名称。

    参数
    ----
    file_path : str
        文件路径。

    返回
    ----
    str
        检测到的编码名称。
    """
    encodings_to_try: list = ["utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                f.read(4096)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"


# --- 分隔符检测 ---
def detect_delimiter(file_path: str, encoding: str) -> str:
    """
    自动检测文本文件的分隔符。

    通过分析前几行数据，在常见分隔符（逗号、制表符、分号、空格）中选择
    使各行列数最一致的那个。

    参数
    ----
    file_path : str
        文件路径。
    encoding : str
        文件编码。

    返回
    ----
    str
        检测到的分隔符（','、'\\t'、';' 或 ' '）。
    """
    candidates: list = [
        (",", "逗号 (CSV)"),
        ("\t", "制表符 (TSV/TXT)"),
        (";", "分号"),
    ]
    best_delim: str = ","
    best_score: float = 0.0
    detected_name: str = "逗号 (CSV)"

    with open(file_path, "r", encoding=encoding) as f:
        lines = [f.readline() for _ in range(10)]
        lines = [l.rstrip("\n\r") for l in lines if l.strip()]

    if not lines:
        return ",", "逗号 (CSV)"

    for delim, name in candidates:
        n_cols_list = [len(l.split(delim)) for l in lines]
        if len(set(n_cols_list)) <= 1 and n_cols_list[0] >= 2:
            # 所有行列数一致，完美匹配
            return delim, name
        # 计算一致性得分
        mode_count = max(set(n_cols_list), key=n_cols_list.count)
        score = n_cols_list.count(mode_count) / len(n_cols_list)
        if score > best_score and mode_count >= 2:
            best_score = score
            best_delim = delim
            detected_name = name

    return best_delim, detected_name


# --- 检测是否为纯文本格式 ---
def _is_text_file(file_path: str) -> bool:
    """通过读取文件头部字节判断是否为文本文件。"""
    try:
        with open(file_path, "rb") as f:
            head = f.read(4096)
        # 检查是否包含大量非文本字节
        non_text = sum(1 for b in head if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F)
        return (non_text / max(len(head), 1)) < 0.05
    except Exception:
        return False


# --- 尝试作为 JSON 解析 ---
def _try_parse_json(file_path: str) -> Optional[pd.DataFrame]:
    """
    尝试将文件内容解析为 JSON 格式的 DataFrame。

    支持两种常见的水质数据 JSON 结构：
    - 对象数组：[{...}, {...}]
    - 嵌套对象：{"data": [{...}], "records": [{...}]}
    """
    import json
    try:
        with open(file_path, "r", encoding=detect_encoding(file_path)) as f:
            content = f.read().strip()

        # 修复可能的不合法 JSON（如单引号）
        data = json.loads(content)

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # 查找常见的数据键
            for key in ["data", "records", "rows", "items", "results", "监测数据"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            # 如果只有一层键值对，也尽量转换
            if all(isinstance(v, list) for v in data.values()):
                return pd.DataFrame(data)
        return None
    except (json.JSONDecodeError, ValueError, Exception):
        return None


# --- 尝试作为 Excel 解析 ---
def _try_parse_excel(file_path: str) -> Optional[pd.DataFrame]:
    """尝试将文件作为 Excel 读取。"""
    try:
        # 先尝试 openpyxl（.xlsx）
        df = pd.read_excel(file_path, engine="openpyxl")
        if not df.empty:
            return df
    except Exception:
        pass
    try:
        # 再尝试 xlrd（旧 .xls）
        df = pd.read_excel(file_path, engine="xlrd")
        if not df.empty:
            return df
    except Exception:
        pass
    return None


# --- 尝试作为分隔文本解析 ---
def _try_parse_text(file_path: str) -> Optional[pd.DataFrame]:
    """尝试将文件作为分隔文本读取。"""
    encoding = detect_encoding(file_path)
    delim, delim_name = detect_delimiter(file_path, encoding)

    try:
        # 先尝试第一行为表头
        df = pd.read_csv(file_path, sep=delim, encoding=encoding, nrows=500)
        if len(df.columns) >= 2:
            print(f"  [信息] 检测到分隔符: {delim_name}，编码: {encoding.upper()}")
            return df
    except Exception:
        pass

    # 如果第一行解析出的列数太少，尝试无表头模式
    try:
        df = pd.read_csv(file_path, sep=delim, encoding=encoding, header=None, nrows=500)
        # 尝试推断第一行是否为表头：比较第一行和其他行的数据类型
        if len(df.columns) >= 2:
            print(f"  [信息] 检测到分隔符: {delim_name}，编码: {encoding.upper()}（无表头模式）")
            return df
    except Exception:
        pass

    return None


# ============================================================================
# 万能智能导入
# ============================================================================

def smart_import(file_path: str, lake_name: str = "太湖") -> pd.DataFrame:
    """
    万能智能导入：自动识别文件格式、编码、分隔符，完成列名映射和单位转换。

    支持格式（不依赖扩展名，由内容判断）：
    - CSV / TSV / TXT / DAT —— 逗号、制表符、分号分隔的文本
    - Excel —— .xlsx / .xls
    - JSON —— 对象数组或嵌套结构

    自动处理：
    - 编码检测（UTF-8、GBK、GB2312 等）
    - 分隔符检测
    - 列名智能映射（100+ 种常见水质列名）
    - 多段 Excel 自动拆分（同一 Sheet 内含多个月份数据）
    - 低于检出限（＜）自动解析（MDL/2 替代法）
    - 科学计数法文本解析（如"1.09×103" → 1090.0）
    - 度分秒坐标自动转换为十进制度
    - 经纬度颠倒自动纠正（经度/纬度互检）
    - 经纬度缺失沿用 5 月基准自动填补
    - 坐标纠偏（点位坐标漂移超阈值时用基准坐标覆盖）
    - 点位→湖泊自动归属（POINT_LAKE_MAP）
    - 叶绿素单位转换（mg/L → μg/L）
    - 缺失字段补全
    - 监测时段自动推断

    参数
    ----
    file_path : str
        数据文件路径，支持任意扩展名（由内容自动识别格式）。
    lake_name : str
        默认湖泊名称（仅当所有点位都无法匹配 POINT_LAKE_MAP 时使用）。

    返回
    ----
    pd.DataFrame
        标准化数据集，可直接用于平台所有分析模块。
    """
    import re

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件：{file_path}")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError("文件为空，无法导入。")
    print(f"  [信息] 文件大小：{file_size / 1024:.1f} KB")

    raw_df: Optional[pd.DataFrame] = None
    detected_format: str = "未知"

    # --- 策略1：优先按扩展名尝试（快速路径）---
    file_lower = file_path.lower()
    if file_lower.endswith((".xlsx", ".xls")):
        raw_df = _try_parse_excel(file_path)
        if raw_df is not None:
            detected_format = f"Excel ({os.path.splitext(file_path)[1]})"

    # --- 策略2：按内容特征依次尝试 ---
    if raw_df is None and _is_text_file(file_path):
        json_df = _try_parse_json(file_path)
        if json_df is not None:
            raw_df = json_df
            detected_format = "JSON"
        else:
            text_df = _try_parse_text(file_path)
            if text_df is not None:
                raw_df = text_df
                ext = os.path.splitext(file_path)[1].lower() or "文本"
                detected_format = f"分隔文本 ({ext})"

    # --- 策略3：最后兜底 ---
    if raw_df is None:
        raw_df = _try_parse_excel(file_path)
        if raw_df is not None:
            detected_format = "Excel (内容检测)"
        else:
            raise ValueError(
                f"无法识别文件格式。请确认文件内容为以下格式之一：\n"
                f"  • 逗号/制表符分隔的文本（CSV、TSV、TXT、DAT）\n"
                f"  • Excel 工作簿（.xlsx、.xls）\n"
                f"  • JSON 数据文件"
            )

    if raw_df is None or raw_df.empty:
        raise ValueError("文件解析成功但未发现数据。")

    print(f"  [信息] 识别格式: {detected_format}，共 {len(raw_df)} 行 × {len(raw_df.columns)} 列")

    # =========================================================================
    # 转置格式检测（参数在行、采样点在列的特殊格式）
    # =========================================================================
    # 部分 Excel 文件（如长荡湖-玄武湖现场理化参数）采用转置格式：
    # 首列是参数名，列标题是湖泊/点位名。需要先转换为标准格式。

    is_transposed = _detect_transposed_format(raw_df)
    if is_transposed:
        raw_df = _transform_transposed(raw_df, file_path)
        detected_format += "（转置→标准）"
        print(f"  [信息] 转置转换后：共 {len(raw_df)} 行 × {len(raw_df.columns)} 列")

    # =========================================================================
    # 多段 Excel 检测与拆分
    # =========================================================================
    # 检测是否为"多段式"数据：同一 Sheet 内包含多个独立表格，
    # 每个表格有自己的表头行（如"点位（5月）"和"点位（6月）"），
    # 中间以空行分隔。
    #
    # 采用双重检测策略（更鲁棒）：
    #   A. 扫描数据行首列，寻找"点位（X月）"样式的段表头
    #   B. 检测 "Unnamed: 0" 列中的月份标签（如"5月"、"6月"）
    #   C. 以空行为辅助边界标记

    sections = []

    # 检查是否第一个列为 "Unnamed: 0" 且包含月份标签
    col0_name = str(raw_df.columns[0])
    has_unnamed_month = (
        col0_name.startswith("Unnamed") or col0_name == "" or col0_name == "nan"
    ) and any(
        bool(re.search(r'^\d+月$', str(v).strip()))
        for v in raw_df.iloc[:, 0] if pd.notna(v)
    )

    # 策略A：扫描所有行，寻找段表头标记
    # 先检查 raw_df.columns[0] 是否本身就是一个段表头（如"点位（5月）"）
    header_row_indices = []
    first_col = str(raw_df.columns[0])
    if re.search(r'点位[（(]\d+月[）)]', first_col):
        # 第一段的表头在 columns 中，段索引记为 0
        header_row_indices.append(0)

    # 再扫描数据行中是否还有段表头（如后续月份的"点位（6月）"）
    for i in range(len(raw_df)):
        first_cell = str(raw_df.iloc[i, 0])
        if re.search(r'点位[（(]\d+月[）)]', first_cell):
            header_row_indices.append(i)

    # 策略B：检测 "Unnamed: 0" 列中的月份标签作为段边界
    month_boundary_rows = []
    if has_unnamed_month:
        for i in range(len(raw_df)):
            first_cell = str(raw_df.iloc[i, 0]).strip()
            if re.search(r'^\d+月$', first_cell):
                month_boundary_rows.append((i, first_cell))

    # 策略C：辅助用空行（几乎全空的行）
    empty_row_indices = []
    for i in range(len(raw_df)):
        non_null_count = raw_df.iloc[i].notna().sum()
        if non_null_count <= 1:
            empty_row_indices.append(i)

    is_multi_section = len(header_row_indices) >= 1 or len(month_boundary_rows) >= 1

    if is_multi_section:
        # --- 情况1：传统的 "点位（X月）" 段表头 ---
        if len(header_row_indices) >= 1:
            n_sections = len(header_row_indices)

            print(f"  [信息] 检测到多段式数据结构：发现 {n_sections} 个段表头标记。")

            # 构建段边界：每个段从 header_row 开始，到下一个 header_row 或空行或末尾结束
            for hi, hdr_idx in enumerate(header_row_indices):
                # 确定该段的结束位置
                end_row = len(raw_df)
                # 下一个段表头
                if hi + 1 < len(header_row_indices):
                    end_row = min(end_row, header_row_indices[hi + 1])
                # 下一个空行（但至少要有1行数据）
                for ei in empty_row_indices:
                    if ei > hdr_idx and ei < end_row:
                        end_row = ei
                        break

                if hdr_idx == 0:
                    # 第一个段：表头已经在 raw_df.columns 中
                    header_values = [str(c).strip() for c in raw_df.columns]
                    section_data = raw_df.iloc[0:end_row].copy()
                else:
                    # 后续段：表头是该行的值
                    header_row = raw_df.iloc[hdr_idx]
                    header_values = [str(v).strip() if pd.notna(v) else "" for v in header_row]
                    section_data = raw_df.iloc[hdr_idx + 1:end_row].copy()

                if section_data.shape[0] < 1:
                    continue

                # 清理列名
                header_clean = []
                for h in header_values:
                    h = h.strip()
                    if h.startswith("Unnamed") or h == "nan" or h == "None":
                        h = ""
                    header_clean.append(h)

                section_data.columns = header_clean

                # 推断月份
                month_label = None
                header_text = " ".join(header_values)
                month_match = re.search(r'(\d+)\s*月', header_text)
                if month_match:
                    month_label = f"{int(month_match.group(1))}月"

                print(f"  [信息]   第 {len(sections) + 1} 段：{len(section_data)} 行数据，"
                      f"推断月份：{month_label or '未知'}")

                sections.append((section_data, month_label))

        # --- 情况2： "Unnamed: 0" 列中的月份标签作为段边界 ---
        elif len(month_boundary_rows) >= 1:
            print(f"  [信息] 检测到月份标签式分段数据：发现 {len(month_boundary_rows)} 个月份标记。")

            for mi, (boundary_row, month_str) in enumerate(month_boundary_rows):
                # 确定该段的结束位置
                end_row = len(raw_df)
                if mi + 1 < len(month_boundary_rows):
                    end_row = month_boundary_rows[mi + 1][0]
                # 下一个空行
                for ei in empty_row_indices:
                    if ei > boundary_row and ei < end_row:
                        end_row = ei
                        break

                # 表头使用原始 columns
                # 月份标签在 "Unnamed: 0" 列中，该行的其他列可能是数据
                # 边界行的数据从第 1 列开始（跳过 Unnamed: 0 列）
                data_start = boundary_row  # 包含边界行，因为其他列可能是数据
                section_data = raw_df.iloc[data_start:end_row].copy()
                section_data.columns = [str(c).strip() for c in raw_df.columns]

                # 清理列名
                header_clean = []
                for h in section_data.columns:
                    h = h.strip()
                    if h.startswith("Unnamed") or h == "nan" or h == "None":
                        h = ""
                    header_clean.append(h)
                section_data.columns = header_clean

                # 如果边界行第 0 列是月份标签，清除该行该列的月份值（避免干扰数据）
                if boundary_row < len(section_data):
                    first_col_name = str(raw_df.columns[0])
                    if first_col_name.startswith("Unnamed") or first_col_name == "":
                        if "" in section_data.columns:
                            section_data.loc[section_data.index[0], ""] = None

                month_label = month_str

                if section_data.shape[0] >= 1:
                    print(f"  [信息]   第 {len(sections) + 1} 段：{len(section_data)} 行数据，"
                          f"推断月份：{month_label}")
                    sections.append((section_data, month_label))
    else:
        # 单段数据，直接使用
        sections.append((raw_df, None))

    # =========================================================================
    # 逐段处理：列名映射 + 数据清洗
    # =========================================================================

    all_results = []

    for sec_idx, (section_df, month_label) in enumerate(sections):
        sec_label = f"第{sec_idx + 1}段" if len(sections) > 1 else ""

        # --- 列名映射 ---
        mapped: dict = {}
        unmapped: list = []
        unmapped_std: list = []

        for col in section_df.columns:
            col_stripped = str(col).strip()
            if col_stripped in COLUMN_NAME_MAP:
                target = COLUMN_NAME_MAP[col_stripped]
                if target not in mapped:
                    mapped[target] = col
            else:
                unmapped.append(col)
                unmapped_std.append(col_stripped)

        if unmapped and sec_idx == 0:
            print(f"  [信息] {len(unmapped)} 个列名未在映射表中，已保留原列名。")
            if len(unmapped) <= 8:
                print(f"         未映射列: {unmapped_std}")

        # 构建标准化 DataFrame
        result = pd.DataFrame()
        for std_name, src_col in mapped.items():
            result[std_name] = section_df[src_col]

        # 保留未映射的列
        for col in unmapped:
            result[str(col).strip()] = section_df[col]

        # --- 过滤全空行 ---
        # 剔除非关键字段全部为空的行（如纯分隔行）
        key_check_cols = [c for c in ["采样点位", "水温", "pH", "DO", "TN", "TP",
                                        "CODMn", "叶绿素a", "GSM", "2-MIB"]
                          if c in result.columns]
        if key_check_cols:
            result = result.dropna(subset=key_check_cols, how="all")

        if result.empty:
            continue

        # --- 数值单元格统一解析：低于检出限（＜）、科学计数法、普通数值 ---
        # 仅对数值型指标列进行解析，跳过文本列（采样点位、湖泊名称等）
        _text_cols = {"采样点位", "湖泊名称", "监测时段", "采样日期", "经度", "纬度"}
        _parse_cols = [c for c in result.columns if c not in _text_cols]
        for col in _parse_cols:
            result[col] = result[col].apply(_parse_numeric_cell)

        # --- 坐标转换：度分秒 → 十进制度 ---
        for coord_col in ["经度", "纬度"]:
            if coord_col in result.columns:
                result[coord_col] = result[coord_col].apply(_parse_dms_to_decimal)

        # --- 单位转换：叶绿素 mg/L → μg/L ---
        if "叶绿素a_raw_mgL" in result.columns:
            print(f"  [信息] {sec_label} 检测到叶绿素单位为 mg/L，已自动转换为 μg/L（×1000）。")
            result["叶绿素a_raw"] = pd.to_numeric(result["叶绿素a_raw_mgL"], errors="coerce") * 1000
            # 如果已有叶绿素a（来自其他源头），优先保留；否则用转换后的
            if "叶绿素a" not in result.columns or result["叶绿素a"].isna().all():
                result["叶绿素a"] = result["叶绿素a_raw"]
            else:
                # 用转换值填补叶绿素a的空缺
                result["叶绿素a"] = result["叶绿素a"].fillna(result["叶绿素a_raw"])
            result = result.drop(columns=["叶绿素a_raw_mgL", "叶绿素a_raw"], errors="ignore")

        # 如果有送检叶绿素且现场叶绿素为空，用送检值填补
        if "叶绿素a_送检" in result.columns:
            if "叶绿素a" in result.columns:
                result["叶绿素a"] = result["叶绿素a"].fillna(
                    pd.to_numeric(result["叶绿素a_送检"], errors="coerce")
                )
            else:
                result["叶绿素a"] = pd.to_numeric(result["叶绿素a_送检"], errors="coerce")
            result = result.drop(columns=["叶绿素a_送检"], errors="ignore")

        # --- 数据类型自动转换 ---
        numeric_candidates = [
            "水温", "pH", "DO", "浊度", "TN", "TP", "NH3-N", "CODMn",
            "叶绿素a", "GSM", "2-MIB", "藻密度", "电导率", "氧化还原电位",
            "盐度", "TSS", "PC", "fDOM", "fDOM_ppd", "DEP", "ALT",
            "DO饱和度", "DO饱和度RTB", "压力",
        ]
        for col in numeric_candidates:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        # --- Excel 日期序列号转换 ---
        if "采样日期" in result.columns:
            # 检查是否为 Excel 日期序列号（数值范围 ≈1000-100000）
            sample_vals = result["采样日期"].dropna()
            if len(sample_vals) > 0:
                try:
                    test_val = float(sample_vals.iloc[0])
                    if 1000 < test_val < 100000:
                        result["采样日期"] = result["采样日期"].apply(_excel_serial_to_date)
                        print(f"  [信息] {sec_label} 检测到 Excel 日期序列号，已自动转换。")
                except (ValueError, TypeError):
                    pass

        # --- 点位→湖泊自动归属 ---
        if "采样点位" in result.columns:
            result["湖泊名称"] = result["采样点位"].apply(resolve_lake_from_point)

        # 如果还有未识别的湖泊，用默认值
        if "湖泊名称" in result.columns:
            result.loc[result["湖泊名称"].isin(["", "未知水体"]), "湖泊名称"] = lake_name
        else:
            result["湖泊名称"] = lake_name

        # --- 补全缺失的必需字段 ---
        required_cols: list = [
            "湖泊名称", "采样点位", "监测时段", "采样日期",
            "水温", "pH", "DO", "浊度", "TN", "TP", "NH3-N", "CODMn", "叶绿素a",
            "GSM", "2-MIB",
        ]
        for col in required_cols:
            if col not in result.columns:
                result[col] = float("nan")

        # --- 自动推断监测时段 ---
        if result["监测时段"].isna().all():
            month_num = None
            # 策略1：从段标签提取月份
            if month_label:
                m = re.search(r'(\d+)', month_label)
                if m:
                    month_num = int(m.group(1))
            # 策略2：从文件名提取月份
            if month_num is None:
                for kw, mn in [("5月", 5), ("6月", 6), ("7月", 7), ("8月", 8),
                                ("9月", 9), ("10月", 10), ("11月", 11),
                                ("3月", 3), ("4月", 4)]:
                    if kw in file_path:
                        month_num = mn
                        break
            # 策略3：用当前月份
            if month_num is None:
                month_num = datetime.datetime.now().month

            if 3 <= month_num <= 5:
                period = "平水期（3-5月）"
            elif 6 <= month_num <= 8:
                period = "藻类生长期（6-8月）"
            else:
                period = "爆发期（9-11月）"
            result["监测时段"] = period

        # --- 自动添加采样日期 ---
        if result["采样日期"].isna().all():
            if month_label:
                m = re.search(r'(\d+)', month_label)
                if m:
                    default_date = f"2026-{int(m.group(1)):02d}-15"
                else:
                    default_date = "2026-05-15"
            else:
                default_date = "2026-05-15"
            result["采样日期"] = default_date

        all_results.append(result)

    # =========================================================================
    # 合并所有段的结果
    # =========================================================================

    if not all_results:
        raise ValueError("所有数据段处理后均为空，请检查数据内容。")

    result = pd.concat(all_results, ignore_index=True)

    # --- 重新排序列 ---
    preferred_order = [
        "湖泊名称", "采样点位", "监测时段", "采样日期",
        "水温", "pH", "DO", "浊度",
        "TN", "TP", "NH3-N", "CODMn",
        "叶绿素a", "藻密度", "电导率", "氧化还原电位",
        "GSM", "2-MIB",
        "经度", "纬度",
    ]
    final_cols = [c for c in preferred_order if c in result.columns]
    remaining = [c for c in result.columns if c not in final_cols]
    result = result[final_cols + remaining]

    # --- 经纬度互检：纠正经度/纬度颠倒 ---
    result = _correct_swapped_coordinates(result)

    # --- 经纬度缺失填补：6/7 月沿用该点位 5 月基准经纬度 ---
    result = fill_coordinates_from_baseline(result)

    # --- 参考表坐标纠偏：纠正点位坐标填错（漂移超阈值）的情况 ---
    result = _snap_coordinates_to_reference(result)

    # =========================================================================
    # 汇总报告
    # =========================================================================

    print(f"  [信息] 智能导入完成：{len(result)} 条记录，{len(result.columns)} 个字段。")
    print(f"         识别格式：{detected_format}"
          f"{'（多段合并）' if is_multi_section else ''}")

    # 湖泊分布
    lake_counts = result["湖泊名称"].value_counts()
    lake_summary = "、".join([f"{lk}({cnt})" for lk, cnt in lake_counts.items()])
    print(f"         水体分布：{lake_summary}")

    # 监测时段分布
    period_counts = result["监测时段"].value_counts()
    period_summary = "、".join([f"{p}({c})" for p, c in period_counts.items()])
    print(f"         监测时段：{period_summary}")

    # 嗅味数据检测
    has_odor = not (result["GSM"].isna().all() and result["2-MIB"].isna().all())
    if not has_odor:
        print("  [提示] 未检测到 GSM/2-MIB 数据，嗅味风险预警模块将不可用。")
        print("         其他分析（可视化、相关性、回归）可正常运行。")
    else:
        gsm_count = result["GSM"].notna().sum()
        mib_count = result["2-MIB"].notna().sum()
        print(f"  [信息] 已检测到嗅味物质数据：GSM {gsm_count} 条、2-MIB {mib_count} 条。"
              f"全部分析模块可用。")

    return result


def _extract_month_from_date(value) -> Optional[int]:
    """
    从采样日期中提取月份（1-12）。

    支持字符串（如"2026-06-15"、"2026年6月"）与 datetime/date 对象。

    参数
    ----
    value : any
        采样日期值。

    返回
    ----
    Optional[int]
        提取到的月份，无法提取则返回 None。
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.month

    import re
    s = str(value).strip()
    # 优先匹配 "YYYY-MM-DD" / "YYYY/MM" / "YYYY年M月"
    m = re.search(r'(\d{4})[-/年](\d{1,2})', s)
    if m:
        month = int(m.group(2))
        if 1 <= month <= 12:
            return month
    # 兜底：直接匹配 "M月"
    m = re.search(r'(\d{1,2})\s*月', s)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return month
    return None


def fill_coordinates_from_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    经纬度缺失填补：6月/7月某点位经纬度缺失时，沿用该点位 5 月基准经纬度。

    逻辑：
    1. 从"采样日期"提取月份；
    2. 对每个采样点位，取 5 月的首个非空经纬度作为基准；
    3. 用该基准填补同一采样点位在 6月/7月（或任何非 5 月）缺失的经纬度。

    参数
    ----
    df : pd.DataFrame
        标准化数据集，需含"采样点位"、"采样日期"及"经度"/"纬度"列。

    返回
    ----
    pd.DataFrame
        填补后的数据框（原地返回副本，不修改入参）。
    """
    validate_dataframe(df)

    if "采样点位" not in df.columns or "采样日期" not in df.columns:
        return df
    coord_cols = [c for c in ["经度", "纬度"] if c in df.columns]
    if not coord_cols:
        return df

    result = df.copy()

    # --- 提取各行的月份 ---
    month_series = result["采样日期"].apply(_extract_month_from_date)

    # --- 构建 5 月基准坐标：点位 → {经度/纬度: 首个非空值} ---
    baseline: dict = {}
    may_mask = month_series == 5
    for point, grp in result[may_mask].groupby("采样点位"):
        for cc in coord_cols:
            non_null = grp[cc].dropna()
            if not non_null.empty:
                baseline.setdefault(point, {})[cc] = non_null.iloc[0]

    if not baseline:
        return result

    # --- 对非 5 月且坐标缺失的行进行填补 ---
    fill_mask = (month_series != 5) & (~month_series.isna())
    for idx in result.index[fill_mask]:
        point = result.loc[idx, "采样点位"]
        if point not in baseline:
            continue
        for cc in coord_cols:
            if cc in baseline[point] and pd.isna(result.loc[idx, cc]):
                result.loc[idx, cc] = baseline[point][cc]

    return result


# 中国境内经纬度合理范围（用于经纬度互检纠正）
_COORD_LON_RANGE: Tuple[float, float] = (73.0, 135.0)
_COORD_LAT_RANGE: Tuple[float, float] = (18.0, 54.0)


def _correct_swapped_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    经纬度互检：检测并纠正「经度/纬度」两列被颠倒的情况。

    部分源数据将纬度值误填到经度列（反之亦然）。判定规则：
    当某行的「经度」落在纬度合理区间（18~54）且「纬度」落在经度合理区间（73~135）
    时，判定为颠倒并交换两列。

    参数
    ----
    df : pd.DataFrame
        标准化数据集，需含"经度"/"纬度"列。

    返回
    ----
    pd.DataFrame
        纠正后的数据框（返回副本，不修改入参）。
    """
    if "经度" not in df.columns or "纬度" not in df.columns:
        return df

    result = df.copy()
    lon = pd.to_numeric(result["经度"], errors="coerce")
    lat = pd.to_numeric(result["纬度"], errors="coerce")

    # 颠倒条件：经度值像纬度（18~54）且纬度值像经度（73~135）
    swapped = (
        lon.between(_COORD_LAT_RANGE[0], _COORD_LAT_RANGE[1])
        & lat.between(_COORD_LON_RANGE[0], _COORD_LON_RANGE[1])
    )

    n_swapped = int(swapped.sum())
    if n_swapped > 0:
        result.loc[swapped, ["经度", "纬度"]] = (
            result.loc[swapped, ["纬度", "经度"]].values
        )
        print(f"  [信息] 经纬度互检：已纠正 {n_swapped} 条颠倒的经纬度。")

    return result


# 坐标纠偏阈值（度）：实测坐标与基准坐标偏差超过该值才覆盖
_COORD_SNAP_THRESHOLD: float = 0.5


def _snap_coordinates_to_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    参考表坐标纠偏：当点位命中基准坐标表（POINT_COORDS）且实测坐标与基准
    偏差超过阈值时，用基准坐标覆盖。

    用于纠正源数据中「坐标填错数值」的情况（如点位漂移到错误位置），
    仅对已知点位生效，且偏差须超过阈值，避免误改正常的实测坐标。

    参数
    ----
    df : pd.DataFrame
        标准化数据集，需含"采样点位"、"经度"/"纬度"列。

    返回
    ----
    pd.DataFrame
        纠偏后的数据框（返回副本，不修改入参）。
    """
    if "采样点位" not in df.columns or "经度" not in df.columns or "纬度" not in df.columns:
        return df

    result = df.copy()
    points = result["采样点位"].astype(str).str.strip()
    lon = pd.to_numeric(result["经度"], errors="coerce")
    lat = pd.to_numeric(result["纬度"], errors="coerce")

    for point, (ref_lat, ref_lon) in POINT_COORDS.items():
        in_point = points == point
        if not in_point.any():
            continue

        lon_off = in_point & (lon - ref_lon).abs().gt(_COORD_SNAP_THRESHOLD)
        lat_off = in_point & (lat - ref_lat).abs().gt(_COORD_SNAP_THRESHOLD)

        if lon_off.any() or lat_off.any():
            result.loc[lon_off, "经度"] = ref_lon
            result.loc[lat_off, "纬度"] = ref_lat
            n = int((lon_off | lat_off).sum())
            print(f"  [信息] 坐标纠偏：点位「{point}」纠正 {n} 条偏离基准的坐标。")

    return result


# ============================================================================
# 模块主入口（测试用）
# ============================================================================

if __name__ == "__main__":
    """
    模块自测代码：测试数据清洗和处理功能。
    """
    print("=" * 60)
    print("《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》")
    print("数据处理与清洗模块 - 自测运行")
    print("=" * 60)

    # 从模拟数据模块导入数据
    import sys
    sys.path.insert(0, ".")
    from data_mock import generate_full_mock_dataset

    print("\n[1/5] 生成模拟测试数据...")
    raw_df: pd.DataFrame = generate_full_mock_dataset(samples_per_period=5)

    print("\n[2/5] 检测异常值...")
    cleaned_df: pd.DataFrame = clean_dataset(raw_df, method="iqr", iqr_multiplier=3.0)

    print("\n[3/5] 填补缺失值...")
    filled_df: pd.DataFrame = fill_missing_values(cleaned_df, strategy="mean")

    print("\n[4/5] 按湖泊筛选...")
    filtered_df: pd.DataFrame = filter_by_lake(filled_df, ["千岛湖", "太湖"])

    print("\n[5/5] 按监测时段分组聚合...")
    agg_result: pd.DataFrame = aggregate_by_group(
        filtered_df,
        group_cols=["湖泊名称", "监测时段"],
        agg_func="mean"
    )
    print("\n聚合结果：")
    print(agg_result.head(10).to_string())

    print("\n" + "=" * 60)
    print("自测完成！数据处理与清洗模块所有函数均正常运行。")
    print("=" * 60)
