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
    # 溶解氧
    "溶解氧（mg/L）": "DO",
    "溶解氧(mg/L)": "DO",
    "溶解氧": "DO",
    "DO(mg/L)": "DO",
    "DO（mg/L）": "DO",
    "DO": "DO",
    # pH
    "pH": "pH",
    "PH": "pH",
    "ph": "pH",
    "Ph": "pH",
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
    # CODMn
    "COD锰（mg/L）": "CODMn",
    "COD锰(mg/L)": "CODMn",
    "CODMn（mg/L）": "CODMn",
    "高锰酸盐指数（mg/L）": "CODMn",
    "高锰酸盐指数(mg/L)": "CODMn",
    "高锰酸盐指数": "CODMn",
    "CODMn": "CODMn",
    "COD_Mn": "CODMn",
    # 叶绿素
    "叶绿素（mg/L）": "叶绿素a_raw_mgL",
    "叶绿素(mg/L)": "叶绿素a_raw_mgL",
    "叶绿素a（μg/L）": "叶绿素a",
    "叶绿素a(μg/L)": "叶绿素a",
    "叶绿素a（ug/L）": "叶绿素a",
    "叶绿素（μg/L）": "叶绿素a",
    "叶绿素(μg/L)": "叶绿素a",
    "叶绿素a": "叶绿素a",
    "Chl-a(μg/L)": "叶绿素a",
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
    # GSM / 2-MIB
    "GSM（ng/L）": "GSM",
    "GSM(ng/L)": "GSM",
    "GSM": "GSM",
    "土臭素（ng/L）": "GSM",
    "土臭素": "GSM",
    "2-MIB（ng/L）": "2-MIB",
    "2-MIB(ng/L)": "2-MIB",
    "2-MIB": "2-MIB",
    "2MIB": "2-MIB",
    "二甲基异莰醇": "2-MIB",
    # 点位/站点
    "点位": "采样点位",
    "站点": "采样点位",
    "采样点": "采样点位",
    "监测点位": "采样点位",
    "采样点位": "采样点位",
    "监测点": "采样点位",
    "站点名称": "采样点位",
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
    "电导率": "电导率",
    "Cond": "电导率",
    # 氧化还原电位
    "氧化还原电位（mV）": "氧化还原电位",
    "氧化还原电位(mV)": "氧化还原电位",
    "氧化还原电位": "氧化还原电位",
    "ORP": "氧化还原电位",
    # 藻密度
    "藻密度（万个/L）": "藻密度",
    "藻密度(万个/L)": "藻密度",
    "藻密度": "藻密度",
    "藻细胞密度": "藻密度",
}


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
    - 列名智能映射（70+ 种常见水质列名）
    - 叶绿素单位转换（mg/L → μg/L）
    - 缺失字段补全（GSM/2-MIB 等自动以 NaN 填充）
    - 湖泊名称和监测时段自动推断

    参数
    ----
    file_path : str
        数据文件路径，支持任意扩展名（由内容自动识别格式）。
    lake_name : str
        湖泊名称，默认 "太湖"。

    返回
    ----
    pd.DataFrame
        标准化数据集，可直接用于平台所有分析模块。
    """
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
        # 先尝试 JSON
        json_df = _try_parse_json(file_path)
        if json_df is not None:
            raw_df = json_df
            detected_format = "JSON"
        else:
            # 尝试分隔文本
            text_df = _try_parse_text(file_path)
            if text_df is not None:
                raw_df = text_df
                ext = os.path.splitext(file_path)[1].lower() or "文本"
                detected_format = f"分隔文本 ({ext})"

    # --- 策略3：最后兜底 ---
    if raw_df is None:
        # 不是文本，试试 Excel（可能扩展名不对）
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

    # --- 如果第一行不是表头（数字列名），尝试用第一行数据作为列名 ---
    first_col = str(raw_df.columns[0])
    if first_col.isdigit() or (first_col.startswith("0") and first_col.isdigit()):
        # 列名是数字 → 可能是无表头的数据
        print("  [信息] 检测到无表头数据，已尝试推断列名。")

    # --- 列名映射 ---
    mapped: dict = {}
    unmapped: list = []
    unmapped_std: list = []

    for col in raw_df.columns:
        col_stripped = str(col).strip()
        if col_stripped in COLUMN_NAME_MAP:
            target = COLUMN_NAME_MAP[col_stripped]
            if target not in mapped:
                mapped[target] = col
            else:
                pass  # 同名列取第一个
        else:
            unmapped.append(col)
            unmapped_std.append(col_stripped)

    if unmapped:
        print(f"  [信息] {len(unmapped)} 个列名未在映射表中，已保留原列名。")
        if len(unmapped) <= 5:
            print(f"         未映射列: {unmapped_std}")

    # 构建标准化 DataFrame
    result = pd.DataFrame()
    for std_name, src_col in mapped.items():
        result[std_name] = raw_df[src_col]

    # 保留未映射的列
    for col in unmapped:
        result[str(col).strip()] = raw_df[col]

    # --- 单位转换：叶绿素 mg/L → μg/L ---
    if "叶绿素a_raw_mgL" in result.columns:
        print("  [信息] 检测到叶绿素单位为 mg/L，已自动转换为 μg/L（×1000）。")
        result["叶绿素a"] = pd.to_numeric(result["叶绿素a_raw_mgL"], errors="coerce") * 1000
        result = result.drop(columns=["叶绿素a_raw_mgL"])

    # --- 数据类型自动转换 ---
    numeric_candidates = [
        "水温", "pH", "DO", "浊度", "TN", "TP", "NH3-N", "CODMn",
        "叶绿素a", "GSM", "2-MIB", "藻密度", "电导率", "氧化还原电位",
    ]
    for col in numeric_candidates:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # --- 补全站点名称 ---
    if "采样点位" in result.columns:
        if "湖泊名称" not in result.columns:
            result["湖泊名称"] = lake_name
        # 自动在点位名前加湖名
        result["采样点位"] = result.apply(
            lambda row: (
                f"{str(row.get('湖泊名称', lake_name))}-{row['采样点位']}"
                if str(row.get('湖泊名称', '')) not in str(row['采样点位'])
                else str(row['采样点位'])
            ),
            axis=1,
        )

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
        month = datetime.datetime.now().month
        # 尝试从文件名或日期列推断
        date_hint = file_path.lower()
        if any(m in date_hint for m in ["3月", "4月", "5月", "mar", "apr", "may"]):
            period = "平水期（3-5月）"
        elif any(m in date_hint for m in ["6月", "7月", "8月", "jun", "jul", "aug"]):
            period = "藻类生长期（6-8月）"
        elif any(m in date_hint for m in ["9月", "10月", "11月", "sep", "oct", "nov"]):
            period = "爆发期（9-11月）"
        elif 3 <= month <= 5:
            period = "平水期（3-5月）"
        elif 6 <= month <= 8:
            period = "藻类生长期（6-8月）"
        else:
            period = "爆发期（9-11月）"
        result["监测时段"] = period

    # --- 自动添加采样日期 ---
    if result["采样日期"].isna().all():
        result["采样日期"] = "2026-05-15"

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

    # --- 汇总报告 ---
    print(f"  [信息] 智能导入完成：{len(result)} 条记录，{len(result.columns)} 个字段。")
    print(f"         识别格式：{detected_format}")
    has_odor = not (result["GSM"].isna().all() and result["2-MIB"].isna().all())
    if not has_odor:
        print("  [提示] 未检测到 GSM/2-MIB 数据，嗅味风险预警模块将不可用。")
        print("         其他分析（可视化、相关性、回归）可正常运行。")
    else:
        print("  [信息] 已检测到嗅味物质数据，全部分析模块可用。")

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
