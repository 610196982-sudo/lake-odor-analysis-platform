# -*- coding: utf-8 -*-
"""
==============================================================================
模块名称：model.py
所属系统：《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》
功能描述：统计分析与风险预警模型模块
          —— 提供皮尔逊/斯皮尔曼相关性分析、多元线性回归建模、
             随机森林特征重要性排序、嗅味物质超标风险预测、
             方差分析（ANOVA）及模型评估等功能。
==============================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Any
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 全局常量与阈值定义
# ============================================================================

# 嗅味物质超标阈值（ng/L）
# 参考《生活饮用水卫生标准》（GB 5749-2022）及文献报道的感官阈值
ODORANT_THRESHOLDS: dict = {
    "GSM": 10.0,     # 土臭素感官阈值约 4~10 ng/L
    "2-MIB": 10.0,   # 2-甲基异莰醇感官阈值约 5~10 ng/L
}

# 叶绿素a 富营养化阈值（μg/L）
# 参考 OECD 湖泊营养分级标准
CHLA_TROPHIC_THRESHOLDS: dict = {
    "贫营养": 2.5,
    "中营养": 8.0,
    "富营养": 25.0,
    "超富营养": 75.0,
}

# 嗅味风险等级定义
RISK_LEVELS: dict = {
    "🟢 低风险": {
        "描述": "当前水体嗅味物质浓度远低于感官阈值，饮用水口感和气味正常。",
        "建议": "维持常规监测频次，关注水温及营养盐变化趋势。",
        "GSM上限": 4.0,
        "2-MIB上限": 4.0,
    },
    "🟡 中等风险": {
        "描述": "嗅味物质浓度接近感官阈值，部分敏感人群可能察觉异嗅异味。",
        "建议": "建议加密监测频次，排查产嗅藻类可能的增殖区域。",
        "GSM上限": 10.0,
        "2-MIB上限": 10.0,
    },
    "🟠 高风险": {
        "描述": "嗅味物质浓度已超过感官阈值，多数用户可察觉明显异嗅异味。",
        "建议": "建议启动水厂应急监测，考虑投加粉末活性炭（PAC）或高锰酸钾预氧化。",
        "GSM上限": 20.0,
        "2-MIB上限": 20.0,
    },
    "🔴 严重风险": {
        "描述": "嗅味物质浓度严重超标，水体散发强烈霉味或土腥味，威胁供水安全。",
        "建议": "必须立即启动深度处理工艺（臭氧-活性炭联用），通知供水部门和监管机构。",
        "GSM上限": float("inf"),
        "2-MIB上限": float("inf"),
    },
}

# 风险判定阈值列表（按浓度升序）
RISK_THRESHOLDS: list = [
    (4.0,  "🟢 低风险"),
    (10.0, "🟡 中等风险"),
    (20.0, "🟠 高风险"),
    (float("inf"), "🔴 严重风险"),
]


# ============================================================================
# 参数校验辅助函数
# ============================================================================

def _validate_df(df: pd.DataFrame) -> None:
    """校验输入是否为有效且非空的 DataFrame。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"输入必须为 pandas DataFrame，当前传入类型为：{type(df)}")
    if df.empty:
        raise ValueError("输入的 DataFrame 不能为空。")


def _validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """校验指定列是否存在且为数值类型。"""
    _validate_df(df)
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 在 DataFrame 中不存在。")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"列 '{col}' 不是数值类型，当前类型为：{df[col].dtype}")


# ============================================================================
# 相关性分析
# ============================================================================

def run_correlation_analysis(
    df: pd.DataFrame,
    target_cols: List[str],
    predictor_cols: Optional[List[str]] = None,
    method: str = "pearson",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    对目标变量与预测变量执行相关性分析，输出相关系数及显著性检验结果。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    target_cols : List[str]
        目标变量列名列表，如 ['GSM', '2-MIB']。
    predictor_cols : Optional[List[str]]
        预测变量列名列表。若为 None，则自动使用除目标外的所有数值列。
    method : str
        相关系数类型：'pearson'、'spearman' 或 'kendall'。
    alpha : float
        显著性水平阈值，默认 0.05。

    返回
    ----
    pd.DataFrame
        包含相关系数、p 值、显著性标记和样本量的结果表。
    """
    # --- 参数校验 ---
    _validate_df(df)
    valid_methods: set = {"pearson", "spearman", "kendall"}
    if method not in valid_methods:
        raise ValueError(f"method 必须为 {valid_methods} 之一，当前传入：{method}")
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha 应在 (0, 1) 之间，当前传入：{alpha}")

    # --- 确定预测变量 ---
    if predictor_cols is None:
        exclude_set = set(target_cols)
        predictor_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_set
        ]

    _validate_numeric_columns(df, target_cols + predictor_cols)

    # --- 执行相关性分析 ---
    results: list = []
    for target in target_cols:
        for predictor in predictor_cols:
            valid_mask = df[target].notna() & df[predictor].notna()
            x = df.loc[valid_mask, predictor].values
            y = df.loc[valid_mask, target].values

            if len(x) < 3:
                continue

            if method == "pearson":
                corr_func = pearsonr
            elif method == "spearman":
                corr_func = spearmanr
            else:
                corr_func = kendalltau

            try:
                corr_coef, p_value = corr_func(x, y)
            except Exception:
                corr_coef, p_value = np.nan, np.nan

            if pd.isna(p_value):
                significance = "—"
            elif p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < alpha:
                significance = "*"
            else:
                significance = "不显著"

            results.append({
                "目标变量": target,
                "预测变量": predictor,
                "相关系数": round(float(corr_coef), 4),
                "p值": round(float(p_value), 6) if not pd.isna(p_value) else np.nan,
                "显著性": significance,
                "样本量": int(len(x)),
            })

    result_df: pd.DataFrame = pd.DataFrame(results)
    if not result_df.empty:
        result_df["|相关系数|"] = result_df["相关系数"].abs()
        result_df = result_df.sort_values("|相关系数|", ascending=False).drop(columns=["|相关系数|"])

    return result_df.reset_index(drop=True)


# ============================================================================
# 多元线性回归模型
# ============================================================================

def build_linear_regression_model(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    构建多元线性回归模型，量化各环境因子对嗅味物质的驱动贡献。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    target_col : str
        目标变量列名。
    predictor_cols : List[str]
        预测变量列名列表。
    test_size : float
        测试集比例，默认 0.2。
    random_state : int
        随机种子。
    standardize : bool
        是否标准化，默认 True。

    返回
    ----
    Dict[str, Any]
        包含模型对象、评估指标、回归系数等信息的字典。
    """
    _validate_df(df)
    _validate_numeric_columns(df, [target_col] + predictor_cols)
    if test_size <= 0 or test_size >= 1:
        raise ValueError(f"test_size 应在 (0, 1) 之间，当前传入：{test_size}")

    all_cols = [target_col] + predictor_cols
    clean_df = df[all_cols].dropna()

    if len(clean_df) < 10:
        raise ValueError(f"有效样本量不足（当前 {len(clean_df)}），至少需要 10 条。")

    X = clean_df[predictor_cols].values
    y = clean_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler() if standardize else None
    if standardize:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    coefficients = dict(zip(predictor_cols, model.coef_))
    sorted_coefs = sorted(coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True)

    result: Dict[str, Any] = {
        "模型类型": "多元线性回归",
        "目标变量": target_col,
        "预测变量": predictor_cols,
        "截距": round(float(model.intercept_), 4),
        "回归系数": {k: round(v, 4) for k, v in coefficients.items()},
        "系数排序": [(k, round(v, 4)) for k, v in sorted_coefs],
        "训练集_R2": round(r2_score(y_train, y_train_pred), 4),
        "测试集_R2": round(r2_score(y_test, y_test_pred), 4),
        "训练集_RMSE": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
        "测试集_RMSE": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
        "训练集_MAE": round(mean_absolute_error(y_train, y_train_pred), 4),
        "测试集_MAE": round(mean_absolute_error(y_test, y_test_pred), 4),
        "训练样本量": int(len(X_train)),
        "测试样本量": int(len(X_test)),
        "模型对象": model,
        "标准化器": scaler,
    }
    return result


# ============================================================================
# 随机森林特征重要性分析
# ============================================================================

def analyze_feature_importance_rf(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    n_estimators: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    """
    使用随机森林回归模型评估各环境因子对嗅味物质的重要性。

    随机森林不要求数据正态性，能自动捕捉非线性关系。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    target_col : str
        目标变量列名。
    predictor_cols : List[str]
        预测变量列名列表。
    n_estimators : int
        决策树数量，默认 100。
    random_state : int
        随机种子。

    返回
    ----
    pd.DataFrame
        包含各特征重要性得分和排名的数据框。
    """
    _validate_df(df)
    _validate_numeric_columns(df, [target_col] + predictor_cols)
    if n_estimators < 10:
        raise ValueError(f"n_estimators 至少为 10，当前传入：{n_estimators}")

    all_cols = [target_col] + predictor_cols
    clean_df = df[all_cols].dropna()
    if len(clean_df) < 20:
        raise ValueError(f"有效样本量不足（当前 {len(clean_df)}），至少需要 20 条。")

    X = clean_df[predictor_cols].values
    y = clean_df[target_col].values

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X, y)

    importance_df = pd.DataFrame({
        "特征名称": predictor_cols,
        "重要性得分": np.round(rf_model.feature_importances_, 4),
    })
    importance_df = importance_df.sort_values("重要性得分", ascending=False).reset_index(drop=True)
    importance_df["排名"] = range(1, len(importance_df) + 1)
    importance_df["累计重要性"] = importance_df["重要性得分"].cumsum().round(4)

    return importance_df


# ============================================================================
# 嗅味物质超标风险预测
# ============================================================================

def predict_odor_risk(
    water_temp: float,
    ph: float,
    do_val: float,
    turbidity: float,
    tn: float,
    tp: float,
    nh3n: float,
    codmn: float,
    chl_a: float,
    odor_type: str = "GSM",
    model: Optional[LinearRegression] = None,
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """
    基于输入的水质理化指标预测嗅味物质浓度并评估超标风险等级。

    实现从"理化指标输入 → 浓度预测 → 风险定级 → 处理建议"的完整预警链。

    参数
    ----
    water_temp : float
        水温（℃），合理范围 0~40。
    ph : float
        pH 值，合理范围 4~10.5。
    do_val : float
        溶解氧（mg/L），合理范围 0.1~18。
    turbidity : float
        浊度（NTU），合理范围 0~200。
    tn : float
        总氮（mg/L），合理范围 0.01~10。
    tp : float
        总磷（mg/L），合理范围 0.001~2。
    nh3n : float
        氨氮（mg/L），合理范围 0.005~5。
    codmn : float
        高锰酸盐指数（mg/L），合理范围 0.5~30。
    chl_a : float
        叶绿素a（μg/L），合理范围 0.1~300。
    odor_type : str
        嗅味物质类型，'GSM' 或 '2-MIB'。
    model : Optional[LinearRegression]
        预训练的线性回归模型，为 None 时使用经验公式。
    scaler : Optional[StandardScaler]
        与模型配套的标准化器。

    返回
    ----
    Dict[str, Any]
        包含预测浓度、风险等级、风险描述、处理建议的完整字典。
    """
    # --- 参数范围校验 ---
    params_to_validate: dict = {
        "水温": (water_temp, 0.0, 40.0),
        "pH": (ph, 4.0, 10.5),
        "溶解氧": (do_val, 0.1, 18.0),
        "浊度": (turbidity, 0.0, 200.0),
        "TN": (tn, 0.01, 10.0),
        "TP": (tp, 0.001, 2.0),
        "NH3-N": (nh3n, 0.005, 5.0),
        "CODMn": (codmn, 0.5, 30.0),
        "叶绿素a": (chl_a, 0.1, 300.0),
    }

    for param_name, (value, lower, upper) in params_to_validate.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{param_name} 必须为数值类型。")
        if value < lower or value > upper:
            raise ValueError(f"{param_name} = {value}，超出合理范围 [{lower}, {upper}]。")

    if odor_type not in ["GSM", "2-MIB"]:
        raise ValueError(f"odor_type 必须为 'GSM' 或 '2-MIB'，当前传入：{odor_type}")

    # --- 预测嗅味物质浓度 ---
    if model is not None:
        features = np.array([[water_temp, ph, do_val, turbidity, tn, tp, nh3n, codmn, chl_a]])
        if scaler is not None:
            features = scaler.transform(features)
        predicted_concentration = float(model.predict(features)[0])
    else:
        # 简易经验公式估算
        # 基于文献中 GSM/2-MIB 与叶绿素a和水温的典型关系
        chl_factor = chl_a * 0.25
        temp_factor = max(0, water_temp - 15) * 0.6
        tp_factor = tp * 30
        base_prediction = chl_factor + temp_factor + tp_factor

        if odor_type == "GSM":
            predicted_concentration = base_prediction * 1.1
        else:
            predicted_concentration = base_prediction * 0.95

        # 加入 ±10% 随机噪声模拟预测不确定性
        np.random.seed(abs(hash(f"{water_temp}{ph}{chl_a}{odor_type}")) % (2**31))
        noise = np.random.normal(0, max(predicted_concentration * 0.1, 0.1))
        predicted_concentration += noise

    predicted_concentration = max(0.01, round(predicted_concentration, 2))

    # --- 判定风险等级 ---
    risk_level: str = "🔴 严重风险"
    risk_info: dict = RISK_LEVELS["🔴 严重风险"]
    for upper_bound, level_name in RISK_THRESHOLDS:
        if predicted_concentration <= upper_bound:
            risk_level = level_name
            risk_info = RISK_LEVELS[level_name]
            break

    result: Dict[str, Any] = {
        "嗅味物质类型": odor_type,
        "预测浓度": predicted_concentration,
        "浓度单位": "ng/L",
        "感官阈值": ODORANT_THRESHOLDS.get(odor_type, 10.0),
        "风险等级": risk_level,
        "风险描述": risk_info["描述"],
        "处理建议": risk_info["建议"],
        "输入参数": {
            "水温": water_temp, "pH": ph, "溶解氧": do_val,
            "浊度": turbidity, "TN": tn, "TP": tp,
            "NH3-N": nh3n, "CODMn": codmn, "叶绿素a": chl_a,
        },
        "预测模型": "经验公式估算" if model is None else "多元线性回归模型",
    }
    return result


# ============================================================================
# 方差分析（ANOVA）
# ============================================================================

def run_anova_analysis(
    df: pd.DataFrame,
    value_col: str,
    group_col: str
) -> Dict[str, Any]:
    """
    执行单因素方差分析（ANOVA），检验不同分组间某指标均值是否存在显著差异。

    同时提供 Kruskal-Wallis 非参数检验作为补充
    （不要求数据满足正态性和方差齐性假设）。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    value_col : str
        待检验的数值变量列名。
    group_col : str
        分组变量列名。

    返回
    ----
    Dict[str, Any]
        包含 F 统计量、p 值、各组描述统计等信息。
    """
    _validate_df(df)
    if value_col not in df.columns:
        raise ValueError(f"列 '{value_col}' 在数据中不存在。")
    if group_col not in df.columns:
        raise ValueError(f"列 '{group_col}' 在数据中不存在。")

    groups = df[group_col].unique()
    if len(groups) < 2:
        raise ValueError(f"分组变量至少需要 2 个不同水平，当前只有 {len(groups)} 个。")

    group_data_list: list = []
    group_stats: dict = {}
    for grp_name in groups:
        grp_values = df.loc[df[group_col] == grp_name, value_col].dropna().values
        if len(grp_values) < 3:
            raise ValueError(f"分组 '{grp_name}' 有效样本量不足（{len(grp_values)} < 3）。")
        group_data_list.append(grp_values)
        group_stats[str(grp_name)] = {
            "样本量": int(len(grp_values)),
            "均值": round(float(np.mean(grp_values)), 3),
            "标准差": round(float(np.std(grp_values, ddof=1)), 3),
            "最小值": round(float(np.min(grp_values)), 3),
            "最大值": round(float(np.max(grp_values)), 3),
        }

    f_stat, anova_p = f_oneway(*group_data_list)
    h_stat, kw_p = kruskal(*group_data_list)

    def _sig_label(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return "不显著"

    result: Dict[str, Any] = {
        "检验变量": value_col,
        "分组变量": group_col,
        "分组水平": list(group_stats.keys()),
        "ANOVA_F统计量": round(float(f_stat), 4),
        "ANOVA_p值": round(float(anova_p), 6),
        "ANOVA显著性": _sig_label(anova_p),
        "KruskalWallis_H统计量": round(float(h_stat), 4),
        "KruskalWallis_p值": round(float(kw_p), 6),
        "KruskalWallis显著性": _sig_label(kw_p),
        "分组描述统计": group_stats,
    }
    return result


# ============================================================================
# 模型交叉验证
# ============================================================================

def evaluate_model_cv(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    执行 K 折交叉验证，评估线性回归模型的稳健性和泛化能力。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    target_col : str
        目标变量列名。
    predictor_cols : List[str]
        预测变量列名列表。
    cv_folds : int
        交叉验证折数，默认 5。
    random_state : int
        随机种子。

    返回
    ----
    Dict[str, Any]
        包含各折 R² 得分、平均 R² 和标准差等信息的字典。
    """
    _validate_df(df)
    _validate_numeric_columns(df, [target_col] + predictor_cols)

    max_folds = min(20, len(df) // 3)
    if cv_folds < 2 or cv_folds > max_folds:
        raise ValueError(f"cv_folds 应在 [2, {max_folds}] 之间，当前传入：{cv_folds}")

    all_cols = [target_col] + predictor_cols
    clean_df = df[all_cols].dropna()
    if len(clean_df) < cv_folds * 3:
        raise ValueError(f"有效样本量不足（{len(clean_df)}）以进行 {cv_folds} 折交叉验证。")

    X = clean_df[predictor_cols].values
    y = clean_df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    r2_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="r2")
    neg_mse_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse_scores)

    result: Dict[str, Any] = {
        "交叉验证折数": cv_folds,
        "各折R2得分": [round(float(s), 4) for s in r2_scores],
        "平均R2": round(float(r2_scores.mean()), 4),
        "R2标准差": round(float(r2_scores.std(ddof=1)), 4),
        "各折RMSE": [round(float(s), 4) for s in rmse_scores],
        "平均RMSE": round(float(rmse_scores.mean()), 4),
        "RMSE标准差": round(float(rmse_scores.std(ddof=1)), 4),
        "样本总量": int(len(clean_df)),
    }
    return result


# ============================================================================
# 模块主入口（测试用）
# ============================================================================

if __name__ == "__main__":
    """模块自测代码：测试统计分析和风险预测功能。"""
    print("=" * 60)
    print("《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》")
    print("统计分析与风险预警模型模块 - 自测运行")
    print("=" * 60)

    import sys
    sys.path.insert(0, ".")
    from data_mock import generate_full_mock_dataset
    from process_data import clean_dataset, fill_missing_values

    print("\n[1/6] 生成模拟数据...")
    raw_df = generate_full_mock_dataset(samples_per_period=15)
    df = clean_dataset(raw_df)
    df = fill_missing_values(df, strategy="mean")

    print("\n[2/6] 执行相关性分析...")
    corr_result = run_correlation_analysis(df, target_cols=["GSM", "2-MIB"], method="pearson")
    print(f"相关性分析完成，共 {len(corr_result)} 对变量关系。")
    print("Top 5 相关关系：")
    print(corr_result.head().to_string())

    print("\n[3/6] 构建线性回归模型...")
    lr_result = build_linear_regression_model(
        df, target_col="GSM",
        predictor_cols=["水温", "pH", "DO", "浊度", "TN", "TP", "NH3-N", "CODMn", "叶绿素a"],
    )
    print(f"训练集 R² = {lr_result['训练集_R2']}, 测试集 R² = {lr_result['测试集_R2']}")

    print("\n[4/6] 分析特征重要性...")
    importance_df = analyze_feature_importance_rf(
        df, target_col="GSM",
        predictor_cols=["水温", "pH", "DO", "浊度", "TN", "TP", "NH3-N", "CODMn", "叶绿素a"],
    )
    print(importance_df.to_string())

    print("\n[5/6] 执行嗅味物质超标风险预测...")
    risk_result = predict_odor_risk(
        water_temp=26.0, ph=8.2, do_val=5.5, turbidity=25.0,
        tn=2.5, tp=0.15, nh3n=0.35, codmn=6.0, chl_a=45.0, odor_type="GSM",
    )
    print(f"预测浓度：{risk_result['预测浓度']} ng/L → {risk_result['风险等级']}")

    print("\n[6/6] 执行方差分析...")
    anova_result = run_anova_analysis(df, value_col="GSM", group_col="监测时段")
    print(f"ANOVA：F={anova_result['ANOVA_F统计量']}, p={anova_result['ANOVA_p值']} ({anova_result['ANOVA显著性']})")

    print("\n" + "=" * 60)
    print("自测完成！统计分析与风险预警模型模块所有函数均正常运行。")
    print("=" * 60)
