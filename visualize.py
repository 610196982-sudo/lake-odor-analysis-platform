# -*- coding: utf-8 -*-
"""
==============================================================================
模块名称：visualize.py
所属系统：《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》
功能描述：科学可视化模块
          —— 基于 Matplotlib 和 Seaborn 提供科研级数据可视化功能，
             包括时空分布折线图、箱线图、相关性热力图、散点拟合图、
             多面板组合图等。所有图表均使用中文字体，配色方案采用
             学术期刊常用的 Nature 风格调色板。
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import seaborn as sns
from typing import Optional, List, Tuple, Union
import warnings

# 忽略非关键警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 全局可视化配置
# ============================================================================

# --- 中文字体配置（鲁棒跨平台方案）---

# 全平台中文字体回退链：即使主检测失败，matplotlib 也会按顺序尝试，
# 命中任意已安装的 CJK 字体即可正常渲染中文（无需依赖单一字体）。
_CJK_FONT_STACK: list = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "PingFang SC",
    "Hiragino Sans GB", "Source Han Sans SC", "Droid Sans Fallback",
]


def _setup_chinese_font() -> str:
    """
    检测并配置中文字体，确保在 Windows / macOS / Linux（含 Streamlit Cloud）上正确渲染中文。

    检测策略：
    1. 扫描 matplotlib 已知的所有 TrueType 字体，寻找 CJK 字体
    2. 若未找到，依次检查常见系统路径
    3. 若仍未找到，尝试从 Google Fonts 下载 Noto Sans SC（仅限 Linux/云端环境）
    4. 全部失败则返回 sans-serif 并发出警告

    返回
    ----
    str
        实际可用的字体名称。
    """
    from matplotlib.font_manager import fontManager, FontProperties
    import os
    import sys

    # -- 策略0（云端优先）：加载仓库内置中文字体，不依赖 apt 安装与网络下载 --
    # 本地 Windows/macOS 自带优质中文字体，走后面的策略即可；云端 Linux 缺少
    # 中文字体，且 apt/下载均不稳定，因此优先从项目 fonts/ 目录加载静态字体。
    if sys.platform not in ("win32", "darwin"):
        bundled_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fonts", "NotoSansSC-Regular.otf",
        )
        if os.path.exists(bundled_path):
            try:
                fontManager.addfont(bundled_path)
                bundled_name = FontProperties(fname=bundled_path).get_name()
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [bundled_name, "sans-serif"]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"  [信息] 已加载仓库内置中文字体：{bundled_name}（{bundled_path}）")
                return bundled_name
            except Exception as e:
                print(f"  [警告] 加载仓库内置字体失败：{e}")

    # -- 策略1：按优先级扫描 matplotlib 已注册的字体 ---
    # 优先选择已知渲染效果好的无衬线中文字体，避免装饰性字体（如方正大标宋）
    cjk_priority = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "PingFang SC", "Hiragino Sans GB",
        "Source Han Sans SC", "Droid Sans Fallback",
    ]
    for preferred in cjk_priority:
        for font in fontManager.ttflist:
            if preferred.lower() == font.name.lower():
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [font.name, "sans-serif"]
                print(f"  [信息] 已启用中文字体：{font.name}（来源：{font.fname}）")
                return font.name

    # 兜底：模糊匹配其他 CJK 字体（排除装饰/衬线字体，优先无衬线）
    cjk_keywords = [
        "Hei", "YaHei", "Noto Sans", "WenQuanYi", "PingFang", "Hiragino",
        "Droid Sans", "Microsoft JhengHei",
    ]
    for font in fontManager.ttflist:
        name = font.name
        for kw in cjk_keywords:
            if kw.lower() in name.lower():
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [name, "sans-serif"]
                print(f"  [信息] 已启用中文字体（兜底匹配）：{name}（来源：{font.fname}）")
                return name

    # -- 策略2：检查常见系统字体路径 ---
    _candidate_paths = []
    if sys.platform == "win32":
        _candidate_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
    elif sys.platform == "darwin":
        _candidate_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:  # Linux / Streamlit Cloud
        _candidate_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
            "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
        ]

    for path in _candidate_paths:
        if os.path.exists(path):
            fontManager.addfont(path)
            fp = FontProperties(fname=path)
            name = fp.get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "sans-serif"]
            print(f"  [信息] 已启用中文字体：{name}（路径：{path}）")
            return name

    # -- 策略3：尝试从网络下载（仅 Linux 环境）---
    if sys.platform != "win32" and sys.platform != "darwin":
        try:
            import urllib.request
            import shutil
            import glob as _glob

            font_dir = os.path.expanduser("~/.fonts")
            os.makedirs(font_dir, exist_ok=True)
            font_path = os.path.join(font_dir, "NotoSansSC-Regular.ttf")

            if not os.path.exists(font_path):
                # 使用多个备用 URL（TTF 格式，兼容性最好）
                urls = [
                    "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf",
                    "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf",
                    "https://cdn.jsdelivr.net/gh/AimeeMao/Fonts@main/NotoSansSC/NotoSansSC-Regular.ttf",
                ]
                downloaded = False
                for url in urls:
                    try:
                        print("  [信息] 正在下载中文字体...")
                        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            with open(font_path, "wb") as f:
                                f.write(resp.read())
                        # 验证文件大小（至少 1MB 才算有效字体）
                        if os.path.getsize(font_path) > 100000:
                            downloaded = True
                            break
                        else:
                            os.remove(font_path)
                    except Exception:
                        continue
                if not downloaded:
                    raise RuntimeError("所有字体下载源均不可用")
                print(f"  [信息] 字体已下载至：{font_path}（{os.path.getsize(font_path) / 1024:.0f} KB）")

            # --- 注册字体并强制刷新缓存 ---
            fontManager.addfont(font_path)

            # 清除 matplotlib 字体缓存，强制重新扫描
            try:
                cache_dir = matplotlib.get_cachedir()
                for cache_file in _glob.glob(os.path.join(cache_dir, "fontlist*.json")):
                    os.remove(cache_file)
            except Exception:
                pass

            # 重新加载字体列表
            try:
                fontManager.__init__()
            except Exception:
                pass

            # 获取字体名称并设置
            fp = FontProperties(fname=font_path)
            name = fp.get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  [信息] 已启用中文字体：{name}")
            return name
        except Exception as e:
            print(f"  [警告] 字体下载/注册失败：{e}")

    # -- 全部失败---
    print(
        "  [警告] 未找到任何中文字体！图表中的中文标签可能显示为方框。\n"
        "         Linux 用户请执行：sudo apt install fonts-noto-cjk"
    )
    # 即使未主动检测到，也设置完整回退链，交给 matplotlib 自行匹配
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _CJK_FONT_STACK + ["sans-serif"]
    return "sans-serif"


AVAILABLE_FONT: str = _setup_chinese_font()

# --- 全局 matplotlib 参数设置（学术期刊风格）---
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["xtick.major.size"] = 3.5
plt.rcParams["ytick.major.size"] = 3.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linewidth"] = 0.4
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#fcfcfc"

# --- 学术配色方案 ---
# Nature Reviews 风格：柔和、高区分度、色盲友好
NATURE_PALETTE: list = [
    "#CC5542",  # 暖砖红
    "#3E8EBA",  # 钴蓝
    "#55A87C",  # 森林绿
    "#4A5F82",  # 石板蓝
    "#E8916A",  # 暖杏橙
    "#7A8EAE",  # 雾蓝灰
    "#6FB89B",  # 鼠尾草绿
    "#B8464A",  # 暗玫红
    "#8B7355",  # 可可棕
    "#A69C8E",  # 暖灰
]

# 设置 seaborn 学术风格
# 注意：sns.set_style / sns.set_context 会重置 font.sans-serif 与各项字号，
# 把上方 _setup_chinese_font() 的中文字体设置覆盖掉，导致中文显示为方框。
# 因此必须在 seaborn 调用之后，重新应用中文字体与学术字号配置。
sns.set_style("ticks")
sns.set_palette(sns.color_palette(NATURE_PALETTE))
sns.set_context("paper", font_scale=1.0)

# --- 重新应用中文字体与学术字号（防止被 seaborn 覆盖）---
plt.rcParams["font.family"] = "sans-serif"
# 检测到的字体置顶，其后接完整回退链，最后以 sans-serif 兜底
plt.rcParams["font.sans-serif"] = (
    ([AVAILABLE_FONT] if AVAILABLE_FONT and AVAILABLE_FONT != "sans-serif" else [])
    + _CJK_FONT_STACK
    + ["sans-serif"]
)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8


# ============================================================================
# 辅助校验函数
# ============================================================================

def _validate_df_and_cols(
    df: pd.DataFrame,
    required_cols: List[str]
) -> None:
    """
    校验 DataFrame 有效且包含所需列。

    参数
    ----
    df : pd.DataFrame
        待校验的数据框。
    required_cols : List[str]
        需要存在的列名列表。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"输入必须为 pandas DataFrame，当前传入类型为：{type(df)}"
        )
    if df.empty:
        raise ValueError("输入的 DataFrame 不能为空。")
    missing_cols: list = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"以下必需列在数据中不存在：{missing_cols}")


def _validate_custom_params(
    title: str,
    xlabel: str,
    ylabel: str
) -> None:
    """
    校验图表标题和轴标签参数。
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("图表标题不能为空。")
    if not isinstance(xlabel, str):
        raise ValueError("X 轴标签必须为字符串。")
    if not isinstance(ylabel, str):
        raise ValueError("Y 轴标签必须为字符串。")


# ============================================================================
# 时空分布折线图
# ============================================================================

def plot_temporal_trend(
    df: pd.DataFrame,
    x_col: str = "采样日期",
    y_col: str = "GSM",
    group_col: str = "湖泊名称",
    title: str = "各湖泊嗅味物质（GSM）时间变化趋势",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    绘制多组时间序列折线图，展示不同湖泊/点位在时间维度上的指标变化趋势。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    x_col : str
        作为 X 轴（时间）的列名，默认为"采样日期"。
    y_col : str
        作为 Y 轴（指标值）的列名，默认为"GSM"。
    group_col : str
        分组变量（用于绘制多条折线），默认为"湖泊名称"。
    title : str
        图表标题。
    figsize : Tuple[int, int]
        图表尺寸（宽, 高），单位为英寸。

    返回
    ----
    plt.Figure
        matplotlib Figure 对象。
    """
    # --- 参数校验 ---
    _validate_df_and_cols(df, [x_col, y_col, group_col])
    _validate_custom_params(title, "采样日期", y_col)
    if figsize[0] <= 0 or figsize[1] <= 0:
        raise ValueError(f"figsize 必须为正数，当前传入：{figsize}")

    # --- 创建图表 ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 按分组变量分别绘制折线 ---
    groups = df[group_col].unique()
    for idx, group_name in enumerate(groups):
        group_data = df[df[group_col] == group_name].copy()

        # 如果 X 轴是日期，进行排序
        if x_col == "采样日期":
            group_data[x_col] = pd.to_datetime(group_data[x_col])
            group_data = group_data.sort_values(x_col)

        # 按 X 聚合（取均值）
        if x_col == "采样日期":
            trend = group_data.groupby(x_col)[y_col].mean()
            color = NATURE_PALETTE[idx % len(NATURE_PALETTE)]
            ax.plot(
                trend.index,
                trend.values,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=color,
                label=str(group_name),
                alpha=0.85,
            )
        else:
            trend = group_data.groupby(x_col)[y_col].mean()
            color = NATURE_PALETTE[idx % len(NATURE_PALETTE)]
            ax.plot(
                range(len(trend)),
                trend.values,
                marker="s",
                markersize=4,
                linewidth=1.8,
                color=color,
                label=str(group_name),
                alpha=0.85,
            )
            ax.set_xticks(range(len(trend)))
            ax.set_xticklabels(trend.index, rotation=45, ha="right", fontsize=8)

    # --- 设置标签和标题 ---
    ax.set_xlabel("采样日期", fontsize=12, fontweight="medium")
    ax.set_ylabel(y_col, fontsize=12, fontweight="medium")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # --- 设置图例 ---
    ax.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
        title=group_col,
        title_fontsize=10,
    )

    # --- 设置网格 ---
    ax.grid(True, alpha=0.3, linestyle="--")

    # --- 紧凑布局 ---
    fig.tight_layout()

    return fig


# ============================================================================
# 箱线图（时空分布对比）
# ============================================================================

def plot_boxplot_comparison(
    df: pd.DataFrame,
    x_col: str = "监测时段",
    y_col: str = "GSM",
    hue_col: Optional[str] = "湖泊名称",
    title: str = "不同监测时段嗅味物质（GSM）浓度分布对比",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    绘制分组箱线图，对比不同类别下的指标分布特征。

    箱线图展示的信息：
    - 中位数（箱内横线）
    - 四分位距 IQR（箱体）
    - 1.5×IQR 范围内的须线
    - 离群点（须线外的散点）

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    x_col : str
        X 轴分类变量。
    y_col : str
        Y 轴数值变量。
    hue_col : Optional[str]
        色调分组变量（嵌套在 X 分类内）。
    title : str
        图表标题。
    figsize : Tuple[int, int]
        图表尺寸。

    返回
    ----
    plt.Figure
        matplotlib Figure 对象。
    """
    # --- 参数校验 ---
    _validate_df_and_cols(df, [x_col, y_col])
    if hue_col is not None and hue_col not in df.columns:
        raise ValueError(f"hue_col '{hue_col}' 在数据中不存在。")
    _validate_custom_params(title, x_col, y_col)

    # --- 创建图表 ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 绘制箱线图 ---
    if hue_col is not None:
        sns.boxplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette=NATURE_PALETTE[:len(df[hue_col].unique())],
            ax=ax,
            width=0.7,
            linewidth=1.0,
            flierprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markersize": 4,
                "alpha": 0.5,
            },
        )
    else:
        sns.boxplot(
            data=df,
            x=x_col,
            y=y_col,
            palette=NATURE_PALETTE[:len(df[x_col].unique())],
            ax=ax,
            width=0.5,
            linewidth=1.0,
            flierprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markersize": 4,
                "alpha": 0.5,
            },
        )

    # --- 叠加散点（展示数据分布密度） ---
    if hue_col is not None:
        sns.stripplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette=NATURE_PALETTE[:len(df[hue_col].unique())],
            ax=ax,
            dodge=True,
            size=3,
            alpha=0.3,
            legend=False,
        )

    # --- 设置标签 ---
    ax.set_xlabel(x_col, fontsize=12, fontweight="medium")
    ax.set_ylabel(y_col, fontsize=12, fontweight="medium")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # --- 优化图例 ---
    if hue_col is not None:
        ax.legend(
            loc="upper right",
            frameon=True,
            fontsize=8,
            title=hue_col,
            title_fontsize=9,
        )

    # --- 网格 ---
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    fig.tight_layout()
    return fig


# ============================================================================
# 相关性热力图
# ============================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    title: str = "各理化指标与嗅味物质相关性热力图",
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
    cmap: str = "RdBu_r"
) -> plt.Figure:
    """
    绘制数值指标之间的相关性系数热力图。

    支持三种相关系数：
    - 'pearson'：皮尔逊线性相关系数（要求正态分布）。
    - 'spearman'：斯皮尔曼秩相关系数（对异常值不敏感）。
    - 'kendall'：肯德尔秩相关系数（适合小样本）。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    columns : Optional[List[str]]
        参与计算相关性的列名列表。若为 None，则使用所有数值列。
    method : str
        相关系数类型，默认 'pearson'。
    title : str
        图表标题。
    figsize : Tuple[int, int]
        图表尺寸。
    annot : bool
        是否在热力图方格中标注数值，默认 True。
    cmap : str
        颜色映射方案。

    返回
    ----
    plt.Figure
        matplotlib Figure 对象。
    """
    # --- 参数校验 ---
    _validate_df_and_cols(df, [])
    valid_methods: list = ["pearson", "spearman", "kendall"]
    if method not in valid_methods:
        raise ValueError(
            f"method 必须为 {valid_methods} 之一，当前传入：{method}"
        )

    # --- 选择数值列 ---
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 在数据中不存在。")
        numeric_df = df[columns].select_dtypes(include=[np.number])

    # --- 计算相关矩阵 ---
    corr_matrix: pd.DataFrame = numeric_df.corr(method=method)

    # --- 创建图表 ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 绘制热力图 ---
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "shrink": 0.8,
            "label": f"{method.upper()} 相关系数",
        },
        ax=ax,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
    )

    # --- 设置标签 ---
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=9,
    )

    fig.tight_layout()
    return fig


# ============================================================================
# 散点拟合图（回归分析）
# ============================================================================

def plot_scatter_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = "湖泊名称",
    add_regression: bool = True,
    title: str = "",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    绘制散点图并叠加线性回归拟合线，展示两变量之间的定量关系。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    x_col : str
        X 轴变量（如"TN"）。
    y_col : str
        Y 轴变量（如"GSM"）。
    hue_col : Optional[str]
        分组着色变量。
    add_regression : bool
        是否叠加回归拟合线，默认 True。
    title : str
        图表标题。若为空字符串，则自动生成。
    figsize : Tuple[int, int]
        图表尺寸。

    返回
    ----
    plt.Figure
        matplotlib Figure 对象。
    """
    # --- 参数校验 ---
    _validate_df_and_cols(df, [x_col, y_col])
    if hue_col is not None and hue_col not in df.columns:
        raise ValueError(f"hue_col '{hue_col}' 在数据中不存在。")

    if not title:
        title = f"{x_col} 与 {y_col} 的散点关系图"

    # --- 创建图表 ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 绘制散点 ---
    if hue_col is not None:
        groups = df[hue_col].unique()
        for idx, group_name in enumerate(groups):
            group_data = df[df[hue_col] == group_name]
            color = NATURE_PALETTE[idx % len(NATURE_PALETTE)]
            ax.scatter(
                group_data[x_col],
                group_data[y_col],
                c=[color],
                label=str(group_name),
                alpha=0.5,
                s=30,
                edgecolors="white",
                linewidth=0.5,
            )
            # 分组绘制回归线（x 非恒定才拟合，退化数据跳过，避免 SVD 奇异）
            if add_regression and len(group_data) >= 3:
                x_vals = group_data[x_col].values
                y_vals = group_data[y_col].values
                # 过滤 NaN / 非数值
                mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                x_vals = x_vals[mask]
                y_vals = y_vals[mask]
                if len(x_vals) >= 3 and np.unique(x_vals).size >= 2:
                    try:
                        coeffs = np.polyfit(x_vals, y_vals, 1)
                        poly_fn = np.poly1d(coeffs)
                        x_sorted = np.sort(x_vals)
                        ax.plot(
                            x_sorted,
                            poly_fn(x_sorted),
                            color=color,
                            linewidth=2,
                            linestyle="--",
                            alpha=0.8,
                        )
                    except (np.linalg.LinAlgError, ValueError):
                        pass
    else:
        ax.scatter(
            df[x_col],
            df[y_col],
            c=NATURE_PALETTE[0],
            alpha=0.5,
            s=30,
            edgecolors="white",
            linewidth=0.5,
        )
        # 绘制总体回归线（并集掩码对齐 x/y，x 非恒定才拟合，退化数据跳过）
        if add_regression:
            valid_mask = (
                pd.to_numeric(df[x_col], errors="coerce").notna()
                & pd.to_numeric(df[y_col], errors="coerce").notna()
            )
            x_vals = pd.to_numeric(df.loc[valid_mask, x_col], errors="coerce").values
            y_vals = pd.to_numeric(df.loc[valid_mask, y_col], errors="coerce").values
            if len(x_vals) >= 3 and np.unique(x_vals).size >= 2:
                try:
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    poly_fn = np.poly1d(coeffs)
                    x_sorted = np.sort(x_vals)
                    ax.plot(
                        x_sorted,
                        poly_fn(x_sorted),
                        color="black",
                        linewidth=2,
                        linestyle="--",
                        alpha=0.7,
                        label=f"线性拟合 (y={coeffs[0]:.3f}x + {coeffs[1]:.3f})",
                    )
                except (np.linalg.LinAlgError, ValueError):
                    pass

    # --- 设置标签和标题 ---
    ax.set_xlabel(x_col, fontsize=12, fontweight="medium")
    ax.set_ylabel(y_col, fontsize=12, fontweight="medium")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # --- 图例 ---
    if hue_col is not None:
        ax.legend(
            loc="best",
            frameon=True,
            fontsize=8,
            title=hue_col,
            title_fontsize=9,
            markerscale=1.5,
        )
    elif add_regression and hue_col is None:
        ax.legend(loc="best", fontsize=9)

    # --- 网格 ---
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    return fig


# ============================================================================
# 多面板组合图
# ============================================================================

def plot_multi_panel_dashboard(
    df: pd.DataFrame,
    target_col: str = "GSM",
    predictor_cols: Optional[List[str]] = None,
    title: str = "嗅味物质（GSM）与环境驱动因子多面板分析",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    绘制多面板仪表板图，在一个图中同时展示目标变量与多个预测变量之间的关系。

    适用于快速浏览各环境因子对嗅味物质的潜在驱动作用。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    target_col : str
        目标变量（如"GSM"）。
    predictor_cols : Optional[List[str]]
        预测变量列表。若为 None，则自动使用除嗅味物质外的所有数值指标。
    title : str
        总图表标题。
    figsize : Tuple[int, int]
        图表尺寸。

    返回
    ----
    plt.Figure
        matplotlib Figure 对象。
    """
    # --- 参数校验 ---
    _validate_df_and_cols(df, [target_col])

    # --- 确定预测变量 ---
    if predictor_cols is None:
        # 排除分类列和嗅味物质列
        exclude_cols: set = {
            target_col, "GSM", "2-MIB",
            "湖泊名称", "采样点位", "监测时段", "采样日期",
        }
        predictor_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

    if not predictor_cols:
        raise ValueError("没有可用的预测变量列。")

    n_predictors: int = len(predictor_cols)
    n_cols: int = min(3, n_predictors)
    n_rows: int = int(np.ceil(n_predictors / n_cols))

    # --- 创建子图网格 ---
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        squeeze=False,
    )

    # --- 为每个预测变量绘制子图 ---
    for idx, pred_col in enumerate(predictor_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        # 散点图 + 回归线
        # 用「并集掩码」同时剔除 x/y 中的缺失与非数值，保证两者严格对齐
        valid_mask = (
            pd.to_numeric(df[pred_col], errors="coerce").notna()
            & pd.to_numeric(df[target_col], errors="coerce").notna()
        )
        x_vals = pd.to_numeric(df.loc[valid_mask, pred_col], errors="coerce").values
        y_vals = pd.to_numeric(df.loc[valid_mask, target_col], errors="coerce").values
        n_pts = len(x_vals)

        ax.scatter(
            x_vals,
            y_vals,
            c=NATURE_PALETTE[idx % len(NATURE_PALETTE)],
            alpha=0.4,
            s=20,
            edgecolors="none",
        )

        # 回归拟合（仅当点数足够且 x 非恒定，否则跳过以避免 SVD 奇异）
        if n_pts >= 3 and np.unique(x_vals).size >= 2:
            try:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                poly_fn = np.poly1d(coeffs)
                x_sorted = np.sort(x_vals)
                ax.plot(
                    x_sorted,
                    poly_fn(x_sorted),
                    color="black",
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                )

                # 添加 R² 标注
                residuals = y_vals - poly_fn(x_vals)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                ax.text(
                    0.05, 0.95,
                    f"R² = {r_squared:.3f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.8,
                    },
                )
            except (np.linalg.LinAlgError, ValueError):
                # 数据退化（如常数列/奇异矩阵），跳过回归线，仅保留散点
                pass

        ax.set_xlabel(pred_col, fontsize=9)
        ax.set_ylabel(target_col, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    # --- 隐藏多余的空子图 ---
    for idx in range(n_predictors, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    # --- 总标题 ---
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    return fig


# ============================================================================
# 各湖泊指标对比柱状图
# ============================================================================

def plot_bar_comparison(
    df: pd.DataFrame,
    x_col: str = "湖泊名称",
    y_col: str = "GSM",
    hue_col: Optional[str] = "监测时段",
    agg_func: str = "mean",
    title: str = "各湖泊嗅味物质（GSM）平均浓度对比",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    绘制分组柱状图，直观对比不同湖泊/时期/点位的指标均值。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    x_col : str
        X 轴分类变量。
    y_col : str
        Y 轴数值变量。
    hue_col : Optional[str]
        嵌套分组变量。
    agg_func : str
        聚合函数，默认 'mean'。
    title : str
        图表标题。
    figsize : Tuple[int, int]
        图表尺寸。

    返回
    ----
    plt.Figure
        matplotlib Figure 对象。
    """
    # --- 参数校验 ---
    _validate_df_and_cols(df, [x_col, y_col])
    if hue_col is not None and hue_col not in df.columns:
        raise ValueError(f"hue_col '{hue_col}' 在数据中不存在。")
    _validate_custom_params(title, x_col, y_col)

    # --- 聚合数据 ---
    agg_cols = [x_col, y_col]
    if hue_col is not None:
        agg_cols.append(hue_col)
    group_cols = [c for c in agg_cols if c != y_col]
    agg_df = df.groupby(group_cols, as_index=False)[y_col].agg(agg_func)

    # --- 创建图表 ---
    fig, ax = plt.subplots(figsize=figsize)

    if hue_col is not None:
        # 分组柱状图
        x_categories = agg_df[x_col].unique()
        hue_categories = agg_df[hue_col].unique()
        n_x = len(x_categories)
        n_hue = len(hue_categories)
        bar_width = 0.8 / n_hue

        for idx, hue_cat in enumerate(hue_categories):
            subset = agg_df[agg_df[hue_col] == hue_cat]
            # 对齐到正确的 X 位置
            positions = np.arange(n_x) + (idx - (n_hue - 1) / 2) * bar_width
            heights = []
            for x_cat in x_categories:
                row = subset[subset[x_col] == x_cat]
                if not row.empty:
                    heights.append(row[y_col].values[0])
                else:
                    heights.append(0)
            color = NATURE_PALETTE[idx % len(NATURE_PALETTE)]
            ax.bar(
                positions,
                heights,
                width=bar_width,
                color=color,
                label=str(hue_cat),
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(np.arange(n_x))
        ax.set_xticklabels(x_categories, rotation=30, ha="right", fontsize=9)
        ax.legend(
            title=hue_col,
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )
    else:
        # 简单柱状图
        colors = [
            NATURE_PALETTE[i % len(NATURE_PALETTE)]
            for i in range(len(agg_df))
        ]
        ax.bar(
            agg_df[x_col],
            agg_df[y_col],
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.tick_params(axis="x", rotation=30, labelsize=9)

    # --- 添加数值标签 ---
    if hue_col is None:
        for i, val in enumerate(agg_df[y_col]):
            ax.text(
                i, val,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # --- 标签和标题 ---
    ax.set_xlabel(x_col, fontsize=12, fontweight="medium")
    ax.set_ylabel(y_col, fontsize=12, fontweight="medium")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    fig.tight_layout()
    return fig


# ============================================================================
# 空间 GIS 地图（采样断面分布）
# ============================================================================

# 基准坐标表统一从 process_data 引入（单一数据源，避免重复维护）
from process_data import POINT_COORDS  # noqa: E402

# 嗅味浓度分级阈值（ng/L）
_ODOR_LIGHT: float = 10.0
_ODOR_SEVERE: float = 50.0

# 风险等级配色：正常（绿）/ 轻度（蓝）/ 严重（红）/ 未知（灰）
_ODOR_LEVEL_COLORS: dict = {
    "正常":     "#2e8b57",
    "轻度超标": "#2b7bba",
    "严重超标": "#c0392b",
    "未知":     "#9aa5b1",
}


def _classify_odor_level(value: Optional[float]) -> str:
    """
    按嗅味浓度分级：<10 正常、10~50 轻度超标、>50 严重超标。
    """
    if value is None:
        return "未知"
    if isinstance(value, float) and np.isnan(value):
        return "未知"
    if value < _ODOR_LIGHT:
        return "正常"
    if value <= _ODOR_SEVERE:
        return "轻度超标"
    return "严重超标"


def _build_point_geo_frame(
    df: pd.DataFrame,
    odor_col: str,
) -> pd.DataFrame:
    """
    构建「每采样点一行」的地理聚合表，用于 GIS 地图绘制。

    包含字段：采样点位、湖泊名称、纬度、经度、嗅味浓度均值、风险等级，
    以及各理化指标的均值（用于悬浮弹窗）。
    """
    point_col = "采样点位" if "采样点位" in df.columns else "湖泊名称"
    lake_col = "湖泊名称" if "湖泊名称" in df.columns else point_col

    rows: list = []
    for name, grp in df.groupby(point_col):
        # --- 解析坐标：优先实测均值，其次内置基准 ---
        lat: Optional[float] = None
        lon: Optional[float] = None
        if "纬度" in grp.columns and "经度" in grp.columns:
            lat_vals = pd.to_numeric(grp["纬度"], errors="coerce").dropna()
            lon_vals = pd.to_numeric(grp["经度"], errors="coerce").dropna()
            if not lat_vals.empty and not lon_vals.empty:
                lat = float(lat_vals.mean())
                lon = float(lon_vals.mean())
        if (lat is None or lon is None) and name in POINT_COORDS:
            lat, lon = POINT_COORDS[name]
        if lat is None or lon is None:
            continue  # 无坐标的点跳过

        # --- 嗅味浓度均值 ---
        odor_val: Optional[float] = None
        if odor_col in grp.columns:
            odor_vals = pd.to_numeric(grp[odor_col], errors="coerce").dropna()
            if not odor_vals.empty:
                odor_val = float(odor_vals.mean())

        # --- 理化指标均值（悬浮弹窗用） ---
        info_cols = [
            "水温", "DO", "pH", "浊度", "TN", "TP", "NH3-N", "CODMn",
            "叶绿素a", "藻密度", "电导率", "GSM", "2-MIB",
        ]
        info: dict = {}
        for c in info_cols:
            if c in grp.columns:
                vals = pd.to_numeric(grp[c], errors="coerce").dropna()
                info[c] = round(float(vals.mean()), 3) if not vals.empty else None

        lake = grp[lake_col].iloc[0] if lake_col in grp.columns else name

        rows.append({
            "采样点位": name,
            "湖泊名称": lake,
            "纬度": lat,
            "经度": lon,
            odor_col: odor_val if odor_val is not None else 0.0,
            "风险等级": _classify_odor_level(odor_val),
            **info,
        })

    return pd.DataFrame(rows)


def _plot_gis_map_matplotlib(
    geo: pd.DataFrame,
    odor_col: str,
    title: str,
) -> plt.Figure:
    """
    GIS 地图的 matplotlib 兜底实现（无 basemap，以经纬度散点呈现）。
    仅在未安装 plotly 时使用。
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    for level, color in _ODOR_LEVEL_COLORS.items():
        sub = geo[geo["风险等级"] == level]
        if sub.empty:
            continue
        sizes = pd.to_numeric(sub[odor_col], errors="coerce").fillna(0.0)
        max_s = float(sizes.max()) if sizes.max() > 0 else 1.0
        marker_size = 60 + 320 * (sizes / max_s)
        ax.scatter(
            sub["经度"], sub["纬度"],
            s=marker_size,
            c=color,
            alpha=0.75,
            edgecolors="white",
            linewidth=0.8,
            label=level,
            zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                str(r["采样点位"]),
                (r["经度"], r["纬度"]),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
            )

    ax.set_xlabel("经度（°E）", fontsize=12, fontweight="medium")
    ax.set_ylabel("纬度（°N）", fontsize=12, fontweight="medium")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(title="风险等级", fontsize=9, title_fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def plot_gis_map(
    df: pd.DataFrame,
    odor_col: str = "2-MIB",
    title: str = "长三角湖库采样断面空间分布与嗅味风险",
    zoom: int = 6,
    center: Tuple[float, float] = (31.0, 120.0),
):
    """
    绘制采样断面的空间 GIS 地图。

    - 气泡大小对应嗅味浓度（odor_col）。
    - 颜色分级：正常（<10，绿）、轻度（10~50，黄）、严重（>50，红）。
    - 悬浮弹窗展示该点位的理化指标均值。

    优先使用 plotly.express.scatter_mapbox（免费 open-street-map 彩色底图，无需 token）；
    若未安装 plotly，则回退为 matplotlib 散点图。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    odor_col : str
        用于着色的嗅味物质列（如"2-MIB"或"GSM"）。
    title : str
        图表标题。
    zoom : int
        地图缩放级别（仅 plotly 有效）。
    center : Tuple[float, float]
        地图中心（纬度, 经度）（仅 plotly 有效）。

    返回
    ----
    plotly.graph_objects.Figure 或 plt.Figure
        地图对象（类型取决于是否安装了 plotly）。
    """
    _validate_df_and_cols(df, [])

    # 嗅味列兜底：优先 GSM/2-MIB
    if odor_col not in df.columns:
        for cand in ("2-MIB", "GSM"):
            if cand in df.columns and not df[cand].isna().all():
                odor_col = cand
                break

    geo = _build_point_geo_frame(df, odor_col)
    if geo.empty:
        raise ValueError(
            "数据中没有可定位的采样点位（既无经度/纬度列，点位也不在内置坐标基准中）。"
        )

    # --- 优先 plotly ---
    try:
        import plotly.express as px  # noqa: F401
    except ImportError:
        px = None

    if px is not None:
        hover_cols = [
            c for c in (
                "水温", "DO", "pH", "浊度", "TN", "TP", "NH3-N", "CODMn",
                "叶绿素a", "藻密度", "电导率", "GSM", "2-MIB",
            ) if c in geo.columns and c != odor_col
        ]
        fig = px.scatter_mapbox(
            geo,
            lat="纬度",
            lon="经度",
            color="风险等级",
            color_discrete_map=_ODOR_LEVEL_COLORS,
            size=odor_col,
            size_max=40,
            hover_name="采样点位",
            hover_data={
                "纬度": False,
                "经度": False,
                "风险等级": False,
                **{c: ":.3f" for c in hover_cols},
            },
            zoom=zoom,
            center={"lat": center[0], "lon": center[1]},
            mapbox_style="open-street-map",
            title=title,
            category_orders={"风险等级": ["正常", "轻度超标", "严重超标", "未知"]},
        )
        fig.update_layout(
            margin={"r": 0, "t": 60, "l": 0, "b": 0},
            legend=dict(
                title="风险等级",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            height=580,
        )
        return fig

    # --- matplotlib 兜底 ---
    return _plot_gis_map_matplotlib(geo, odor_col, title)


def is_plotly_figure(fig) -> bool:
    """
    判断 Figure 是否为 plotly 对象，用于选择 st.plotly_chart 或 st.pyplot 渲染。

    参数
    ----
    fig : object
        待判断的图表对象。

    返回
    ----
    bool
        若为 plotly 对象返回 True，否则 False。
    """
    return fig is not None and fig.__class__.__module__.startswith("plotly")


# ============================================================================
# 双 Y 轴对比图
# ============================================================================

def plot_dual_axis(
    df: pd.DataFrame,
    x_col: str = "监测时段",
    left_col: str = "水温",
    right_col: str = "GSM",
    title: str = "",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    绘制双 Y 轴对比图：左轴展示一个指标（如水温/TN），右轴展示另一个指标
    （如 2-MIB/土臭素），用于观察两变量在不同类别下的协同变化。

    参数
    ----
    df : pd.DataFrame
        监测数据集。
    x_col : str
        X 轴分类变量（如"监测时段"、"湖泊名称"）。
    left_col : str
        左 Y 轴指标（如"水温"、"TN"）。
    right_col : str
        右 Y 轴指标（如"2-MIB"、"GSM"）。
    title : str
        图表标题；为空时自动生成。
    figsize : Tuple[int, int]
        图表尺寸。

    返回
    ----
    plt.Figure
        matplotlib Figure 对象。
    """
    _validate_df_and_cols(df, [x_col, left_col, right_col])
    if not title:
        title = f"{left_col}（左轴）与 {right_col}（右轴）的对比"

    # 按 X 分类聚合均值
    agg = df.groupby(x_col, as_index=False)[[left_col, right_col]].mean()
    x_pos = np.arange(len(agg))

    fig, ax1 = plt.subplots(figsize=figsize)

    # --- 左轴（线 + 圆形标记） ---
    ax1.plot(
        x_pos, agg[left_col],
        marker="o", markersize=5, linewidth=2,
        color=NATURE_PALETTE[1], label=left_col,
    )
    ax1.set_xlabel(x_col, fontsize=12, fontweight="medium")
    ax1.set_ylabel(left_col, fontsize=12, fontweight="medium", color=NATURE_PALETTE[1])
    ax1.tick_params(axis="y", labelcolor=NATURE_PALETTE[1])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(agg[x_col], rotation=45, ha="right", fontsize=9)

    # --- 右轴（线 + 方形标记，虚线） ---
    ax2 = ax1.twinx()
    ax2.plot(
        x_pos, agg[right_col],
        marker="s", markersize=5, linewidth=2, linestyle="--",
        color=NATURE_PALETTE[0], label=right_col,
    )
    ax2.set_ylabel(right_col, fontsize=12, fontweight="medium", color=NATURE_PALETTE[0])
    ax2.tick_params(axis="y", labelcolor=NATURE_PALETTE[0])

    # --- 合并双轴图例 ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


# ============================================================================
# 图表导出函数
# ============================================================================

def save_figure(
    fig: plt.Figure,
    file_path: str,
    dpi: int = 300
) -> str:
    """
    将 matplotlib Figure 保存为高分辨率图片文件。

    支持格式：PNG、JPG、SVG、PDF。

    参数
    ----
    fig : plt.Figure
        待保存的图片对象。
    file_path : str
        目标文件路径（含扩展名）。
    dpi : int
        输出分辨率，默认 300 dpi。

    返回
    ----
    str
        保存成功的文件路径。
    """
    # --- 参数校验 ---
    if not isinstance(fig, plt.Figure):
        raise TypeError(
            f"fig 必须为 matplotlib Figure，当前传入类型：{type(fig)}"
        )
    if not isinstance(file_path, str) or file_path.strip() == "":
        raise ValueError("file_path 不能为空。")
    if dpi < 72 or dpi > 1200:
        raise ValueError(f"dpi 应在 [72, 1200] 之间，当前传入：{dpi}")

    # --- 保存图片 ---
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  [信息] 图表已保存至：{file_path}")

    return file_path


# ============================================================================
# 模块主入口（测试用）
# ============================================================================

if __name__ == "__main__":
    """
    模块自测代码：测试所有绘图函数。
    """
    print("=" * 60)
    print("《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》")
    print("科学可视化模块 - 自测运行")
    print("=" * 60)

    import sys
    sys.path.insert(0, ".")
    from data_mock import generate_full_mock_dataset
    from process_data import clean_dataset, fill_missing_values

    # 生成测试数据
    print("\n[1/5] 生成模拟数据...")
    raw_df = generate_full_mock_dataset(samples_per_period=10)
    df = clean_dataset(raw_df)
    df = fill_missing_values(df, strategy="mean")

    # 测试 1：时间趋势折线图
    print("\n[2/5] 绘制时间趋势折线图...")
    fig1 = plot_temporal_trend(
        df,
        x_col="采样日期",
        y_col="GSM",
        group_col="湖泊名称",
        title="各湖泊嗅味物质（GSM）时间变化趋势",
    )

    # 测试 2：箱线图
    print("\n[3/5] 绘制监测时段箱线图...")
    fig2 = plot_boxplot_comparison(
        df,
        x_col="监测时段",
        y_col="GSM",
        hue_col="湖泊名称",
        title="不同监测时段嗅味物质（GSM）浓度分布对比",
    )

    # 测试 3：相关性热力图
    print("\n[4/5] 绘制相关性热力图...")
    fig3 = plot_correlation_heatmap(
        df,
        title="各理化指标与嗅味物质相关性热力图（Pearson）",
    )

    # 测试 4：散点拟合图
    print("\n[5/5] 绘制散点拟合图...")
    fig4 = plot_scatter_with_regression(
        df,
        x_col="TN",
        y_col="GSM",
        hue_col="湖泊名称",
        title="总氮（TN）与土臭素（GSM）的散点关系图",
    )

    print("\n" + "=" * 60)
    print("自测完成！可视化模块所有绘图函数均正常运行。")
    print("=" * 60)
