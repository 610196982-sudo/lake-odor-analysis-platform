# -*- coding: utf-8 -*-
"""
==============================================================================
模块名称：main.py
所属系统：《长三角湖库水体理化特征与嗅味污染多维分析平台 V1.0》
功能描述：系统主程序
          —— 基于 Streamlit 框架的 Web 交互界面，整合数据导入、数据清洗、
             时空特征可视化、驱动因子相关性分析、嗅味物质超标风险预警
             等全部功能模块。所有界面元素使用标准中文学术表达。
==============================================================================

启动方式：
    streamlit run main.py

依赖库：
    - streamlit
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scipy
    - scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ============================================================================
# Streamlit 页面全局配置
# ============================================================================

st.set_page_config(
    page_title="湖库水体嗅味污染多维分析平台 V1.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 导入自定义模块
# ============================================================================

# 确保当前目录在 Python 搜索路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_mock import (
    generate_full_mock_dataset,
    export_dataset_to_csv,
    get_dataset_summary,
    LAKE_NAMES,
    SAMPLING_POINTS,
    HYDROLOGICAL_PERIODS,
)
from process_data import (
    load_dataset_from_csv,
    load_dataset_from_excel,
    filter_by_lake,
    filter_by_period,
    aggregate_by_group,
    normalize_columns,
    smart_import,
    DRIVER_COLUMNS,
)
from visualize import (
    plot_temporal_trend,
    plot_boxplot_comparison,
    plot_correlation_heatmap,
    plot_scatter_with_regression,
    plot_multi_panel_dashboard,
    plot_bar_comparison,
    plot_gis_map,
    plot_dual_axis,
    is_plotly_figure,
    save_figure,
)
from model import (
    run_correlation_analysis,
    build_linear_regression_model,
    analyze_feature_importance_rf,
    predict_odor_risk,
    run_anova_analysis,
    evaluate_model_cv,
)


# ============================================================================
# 会话状态初始化
# ============================================================================

def init_session_state() -> None:
    """
    初始化 Streamlit 会话状态，用于在页面切换间保持数据。
    """
    # 初始化主数据集
    if "main_dataset" not in st.session_state:
        st.session_state["main_dataset"] = None

    # 初始化模型训练结果
    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None

    # 初始化标准化器
    if "scaler" not in st.session_state:
        st.session_state["scaler"] = None

    # 初始化数据来源标记
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "尚未导入数据"


# ============================================================================
# CSS 自定义样式
# ============================================================================

def apply_custom_css() -> None:
    """
    注入自定义 CSS 样式，采用现代科技风格。
    """
    st.markdown(
        """
        <style>
        /* 注：不加载 Google Fonts（国内被墙），直接用系统自带中文字体 */
        html, body, [class*="css"] {
            font-family: 'Microsoft YaHei', 'PingFang SC', 'SimHei', 'Noto Sans SC', sans-serif;
        }

        /* 主标题 */
        .main-title {
            font-size: 2.0rem;
            font-weight: 600;
            color: #16425b;
            text-align: center;
            padding: 30px 0 8px 0;
            margin-bottom: 6px;
            letter-spacing: 2px;
        }

        /* 副标题 */
        .sub-title {
            text-align: center;
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 30px;
            letter-spacing: 1px;
        }

        /* 模块标题 */
        .module-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #16425b;
            border-left: 3px solid #0e7490;
            padding-left: 14px;
            margin: 28px 0 14px 0;
        }

        /* 信息框 */
        .info-box {
            background-color: #f2f8f7;
            border: 1px solid #d7e8e6;
            padding: 16px 20px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 0.92rem;
            line-height: 1.8;
            color: #475569;
        }

        /* 成功提示框 */
        .success-box {
            background-color: #f0f7f4;
            border: 1px solid #a3cfbb;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 10px 0;
            color: #1e6e3e;
        }

        /* 警告提示框 */
        .warning-box {
            background-color: #fefaf0;
            border: 1px solid #f0c78e;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 10px 0;
            color: #8a6d14;
        }

        /* 错误提示框 */
        .error-box {
            background-color: #fdf5f5;
            border: 1px solid #e0a6a6;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 10px 0;
            color: #a94442;
        }

        /* 数据统计数字卡片 */
        .stat-card {
            background: #ffffff;
            border: 1px solid #e3e8e6;
            border-top: 3px solid #0e7490;
            border-radius: 10px;
            padding: 22px 18px;
            text-align: center;
            color: #16425b;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 2px 6px rgba(14, 116, 144, 0.07);
        }
        .stat-card .stat-value {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: #0e7490;
        }
        .stat-card .stat-label {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 4px;
            letter-spacing: 1px;
        }

        /* 功能模块卡片 */
        .func-card {
            background: #ffffff;
            border: 1px solid #d7e8e6;
            border-radius: 10px;
            padding: 24px 14px;
            text-align: center;
            transition: all 0.25s ease;
            cursor: default;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 120px;
            height: 100%;
        }
        .func-card:hover {
            border-color: #0e7490;
            box-shadow: 0 4px 14px rgba(14, 116, 144, 0.14);
            transform: translateY(-2px);
        }
        .func-card .func-name {
            font-size: 0.95rem;
            font-weight: 600;
            color: #16425b;
            margin-bottom: 6px;
        }
        .func-card .func-desc {
            font-size: 0.78rem;
            color: #64748b;
            line-height: 1.6;
        }

        /* 工作流步骤 */
        .workflow-step {
            display: inline-block;
            text-align: center;
            padding: 12px 20px;
        }
        .workflow-step .step-num {
            display: inline-block;
            width: 36px;
            height: 36px;
            line-height: 36px;
            border-radius: 50%;
            background: #0e7490;
            color: #fff;
            font-weight: 700;
            font-size: 1.0rem;
            margin-bottom: 6px;
        }
        .workflow-step .step-text {
            font-size: 0.8rem;
            color: #475569;
            font-weight: 500;
        }

        /* 页脚 */
        .footer {
            text-align: center;
            color: #94a3b8;
            padding: 28px 0;
            border-top: 1px solid #d7e8e6;
            margin-top: 50px;
            font-size: 0.78rem;
            letter-spacing: 1px;
        }

        /* ======== 侧边栏高级样式 ======== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e9f4f3 0%, #ffffff 100%);
            border-right: 1px solid #d7e8e6;
        }
        .sidebar-brand {
            text-align: center;
            padding: 20px 12px 14px 12px;
        }
        .sidebar-brand-icon {
            font-size: 2.2rem;
            margin-bottom: 6px;
        }
        .sidebar-brand-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #16425b;
            margin: 0;
            letter-spacing: 1.5px;
        }
        .sidebar-brand-sub {
            font-size: 0.68rem;
            color: #7ea0a8;
            margin: 4px 0 0 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .sidebar-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #e3e8e6, transparent);
            margin: 6px 12px;
        }

        /* 导航 radio — 全宽一致卡片式 */
        section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
            flex-direction: column !important;
            gap: 3px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] > div > div {
            width: 100% !important;
            flex: 1 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            display: flex !important;
            width: 100% !important;
            padding: 10px 14px !important;
            border-radius: 8px !important;
            font-size: 0.9rem !important;
            color: #475569 !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            box-sizing: border-box !important;
            margin: 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background-color: #e6f3f1 !important;
            color: #0e7490 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"]:checked + *,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background-color: #d5efee !important;
            color: #0e7490 !important;
            font-weight: 600 !important;
        }

        .sidebar-footer {
            font-size: 0.7rem;
            color: #94a3b8;
            text-align: center;
            line-height: 1.6;
        }
        .sidebar-status {
            background: #e8f5f3;
            border-left: 3px solid #0e7490;
            border-radius: 4px;
            padding: 8px 12px;
            margin-top: 10px;
        }
        .sidebar-status p {
            font-size: 0.72rem;
            color: #0e7490;
            margin: 0;
        }

        /* Streamlit 默认红色 → 亮蓝 */
        :root {
            --primary-color: #0e7490;
        }
        .stButton > button {
            background-color: #0e7490 !important;
            border-color: #0e7490 !important;
            color: #fff !important;
        }
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus {
            border-color: #0e7490 !important;
            box-shadow: 0 0 0 1px #0e7490 !important;
        }
        input[type="checkbox"]:checked,
        input[type="radio"]:checked {
            accent-color: #0e7490;
        }
        .stSlider > div > div > div > div {
            background-color: #0e7490 !important;
        }
        .stProgress > div > div > div {
            background-color: #0e7490 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 页面：系统首页
# ============================================================================

def render_home_page() -> None:
    """
    渲染系统首页，科技感数据驾驶舱风格。
    """
    # --- Hero 区域 ---
    st.markdown(
        '<div class="main-title">湖库水体嗅味污染多维分析平台</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-title">'
        '基于多源环境理化指标 · 驱动因子识别 · 风险智能预警</p>',
        unsafe_allow_html=True,
    )

    # --- 实时数据状态仪表盘 ---
    st.markdown('<div class="module-title">数据概览</div>', unsafe_allow_html=True)

    df = st.session_state.get("main_dataset")
    cleaned = st.session_state.get("main_dataset")

    if df is not None:
        # 有数据时展示实时统计
        sites = df["采样点位"].nunique() if "采样点位" in df.columns else len(df)
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        param_count = len(numeric_cols)
        has_odor = (
            "GSM" in df.columns and not df["GSM"].isna().all()
        ) or (
            "2-MIB" in df.columns and not df["2-MIB"].isna().all()
        )
        lake_list = df["湖泊名称"].unique().tolist() if "湖泊名称" in df.columns else ["—"]

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{len(df):,}</div>'
                f'<div class="stat-label">监测记录</div></div>',
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{sites}</div>'
                f'<div class="stat-label">采样点位</div></div>',
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{param_count}</div>'
                f'<div class="stat-label">监测指标</div></div>',
                unsafe_allow_html=True,
            )
        with s4:
            odor_status = "已收录" if has_odor else "待补充"
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value" style="font-size:1.3rem;">{odor_status}</div>'
                f'<div class="stat-label">嗅味物质 GSM / 2-MIB</div></div>',
                unsafe_allow_html=True,
            )

        st.caption(
            f"当前数据来源：{st.session_state.get('data_source', '—')}"
            f"  |  研究水体：{'、'.join(lake_list)}"
        )
    else:
        # 无数据时展示占位
        s1, s2, s3, s4 = st.columns(4)
        for col, label in zip(
            [s1, s2, s3, s4],
            ["监测记录", "采样点位", "监测指标", "嗅味物质"]
        ):
            with col:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-value" style="font-size:1.2rem;opacity:0.5;">—</div>'
                    f'<div class="stat-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )
        st.caption("尚未导入数据。请通过左侧导航进入「数据导入」上传您的监测数据。")

    # --- 功能模块 ---
    st.markdown('<div class="module-title">核心功能</div>', unsafe_allow_html=True)

    modules = [
        {
            "name": "数据导入",
            "desc": "多格式自动识别 · 列名智能映射<br>转置格式检测 · 数据验证",
        },
        {
            "name": "时空特征可视化",
            "desc": "时间序列 · 箱线对比<br>热力图 · 散点拟合 · 多面板联动",
        },
        {
            "name": "驱动因子分析",
            "desc": "Pearson / Spearman 相关<br>多元线性回归 · 随机森林排序",
        },
        {
            "name": "风险预警评估",
            "desc": "理化指标实时输入<br>嗅味浓度预测 · 四级定级 · 处理建议",
        },
    ]

    cols = st.columns(4)
    for col, mod in zip(cols, modules):
        with col:
            st.markdown(
                f'<div class="func-card">'
                f'<div class="func-name">{mod["name"]}</div>'
                f'<div class="func-desc">{mod["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- 分析工作流 ---
    st.markdown('<div class="module-title">分析流程</div>', unsafe_allow_html=True)

    steps = ["数据导入", "数据验证", "特征可视化", "因子建模", "风险预警"]
    workflow_html = (
        '<div style="display:flex;justify-content:center;align-items:center;'
        'flex-wrap:wrap;gap:10px;padding:20px 0;">'
    )
    for i, step in enumerate(steps):
        workflow_html += (
            f'<div class="workflow-step">'
            f'<div class="step-num">{i + 1}</div>'
            f'<div class="step-text">{step}</div>'
            f'</div>'
        )
        if i < len(steps) - 1:
            workflow_html += (
                '<div style="font-size:1.2rem;color:#c0c4cc;padding:0 4px;">→</div>'
            )
    workflow_html += '</div>'

    st.markdown(workflow_html, unsafe_allow_html=True)

    # --- 技术栈 ---
    st.markdown('<div class="module-title">技术架构</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.columns(5)
    tech_items = [
        ("Python 3", "核心语言"),
        ("Streamlit", "Web 交互框架"),
        ("scikit-learn", "机器学习引擎"),
        ("Matplotlib", "科学可视化"),
        ("pandas / numpy", "数据处理层"),
    ]
    for col, (name, desc) in zip([t1, t2, t3, t4, t5], tech_items):
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:10px;">'
                f'<p style="font-weight:600;color:#16425b;margin:0;font-size:0.9rem;">{name}</p>'
                f'<p style="color:#64748b;margin:4px 0 0 0;font-size:0.75rem;">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- 页脚 ---
    st.markdown(
        '<div class="footer">'
        'Lake Water Quality & Odor Analysis Platform  ·  Python + Streamlit'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# 页面：数据导入
# ============================================================================

def render_data_import_page() -> None:
    """
    渲染数据导入页面。
    """
    st.markdown(
        '<div class="main-title">数据导入</div>',
        unsafe_allow_html=True,
    )

    # --- 数据导入方式选择 ---
    st.markdown('<div class="module-title">步骤一：选择数据来源</div>', unsafe_allow_html=True)

    data_option = st.radio(
        "请选择数据导入方式：",
        ["使用系统内置模拟数据（快速体验）", "上传本地数据文件（自动识别格式）"],
        horizontal=True,
        help="上传模式支持 CSV、Excel、TXT、TSV、DAT、JSON 等多种格式，系统自动识别。",
    )

    df = None

    if data_option.startswith("使用系统内置模拟数据"):
        # --- 使用内置模拟数据 ---
        col1, col2 = st.columns(2)
        with col1:
            samples_count = st.slider(
                "模拟数据量（每点位/监测时段）",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                help="数值越大数据越丰富。推荐 15~20。",
            )
        with col2:
            random_seed = st.number_input(
                "随机种子（保证结果可复现）",
                min_value=1,
                max_value=9999,
                value=42,
            )

        if st.button("生成模拟监测数据", type="primary", use_container_width=True):
            with st.spinner("正在生成模拟数据..."):
                df = generate_full_mock_dataset(
                    samples_per_period=samples_count,
                    random_seed=random_seed,
                )
                st.session_state["main_dataset"] = df
                st.session_state["data_source"] = (
                    f"内置模拟数据（{samples_count} 条/点位/监测时段，种子={random_seed}）"
                )

            st.markdown(
                f'<div class="success-box">'
                f'数据生成成功——共 <b>{len(df)}</b> 条记录，'
                f'<b>{len(df.columns)}</b> 个字段。</div>',
                unsafe_allow_html=True,
            )

    else:
        # --- 上传任意格式文件 ---
        st.markdown(
            '<div class="info-box">'
            '<b>支持的文件格式：</b>'
            'CSV（逗号分隔）、TSV/TXT/DAT（制表符分隔）、Excel（.xlsx/.xls）、JSON'
            '<br><b>自动识别：</b>编码、分隔符、多段数据拆分、低于检出限(＜)解析、'
            '度分秒坐标转换、列名智能映射、点位→湖泊自动归属'
            '</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "拖拽或点击选择数据文件",
            type=None,  # 不限制扩展名
            accept_multiple_files=False,
            help="支持 CSV / TSV / TXT / DAT / Excel / JSON 等，由内容自动识别格式。"
                 "系统将自动识别各监测点位所属的湖泊/水库。",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                # 保存到临时文件
                tmp_path = f"_tmp_upload_{uploaded_file.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 万能智能导入（湖泊名称自动识别）
                df = smart_import(tmp_path)
                os.remove(tmp_path)

                st.session_state["main_dataset"] = df
                st.session_state["data_source"] = f"上传文件：{uploaded_file.name}"

                # 显示导入详情
                format_info = ""
                ext = uploaded_file.name.rsplit(".", 1)[-1].upper() if "." in uploaded_file.name else "未知"
                format_info = f"检测格式：{ext}"

                has_odor = (
                    "GSM" in df.columns and not df["GSM"].isna().all()
                ) or (
                    "2-MIB" in df.columns and not df["2-MIB"].isna().all()
                )

                st.markdown(
                    f'<div class="success-box">'
                    f'文件导入成功——<b>{uploaded_file.name}</b> | '
                    f'{format_info} | 共 <b>{len(df)}</b> 条记录，<b>{len(df.columns)}</b> 个字段'
                    f'{" | ✅ 含嗅味数据" if has_odor else " | ⚠️ 无嗅味数据（不影响其他分析）"}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # 列映射报告
                with st.expander("查看列名映射详情"):
                    st.caption("系统自动识别的指标及其映射关系：")
                    mapped_info = []
                    expected = ["DO", "pH", "浊度", "TN", "TP", "NH3-N", "CODMn", "叶绿素a", "GSM", "2-MIB", "水温"]
                    for col in expected:
                        if col in df.columns:
                            val = "✅ 已识别" if not df[col].isna().all() else "⚠️ 已识别（数据为空）"
                            mapped_info.append({"标准字段": col, "状态": val})
                    for col in df.columns:
                        if col not in expected + ["湖泊名称", "采样点位", "监测时段", "采样日期", "经度", "纬度"]:
                            mapped_info.append({"标准字段": col, "状态": "📎 保留原始列"})
                    st.dataframe(pd.DataFrame(mapped_info), use_container_width=True)

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">文件读取失败：{e}</div>',
                    unsafe_allow_html=True,
                )

    # --- 如果已有数据，显示数据预览和清洗选项 ---
    if st.session_state["main_dataset"] is not None:
        df = st.session_state["main_dataset"]

        st.markdown('<div class="module-title">步骤二：数据总览</div>', unsafe_allow_html=True)

        # 数据基本信息
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("📋 记录总数", f"{len(df)} 条")
        with col_b:
            st.metric("📊 字段数量", f"{len(df.columns)} 个")
        with col_c:
            missing_total = int(df.isna().sum().sum())
            st.metric("❓ 缺失值总数", f"{missing_total} 个")
        with col_d:
            st.metric("📂 数据来源", st.session_state["data_source"])

        # 数据表格预览
        st.markdown("**数据预览（前 10 行）：**")
        st.dataframe(df.head(10), use_container_width=True)

        # 各列统计信息
        with st.expander("📊 查看各数值指标的描述性统计"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(
                    df[numeric_cols].describe().round(3),
                    use_container_width=True,
                )

        # --- 数据验证与概览 ---
        st.markdown('<div class="module-title">步骤三：数据验证</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="info-box">'
            '<b>实测数据说明：</b>所有数据均为现场实测值，系统不会对原始数据进行任何修改、'
            '剔除或填补操作。以下展示各项指标的数据完整度，供您了解数据质量状况。'
            '</div>',
            unsafe_allow_html=True,
        )

        # 缺失值概览
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_info = []
        for col in numeric_cols:
            n_miss = int(df[col].isna().sum())
            n_total = len(df)
            pct = n_miss / n_total * 100 if n_total > 0 else 0
            missing_info.append({
                "指标": col,
                "有效值": n_total - n_miss,
                "缺失值": n_miss,
                "完整率": f"{100 - pct:.1f}%",
            })
        # 也检查文本列
        for col in df.columns:
            if col not in numeric_cols:
                n_miss = int(df[col].isna().sum())
                n_total = len(df)
                pct = n_miss / n_total * 100 if n_total > 0 else 0
                missing_info.append({
                    "指标": col,
                    "有效值": n_total - n_miss,
                    "缺失值": n_miss,
                    "完整率": f"{100 - pct:.1f}%",
                })

        if missing_info:
            st.markdown("**各字段数据完整度：**")
            st.dataframe(
                pd.DataFrame(missing_info).sort_values("完整率"),
                use_container_width=True,
                hide_index=True,
            )

        # --- 数据导出 ---
        st.markdown('<div class="module-title">步骤四：导出数据</div>', unsafe_allow_html=True)
        import io as _io
        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            st.download_button(
                label="📥 下载 CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="清洗后水质数据.csv",
                mime="text/csv",
            )
        with col_xlsx:
            buf = _io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="数据")
            st.download_button(
                label="📥 下载 Excel",
                data=buf.getvalue(),
                file_name="清洗后水质数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ============================================================================
# 页面：时空特征可视化
# ============================================================================

def render_visualization_page() -> None:
    """
    渲染时空特征可视化页面。
    """
    st.markdown(
        '<div class="main-title">时空特征可视化</div>',
        unsafe_allow_html=True,
    )

    # 检查是否有可用数据
    df = st.session_state.get("main_dataset")

    if df is None:
        st.markdown(
            '<div class="warning-box">⚠️ <b>暂无可用数据。</b>'
            '请先在「数据导入」页面导入或生成数据。</div>',
            unsafe_allow_html=True,
        )
        return

    # --- 数据筛选控件 ---
    st.markdown('<div class="module-title">数据筛选条件</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_lakes = st.multiselect(
            "选择湖泊",
            options=df["湖泊名称"].unique().tolist(),
            default=df["湖泊名称"].unique().tolist()[:3],
        )
    with col2:
        selected_periods = st.multiselect(
            "选择监测时段",
            options=df["监测时段"].unique().tolist() if "监测时段" in df.columns else [],
            default=df["监测时段"].unique().tolist() if "监测时段" in df.columns else [],
        )
    with col3:
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # 优先选有数据的指标，GSM/2-MIB 全空则跳过
        odor_candidates = ["GSM", "2-MIB"]
        has_odor = any(c in all_numeric and not df[c].isna().all() for c in odor_candidates)
        if has_odor:
            default_y = "GSM" if "GSM" in all_numeric and not df["GSM"].isna().all() else "2-MIB"
        else:
            default_y = "叶绿素a" if "叶绿素a" in all_numeric else all_numeric[0]
        y_variable = st.selectbox(
            "选择分析指标",
            options=all_numeric,
            index=all_numeric.index(default_y) if default_y in all_numeric else 0,
        )

    # 应用筛选
    filtered_df = df.copy()
    if selected_lakes:
        filtered_df = filtered_df[filtered_df["湖泊名称"].isin(selected_lakes)]
    if selected_periods and "监测时段" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["监测时段"].isin(selected_periods)]

    if filtered_df.empty:
        st.warning("⚠️ 当前筛选条件下无数据，请调整筛选条件。")
        return

    st.markdown(
        f'<div class="info-box">📌 当前筛选结果：<b>{len(filtered_df)}</b> 条记录</div>',
        unsafe_allow_html=True,
    )

    # --- 图表类型选择 ---
    st.markdown('<div class="module-title">可视化图表</div>', unsafe_allow_html=True)

    viz_type = st.selectbox(
        "请选择可视化类型：",
        options=[
            "GIS 空间分布地图",
            "时间变化趋势折线图",
            "分组箱线图对比",
            "相关性热力图",
            "散点拟合关系图",
            "多面板驱动因子分析",
            "柱状图对比",
            "双 Y 轴对比图",
        ],
    )

    fig = None

    if viz_type == "时间变化趋势折线图":
        fig = plot_temporal_trend(
            filtered_df,
            y_col=y_variable,
            title=f"各湖泊 {y_variable} 的时间变化趋势",
        )

    elif viz_type == "分组箱线图对比":
        x_choice = st.selectbox(
            "X 轴分组变量",
            options=["监测时段", "湖泊名称"],
        )
        hue_choice = st.selectbox(
            "嵌套分组（色调）",
            options=["湖泊名称", "监测时段"],
            index=1 if x_choice == "监测时段" else 0,
        )
        fig = plot_boxplot_comparison(
            filtered_df,
            x_col=x_choice,
            y_col=y_variable,
            hue_col=hue_choice,
            title=f"不同{x_choice}下 {y_variable} 的浓度分布对比",
        )

    elif viz_type == "相关性热力图":
        fig = plot_correlation_heatmap(
            filtered_df,
            title=f"各理化指标与嗅味物质相关性热力图",
        )

    elif viz_type == "散点拟合关系图":
        x_var = st.selectbox(
            "X 轴变量（预测因子）",
            options=[c for c in all_numeric if c != y_variable],
        )
        fig = plot_scatter_with_regression(
            filtered_df,
            x_col=x_var,
            y_col=y_variable,
            title=f"{x_var} 与 {y_variable} 的散点关系图",
        )

    elif viz_type == "多面板驱动因子分析":
        predictor_candidates = [c for c in DRIVER_COLUMNS if c in all_numeric and c != y_variable]
        fig = plot_multi_panel_dashboard(
            filtered_df,
            target_col=y_variable,
            predictor_cols=predictor_candidates,
            title=f"{y_variable} 与各环境驱动因子的多面板分析",
        )

    elif viz_type == "柱状图对比":
        x_choice = st.selectbox("X 轴变量", options=["湖泊名称", "监测时段"])
        hue_choice = st.selectbox(
            "嵌套分组",
            options=["监测时段", "湖泊名称"],
            index=1 if x_choice == "湖泊名称" else 0,
        )
        fig = plot_bar_comparison(
            filtered_df,
            x_col=x_choice,
            y_col=y_variable,
            hue_col=hue_choice,
            title=f"各{x_choice} {y_variable} 平均浓度对比",
        )

    elif viz_type == "GIS 空间分布地图":
        odor_options = [c for c in ["2-MIB", "GSM"] if c in filtered_df.columns and not filtered_df[c].isna().all()]
        if not odor_options:
            st.warning("⚠️ 当前数据中无嗅味物质（2-MIB / GSM）数据，无法按嗅味浓度着色。")
        else:
            odor_col = st.selectbox(
                "嗅味指标（用于气泡大小与颜色分级）",
                options=odor_options,
                index=0,
            )
            try:
                fig = plot_gis_map(
                    filtered_df,
                    odor_col=odor_col,
                    title=f"长三角湖库采样断面空间分布与嗅味风险（{odor_col}）",
                )
            except ValueError as e:
                st.warning(f"⚠️ {e}")
                fig = None

    elif viz_type == "双 Y 轴对比图":
        non_odor_numeric = [c for c in all_numeric if c not in {"GSM", "2-MIB"}]
        left_options = non_odor_numeric or all_numeric
        right_options = all_numeric
        default_left = "水温" if "水温" in left_options else (left_options[0] if left_options else None)
        default_right = y_variable if y_variable in right_options else (right_options[0] if right_options else None)
        left_col = st.selectbox(
            "左 Y 轴指标（如水温 / 总氮）",
            options=left_options,
            index=left_options.index(default_left) if default_left in left_options else 0,
        )
        right_col = st.selectbox(
            "右 Y 轴指标（如 2-MIB / 土臭素）",
            options=right_options,
            index=right_options.index(default_right) if default_right in right_options else 0,
        )
        x_choice = st.selectbox("X 轴变量", options=["监测时段", "湖泊名称"])
        fig = plot_dual_axis(
            filtered_df,
            x_col=x_choice,
            left_col=left_col,
            right_col=right_col,
        )

    # --- 显示图表（plotly 与 matplotlib 分流） ---
    if fig is not None:
        if is_plotly_figure(fig):
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig)

    # --- 图表导出（直接下载到浏览器：matplotlib → PNG，plotly → 交互式 HTML） ---
    if fig is not None and not is_plotly_figure(fig):
        import io as _io
        _buf = _io.BytesIO()
        fig.savefig(_buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        st.download_button(
            label="📥 下载图表（高清 PNG）",
            data=_buf.getvalue(),
            file_name=f"{viz_type}.png",
            mime="image/png",
        )
    elif fig is not None and is_plotly_figure(fig):
        st.download_button(
            label="📥 下载地图（交互式 HTML）",
            data=fig.to_html(),
            file_name=f"{viz_type}.html",
            mime="text/html",
        )


# ============================================================================
# 页面：驱动因子分析
# ============================================================================

def render_analysis_page() -> None:
    """
    渲染驱动因子分析页面。
    """
    st.markdown(
        '<div class="main-title">驱动因子分析</div>',
        unsafe_allow_html=True,
    )

    df = st.session_state.get("main_dataset")

    if df is None:
        st.markdown(
            '<div class="warning-box">⚠️ <b>暂无可用数据。</b>'
            '请先在「数据导入」页面导入或生成数据。</div>',
            unsafe_allow_html=True,
        )
        return

    # --- 分析类型选择 ---
    analysis_type = st.selectbox(
        "请选择分析类型：",
        options=[
            "皮尔逊相关性分析",
            "多元线性回归建模",
            "随机森林特征重要性排序",
            "交叉验证模型评估",
            "方差分析（ANOVA）",
        ],
    )

    if analysis_type == "皮尔逊相关性分析":
        st.markdown('<div class="module-title">皮尔逊相关性分析</div>', unsafe_allow_html=True)

        target = st.multiselect(
            "选择目标变量（嗅味物质）",
            options=["GSM", "2-MIB"],
            default=["GSM", "2-MIB"],
        )
        method = st.selectbox(
            "选择相关系数类型",
            options=["pearson", "spearman", "kendall"],
            format_func=lambda x: {
                "pearson": "皮尔逊（Pearson）— 线性相关",
                "spearman": "斯皮尔曼（Spearman）— 秩相关，对异常值不敏感",
                "kendall": "肯德尔（Kendall）— 适合小样本",
            }.get(x, x),
        )

        if st.button("🔍 执行相关性分析", type="primary"):
            if not target:
                st.warning("⚠️ 请至少选择一个目标变量。")
            else:
                with st.spinner("正在计算相关系数..."):
                    driver_cols = [c for c in DRIVER_COLUMNS if c in df.columns]
                    corr_df = run_correlation_analysis(
                        df, target_cols=target, predictor_cols=driver_cols, method=method,
                    )
                st.markdown(
                    f'<div class="success-box">✅ 分析完成！共计算 <b>{len(corr_df)}</b> 对变量关系。</div>',
                    unsafe_allow_html=True,
                )

                # 显著性筛选
                sig_only = st.checkbox("仅显示显著相关的结果", value=False)
                if sig_only:
                    corr_df = corr_df[corr_df["显著性"] != "不显著"]

                st.dataframe(corr_df, use_container_width=True)

                # 下载结果
                csv_data = corr_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 下载相关性分析结果（CSV）",
                    data=csv_data,
                    file_name="相关性分析结果.csv",
                    mime="text/csv",
                )

    elif analysis_type == "多元线性回归建模":
        st.markdown('<div class="module-title">多元线性回归建模</div>', unsafe_allow_html=True)

        target_options = ["GSM", "2-MIB"]
        target_col = st.selectbox("选择目标变量", options=target_options)

        all_num = df.select_dtypes(include=[np.number]).columns.tolist()
        predictor_options = [c for c in DRIVER_COLUMNS if c in all_num]
        predictor_cols = st.multiselect(
            "选择预测变量（环境因子）",
            options=predictor_options,
            default=predictor_options[:min(6, len(predictor_options))],
        )

        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)

        if st.button("📈 构建回归模型", type="primary"):
            if len(predictor_cols) < 2:
                st.warning("⚠️ 请至少选择 2 个预测变量。")
            else:
                with st.spinner("正在训练模型..."):
                    result = build_linear_regression_model(
                        df, target_col=target_col, predictor_cols=predictor_cols,
                        test_size=test_size,
                    )
                    st.session_state["trained_model"] = result["模型对象"]
                    st.session_state["scaler"] = result["标准化器"]

                # 展示结果
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("训练集 R²", f"{result['训练集_R2']:.4f}")
                with col2:
                    st.metric("测试集 R²", f"{result['测试集_R2']:.4f}")
                with col3:
                    st.metric("测试集 RMSE", f"{result['测试集_RMSE']:.4f}")

                st.markdown("**回归系数排序（按绝对值）：**")
                coef_data = []
                for name, val in result["系数排序"]:
                    direction = "↑ 正驱动" if val > 0 else "↓ 负驱动"
                    coef_data.append({
                        "变量": name,
                        "回归系数": val,
                        "驱动方向": direction,
                    })
                st.dataframe(pd.DataFrame(coef_data), use_container_width=True)

    elif analysis_type == "随机森林特征重要性排序":
        st.markdown('<div class="module-title">随机森林特征重要性排序</div>', unsafe_allow_html=True)

        target_col = st.selectbox("选择目标变量", options=["GSM", "2-MIB"])
        all_num = df.select_dtypes(include=[np.number]).columns.tolist()
        driver_opts = [c for c in DRIVER_COLUMNS if c in all_num]
        predictor_cols = st.multiselect(
            "选择预测变量",
            options=driver_opts,
            default=driver_opts[:min(6, len(driver_opts))],
        )
        n_trees = st.slider("决策树数量", 50, 500, 100, 50)

        if st.button("🌲 分析特征重要性", type="primary"):
            if len(predictor_cols) < 2:
                st.warning("⚠️ 请至少选择 2 个预测变量。")
            else:
                with st.spinner("正在训练随机森林模型..."):
                    importance_df = analyze_feature_importance_rf(
                        df, target_col=target_col,
                        predictor_cols=predictor_cols, n_estimators=n_trees,
                    )
                st.dataframe(importance_df, use_container_width=True)

                # 简易柱状图
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = plt.cm.Reds_r(
                    np.linspace(0.3, 0.9, len(importance_df))
                )
                ax.barh(
                    importance_df["特征名称"],
                    importance_df["重要性得分"],
                    color=colors,
                    edgecolor="white",
                )
                ax.set_xlabel("重要性得分", fontsize=11)
                ax.set_title(
                    f"各环境因子对 {target_col} 的重要性排序（随机森林）",
                    fontsize=13,
                    fontweight="bold",
                )
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis="x", linestyle="--")
                plt.tight_layout()
                st.pyplot(fig)

    elif analysis_type == "交叉验证模型评估":
        st.markdown('<div class="module-title">交叉验证模型评估</div>', unsafe_allow_html=True)

        target_col = st.selectbox("选择目标变量", options=["GSM", "2-MIB"])
        all_num = df.select_dtypes(include=[np.number]).columns.tolist()
        driver_opts = [c for c in DRIVER_COLUMNS if c in all_num]
        predictor_cols = st.multiselect(
            "选择预测变量",
            options=driver_opts,
            default=driver_opts[:min(5, len(driver_opts))],
        )
        cv_folds = st.slider("交叉验证折数", 3, 10, 5)

        if st.button("📉 执行交叉验证", type="primary"):
            if len(predictor_cols) < 2:
                st.warning("⚠️ 请至少选择 2 个预测变量。")
            else:
                with st.spinner("正在执行交叉验证..."):
                    cv_result = evaluate_model_cv(
                        df, target_col=target_col,
                        predictor_cols=predictor_cols, cv_folds=cv_folds,
                    )
                st.metric("平均 R²", f"{cv_result['平均R2']:.4f}")
                st.metric("R² 标准差", f"{cv_result['R2标准差']:.4f}")

                st.markdown("**各折详细结果：**")
                fold_data = pd.DataFrame({
                    "折次": list(range(1, cv_folds + 1)),
                    "R²": cv_result["各折R2得分"],
                    "RMSE": cv_result["各折RMSE"],
                })
                st.dataframe(fold_data, use_container_width=True)

    elif analysis_type == "方差分析（ANOVA）":
        st.markdown('<div class="module-title">方差分析（ANOVA）</div>', unsafe_allow_html=True)

        value_col = st.selectbox(
            "选择检验变量",
            options=["GSM", "2-MIB"],
        )
        group_col = st.selectbox(
            "选择分组变量",
            options=["监测时段", "湖泊名称"],
        )

        if st.button("📋 执行方差分析", type="primary"):
            with st.spinner("正在执行方差分析..."):
                anova_result = run_anova_analysis(df, value_col=value_col, group_col=group_col)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔹 单因素 ANOVA 结果**")
                st.metric("F 统计量", f"{anova_result['ANOVA_F统计量']:.4f}")
                st.metric("p 值", f"{anova_result['ANOVA_p值']:.6f}")
                st.metric("显著性", anova_result["ANOVA显著性"])
            with col2:
                st.markdown("**🔹 Kruskal-Wallis 检验结果**")
                st.metric("H 统计量", f"{anova_result['KruskalWallis_H统计量']:.4f}")
                st.metric("p 值", f"{anova_result['KruskalWallis_p值']:.6f}")
                st.metric("显著性", anova_result["KruskalWallis显著性"])

            st.markdown("**各分组描述统计：**")
            stats_data = []
            for grp, stats in anova_result["分组描述统计"].items():
                stats_data.append({
                    "分组": grp,
                    "样本量": stats["样本量"],
                    "均值": stats["均值"],
                    "标准差": stats["标准差"],
                    "最小值": stats["最小值"],
                    "最大值": stats["最大值"],
                })
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


# ============================================================================
# 页面：风险预警评估
# ============================================================================

def render_risk_warning_page() -> None:
    """
    渲染风险预警评估页面。
    """
    st.markdown(
        '<div class="main-title">风险预警评估</div>',
        unsafe_allow_html=True,
    )

    df = st.session_state.get("main_dataset")

    # 检查是否有嗅味物质数据
    has_odor_data = False
    if df is not None:
        gsm_ok = "GSM" in df.columns and not df["GSM"].isna().all()
        mib_ok = "2-MIB" in df.columns and not df["2-MIB"].isna().all()
        has_odor_data = gsm_ok or mib_ok

    if df is None:
        st.markdown(
            '<div class="warning-box">请先在「数据导入」页面导入或生成数据。</div>',
            unsafe_allow_html=True,
        )
        return

    if not has_odor_data:
        st.markdown(
            '<div class="warning-box">'
            '<b>当前数据中尚未包含嗅味物质（GSM / 2-MIB）实测浓度。</b><br><br>'
            '风险预警模型需要嗅味物质数据作为训练目标变量。'
            '待嗅味物质检测数据出具后，更新数据文件即可启用本模块。<br><br>'
            '<b>以下功能当前可用：</b>'
            '<ul>'
            '<li>理化指标相关性分析（「驱动因子分析」页面）</li>'
            '<li>时空特征可视化（「时空特征可视化」页面）</li>'
            '<li>各站点水质指标对比</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )
        return

    # --- 参数输入区域 ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌡️ 物理指标**")
        water_temp = st.slider(
            "水温（℃）",
            min_value=0.0,
            max_value=40.0,
            value=25.0,
            step=0.5,
            help="湖库表层水温，范围 0~40℃。",
        )
        turbidity = st.slider(
            "浊度（NTU）",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=1.0,
            help="浊度，反映水体中悬浮颗粒物含量。",
        )
        do_val = st.slider(
            "溶解氧 DO（mg/L）",
            min_value=0.1,
            max_value=18.0,
            value=6.0,
            step=0.1,
            help="溶解氧浓度。夏季高温时通常较低。",
        )

    with col2:
        st.markdown("**🧪 化学指标**")
        ph = st.slider(
            "pH 值",
            min_value=4.0,
            max_value=10.5,
            value=7.5,
            step=0.1,
            help="水体酸碱度。藻类大量增殖时 pH 可升高至 8.5 以上。",
        )
        tn = st.slider(
            "总氮 TN（mg/L）",
            min_value=0.01,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="总氮浓度，反映水体氮污染水平。",
        )
        tp = st.slider(
            "总磷 TP（mg/L）",
            min_value=0.001,
            max_value=2.0,
            value=0.08,
            step=0.001,
            format="%.3f",
            help="总磷浓度。磷通常是淡水湖库藻类生长的限制因子。",
        )

    with col3:
        st.markdown("**🧬 生物与综合指标**")
        nh3n = st.slider(
            "氨氮 NH₃-N（mg/L）",
            min_value=0.005,
            max_value=5.0,
            value=0.25,
            step=0.005,
            format="%.3f",
            help="氨氮浓度，反映含氮有机物分解程度。",
        )
        codmn = st.slider(
            "高锰酸盐指数 CODMn（mg/L）",
            min_value=0.5,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="反映水体中有机物污染程度的综合指标。",
        )
        chl_a = st.slider(
            "叶绿素a（μg/L）",
            min_value=0.1,
            max_value=300.0,
            value=25.0,
            step=1.0,
            help="叶绿素a 浓度，反映浮游植物（藻类）的生物量水平。",
        )

    # --- 预测设置 ---
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        odor_type = st.radio(
            "选择预测的嗅味物质类型",
            options=["GSM", "2-MIB"],
            horizontal=True,
            help="GSM（土臭素）：典型土霉味；2-MIB（2-甲基异莰醇）：典型土腥味。",
        )
    with col_b:
        use_trained_model = st.checkbox(
            "使用已训练的回归模型进行预测",
            value=False,
            help="若在「驱动因子分析」页面已训练模型，可勾选此项使用模型预测。"
            "否则使用内置经验公式。",
        )

    # --- 执行预测 ---
    if st.button("⚠️ 执行风险预测", type="primary", use_container_width=True):
        with st.spinner("正在计算嗅味物质浓度并评估风险等级..."):
            model = None
            scaler = None
            if use_trained_model:
                model = st.session_state.get("trained_model")
                scaler = st.session_state.get("scaler")
                if model is None:
                    st.warning("⚠️ 尚未训练模型，将使用经验公式进行预测。")

            risk_result = predict_odor_risk(
                water_temp=water_temp,
                ph=ph,
                do_val=do_val,
                turbidity=turbidity,
                tn=tn,
                tp=tp,
                nh3n=nh3n,
                codmn=codmn,
                chl_a=chl_a,
                odor_type=odor_type,
                model=model,
                scaler=scaler,
            )

        # --- 显示预测结果 ---
        st.markdown("---")
        st.markdown('<div class="module-title">预测结果</div>', unsafe_allow_html=True)

        # 风险等级颜色映射
        risk_color_map = {
            "🟢 低风险": "#27ae60",
            "🟡 中等风险": "#f39c12",
            "🟠 高风险": "#e67e22",
            "🔴 严重风险": "#e74c3c",
        }

        risk_color = risk_color_map.get(risk_result["风险等级"], "#333333")

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric(
                label=f"预测 {odor_type} 浓度",
                value=f"{risk_result['预测浓度']} ng/L",
                delta=f"阈值：{risk_result['感官阈值']} ng/L",
            )
        with col_r2:
            st.markdown(
                f'<div style="background-color:{risk_color};border-radius:10px;'
                f'padding:20px;text-align:center;color:white;">'
                f'<p style="font-size:0.9rem;margin:0;">风险等级</p>'
                f'<p style="font-size:1.8rem;margin:5px 0;font-weight:bold;">'
                f'{risk_result["风险等级"]}</p></div>',
                unsafe_allow_html=True,
            )
        with col_r3:
            st.markdown(
                f'<div style="background-color:#f7f9fc;border-radius:10px;'
                f'padding:15px;border:1px solid #d5dbdb;">'
                f'<p style="font-weight:bold;color:#1a5276;margin:0;">📋 处理建议</p>'
                f'<p style="font-size:0.85rem;margin-top:8px;line-height:1.6;">'
                f'{risk_result["处理建议"]}</p></div>',
                unsafe_allow_html=True,
            )

        # 详细风险描述
        st.markdown("**📝 风险详细描述：**")
        st.info(risk_result["风险描述"])

        # 输入参数回顾
        with st.expander("📋 查看本次预测的输入参数"):
            input_params = risk_result["输入参数"]
            params_df = pd.DataFrame({
                "参数名称": list(input_params.keys()),
                "输入值": list(input_params.values()),
                "单位": ["℃", "—", "mg/L", "NTU", "mg/L", "mg/L", "mg/L", "mg/L", "μg/L"],
            })
            st.dataframe(params_df, use_container_width=True)

        # 嗅味风险等级说明
        with st.expander("📖 嗅味风险等级划分标准"):
            for level_name, level_info in RISK_LEVELS.items():
                level_color = risk_color_map.get(level_name, "#333")
                st.markdown(
                    f'<div style="border-left:5px solid {level_color};'
                    f'padding:10px;margin:8px 0;background-color:#fafafa;">'
                    f'<b>{level_name}</b>：{level_info["描述"]}<br>'
                    f'<i>建议：{level_info["建议"]}</i></div>',
                    unsafe_allow_html=True,
                )


# ============================================================================
# 主程序入口
# ============================================================================

def main() -> None:
    """
    系统主入口函数。
    初始化会话状态、渲染自定义样式、创建侧边栏导航，
    根据用户选择切换到对应的功能页面。
    """
    # --- 初始化 ---
    init_session_state()
    apply_custom_css()

    # --- 侧边栏导航 ---
    with st.sidebar:
        # 品牌标识区
        st.markdown(
            """
            <div class="sidebar-brand">
            <div class="sidebar-brand-icon"></div>
            <p class="sidebar-brand-title">嗅味污染分析平台</p>
            <p class="sidebar-brand-sub">Lake Odor Analysis Platform</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # 导航菜单
        page = st.radio(
            "导航",
            options=[
                "系统首页",
                "数据导入",
                "时空特征可视化",
                "驱动因子分析",
                "风险预警评估",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # 侧边栏底部信息
        st.markdown(
            '<p class="sidebar-footer">'
            '© 2026 湖库水体嗅味污染<br>多维分析平台<br>'
            '基于实测数据的溯源解析</p>',
            unsafe_allow_html=True,
        )

        # 显示当前数据状态
        if st.session_state["main_dataset"] is not None:
            st.markdown(
                f'<div class="sidebar-status">'
                f'<p>✅ 数据已加载<br>'
                f'{st.session_state["data_source"]}</p></div>',
                unsafe_allow_html=True,
            )

    # --- 根据导航选择渲染对应页面 ---
    if page == "系统首页":
        render_home_page()
    elif page == "数据导入":
        render_data_import_page()
    elif page == "时空特征可视化":
        render_visualization_page()
    elif page == "驱动因子分析":
        render_analysis_page()
    elif page == "风险预警评估":
        render_risk_warning_page()


# ============================================================================
# 运行时入口
# ============================================================================

if __name__ == "__main__":
    main()
