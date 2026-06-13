"""
MS_ITS_GUI_APP - Phase 1 MVP
Multi-target Prediction GUI: Marshall Stability (MS) and Indirect Tensile Strength (ITS)

Features:
- Tab 1: Dual-target prediction + Performance trade-off plot
- Tab 2: Dual-target SHAP interpretation + Consistency analysis
"""

import os
import warnings

# Streamlit Community Cloud usually runs on CPU-only machines. Hide CUDA before
# importing torch-backed model packages so persisted TabICL artifacts do not try
# to restore themselves onto a GPU device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

warnings.filterwarnings("ignore")

from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 导入工具函数
from utils.prediction import (
    transform_input,
    predict_ms_its,
    get_ft_options_from_preprocessor,
)
from utils.shap_analysis import (
    plot_waterfall_from_explanation,
    analyze_feature_consistency,
)
from utils.visualization import (
    plot_ms_its_tradeoff,
    plot_dual_waterfall,
    configure_matplotlib,
)
from utils.feature_aggregation import create_aggregated_shap_explanation

# 配置 matplotlib
configure_matplotlib()

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model_MS_ITS_RF.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor_MS_ITS.joblib"

# SHAP explainer paths (预训练的 TreeExplainer)
SHAP_EXPLAINER_MS_PATH = BASE_DIR / "models" / "shap_explainer_MS_RF.pkl"
SHAP_EXPLAINER_ITS_PATH = BASE_DIR / "models" / "shap_explainer_ITS_RF.pkl"

# 数据文件（可选）
DATASET_PATH = BASE_DIR / "data" / "Dataset_cleaned.xlsx"
PARETO_PATH = BASE_DIR / "data" / "pareto_solutions.xlsx"

# ============================================================
# 输入特征配置
# ============================================================
RAW_FEATURES = [
    "Pe", "Du", "SP", "AC", "AV", "VMA", "VFA",
    "Ag2.36", "Ag4.75", "Ag9.5", "FT", "FC", "FL", "TS", "MT"
]

DEFAULT_INPUT = {
    "Pe": 71.0,
    "Du": 100.0,
    "SP": 49.0,
    "AC": 5.1,
    "AV": 4.3,
    "VMA": 16.3,
    "VFA": 72.7,
    "Ag2.36": 37.0,
    "Ag4.75": 53.6,
    "Ag9.5": 79.0,
    "FT": "No_fiber",
    "FC": 0.0,
    "FL": 0.0,
    "TS": 0.0,
    "MT": 165.0,
}

RANGES = {
    "Pe": (30.0, 100.0, 0.1),
    "Du": (80.0, 200.0, 1.0),
    "SP": (40.0, 80.0, 0.1),
    "AC": (3.0, 10.0, 0.1),
    "AV": (1.0, 10.0, 0.1),
    "VMA": (12.0, 25.0, 0.1),
    "VFA": (15.0, 95.0, 0.1),
    "Ag2.36": (10.0, 90.0, 0.1),
    "Ag4.75": (20.0, 100.0, 0.1),
    "Ag9.5": (40.0, 100.0, 0.1),
    "FC": (0.0, 3.0, 0.01),
    "FL": (0.0, 30.0, 0.1),
    "TS": (0.0, 5000.0, 10.0),
    "MT": (0.0, 220.0, 1.0),
}

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="FRAC MS & ITS Prediction GUI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown(
    """
    <style>
    /* 数值输入框字体 */
    [data-testid="stNumberInput"] input {
        font-size: 20px !important;
    }
    /* 下拉框字体 */
    div[data-baseweb="select"] > div {
        font-size: 18px !important;
    }
    /* 标签字体 */
    label[data-testid="stWidgetLabel"] p {
        font-size: 22px !important;
        font-weight: normal !important;
    }
    /* Tab 标签字体 - 多种选择器确保生效 */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 26px !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 26px !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 26px !important;
        font-weight: 600 !important;
    }
    /* 主要按钮字体（Predict 和 Run SHAP Analysis）*/
    button[kind="primary"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        padding: 1rem 1.5rem !important;
        height: auto !important;
    }
    button[kind="primary"] p {
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    /* 普通按钮字体 */
    button {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown(
    """
    <div style='background: linear-gradient(90deg, #0B5ED7 0%, #0D8BFF 100%);
                padding:5px 12px;border-radius:10px;text-align:center;margin-bottom:20px;'>
        <h1 style='color:white;margin:0;line-height:1.1;padding:0;'>🎯 Fiber-Reinforced Asphalt Concrete</h1>
        <h2 style='color:white;margin:0;line-height:1.1;padding:0;'>Marshall Stability & Indirect Tensile Strength Prediction</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    This platform predicts **MS (Marshall Stability)** and **ITS (Indirect Tensile Strength)**
    based on 15 input features, with SHAP-based interpretation for model transparency.
    """
)

# ============================================================
# Resource Loading
# ============================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, preprocessor, dataset, and pre-trained TreeExplainers"""
    import pickle

    info = {
        "model": None,
        "preprocessor": None,
        "dataset": None,
        "pareto": None,
        "shap_explainer_ms": None,
        "shap_explainer_its": None,
        "errors": {},
    }

    # Load model
    try:
        if MODEL_PATH.exists():
            info["model"] = joblib.load(MODEL_PATH)
            # RF model contains preprocessor, extract it
            if isinstance(info["model"], dict) and "preprocessor" in info["model"]:
                info["preprocessor"] = info["model"]["preprocessor"]
        else:
            info["errors"]["model"] = f"Model file not found: {MODEL_PATH}"
    except Exception:
        info["errors"]["model"] = traceback.format_exc()

    # Fallback: load standalone preprocessor if not extracted from model
    if info["preprocessor"] is None:
        try:
            if PREPROCESSOR_PATH.exists():
                info["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
            else:
                info["errors"]["preprocessor"] = f"Preprocessor file not found: {PREPROCESSOR_PATH}"
        except Exception:
            info["errors"]["preprocessor"] = traceback.format_exc()

    # Load pre-trained TreeExplainers (fast SHAP computation for RF models)
    try:
        if SHAP_EXPLAINER_MS_PATH.exists():
            with open(SHAP_EXPLAINER_MS_PATH, "rb") as f:
                info["shap_explainer_ms"] = pickle.load(f)
        else:
            info["errors"]["shap_ms"] = f"SHAP explainer MS not found: {SHAP_EXPLAINER_MS_PATH}"
    except Exception:
        info["errors"]["shap_ms"] = traceback.format_exc()

    try:
        if SHAP_EXPLAINER_ITS_PATH.exists():
            with open(SHAP_EXPLAINER_ITS_PATH, "rb") as f:
                info["shap_explainer_its"] = pickle.load(f)
        else:
            info["errors"]["shap_its"] = f"SHAP explainer ITS not found: {SHAP_EXPLAINER_ITS_PATH}"
    except Exception:
        info["errors"]["shap_its"] = traceback.format_exc()

    # Load dataset (optional, for background display)
    try:
        if DATASET_PATH.exists():
            info["dataset"] = pd.read_excel(DATASET_PATH)
    except Exception:
        pass

    # Load Pareto front (optional)
    try:
        if PARETO_PATH.exists():
            info["pareto"] = pd.read_excel(PARETO_PATH)
    except Exception:
        pass

    return info


artifacts = load_artifacts()

# Display loading errors
if artifacts["errors"]:
    with st.sidebar:
        st.warning("⚠️ Some resources failed to load")
        with st.expander("View Details"):
            for k, v in artifacts["errors"].items():
                st.error(f"**{k}**")
                st.code(v, language="text")

# ============================================================
# Create Tabs
# ============================================================
tab_pred, tab_shap = st.tabs([
    "🎯 Multi-target Prediction",
    "📊 SHAP Interpretation"
])

# ============================================================
# Tab 1: Multi-target Prediction
# ============================================================
with tab_pred:
    # Input area
    st.markdown("### 📝 Input Parameters")

    ft_options = get_ft_options_from_preprocessor(artifacts["preprocessor"])

    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        pe = st.number_input("Pe (0.1 mm)", *RANGES["Pe"][:2], value=DEFAULT_INPUT["Pe"], step=RANGES["Pe"][2])
    with col2:
        du = st.number_input("Du (cm)", *RANGES["Du"][:2], value=DEFAULT_INPUT["Du"], step=RANGES["Du"][2])
    with col3:
        sp = st.number_input("SP (°C)", *RANGES["SP"][:2], value=DEFAULT_INPUT["SP"], step=RANGES["SP"][2])

    # Row 2
    col1, col2, col3 = st.columns(3)
    with col1:
        ac = st.number_input("AC (wt.%)", *RANGES["AC"][:2], value=DEFAULT_INPUT["AC"], step=RANGES["AC"][2])
    with col2:
        av = st.number_input("AV (%)", *RANGES["AV"][:2], value=DEFAULT_INPUT["AV"], step=RANGES["AV"][2])
    with col3:
        vma = st.number_input("VMA (%)", *RANGES["VMA"][:2], value=DEFAULT_INPUT["VMA"], step=RANGES["VMA"][2])

    # Row 3
    col1, col2, col3 = st.columns(3)
    with col1:
        vfa = st.number_input("VFA (%)", *RANGES["VFA"][:2], value=DEFAULT_INPUT["VFA"], step=RANGES["VFA"][2])
    with col2:
        ag236 = st.number_input("Ag2.36 (%)", *RANGES["Ag2.36"][:2], value=DEFAULT_INPUT["Ag2.36"], step=RANGES["Ag2.36"][2])
    with col3:
        ag475 = st.number_input("Ag4.75 (%)", *RANGES["Ag4.75"][:2], value=DEFAULT_INPUT["Ag4.75"], step=RANGES["Ag4.75"][2])

    # Row 4
    col1, col2, col3 = st.columns(3)
    with col1:
        ag95 = st.number_input("Ag9.5 (%)", *RANGES["Ag9.5"][:2], value=DEFAULT_INPUT["Ag9.5"], step=RANGES["Ag9.5"][2])
    with col2:
        default_ft = str(DEFAULT_INPUT["FT"]).strip()
        default_idx = 0
        for i, opt in enumerate(ft_options):
            if str(opt).strip().lower() == default_ft.lower():
                default_idx = i
                break
        ft = st.selectbox("FT (Fiber Type)", options=ft_options, index=default_idx)
    with col3:
        fc = st.number_input("FC (wt.%)", *RANGES["FC"][:2], value=DEFAULT_INPUT["FC"], step=RANGES["FC"][2])

    # Row 5
    col1, col2, col3 = st.columns(3)
    with col1:
        fl = st.number_input("FL (mm)", *RANGES["FL"][:2], value=DEFAULT_INPUT["FL"], step=RANGES["FL"][2])
    with col2:
        ts = st.number_input("TS (MPa)", *RANGES["TS"][:2], value=DEFAULT_INPUT["TS"], step=RANGES["TS"][2])
    with col3:
        mt = st.number_input("MT (°C)", *RANGES["MT"][:2], value=DEFAULT_INPUT["MT"], step=RANGES["MT"][2])

    raw_input_df = pd.DataFrame([{
        "Pe": pe, "Du": du, "SP": sp, "AC": ac, "AV": av,
        "VMA": vma, "VFA": vfa, "Ag2.36": ag236, "Ag4.75": ag475, "Ag9.5": ag95,
        "FT": ft, "FC": fc, "FL": fl, "TS": ts, "MT": mt,
    }])

    # Predict button
    st.markdown("---")
    predict_btn = st.button("🔮 Predict MS & ITS", use_container_width=True, type="primary")

    if predict_btn:
        if artifacts["model"] is None or artifacts["preprocessor"] is None:
            st.error("❌ Model or preprocessor not loaded, cannot predict.")
        else:
            try:
                with st.spinner("🔄 Predicting..."):
                    X_input = transform_input(artifacts["preprocessor"], raw_input_df)
                    ms_pred, its_pred = predict_ms_its(artifacts["model"], X_input)

                    # Save to session_state
                    st.session_state["raw_input_df"] = raw_input_df.copy()
                    st.session_state["X_input"] = X_input.copy()
                    st.session_state["ms_pred"] = ms_pred
                    st.session_state["its_pred"] = its_pred

                    # Clear old SHAP results (avoid displaying outdated SHAP)
                    for key in ["shap_computed", "aggregated_shap_ms", "aggregated_shap_its", "consistency_df"]:
                        if key in st.session_state:
                            del st.session_state[key]

                st.success("✅ Prediction completed!")

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

    # Display prediction results
    if "ms_pred" in st.session_state and "its_pred" in st.session_state:
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        col_ms, col_its = st.columns(2)

        with col_ms:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding:8px 20px;border-radius:10px;text-align:center;">
                    <h3 style='color:white;margin:0;line-height:1.1;padding:0;'>Marshall Stability (MS)</h3>
                    <h1 style='color:white;margin:0;line-height:1.1;padding:0;'>{st.session_state["ms_pred"]:.2f} kN</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_its:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding:8px 20px;border-radius:10px;text-align:center;">
                    <h3 style='color:white;margin:0;line-height:1.1;padding:0;'>Indirect Tensile Strength (ITS)</h3>
                    <h1 style='color:white;margin:0;line-height:1.1;padding:0;'>{st.session_state["its_pred"]:.2f} MPa</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Performance trade-off plot
        st.markdown("### 📊 Performance Trade-off Visualization")
        try:
            fig = plot_ms_its_tradeoff(
                st.session_state["ms_pred"],
                st.session_state["its_pred"],
                dataset_df=artifacts["dataset"],
                pareto_df=artifacts["pareto"]
            )
            # 使用居中容器
            st.markdown(
                '<style>div[data-testid="stImage"] { display: flex; justify-content: center; }</style>',
                unsafe_allow_html=True
            )
            st.pyplot(fig, use_container_width=False)
        except Exception as e:
            st.warning(f"⚠️ Trade-off plot failed: {str(e)}")

        # Display raw input (collapsible)
        with st.expander("🔍 Show Input Features"):
            st.markdown("**User Input Feature Values**")
            # Transpose display: feature names as rows, values as columns
            display_df = st.session_state["raw_input_df"].T
            display_df.columns = ["Value"]
            st.dataframe(display_df, use_container_width=True)

    else:
        st.info("👆 Please click the **Predict** button to generate prediction results.")

# ============================================================
# Tab 2: SHAP Interpretation
# ============================================================
with tab_shap:
    # 1. Enhanced readiness check (check all required session keys)
    required_keys = ["raw_input_df", "X_input", "ms_pred", "its_pred"]
    if not all(k in st.session_state for k in required_keys):
        st.warning("⚠️ Please complete prediction in the **Multi-target Prediction** tab first.")
        st.stop()

    # 2. Display current prediction summary
    st.info(f"""
    **Current Prediction Results**:
    - MS = {st.session_state['ms_pred']:.2f} kN
    - ITS = {st.session_state['its_pred']:.2f} MPa
    """)

    # 3. Button to trigger runtime SHAP computation
    compute_shap_btn = st.button(
        "🔍 Run SHAP Analysis",
        key="compute_shap_button",
        type="primary",
        use_container_width=True
    )

    if compute_shap_btn:
        try:
            # Use pre-trained TreeExplainers for instant SHAP computation
            X_input = st.session_state["X_input"]
            raw_input_df = st.session_state["raw_input_df"]

            explainer_ms = artifacts.get("shap_explainer_ms")
            explainer_its = artifacts.get("shap_explainer_its")

            if explainer_ms is None or explainer_its is None:
                st.error("❌ SHAP explainers not loaded. Please check model artifacts.")
                st.stop()

            with st.spinner("Computing SHAP values with TreeExplainer..."):
                # TreeExplainer: instant computation (< 1 second)
                shap_values_ms = explainer_ms(X_input)
                shap_values_its = explainer_its(X_input)

            # Aggregate FT features
            aggregated_shap_ms = create_aggregated_shap_explanation(shap_values_ms, raw_input_df)
            aggregated_shap_its = create_aggregated_shap_explanation(shap_values_its, raw_input_df)

            # Calculate consistency analysis
            consistency_df = analyze_feature_consistency(
                aggregated_shap_ms.values,
                aggregated_shap_its.values,
                aggregated_shap_ms.feature_names
            )

            # Cache to session_state
            st.session_state["shap_computed"] = True
            st.session_state["aggregated_shap_ms"] = aggregated_shap_ms
            st.session_state["aggregated_shap_its"] = aggregated_shap_its
            st.session_state["consistency_df"] = consistency_df

            st.success("✅ SHAP analysis completed!")

        except Exception as e:
            st.error(f"❌ SHAP computation failed: {str(e)}")
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())

    # 4. Display cached SHAP results
    if st.session_state.get("shap_computed", False):
        aggregated_shap_ms = st.session_state["aggregated_shap_ms"]
        aggregated_shap_its = st.session_state["aggregated_shap_its"]
        consistency_df = st.session_state["consistency_df"]

        # Waterfall Plots
        st.markdown("### 📊 Waterfall Plots (Feature Contributions)")
        try:
            col_ms, col_its = st.columns(2)

            with col_ms:
                st.markdown("**MS Contribution**")
                fig_ms = plot_waterfall_from_explanation(aggregated_shap_ms, max_display=10)
                st.pyplot(fig_ms, use_container_width=True)

            with col_its:
                st.markdown("**ITS Contribution**")
                fig_its = plot_waterfall_from_explanation(aggregated_shap_its, max_display=10)
                st.pyplot(fig_its, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Waterfall plots rendering failed: {str(e)}")
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())

        # Feature Consistency Analysis
        st.markdown("### 🔍 Feature Consistency Analysis")
        st.markdown("Analyze whether features have consistent effects on MS and ITS:")

        st.dataframe(
            consistency_df[['Feature', 'MS_SHAP', 'ITS_SHAP', 'Consistency']],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        **Legend:**
        - ✅ **Aligned**: Feature has consistent effects on both targets (both increase or both decrease)
        - ⚠️ **Opposite**: Feature has opposite effects on the two targets (one increases, one decreases)
        - ⚪ **Neutral**: Feature has minimal impact on both targets
        """)

    else:
        st.info("👆 Click the **Run SHAP Analysis** button to start analysis.")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:gray;font-size:12px;'>
        <p>🎓 Developed for fiber-reinforced asphalt concrete research</p>
        <p>Powered by TabPFN + SHAP | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)
