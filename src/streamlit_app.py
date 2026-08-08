"""
streamlit_app.py
管道腐蚀预测与标准问答系统 - Streamlit Web 界面
Streamlit Community Cloud 部署入口

启动方式: streamlit run src/streamlit_app.py
"""

import sys
import os
import io
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from corrosion_model import CorrosionPredictor
from rag_engine import CorrosionRAG
from integrity_tools import b31g_calculate, recommend_inhibitor, risk_matrix, RISK_MATRIX_COLORS
from environment_models import (
    soil_corrosion, seawater_corrosion, mic_corrosion,
    galvanic_corrosion, corrosion_cost_estimate,
    mic_multi_organism, mic_biocide_program,
    mic_material_upgrade, mic_monitoring_plan,
)
from engineering_models import (
    co2_corrosion, co2_corrosion_curve,
    erosion_critical_velocity, erosion_rate_salama, EROSION_C,
    h2s_ssc_screening, scc_susceptibility, pren_all,
    scc_excavation_priority, scc_crack_life,
    scc_mitigation_tree, scc_risk_overlay,
)
from scc_morphology import simulate_crack_population, build_morphology_figures
from mic_ml import (
    get_trained_models, predict_mic_risk, MIC_RISK_LABELS, MIC_FEATURES, DEFAULT_FEATURES,
)
from data_calibration import (
    sample_template_df, demo_synthetic_df, parse_uploaded_csv, calibrate_with_data,
)
from pipeline_types import get_pipeline_presets, get_preset
from ndt_knowledge import NDT_METHODS, recommend_ndt, ILI_THREAT_MAP
from consequence_remediation import consequence_analysis, recommend_remediation
from styles import apply_theme

# ----------------------
# 页面配置
# ----------------------
st.set_page_config(
    page_title="管道腐蚀预测与标准问答系统",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------
# 从 Streamlit Secrets 读取 Dify 配置
# ----------------------
try:
    if "dify" in st.secrets:
        os.environ["DIFY_API_URL"] = st.secrets["dify"]["api_url"]
        os.environ["DIFY_API_KEY"] = st.secrets["dify"]["api_key"]
except Exception:
    pass

# ----------------------
# 侧边栏：设置
# ----------------------
with st.sidebar:
    st.markdown("### ⚙️ 设置")

    # 暗黑模式
    dark_mode = st.toggle("🌙 暗黑模式", value=False)

    # 语言选择
    lang = st.radio("🌐 语言", ["中文", "English"], horizontal=True)

    st.markdown("---")
    st.markdown("### 📊 快速导航")
    st.markdown("""
    - [腐蚀预测](#tab1)
    - [标准问答](#tab2)
    - [数据探索](#tab3)
    - [完整性工具](#tab4)
    - [腐蚀环境分析](#tab5)
    - [机理与模型](#tab6)
    - [🔍 无损检测(NDT)](#tab9)
    - [📡 实测数据标定](#tab8)
    - [关于](#tab7)
    """)
    st.markdown("---")
    st.caption("MIT License")

# 暗黑模式 CSS
apply_theme(dark_mode)

# 多语言文本
I18N = {
    "中文": {
        "title": "🔧 管道腐蚀预测与标准问答系统",
        "subtitle": "Pipeline Corrosion Prediction & Standards Q&A",
        "tabs": ["📊 腐蚀预测", "💬 标准问答", "📈 数据探索", "🔧 完整性工具", "🌍 腐蚀环境分析", "🧪 机理与模型", "🔍 无损检测(NDT)", "📡 实测数据标定", "ℹ️ 关于"],
        "input_params": "输入管道参数",
        "predict_btn": "🔍 预测腐蚀速率",
        "compare_btn": "📊 对比材料",
        "trend_btn": "📈 趋势分析",
        "batch_btn": "📋 批量预测",
        "export_btn": "💾 导出报告",
        "result_title": "预测结果",
        "confidence": "置信区间",
        "suggestion": "建议",
        "material_advice": "材料建议",
        "prediction_history": "预测历史",
        "model_comparison": "模型对比",
        "feature_importance": "特征重要性",
        "data_overview": "数据概览",
        "distribution": "分布分析",
        "correlation": "相关性分析",
    },
    "English": {
        "title": "🔧 Pipeline Corrosion Prediction & Standards Q&A",
        "subtitle": "AI-Powered Corrosion Management",
        "tabs": ["📊 Prediction", "💬 Q&A", "📈 Data Explorer", "🔧 Integrity Tools", "🌍 Env. Analysis", "🧪 Mechanisms", "🔍 NDT", "📡 Calibration", "ℹ️ About"],
        "input_params": "Input Parameters",
        "predict_btn": "🔍 Predict Corrosion Rate",
        "compare_btn": "📊 Compare Materials",
        "trend_btn": "📈 Trend Analysis",
        "batch_btn": "📋 Batch Predict",
        "export_btn": "💾 Export Report",
        "result_title": "Prediction Result",
        "confidence": "Confidence Interval",
        "suggestion": "Recommendation",
        "material_advice": "Material Advice",
        "prediction_history": "Prediction History",
        "model_comparison": "Model Comparison",
        "feature_importance": "Feature Importance",
        "data_overview": "Data Overview",
        "distribution": "Distribution Analysis",
        "correlation": "Correlation Analysis",
    },
}
T = I18N[lang]

# ----------------------
# 模型初始化（缓存）
# ----------------------
@st.cache_resource
def get_predictor():
    return CorrosionPredictor()

@st.cache_resource
def get_rag():
    return CorrosionRAG()

@st.cache_data
def get_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "corrosion_dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    from data_processor import generate_corrosion_data
    return generate_corrosion_data(n_samples=500)

@st.cache_data
def get_model_comparison():
    p = CorrosionPredictor()
    return p.train_multiple_models()

predictor = get_predictor()
rag = get_rag()
dataset = get_dataset()

MATERIAL_CHOICES = {
    "碳钢 (Carbon Steel)": "carbon_steel",
    "316不锈钢 (316 SS)": "stainless_316",
    "13Cr马氏体不锈钢 (13Cr)": "13cr",
    "超级13Cr (Super 13Cr)": "super_13cr",
    "2205双相不锈钢 (Duplex 2205)": "duplex_2205",
    "2507超级双相不锈钢 (Super Duplex 2507)": "duplex_2507",
    "825合金 (Alloy 825)": "alloy_825",
    "625合金 (Inconel 625)": "alloy_625",
    "C-276合金 (Hastelloy C-276)": "alloy_c276",
    "钛合金 (Titanium Gr.2)": "titanium",
}
MATERIAL_CHOICES_REVERSE = {v: k for k, v in MATERIAL_CHOICES.items()}

# 管线类型 → 介质（用于后果分析释放说明）
PIPELINE_PRODUCT = {
    "gas_transmission": "天然气", "city_gas": "天然气", "hydrogen": "天然气",
    "oil_transmission": "原油", "gathering": "原油", "subsea": "原油",
    "water_injection": "注水", "water_supply": "海水",
    "chemical": "化工介质", "sour_gas": "天然气",
}
# 风险等级 → 维护建议严重度
RISK_TO_SEVERITY = {"低风险": "低", "中风险": "中", "高风险": "高", "严重风险": "严重"}

RISK_STYLES = {
    "低风险": ("#27ae60", "🟢"),
    "中风险": ("#f39c12", "🟡"),
    "高风险": ("#e74c3c", "🔴"),
    "严重风险": ("#c0392b", "🔴"),
}

# ----------------------
# 通用：渲染多环境腐蚀模型结果卡片
# ----------------------
def _show_env_result(label, res, *pairs):
    """渲染多环境腐蚀模型结果。pairs: (结果字典key, 显示名) 元组列表。"""
    st.markdown(f"#### {label}估算结果")
    if pairs:
        cols = st.columns(len(pairs))
        for i, (k, name) in enumerate(pairs):
            cols[i].metric(name, str(res.get(k)))

    color = res.get("color", "#333")
    severity = res.get("severity") or res.get("risk") or res.get("corrosivity") or res.get("level")
    if severity:
        st.markdown(
            f'<div style="background:{color}15; border:2px solid {color}; border-radius:10px; padding:12px; margin:10px 0; text-align:center;">'
            f'<span style="font-weight:700; color:{color}; font-size:1.1rem;">{severity}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    detail = res.get("detail") or res.get("factors")
    if detail:
        with st.expander("📋 评分明细"):
            for dk, dv in detail.items():
                st.markdown(f"- {dk}: {dv}")
    if "advice" in res:
        st.caption("💡 " + res["advice"])


# ----------------------
# 页头
# ----------------------
st.markdown(f'<div class="main-title">{T["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{T["subtitle"]}</div>', unsafe_allow_html=True)

# 分享链接
if st.query_params.get("shared") == "1":
    st.success("📎 分享链接已加载预设参数")

# ----------------------
# 选项卡
# ----------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(T["tabs"])

# ======================
# Tab 1: 腐蚀预测 (增强版)
# ======================
with tab1:
    col_input, col_result = st.columns(2)

    with col_input:
        st.markdown(f"### {T['input_params']}")

        # 管线类型预设（联动典型工况与主导威胁）
        presets = get_pipeline_presets()
        ptype_label = st.selectbox("管线类型 / Pipeline Type", options=list(presets.keys()),
                                   format_func=lambda k: presets[k], key="pt_type")
        ptype = ptype_label
        preset = get_preset(ptype)

        if st.button("📋 套用该类型典型工况", width="stretch", key="apply_preset"):
            ps = preset["env"]
            st.session_state["pt_temp"] = int(round(ps["temp"]))
            st.session_state["pt_ph"] = float(ps["ph"])
            st.session_state["pt_co2"] = float(ps["co2_pressure"])
            st.session_state["pt_h2s"] = int(round(ps["h2s"]))
            st.session_state["pt_flow"] = float(ps["flow"])
            st.session_state["pt_cl"] = int(round(ps["chloride"]))
            dm = MATERIAL_CHOICES_REVERSE.get(preset["default_material"])
            if dm:
                st.session_state["pt_mat"] = dm
            st.toast(f"已套用「{preset['label']}」典型工况，主导威胁：{('、'.join(preset['dominant_threats']))}")

        material_label = st.selectbox("管材类型 / Material", options=list(MATERIAL_CHOICES.keys()),
                                      key="pt_mat")
        material = MATERIAL_CHOICES[material_label]

        temperature = st.slider("温度 Temperature (°C)", 0, 150, 80, step=1, key="pt_temp")
        ph = st.slider("pH 值", 3.0, 10.0, 6.0, step=0.1, key="pt_ph")
        co2_pressure = st.slider("CO2 分压 (MPa)", 0.0, 10.0, 1.0, step=0.1, key="pt_co2")
        h2s_concentration = st.slider("H2S 浓度 (ppm)", 0, 1000, 50, step=10, key="pt_h2s")
        flow_rate = st.slider("流速 Flow Rate (m/s)", 0.0, 10.0, 3.0, step=0.1, key="pt_flow")
        chloride_content = st.slider("氯离子 Cl- (ppm)", 0, 100000, 5000, step=500, key="pt_cl")

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            predict_btn = st.button(T["predict_btn"], type="primary", width="stretch")
        with col_btn2:
            compare_btn = st.button(T["compare_btn"], width="stretch")

        col_btn3, col_btn4 = st.columns([1, 1])
        with col_btn3:
            trend_btn = st.button(T["trend_btn"], width="stretch")
        with col_btn4:
            share_btn = st.button("🔗 分享", width="stretch")

    with col_result:
        st.markdown(f"### {T['result_title']}")

        if predict_btn:
            result = predictor.predict_with_confidence(
                material=material,
                temperature=float(temperature),
                ph=float(ph),
                co2_pressure=float(co2_pressure),
                h2s_concentration=float(h2s_concentration),
                flow_rate=float(flow_rate),
                chloride_content=float(chloride_content),
            )

            color, icon = RISK_STYLES.get(result["risk_level"], ("#333", "⚪"))
            rate = result["corrosion_rate"]
            gauge_pct = min(rate / 2.0 * 100, 100)

            # 仪表盘
            st.markdown(f"""
            <div style="background: {'#1a1a2e' if dark_mode else '#f0f2f6'}; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                <div style="text-align: center; margin-bottom: 10px;">
                    <span style="font-size: 1.1rem; color: {'#aaa' if dark_mode else '#666'};">腐蚀速率 Corrosion Rate</span>
                </div>
                <div style="background: {'#333' if dark_mode else '#ddd'}; border-radius: 10px; height: 30px; overflow: hidden; position: relative;">
                    <div style="background: linear-gradient(90deg, #27ae60 0%, #f39c12 40%, #e74c3c 70%, #c0392b 100%); height: 100%; width: {gauge_pct}%; border-radius: 10px; transition: width 0.5s;"></div>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: 700; color: #fff; font-size: 1.2rem;">{rate:.2f} mm/a</div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.75rem; color: {'#888' if dark_mode else '#999'};">
                    <span>0 (安全)</span><span>0.5</span><span>1.0</span><span>2.0+ (严重)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 置信区间
            st.markdown(f"""
            <div style="background: {'#1a1a2e' if dark_mode else '#eef'}; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                <span style="font-size: 0.85rem; color: {'#aaa' if dark_mode else '#666'};">📊 {T['confidence']} (95%)</span><br>
                <span style="font-size: 1.1rem; font-weight: 700; color: {color};">
                    [{result['confidence_lower']:.3f} — {result['confidence_upper']:.3f}] mm/a
                </span>
                <span style="font-size: 0.8rem; color: {'#888' if dark_mode else '#999'};"> (±{result['mae'] if 'mae' in result else predictor.mae:.3f})</span>
            </div>
            """, unsafe_allow_html=True)

            # 风险等级 + 材料卡片
            col_risk, col_mat = st.columns(2)
            with col_risk:
                st.markdown(
                    f'<div style="background:{color}15; border: 2px solid {color}; border-radius: 10px; padding: 12px; text-align: center;">'
                    f'<span style="font-size: 1.5rem;">{icon}</span><br>'
                    f'<span style="font-weight: 700; color: {color}; font-size: 1.1rem;">{result["risk_level"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_mat:
                st.metric("管材 Material", result["material_label"])

            st.markdown(f"#### 📋 {T['suggestion']}")
            st.info(result["suggestion"])
            st.markdown(f"**🔧 {T['material_advice']}**: {result['material_advice']}")

            # ===== 后果分析与维护建议（依据管线类型 + 预测结果）=====
            with st.expander("🎯 后果分析与维护建议", expanded=False):
                st.caption(f"管线类型：**{preset['label']}** ｜ 主导威胁：{('、'.join(preset['dominant_threats']))} ｜ 介质危险：{preset['product_hazard']}")
                cc1, cc2, cc3, cc4 = st.columns(4)
                dia = cc1.slider("管径 (mm)", 50, 1400, 500, 10, key="cons_dia")
                pres = cc2.slider("操作压力 (MPa)", 0.1, 15.0, 6.0, 0.1, key="cons_pres")
                loc = cc3.selectbox("位置类型", ["一般区域", "人口密集区(HCA)", "水体/环境敏感区", "荒野"], key="cons_loc")
                wloss = cc4.slider("最大壁损 (%)", 0, 100, 20, 1, key="cons_wloss")

                sev = RISK_TO_SEVERITY.get(result["risk_level"], "中")
                product = PIPELINE_PRODUCT.get(ptype, "天然气")
                cons = consequence_analysis(ptype, dia, pres, product, loc, wloss)

                st.markdown(
                    f"**后果等级：** <span style='color:{cons['color']};font-weight:700;'>【{cons['level']}】</span> （综合评分 {cons['score']}）",
                    unsafe_allow_html=True,
                )
                st.info(cons["summary"] + "  " + cons["release_note"])

                threat_opts = list(dict.fromkeys(
                    preset["dominant_threats"] + ["CO₂内腐蚀", "H₂S开裂", "外部腐蚀", "SCC", "MIC", "冲蚀", "电偶腐蚀"]
                ))
                threat = st.selectbox("选择需给出维护建议的威胁", threat_opts, key="rem_threat")
                rem = recommend_remediation(threat, sev, ptype, wloss, material)
                st.markdown(f"**优先级：** `{rem['priority']}` ｜ 威胁：{rem['threat']}")
                st.warning(rem["b31g_note"])
                st.markdown("**🚑 立即措施：**")
                for it in rem["immediate"]:
                    st.markdown(f"- {it}")
                st.markdown(f"**🛠️ 工程修复（{sev}级）：** {rem['repair']}")
                st.markdown(f"**🔎 监测方案：** {rem['monitor']}")
                st.markdown(f"**📚 标准依据：** {rem['standard']}")
                st.caption(rem["pipeline_context"] + "  " + rem["reference"])

            # 导出按钮
            export_df = pd.DataFrame([{
                "材料": result["material_label"],
                "温度(°C)": temperature,
                "pH": ph,
                "CO2分压(MPa)": co2_pressure,
                "H2S(ppm)": h2s_concentration,
                "流速(m/s)": flow_rate,
                "氯离子(ppm)": chloride_content,
                "腐蚀速率(mm/a)": rate,
                "置信下限": result["confidence_lower"],
                "置信上限": result["confidence_upper"],
                "风险等级": result["risk_level"],
            }])
            st.dataframe(export_df, width="stretch", hide_index=True)

            csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "💾 下载 CSV 报告", data=csv_data,
                file_name=f"corrosion_report_{int(time.time())}.csv",
                mime="text/csv", width="stretch",
            )

            # ===== 检测周期推荐（基于NACE SP0775 / API 570）=====
            st.markdown("---")
            st.markdown("#### 📅 检测周期推荐")

            inspection_intervals = {
                "低风险": {"ili": "10-14 年", "ecda": "7-10 年", "cp": "5-7 年", "action": "常规监测周期"},
                "中风险": {"ili": "7-10 年", "ecda": "5-7 年", "cp": "3-5 年", "action": "加强监测，增加缓蚀剂"},
                "高风险": {"ili": "5-7 年", "ecda": "3-5 年", "cp": "1-2 年", "action": "评估剩余强度，制定维修计划"},
                "严重风险": {"ili": "3-5 年或立即评估", "ecda": "1-3 年", "cp": "6-12 个月", "action": "立即降压运行或维修"},
            }
            interval = inspection_intervals.get(result["risk_level"], inspection_intervals["低风险"])

            col_ili, col_ecda, col_cp = st.columns(3)
            col_ili.metric("🔧 内检测(ILI)", interval["ili"])
            col_ecda.metric("🔍 外腐蚀评估(ECDA)", interval["ecda"])
            col_cp.metric("⚡ 阴保监测(CP)", interval["cp"])
            st.caption(f"📋 基于 NACE SP0775 风险分类标准 | 建议措施：{interval['action']}")

            # ===== 剩余寿命预测器 =====
            st.markdown("#### ⏳ 剩余寿命评估")
            st.markdown("输入管道壁厚参数，计算管道剩余服役寿命：")

            col_life1, col_life2, col_life3 = st.columns(3)
            with col_life1:
                nominal_wt = st.number_input("公称壁厚 (mm)", min_value=3.0, max_value=50.0, value=8.0, step=0.5, key="nominal_wt")
            with col_life2:
                measured_wt = st.number_input("实测壁厚 (mm)", min_value=1.0, max_value=50.0, value=7.2, step=0.1, key="measured_wt")
            with col_life3:
                min_wt = st.number_input("最小允许壁厚 (mm)", min_value=1.0, max_value=30.0, value=4.0, step=0.5, key="min_wt",
                                         help="通常取公称壁厚的40%-50%，或按ASME B31G计算")

            if st.button("🔢 计算剩余寿命", key="calc_life", width="stretch"):
                if rate > 0.001:
                    remaining_wall = measured_wt - min_wt
                    if remaining_wall > 0:
                        remaining_life = remaining_wall / rate
                        corrosion_margin = nominal_wt - measured_wt
                        years_since_last = corrosion_margin / rate if rate > 0 else 0

                        if remaining_life > 10:
                            life_color, life_icon, life_msg = "#27ae60", "🟢", f"剩余寿命充足（{remaining_life:.1f} 年），按常规周期监测"
                        elif remaining_life > 5:
                            life_color, life_icon, life_msg = "#f39c12", "🟡", f"剩余寿命中等（{remaining_life:.1f} 年），制定中期维修计划"
                        elif remaining_life > 1:
                            life_color, life_icon, life_msg = "#e74c3c", "🔴", f"剩余寿命不足（{remaining_life:.1f} 年），优先安排维修/更换"
                        else:
                            life_color, life_icon, life_msg = "#c0392b", "🔴", f"⚠️ 管道已临近失效（剩余 {remaining_life:.1f} 年），需立即维修或降压运行"

                        st.markdown(f"""
                        <div style="background: {life_color}15; border: 2px solid {life_color}; border-radius: 10px; padding: 16px; margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 1.3rem;">{life_icon}</span>
                                <span style="font-size: 1.8rem; font-weight: 700; color: {life_color};">{remaining_life:.1f} 年</span>
                            </div>
                            <div style="margin-top: 8px; font-size: 0.9rem;">{life_msg}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        col_detail1, col_detail2 = st.columns(2)
                        col_detail1.metric("已腐蚀壁厚", f"{nominal_wt - measured_wt:.1f} mm")
                        col_detail2.metric("可用壁厚余量", f"{remaining_wall:.1f} mm")

                        if years_since_last > 0:
                            st.caption(f"📌 推算距上次检测约 {years_since_last:.1f} 年 | 预计下次检测时间：{interval['ili']}")
                    else:
                        st.error(f"⚠️ 实测壁厚({measured_wt:.1f}mm)已低于最小允许壁厚({min_wt:.1f}mm)，管道不满足安全运行条件，需立即维修或更换！")
                else:
                    st.success("✅ 腐蚀速率极低（<0.001 mm/a），剩余寿命可视为无限期，按常规周期监测即可。")

            # 保存到预测历史
            if "prediction_history" not in st.session_state:
                st.session_state["prediction_history"] = []
            st.session_state["prediction_history"].append({
                "material": result["material_label"],
                "rate": rate,
                "risk": result["risk_level"],
                "temp": temperature,
                "ph": ph,
                "co2": co2_pressure,
            })

        elif compare_btn:
            st.markdown("#### 📊 材料对比（相同工况）")
            compare_results = []
            for mat_label, mat_code in MATERIAL_CHOICES.items():
                r = predictor.predict(
                    material=mat_code,
                    temperature=float(temperature),
                    ph=float(ph),
                    co2_pressure=float(co2_pressure),
                    h2s_concentration=float(h2s_concentration),
                    flow_rate=float(flow_rate),
                    chloride_content=float(chloride_content),
                )
                short_name = mat_label.split(" / ")[0]
                compare_results.append({"材料": short_name, "腐蚀速率(mm/a)": r["corrosion_rate"], "风险": r["risk_level"]})

            # Plotly 柱状图
            fig = px.bar(
                pd.DataFrame(compare_results),
                x="材料", y="腐蚀速率(mm/a)", color="风险",
                color_discrete_map={"低风险": "#27ae60", "中风险": "#f39c12", "高风险": "#e74c3c", "严重风险": "#c0392b"},
                text="腐蚀速率(mm/a)",
                template="plotly_dark" if dark_mode else "plotly_white",
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, width="stretch")

        elif trend_btn:
            st.markdown("#### 📈 趋势分析")
            trend_param = st.selectbox("选择分析参数", ["温度 temperature", "pH值", "CO2分压", "流速 flow_rate"])
            param_map = {"温度 temperature": "temperature", "pH值": "ph", "CO2分压": "co2_pressure", "流速 flow_rate": "flow_rate"}
            param = param_map[trend_param]

            trend_data = predictor.get_trend_data(
                param, material, float(temperature), float(ph),
                float(co2_pressure), float(h2s_concentration),
                float(flow_rate), float(chloride_content),
            )

            if trend_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_data["param_values"], y=trend_data["corrosion_rates"],
                    mode="lines+markers", name="腐蚀速率",
                    line=dict(color="#e74c3c", width=3),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
                ))
                # 标记当前值
                current_val = {"temperature": temperature, "ph": ph, "co2_pressure": co2_pressure, "flow_rate": flow_rate}[param]
                fig.add_vline(x=current_val, line_dash="dash", line_color="blue", annotation_text="当前值")
                fig.update_layout(
                    xaxis_title=trend_data["param_label"],
                    yaxis_title="腐蚀速率 (mm/a)",
                    height=400,
                    template="plotly_dark" if dark_mode else "plotly_white",
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(f"📌 其他参数保持不变：{material_label}，pH={ph}，CO2={co2_pressure}MPa，流速={flow_rate}m/s")

        elif share_btn:
            # 生成分享链接
            params = {
                "shared": "1",
                "mat": material,
                "t": temperature,
                "ph": ph,
                "co2": co2_pressure,
                "h2s": h2s_concentration,
                "fr": flow_rate,
                "cl": chloride_content,
            }
            st.query_params.update(params)
            st.success("✅ 分享链接已生成！复制浏览器地址栏的 URL 即可分享当前参数配置。")

        else:
            st.info("👈 调整左侧参数后，点击按钮查看结果。\n\n**可用的分析模式**：\n- 🔍 单次预测（含置信区间）\n- 📊 材料对比\n- 📈 趋势分析\n- 🔗 分享参数")

    # 预测历史
    if "prediction_history" in st.session_state and st.session_state["prediction_history"]:
        with st.expander(f"📋 {T['prediction_history']}（{len(st.session_state['prediction_history'])} 条）", expanded=False):
            hist_df = pd.DataFrame(st.session_state["prediction_history"][-10:])
            st.dataframe(hist_df, width="stretch", hide_index=True)

    # 批量预测
    with st.expander(f"📋 {T['batch_btn']} - 上传 CSV 文件"):
        st.markdown("""
        **CSV 格式要求**：包含以下列（列名不限顺序）：
        `material, temperature, ph, co2_pressure, h2s_concentration, flow_rate, chloride_content`

        material 可选值：`carbon_steel`, `stainless_316`, `alloy_825`, `duplex_2205`
        """)
        uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"], key="batch_upload")
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                required_cols = ["material", "temperature", "ph", "co2_pressure", "h2s_concentration", "flow_rate", "chloride_content"]
                if not all(col in batch_df.columns for col in required_cols):
                    st.error(f"❌ CSV 缺少必要列。需要：{', '.join(required_cols)}")
                else:
                    results = []
                    progress = st.progress(0, "正在预测...")
                    for i, row in batch_df.iterrows():
                        r = predictor.predict(
                            material=row["material"],
                            temperature=float(row["temperature"]),
                            ph=float(row["ph"]),
                            co2_pressure=float(row["co2_pressure"]),
                            h2s_concentration=float(row["h2s_concentration"]),
                            flow_rate=float(row["flow_rate"]),
                            chloride_content=float(row["chloride_content"]),
                        )
                        results.append({
                            "材料": row["material"],
                            "温度": row["temperature"],
                            "pH": row["ph"],
                            "CO2": row["co2_pressure"],
                            "腐蚀速率(mm/a)": r["corrosion_rate"],
                            "风险等级": r["risk_level"],
                        })
                        progress.progress((i + 1) / len(batch_df), f"预测中... {i+1}/{len(batch_df)}")
                    progress.empty()

                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df, width="stretch", hide_index=True)

                    csv_batch = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "💾 下载批量预测结果", data=csv_batch,
                        file_name=f"batch_prediction_{int(time.time())}.csv",
                        mime="text/csv", width="stretch",
                    )
            except Exception as e:
                st.error(f"❌ 文件解析失败：{e}")

    st.markdown("""
    ---
    **📌 典型工况参考**:
    - 🔴 高腐蚀风险：碳钢 + 80°C + pH 5.5 + CO2 1.5 MPa + H2S 100 ppm
    - 🟢 低腐蚀风险：316不锈钢 + 25°C + pH 7.5 + CO2 0.1 MPa + H2S 0 ppm
    """)

# ======================
# Tab 2: 标准问答 (保持不变)
# ======================
with tab2:
    st.markdown("### 💬 腐蚀标准知识问答")

    mode_labels = {
        "dify": ("🟢 Dify API 智能问答模式", "已连接 Dify Cloud，支持自然语言智能问答"),
        "local": ("🟡 本地向量检索模式", "使用 LangChain + ChromaDB 本地检索"),
        "fallback": ("⚪ 基础模式", "配置 Dify API 后可获得更强问答能力"),
    }
    mode_label, mode_desc = mode_labels.get(rag.mode, mode_labels["fallback"])
    st.caption(f"{mode_label} — {mode_desc}")

    st.markdown("#### 💡 试试这些问题：")
    example_questions = [
        "在什么条件下需要使用 NACE MR0175 规定的抗硫材料？",
        "CO2腐蚀的机理和影响因素是什么？",
        "管道腐蚀速率的风险等级如何划分？",
        "阴极保护的最小保护电位是多少？",
        "ASME B31G 如何评估腐蚀缺陷的剩余强度？",
        "去哪里下载美国 PHMSA 管道事故公开数据？",
        "de Waard-Milliams 的 CO₂ 腐蚀模型公式是什么？",
        "GB/T 23258 对三层PE防腐层厚度有什么要求？",
    ]
    cols = st.columns(len(example_questions))
    for i, (col, q) in enumerate(zip(cols, example_questions)):
        if col.button(q, key=f"example_{i}", width="stretch"):
            st.session_state["pending_question"] = q

    if hasattr(rag, "_cache") and len(rag._cache._cache) > 0:
        st.caption(f"📦 缓存: {len(rag._cache._cache)} 条常见问答（重复问题秒回）")

    with st.expander("📚 问答知识来源（含文献与公开数据）"):
        st.markdown("""
        本系统标准问答的知识来源覆盖**国际标准 + 中国标准 + 研究文献/公开数据**三大类：

        **🇨🇳 中国标准条款库**
        - `chinese_standards_kb.md` — 8 项国标/行标综述（GB/T 23258、SY/T 0087、GB 50251、SY/T 6648、GB/T 21447、GB/T 30582、SY/T 0036）
        - `china_standards_clauses.md`（新增）— 上述标准的**条款级阈值**（防腐层厚度、阴极保护电位、评价等级、设计系数等）

        **🌐 研究资料与公开数据**
        - `research_references.md` — 管线腐蚀 7 大痛点、**PHMSA/PRCI/EGIG/CONCAWE/NTSB/NIST/NETL 等公开数据库链接**、关键论文与标准引用（de Waard 系列、NORSOK M-506、API RP 14E、NACE MR0175、NACE SP0204 等）

        **ℹ️ 关于覆盖率**
        - **本地向量检索模式**：`data/standards/` 下所有 `.md` 自动入库，文献与数据来源问答**已覆盖**。
        - **Dify Cloud 模式（本系统线上默认）**：需将 `research_references.md` 与 `china_standards_clauses.md` 上传至 Dify 知识库后，线上问答才能覆盖文献/细分条款。操作步骤：Dify 控制台 → 知识库 → 创建/选择本应用知识库 → 导入这两个文件 → 重新发布。
        """)

    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    question = st.chat_input("输入你的问题，例如：在什么条件下需要使用抗硫材料？")

    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)
        with chat_container:
            with st.chat_message("assistant"):
                if rag.mode == "dify":
                    status_placeholder = st.empty()
                    status_placeholder.info("🔍 正在检索知识库...")
                    try:
                        status_placeholder.info("✍️ 正在生成回答...")
                        full_answer = st.write_stream(rag.query_stream(question))
                        status_placeholder.empty()
                    except Exception as e:
                        status_placeholder.empty()
                        full_answer = f"⚠️ 回答生成出错: {e}"
                        st.markdown(full_answer)
                else:
                    with st.spinner("正在检索知识库..."):
                        full_answer = rag.query(question)
                    st.markdown(full_answer)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": full_answer})

    col_clear, col_info = st.columns([1, 5])
    with col_clear:
        if st.session_state.messages and st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.rerun()

# ======================
# Tab 3: 数据探索 (新增)
# ======================
with tab3:
    st.markdown("### 📈 数据探索与模型分析")

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 数据概览", "📋 分布分析", "🔥 相关性分析", "🤖 模型对比"
    ])

    # --- 数据概览 ---
    with sub_tab1:
        st.markdown(f"#### {T['data_overview']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("样本总数", f"{len(dataset)}")
        col2.metric("特征数", "7")
        col3.metric("材料类型", "10 种")
        col4.metric("风险等级", "4 级")

        st.markdown("##### 数据预览")
        st.dataframe(dataset.head(20), width="stretch", hide_index=True)

        st.markdown("##### 描述性统计")
        st.dataframe(dataset.describe().round(3), width="stretch")

    # --- 分布分析 ---
    with sub_tab2:
        st.markdown(f"#### {T['distribution']}")

        dist_col = st.selectbox("选择分析列", [
            "corrosion_rate", "temperature", "ph", "co2_pressure",
            "h2s_concentration", "flow_rate", "chloride_content"
        ])

        col_hist, col_box = st.columns(2)

        with col_hist:
            fig_hist = px.histogram(
                dataset, x=dist_col, nbins=40,
                title=f"{dist_col} 分布直方图",
                template="plotly_dark" if dark_mode else "plotly_white",
                color_discrete_sequence=["#e74c3c"],
            )
            fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, width="stretch")

        with col_box:
            fig_box = px.box(
                dataset, x="material", y=dist_col,
                title="按材料分组箱线图",
                template="plotly_dark" if dark_mode else "plotly_white",
                color="material",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_box.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_box, width="stretch")

    # --- 相关性分析 ---
    with sub_tab3:
        st.markdown(f"#### {T['correlation']}")

        corr_cols = ["temperature", "ph", "co2_pressure", "h2s_concentration",
                     "flow_rate", "chloride_content", "corrosion_rate"]
        corr_matrix = dataset[corr_cols].corr()

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu_r",
            zmid=0,
            text=corr_matrix.values.round(3),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig_heatmap.update_layout(
            title="特征相关性热力图",
            height=450,
            template="plotly_dark" if dark_mode else "plotly_white",
            margin=dict(l=60, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_heatmap, width="stretch")

        # 特征重要性
        st.markdown(f"##### {T['feature_importance']}")
        importance = predictor.get_feature_importance()
        if importance:
            imp_df = pd.DataFrame(importance)
            fig_imp = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                title="模型特征重要性 (GradientBoosting)",
                template="plotly_dark" if dark_mode else "plotly_white",
                color="importance", color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_imp, width="stretch")

    # --- 模型对比 ---
    with sub_tab4:
        st.markdown(f"#### {T['model_comparison']}")
        st.caption("训练 7+ 种算法（GBR/RF/DT/LR/MLP/SVR/XGBoost/投票集成）对比性能指标，展示模型选型决策过程")

        comparison = get_model_comparison()

        comp_data = []
        for name, metrics in comparison.items():
            comp_data.append({
                "模型": name,
                "R²": metrics["r2"],
                "MAE (mm/a)": metrics["mae"],
                "RMSE (mm/a)": metrics["rmse"],
            })
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, width="stretch", hide_index=True)

        # R² 对比图
        fig_r2 = px.bar(
            comp_df, x="模型", y="R²",
            title="R² 得分对比（越高越好）",
            template="plotly_dark" if dark_mode else "plotly_white",
            color="R²", color_continuous_scale="Blues",
            text="R²",
        )
        fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_r2.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_r2, width="stretch")

        # MAE 对比图
        fig_mae = px.bar(
            comp_df, x="模型", y="MAE (mm/a)",
            title="MAE 对比（越低越好）",
            template="plotly_dark" if dark_mode else "plotly_white",
            color="MAE (mm/a)", color_continuous_scale="Reds",
            text="MAE (mm/a)",
        )
        fig_mae.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_mae.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_mae, width="stretch")

        st.info("""
        **📌 选型决策**：GradientBoosting 与 VotingEnsemble 在 R² 和 MAE 上均表现最优，
        生产模型选用 GradientBoosting（R²≈0.89）。RandomForest 紧随其后；
        MLP/SVR 在非线性映射上有潜力但需更多数据与调参；LinearRegression 表现最差
        （腐蚀速率与特征间为强非线性关系）。XGBoost 在已安装环境下可用作对比基准。
        """)

# ======================
# Tab 7: 关于 (更新)
# ======================
with tab7:
    st.markdown("""
    ## 关于本系统

    本系统面向管道完整性管理场景，集成腐蚀速率预测与国际标准智能问答功能，
    为管道运维人员提供数据驱动的腐蚀风险评估与标准检索工具。

    ### 功能模块

    **1. 腐蚀预测模块**
    - 基于 500 条模拟腐蚀数据训练的 GradientBoosting 模型（R²=0.89）
    - 输入 7 个管道参数即可预测腐蚀速率和风险等级
    - 支持 10 种工业管材（碳钢/不锈钢/13Cr/双相钢/镍基合金/钛合金）
    - 95% 置信区间、趋势分析、材料对比、批量预测、CSV 导出

    **2. 完整性评估模块**
    - 剩余寿命预测：基于壁厚参数和腐蚀速率计算管道剩余服役寿命
    - 检测周期推荐：依据 NACE SP0775 风险分类推荐内检测/外检测/阴保监测周期

    **3. 完整性工具模块**
    - B31G 剩余强度计算：ASME B31G Level 1 公式，评估缺陷失效压力和剩余强度率
    - 缓蚀剂推荐：基于温度/流速/CO₂/H₂S/介质类型推荐缓蚀剂类型和注入浓度
    - 风险矩阵评估：5×5 概率×后果矩阵，定位管道综合风险等级

    **4. 标准问答模块**
    - Dify Cloud RAG 引擎 + 国际标准（NACE/API/ASME）+ 中国标准（GB/SY/T）知识库
    - 知识库新增**条款级细分**：`china_standards_clauses.md`（防腐层厚度/阴极保护电位/评价等级/设计系数等数值阈值）
    - 新增**文献与公开数据覆盖**：`research_references.md`（PHMSA/PRCI/EGIG 等数据库链接 + 关键论文与标准引用）
    - Streaming 流式响应（首字 2-3 秒）+ LRU 缓存
    - 三级降级策略：Dify API → 本地向量检索（自动收录 data/standards 全部 .md）→ 基础模式

    **5. 数据探索模块**
    - 数据集统计概览与预览
    - 分布分析（直方图 + 箱线图）
    - 相关性热力图 + 特征重要性
    - 多模型对比（GBR / RF / DT / LR / MLP / SVR / XGBoost / 投票集成，7+ 种）

    **6. 腐蚀环境分析模块**
    - 多环境腐蚀模型：土壤（DIN 50929 思路）/海水/微生物(MIC)/电偶腐蚀的简化速率估算
    - 腐蚀成本估算：基于管径/长度/壁厚/腐蚀速率 → 年度金属损失、检测、停产与维修成本
    - 腐蚀失效案例库：CO₂腐蚀/MIC/氯离子SCC/土壤外腐蚀/电偶腐蚀等典型失效案例

    **7. 机理与工程模型模块**
    - CO₂腐蚀(de Waard-Milliams / NORSOK M-506 思路)：温度、CO₂分压、pH 耦合的基础腐蚀速率与 pH 修正
    - 冲蚀临界流速(API RP 14E)与 Salama 含砂冲蚀速率估算
    - H₂S 环境开裂筛查(NACE MR0175 / ISO 15156)：SSC 严苛度区域与硬度上限
    - 应力腐蚀开裂(SCC)敏感性筛查(NACE SP0204)：高 pH / 近中性 pH 两类机理
    - 点蚀抗力当量 PREN：不锈钢/双相钢/镍基合金抗氯离子点蚀能力对比

    **8. 无损检测(NDT)与后果·维护模块**
    - NDT/ILI 知识库：11 种 NDT 方法（MFL/UT-CD/EMAT/Caliper/PAUT/RT/MT/PT/ET/导波/AE）原理/可检缺陷/灵敏度/标准
    - ILI 工具选型：依据主导威胁与管线可检性推荐 MFL/UT-CD 组合 + ECDA/ICDA/SCCDA 直接评估路径
    - API 1163 第3版 POD/POI/尺寸精度量化概念与"组合 run"最佳实践
    - 后果分析：依据管线类型/管径/压力/介质/位置/壁损评估泄漏后果等级
    - 分场景维护建议：覆盖 CO₂内腐蚀/H₂S开裂/外部腐蚀/SCC/MIC/冲蚀/电偶腐蚀 7 类威胁的立即措施·工程修复·监测·标准依据
    - 管线类型预设：10 类管线（长输/集输/注水/海底/城市燃气/化工/酸性气田/输水/掺氢等）联动典型工况与主导威胁

    **📚 研究资料与公开数据源**
    - 管线腐蚀痛点、PHMSA/PRCI/EGIG/CONCAWE/NTSB/NIST/NETL 等公开数据库、关键论文与标准引用见 `data/standards/research_references.md`
    - 中国标准条款级细分（防腐层厚度 / 阴极保护电位 / 评价等级 / 设计系数等数值阈值）见 `data/standards/china_standards_clauses.md`

    ### 技术栈

    | 组件 | 技术 |
    |------|------|
    | Web 界面 | Streamlit + Plotly |
    | 预测模型 | scikit-learn (GradientBoosting / RandomForest / DecisionTree / LinearRegression / MLP / SVR / VotingEnsemble) + 可选 XGBoost |
    | RAG 引擎 | Dify Cloud API (Streaming) + LangChain |
    | 部署平台 | Streamlit Community Cloud |
    | 版本控制 | GitHub |

    ### 联系方式

    GitHub: [项目仓库](https://github.com/phdleo101/pipeline-corrosion-ai)
    Demo: [在线体验](https://pipeline-corrosion-ai.streamlit.app/)

    ---
    *MIT License*
    """)

# ======================
# Tab 4: 完整性工具 (新增 P2)
# ======================
with tab4:
    st.markdown("### 🔧 管道完整性管理工具")

    tool_tab1, tool_tab2, tool_tab3 = st.tabs([
        "📐 B31G剩余强度", "🧪 缓蚀剂推荐", "🎯 风险矩阵"
    ])

    # --- B31G 剩余强度计算器 ---
    with tool_tab1:
        st.markdown("#### ASME B31G 腐蚀缺陷剩余强度评估")
        st.markdown("输入管道几何参数和缺陷尺寸，评估剩余强度和失效压力。")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            b_D = st.number_input("管径 D (mm)", min_value=50.0, max_value=2000.0, value=610.0, step=10.0, key="b_D")
            b_t = st.number_input("壁厚 t (mm)", min_value=2.0, max_value=50.0, value=12.0, step=0.5, key="b_t")
            b_sy = st.number_input("屈服强度 σy (MPa)", min_value=200.0, max_value=700.0, value=415.0, step=5.0, key="b_sy",
                                   help="常见管材：X52=360, X60=415, X65=450, X70=485 MPa")
        with col_b2:
            b_d = st.number_input("缺陷深度 d (mm)", min_value=0.1, max_value=50.0, value=4.0, step=0.1, key="b_d")
            b_L = st.number_input("缺陷长度 L (mm)", min_value=1.0, max_value=2000.0, value=100.0, step=5.0, key="b_L")
            b_P = st.number_input("操作压力 P (MPa，可选)", min_value=0.0, max_value=30.0, value=8.0, step=0.5, key="b_P")

        if st.button("🔢 计算剩余强度", key="b_calc", width="stretch"):
            if b_d >= b_t:
                st.error("⚠️ 缺陷深度不能超过壁厚！")
            else:
                res = b31g_calculate(b_D, b_t, b_d, b_L, b_sy, b_P if b_P > 0 else None)

                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                col_r1.metric("Folias因子 M", f"{res['M']:.2f}")
                col_r2.metric("流动应力 (MPa)", f"{res['sigma_f']:.0f}")
                col_r3.metric("失效压力 (MPa)", f"{res['Pf']:.2f}" if res['Pf'] != float('inf') else "∞")
                col_r4.metric("深度比 d/t", f"{res['dtr']:.2f}")

                if "RSF" in res:
                    st.markdown(f"**剩余强度率 RSF**: {res['RSF']:.2f} — {res['rsf_status']}")

                st.markdown(f"""
                <div style="background: {res['verdict_color']}15; border: 2px solid {res['verdict_color']}; border-radius: 10px; padding: 14px; margin: 10px 0; text-align: center;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: {res['verdict_color']};">{res['verdict']}</span>
                    <div style="margin-top: 6px; font-size: 0.85rem;">{res['verdict_msg']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.caption("📌 依据 ASME B31G Level 1 简化公式：Pf = 2t·σf·(1-d/t) / [D·(1-0.85d/(t·M))]，σf = 1.1×σy")

    # --- 缓蚀剂推荐 ---
    with tool_tab2:
        st.markdown("#### 缓蚀剂选型与加注推荐")
        st.markdown("基于工况条件推荐缓蚀剂类型、注入浓度和预期效果。")

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            i_temp = st.slider("温度 (°C)", 0, 150, 60, step=1, key="i_temp")
            i_flow = st.slider("流速 (m/s)", 0.0, 10.0, 2.0, step=0.1, key="i_flow")
        with col_i2:
            i_co2 = st.slider("CO₂分压 (MPa)", 0.0, 10.0, 1.0, step=0.1, key="i_co2")
            i_h2s = st.slider("H₂S浓度 (ppm)", 0, 1000, 20, step=10, key="i_h2s")
        i_medium = st.selectbox("介质类型", ["湿气", "干气", "产出水", "原油"], key="i_medium")

        if st.button("🔍 推荐缓蚀剂", key="i_rec", width="stretch"):
            rec = recommend_inhibitor(i_temp, i_flow, i_co2, i_h2s, i_medium)

            st.markdown(f"**推荐类型**: {rec['type']}")
            st.caption(f"💡 {rec['type_reason']}")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("连续注入浓度", f"{rec['injection_ppm']} ppm")
            col_b.metric("批处理浓度", f"{rec['batch_ppm']} ppm")
            col_c.metric("预期缓蚀率", rec['expected_efficiency'])

            st.markdown(f"**介质建议**: {rec['medium_advice']}")

            with st.expander("📋 注意事项"):
                for note in rec["notes"]:
                    st.markdown(f"- {note}")

    # --- 风险矩阵 ---
    with tool_tab3:
        st.markdown("#### 管道风险矩阵评估 (5×5)")
        st.markdown("基于腐蚀速率（失效概率）和管道参数（失效后果）评估综合风险等级。")

        col_x1, col_x2 = st.columns(2)
        with col_x1:
            x_rate = st.slider("腐蚀速率 (mm/a)", 0.0, 5.0, 0.5, step=0.05, key="x_rate")
            x_dia = st.number_input("管径 (mm)", min_value=50.0, max_value=2000.0, value=610.0, step=10.0, key="x_dia")
        with col_x2:
            x_pres = st.slider("操作压力 (MPa)", 0.0, 20.0, 8.0, step=0.5, key="x_pres")
            x_loc = st.selectbox("位置类型", ["人口密集区", "一般区域", "荒野"], key="x_loc")

        rm = risk_matrix(x_rate, x_dia, x_pres, x_loc)

        # 5×5 矩阵热力图
        labels = ["极低", "低", "中", "高", "极高"]
        # 风险等级数值矩阵（1=低 2=中 3=高 4=极高），y轴已反转，故矩阵同步反转
        risk_z = [
            [1, 1, 2, 3, 3],
            [1, 2, 2, 3, 4],
            [2, 2, 3, 3, 4],
            [3, 3, 3, 4, 4],
            [3, 4, 4, 4, 4],
        ][::-1]
        risk_text = [
            ["低风险", "低风险", "中风险", "高风险", "高风险"],
            ["低风险", "中风险", "中风险", "高风险", "极高风险"],
            ["中风险", "中风险", "高风险", "高风险", "极高风险"],
            ["高风险", "高风险", "高风险", "极高风险", "极高风险"],
            ["高风险", "极高风险", "极高风险", "极高风险", "极高风险"],
        ][::-1]
        risk_colorscale = [
            [0.0, "#27ae60"], [0.25, "#27ae60"],   # 低风险
            [0.25, "#f39c12"], [0.50, "#f39c12"],   # 中风险
            [0.50, "#e74c3c"], [0.75, "#e74c3c"],   # 高风险
            [0.75, "#c0392b"], [1.00, "#c0392b"],   # 极高风险
        ]
        fig_rm = go.Figure(data=go.Heatmap(
            z=risk_z,
            x=labels,
            y=labels[::-1],
            colorscale=risk_colorscale,
            showscale=False,
            text=risk_text,
            texttemplate="%{text}",
            hoverongaps=False,
            zmin=1,
            zmax=4,
        ))
        # 标注当前位置（y轴反转后，prob_idx 需镜像）
        fig_rm.add_trace(go.Scatter(
            x=[labels[rm["cons_idx"]]],
            y=[labels[::-1][4 - rm["prob_idx"]]],
            mode="markers+text",
            marker=dict(size=18, color="white", symbol="circle", line=dict(width=3, color="black")),
            text="●",
            textfont=dict(size=16, color="black"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_rm.update_layout(
            height=380,
            margin=dict(l=60, r=20, t=20, b=60),
            xaxis_title="失效后果 →",
            yaxis_title="↑ 失效概率",
            template="plotly_dark" if dark_mode else "plotly_white",
        )
        st.plotly_chart(fig_rm, width="stretch")

        col_p, col_c, col_r = st.columns(3)
        col_p.metric("失效概率", rm["prob_level"])
        col_c.metric("失效后果", rm["cons_level"])
        col_r.markdown(f"""
        <div style="background: {rm['risk_color']}15; border: 2px solid {rm['risk_color']}; border-radius: 8px; padding: 8px; text-align: center;">
            <span style="font-weight: 700; color: {rm['risk_color']}; font-size: 0.95rem;">{rm['risk_level']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.caption("📌 概率等级由腐蚀速率决定，后果等级由管径/压力/位置综合评分决定。白圈标注当前管道风险位置。")

# ======================
# Tab 5: 腐蚀环境分析 (新增 P3)
# ======================
with tab5:
    st.markdown("### 🌍 多环境腐蚀分析与成本评估")

    env_tab1, env_tab2, env_tab3 = st.tabs([
        "🌱 多环境腐蚀模型", "💰 腐蚀成本估算", "📚 腐蚀案例库"
    ])

    # --- 多环境腐蚀模型 ---
    with env_tab1:
        st.markdown("#### 多环境腐蚀速率估算")
        st.markdown("选择腐蚀环境类型，输入关键参数估算腐蚀速率与严重程度。公式为工程简化估算，正式评估以现场检测为准。")

        # 管线类型上下文（聚焦主导环境威胁）
        ptype_ctx = st.selectbox("关联管线类型（用于聚焦主导环境威胁）",
                                 list(get_pipeline_presets().keys()),
                                 format_func=lambda k: get_pipeline_presets()[k], key="env_ptype")
        pctx = get_preset(ptype_ctx)
        st.info(f"「{pctx['label']}」主导腐蚀威胁：{('、'.join(pctx['dominant_threats']))}。{pctx['typical']}")

        env_type = st.selectbox("环境类型", ["土壤腐蚀", "海水腐蚀", "微生物腐蚀(MIC)", "电偶腐蚀"], key="env_type")

        if env_type == "土壤腐蚀":
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                s_res = st.number_input("土壤电阻率 (Ω·cm)", min_value=10.0, max_value=20000.0, value=1000.0, step=100.0, key="s_res")
                s_ph = st.number_input("土壤 pH", min_value=2.0, max_value=13.0, value=6.0, step=0.1, key="s_ph")
                s_mat = st.selectbox("管材", list(MATERIAL_CHOICES.keys()), key="s_mat")
            with col_s2:
                s_moist = st.number_input("含水率 (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0, key="s_moist")
                s_cl = st.number_input("氯离子 (ppm)", min_value=0.0, max_value=20000.0, value=200.0, step=50.0, key="s_cl")
                s_so4 = st.number_input("硫酸根 (ppm)", min_value=0.0, max_value=20000.0, value=500.0, step=50.0, key="s_so4")

            if st.button("🔢 计算土壤腐蚀", key="s_calc", width="stretch"):
                res = soil_corrosion(s_res, s_ph, s_moist, s_cl, s_so4, MATERIAL_CHOICES[s_mat])
                _show_env_result("土壤腐蚀", res, ("corrosivity", "腐蚀性分级"), ("rate", "腐蚀速率 (mm/a)"))

        elif env_type == "海水腐蚀":
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                w_ox = st.number_input("溶解氧 (mg/L)", min_value=0.0, max_value=14.0, value=6.0, step=0.5, key="w_ox")
                w_sal = st.number_input("盐度 (‰)", min_value=0.0, max_value=45.0, value=35.0, step=1.0, key="w_sal")
                w_mat = st.selectbox("管材", list(MATERIAL_CHOICES.keys()), key="w_mat")
            with col_w2:
                w_temp = st.number_input("温度 (°C)", min_value=-2.0, max_value=60.0, value=20.0, step=1.0, key="w_temp")
                w_flow = st.number_input("流速 (m/s)", min_value=0.0, max_value=15.0, value=1.5, step=0.5, key="w_flow")

            if st.button("🔢 计算海水腐蚀", key="w_calc", width="stretch"):
                res = seawater_corrosion(w_ox, w_sal, w_temp, w_flow, MATERIAL_CHOICES[w_mat])
                _show_env_result("海水腐蚀", res, ("rate", "腐蚀速率 (mm/a)"), ("severity", "严重程度"))
                if res.get("pitting_risk"):
                    st.markdown(f"**点蚀风险**: {res['pitting_risk']}（PREN≈{res['pren']}）")

        elif env_type == "微生物腐蚀(MIC)":
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                m_srb = st.number_input("SRB 数量 (MPN/mL)", min_value=1.0, max_value=1000000.0, value=10000.0, step=1000.0, key="m_srb")
                m_temp = st.number_input("温度 (°C)", min_value=0.0, max_value=80.0, value=30.0, step=1.0, key="m_temp")
                m_mat = st.selectbox("管材", list(MATERIAL_CHOICES.keys()), key="m_mat")
            with col_m2:
                m_nut = st.selectbox("营养物水平", ["低", "中", "高"], key="m_nut")
                m_ox = st.number_input("溶解氧 (mg/L)", min_value=0.0, max_value=10.0, value=1.0, step=0.5, key="m_ox")

            if st.button("🔢 评估 MIC 风险", key="m_calc", width="stretch"):
                res = mic_corrosion(m_srb, m_nut, m_temp, m_ox, MATERIAL_CHOICES[m_mat])
                _show_env_result("MIC", res, ("risk", "风险等级"), ("rate", "腐蚀速率 (mm/a)"))

            st.divider()
            st.markdown("##### 🦠 MIC-1 多菌属与生物膜热点")
            st.markdown("在 SRB 基础上纳入 APB(产酸)/IRB(铁氧化)/SOB(硫氧化)，并定位生物膜富集热点。")
            mm1, mm2 = st.columns(2)
            with mm1:
                mm_apb = st.number_input("APB 产酸菌 (MPN/mL)", min_value=1.0, max_value=1000000.0, value=1000.0, step=500.0, key="mm_apb")
                mm_irb = st.number_input("IRB 铁氧化菌 (MPN/mL)", min_value=1.0, max_value=1000000.0, value=1000.0, step=500.0, key="mm_irb")
                mm_sob = st.number_input("SOB 硫氧化菌 (MPN/mL)", min_value=1.0, max_value=1000000.0, value=1000.0, step=500.0, key="mm_sob")
            with mm2:
                mm_flow = st.selectbox("流速状态", ["低流速", "正常", "高流速"], key="mm_flow")
                mm_dead = st.checkbox("存在死管/滞留段", value=False, key="mm_dead")
            if st.button("🔢 多菌属评估", key="mm_calc2", width="stretch"):
                mo = mic_multi_organism(m_srb, mm_apb, mm_irb, mm_sob, m_nut, m_temp, m_ox,
                                        flow_regime=mm_flow, dead_leg=mm_dead, material=MATERIAL_CHOICES[m_mat])
                st.markdown(f"**多菌属 MIC 指数**: {mo['mic_index']} ｜ **风险**: {mo['risk']} ｜ **主导菌属**: {mo['dominant']}")
                st.markdown(f"**腐蚀速率(估)**: {mo['rate']} mm/a")
                st.markdown("**生物膜热点**:")
                for h in mo["hotspots"]:
                    st.markdown(f"- {h}")
                st.markdown(f"**建议**: {mo['advice']}")
                st.caption("📚 " + mo["reference"])

            st.divider()
            st.markdown("##### 💊 MIC-2 杀菌剂方案设计")
            mb1, mb2, mb3 = st.columns(3)
            with mb1:
                mb_risk = st.selectbox("MIC 风险等级", ["极低", "中等", "高", "极高"], index=1, key="mb_risk")
            with mb2:
                mb_sys = st.selectbox("系统类型", ["间歇系统", "连续系统"], key="mb_sys")
            with mb3:
                mb_temp = st.number_input("水温 (°C)", min_value=0.0, max_value=90.0, value=30.0, step=1.0, key="mb_temp")
            if st.button("🔢 生成杀菌剂方案", key="mb_calc", width="stretch"):
                bp = mic_biocide_program(mb_risk, system_type=mb_sys, water_temp=mb_temp)
                st.markdown(f"**推荐杀菌剂**: {bp['biocide_type']}")
                st.markdown(f"**投加方式**: {bp['dosing_mode']} ｜ **剂量**: {bp['dose']}")
                st.markdown(f"**轮换策略**: {bp['rotation']}")
                st.markdown(f"**监测**: {bp['monitoring']}")
                if bp["temp_note"]:
                    st.caption("🌡️ " + bp["temp_note"])
                st.caption("📚 " + bp["reference"])

            st.divider()
            st.markdown("##### 🔩 MIC-3 材料升级决策")
            st.markdown("基于 MIC 风险等级与当前材料，结合氯离子/温度给出升级路径（复用 PREN 思路）。")
            mu1, mu2, mu3 = st.columns(3)
            with mu1:
                mu_risk = st.selectbox("MIC 风险等级", ["极低", "中等", "高", "极高"], index=2, key="mu_risk")
                mu_mat = st.selectbox("当前材料", list(MATERIAL_CHOICES.keys()), index=0, key="mu_mat")
            with mu2:
                mu_cl = st.number_input("氯离子 (ppm)", min_value=0.0, max_value=40000.0, value=200.0, step=50.0, key="mu_cl")
                mu_temp = st.number_input("温度 (°C)", min_value=0.0, max_value=120.0, value=30.0, step=1.0, key="mu_temp")
            if st.button("🔢 生成材料升级建议", key="mu_calc", width="stretch"):
                up = mic_material_upgrade(mu_risk, current_material=MATERIAL_CHOICES[mu_mat],
                                          chloride_ppm=mu_cl, temperature=mu_temp)
                st.markdown(f"**当前材料**: {up['current_material']}（PREN≈{up['current_pren']}）")
                st.markdown(f"**建议升级至**: {up['recommended_material']}（PREN≈{up['recommended_pren']}）｜ 目标 PREN≥{up['target_pren']}")
                if up["escalation"]:
                    st.markdown("**升级阶梯**: " + " → ".join([f"{e['material']}({e['PREN']})" for e in up["escalation"]]))
                st.markdown(f"**结论**: {up['verdict']}")
                st.caption("📚 " + up["reference"])

            st.divider()
            st.markdown("##### 🔬 MIC-4 监测与再评估计划")
            st.markdown("基于 MIC 风险等级给出监测手段与再筛查周期（NACE SP0192 / API 570）。")
            mp1, mp2 = st.columns(2)
            with mp1:
                mp_risk = st.selectbox("MIC 风险等级", ["极低", "中等", "高", "极高"], index=2, key="mp_risk")
            with mp2:
                mp_sys = st.selectbox("系统类型", ["间歇系统", "连续系统"], key="mp_sys")
            if st.button("🔢 生成监测计划", key="mp_calc", width="stretch"):
                mp = mic_monitoring_plan(mp_risk, system_type=mp_sys)
                st.markdown(f"**再筛查周期**: {mp['re_screen_interval']} ｜ **在线建议**: {mp['online_recommend']}")
                st.markdown("**监测手段**:")
                for m in mp["methods"]:
                    st.markdown(f"- {m}")
                st.markdown(f"**建议**: {mp['note']}")
                st.caption("💡 " + mp["system_note"])
                st.caption("📚 " + mp["reference"])

            st.divider()
            st.markdown("##### 🤖 MIC 机器学习预测 (随机森林)")
            st.markdown("基于物理约束合成数据集训练的随机森林，从 12 项环境/材料/运行特征预测 MIC 风险等级与腐蚀速率，"
                        "并给出特征重要性。**模型用合成数据训练，正式评估须以现场检测为准。**")
            mic_m = get_trained_models()
            st.caption(f"📊 模型指标: 分类准确率 {mic_m['metrics']['accuracy']*100:.1f}% / 宏F1 {mic_m['metrics']['f1_macro']:.2f}；"
                       f"速率回归 R² {mic_m['metrics']['r2']:.2f} / MAE {mic_m['metrics']['mae']:.3f} mm/a（留出集 {mic_m['metrics']['n_test']} 样本）")
            f1c, f2c = st.columns(2)
            with f1c:
                ml_pH = st.slider("pH", 4.0, 9.0, DEFAULT_FEATURES["pH"], 0.1, key="ml_pH")
                ml_cl = st.number_input("氯离子 (ppm)", 10.0, 60000.0, DEFAULT_FEATURES["chloride_ppm"], 100.0, key="ml_cl")
                ml_srb = st.slider("SRB (log cells/mL)", 0.0, 7.0, DEFAULT_FEATURES["SRB_log"], 0.1, key="ml_srb")
                ml_apb = st.slider("APB (log cells/mL)", 0.0, 6.0, DEFAULT_FEATURES["APB_log"], 0.1, key="ml_apb")
                ml_irb = st.slider("IRB (log cells/mL)", 0.0, 6.0, DEFAULT_FEATURES["IRB_log"], 0.1, key="ml_irb")
                ml_o2 = st.slider("O₂ (ppm)", 0.0, 5.0, DEFAULT_FEATURES["O2_ppm"], 0.1, key="ml_o2")
            with f2c:
                ml_h2s = st.number_input("H₂S (ppm)", 0.0, 50.0, DEFAULT_FEATURES["H2S_ppm"], 0.5, key="ml_h2s")
                ml_temp = st.slider("温度 (°C)", 5.0, 80.0, DEFAULT_FEATURES["temperature"], 1.0, key="ml_temp")
                ml_flow = st.slider("流速 (m/s)", 0.0, 3.0, DEFAULT_FEATURES["flow_velocity"], 0.1, key="ml_flow")
                ml_water = st.slider("含水率 (%)", 0.0, 100.0, DEFAULT_FEATURES["water_cut"], 1.0, key="ml_water")
                ml_sulf = st.number_input("硫酸盐 (ppm)", 0.0, 3000.0, DEFAULT_FEATURES["sulfate_ppm"], 10.0, key="ml_sulf")
                ml_pren = st.slider("材料 PREN", 18.0, 45.0, DEFAULT_FEATURES["pren"], 0.5, key="ml_pren")
            if st.button("🔢 预测 MIC 风险", key="ml_pred", width="stretch"):
                feats = {
                    "pH": ml_pH, "chloride_ppm": ml_cl, "SRB_log": ml_srb, "APB_log": ml_apb,
                    "IRB_log": ml_irb, "O2_ppm": ml_o2, "H2S_ppm": ml_h2s, "temperature": ml_temp,
                    "flow_velocity": ml_flow, "water_cut": ml_water, "sulfate_ppm": ml_sulf, "pren": ml_pren,
                }
                pred = predict_mic_risk(feats)
                st.markdown(f"**MIC 风险等级: <span style='color:#e74c3c;font-weight:bold;font-size:1.3em'>{pred['risk_label']}</span>**"
                            f" ｜ 预测腐蚀速率 **{pred['predicted_rate']:.3f} mm/a**", unsafe_allow_html=True)
                fig_prob = go.Figure(go.Bar(
                    x=[MIC_RISK_LABELS[i] for i in range(4)],
                    y=[p*100 for p in pred["probabilities"]],
                    marker_color=["#27ae60", "#f39c12", "#e67e22", "#e74c3c"],
                    text=[f"{p*100:.0f}%" for p in pred["probabilities"]], textposition="auto"))
                fig_prob.update_layout(height=240, margin=dict(l=40, r=20, t=20, b=30),
                                      yaxis_title="概率 (%)", template="plotly_dark" if dark_mode else "plotly_white")
                st.plotly_chart(fig_prob, width="stretch")
                imp = mic_m["importances"]
                fig_imp = go.Figure(go.Bar(
                    x=[v for _, v in imp][::-1], y=[k for k, _ in imp][::-1], orientation="h",
                    marker_color="#3498db"))
                fig_imp.update_layout(height=360, margin=dict(l=120, r=20, t=20, b=30),
                                     xaxis_title="特征重要性", template="plotly_dark" if dark_mode else "plotly_white")
                st.plotly_chart(fig_imp, width="stretch")
                st.caption("📚 训练数据由 NACE SP0192 思路的'教师规则'生成并叠加噪声，用于演示 ML 流程；"
                           "接入真实检测数据可显著提升精度（见『实测数据标定』Tab）。")

        else:  # 电偶腐蚀
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                g_noble = st.selectbox("阴极性材料（贵金属）", list(MATERIAL_CHOICES.keys()), index=9, key="g_noble")
                g_active = st.selectbox("阳极性材料（活泼金属）", list(MATERIAL_CHOICES.keys()), index=0, key="g_active")
            with col_g2:
                g_ratio = st.number_input("阴极面积/阳极面积", min_value=0.1, max_value=100.0, value=5.0, step=0.5, key="g_ratio")
                g_elec = st.selectbox("介质", ["海水", "淡水", "土壤"], key="g_elec")

            if st.button("🔢 评估电偶腐蚀", key="g_calc", width="stretch"):
                res = galvanic_corrosion(MATERIAL_CHOICES[g_noble], MATERIAL_CHOICES[g_active], g_ratio, g_elec)
                _show_env_result("电偶腐蚀", res, ("level", "严重程度"), ("rate", "腐蚀速率 (mm/a)"))
                st.markdown(f"**电位差 ΔV**: {res['delta_e']} V ｜ **综合严重度**: {res['severity_index']}")

    # --- 腐蚀成本估算 ---
    with env_tab2:
        st.markdown("#### 腐蚀经济性评估")
        st.markdown("输入管道参数与腐蚀速率，估算年度腐蚀相关成本构成。")

        c1, c2, c3 = st.columns(3)
        with c1:
            c_dia = st.number_input("管径 (mm)", min_value=50.0, max_value=2000.0, value=610.0, step=10.0, key="c_dia")
            c_len = st.number_input("管线长度 (km)", min_value=0.1, max_value=5000.0, value=50.0, step=1.0, key="c_len")
            c_wt = st.number_input("壁厚 (mm)", min_value=2.0, max_value=50.0, value=12.0, step=0.5, key="c_wt")
        with c2:
            c_rate = st.number_input("腐蚀速率 (mm/a)", min_value=0.0, max_value=5.0, value=0.2, step=0.01, key="c_rate")
            c_price = st.number_input("管材单价 (¥/kg)", min_value=1.0, max_value=500.0, value=8.0, step=0.5, key="c_price")
            c_insp = st.number_input("单次检测费 (¥)", min_value=0.0, max_value=10000000.0, value=200000.0, step=10000.0, key="c_insp")
        with c3:
            c_down = st.number_input("单日停产损失 (¥/天)", min_value=0.0, max_value=100000000.0, value=500000.0, step=50000.0, key="c_down")
            c_freq = st.number_input("年检测频次", min_value=0.1, max_value=10.0, value=1.0, step=0.5, key="c_freq")
            c_remed = st.number_input("每米维修成本 (¥/m)", min_value=0.0, max_value=100000.0, value=0.0, step=500.0, key="c_remed")

        if st.button("💰 计算腐蚀成本", key="c_calc", width="stretch"):
            cost = corrosion_cost_estimate(c_dia, c_len, c_wt, c_rate, c_price, c_insp, c_down, c_freq, c_remed)

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("年度金属损失", f"{cost['annual_mass_kg']:.0f} kg")
            col_b.metric("预计剩余寿命", f"{cost['life_years']:.1f} 年")
            col_c.metric("年停产天数", f"{cost['downtime_days']:.1f} 天")
            col_d.metric("年度总腐蚀成本", f"¥{cost['total_cost']:,.0f}")

            bd = cost["breakdown"]
            fig_cost = go.Figure(data=[go.Pie(
                labels=list(bd.keys()), values=list(bd.values()),
                hole=0.4, textinfo="label+percent", textfont={"size": 12},
            )])
            fig_cost.update_layout(
                title="年度腐蚀成本构成",
                height=360, margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark" if dark_mode else "plotly_white",
            )
            st.plotly_chart(fig_cost, width="stretch")

            st.markdown("##### 成本明细")
            det_df = pd.DataFrame([{"项目": k, "年度成本(¥)": v} for k, v in bd.items()])
            st.dataframe(det_df, width="stretch", hide_index=True)
            st.caption(f"📌 估算基于管径 {c_dia:.0f}mm × {c_len:.0f}km × 腐蚀速率 {c_rate:.2f} mm/a；金属密度按 7850 kg/m³。结果为方案比选参考，非精确核算。")

    # --- 腐蚀案例库 ---
    with env_tab3:
        st.markdown("#### 腐蚀失效案例库")
        st.markdown("精选典型腐蚀失效模式，用于风险识别与对策参考。")

        case_path = os.path.join(os.path.dirname(__file__), "..", "data", "standards", "corrosion_case_library.md")
        if os.path.exists(case_path):
            with open(case_path, encoding="utf-8") as f:
                case_md = f.read()
            parts = re.split(r"(?m)^## ", case_md)
            intro = parts[0].strip()
            sections = {}
            for p in parts[1:]:
                lines = p.split("\n", 1)
                title = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                sections[title] = body
            st.markdown(intro)
            sel = st.selectbox("选择案例", list(sections.keys()), key="case_sel")
            st.markdown("## " + sel)
            st.markdown(sections[sel])
            with st.expander("📑 查看全部案例与速查表"):
                st.markdown(case_md)
        else:
            st.warning("⚠️ 案例库文件未找到 (data/standards/corrosion_case_library.md)")

# ======================
# Tab 6: 机理与工程模型 (新增 — 基于公开文献与标准)
# ======================
with tab6:
    st.markdown("### 🧪 腐蚀机理与工程模型")
    st.markdown("基于公开文献与行业标准（de Waard-Milliams / NORSOK M-506、API RP 14E、NACE MR0175 / ISO 15156、NACE SP0204）的**工程筛选与估算**模型。正式设计与合规则以现场检测及最新版标准为准。")

    mech_tab1, mech_tab2, mech_tab3, mech_tab4, mech_tab5, mech_tab6 = st.tabs([
        "🌫️ CO₂腐蚀", "💥 冲蚀", "🟡 H₂S开裂", "⚡ SCC敏感性", "🔬 PREN点蚀抗力", "🧬 SCC裂纹形貌"
    ])

    # --- CO2 腐蚀 ---
    with mech_tab1:
        st.markdown("#### CO₂ 腐蚀速率估算 (de Waard-Milliams / NORSOK M-506 思路)")
        st.markdown("基础式 log10(V) = 5.8 − 1710/(T+273) + 0.67·log10(pCO2)，叠加 pH 修正因子 f_pH = 10^[0.32·(pH_sat − pH)]。")
        c1, c2, c3 = st.columns(3)
        with c1:
            co2_T = st.slider("温度 (°C)", 20, 120, 65, step=1, key="co2_T")
        with c2:
            co2_p = st.number_input("CO₂分压 (bar)", min_value=0.001, max_value=100.0, value=2.0, step=0.1, key="co2_p")
        with c3:
            co2_ph = st.number_input("原位 pH (可选)", min_value=0.0, max_value=14.0, value=6.0, step=0.1, key="co2_ph")
        if st.button("🔢 计算 CO₂ 腐蚀", key="co2_calc", width="stretch"):
            r = co2_corrosion(co2_T, co2_p, co2_ph)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("基础速率 (mm/a)", f"{r['rate_base']:.3f}")
            m2.metric("修正后速率 (mm/a)", f"{r['rate_corrected']:.3f}")
            m3.metric("饱和 pH", f"{r['pH_sat']:.2f}" if r['pH_sat'] else "—")
            m4.metric("pH 修正因子", f"{r['f_pH']:.3f}" if r['f_pH'] else "—")
            st.caption("📌 " + r["regime"])
            xs, ys = co2_corrosion_curve(co2_p, pH_actual=co2_ph)
            fig_co2 = go.Figure()
            fig_co2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="腐蚀速率",
                                         line=dict(color="#e74c3c", width=2)))
            fig_co2.add_trace(go.Scatter(x=[co2_T], y=[r["rate_corrected"]],
                                         mode="markers+text", name="当前工况",
                                         marker=dict(size=12, color="#2c3e50", symbol="diamond"),
                                         text=[f"{r['rate_corrected']:.2f}"], textposition="top center"))
            fig_co2.update_layout(height=340, margin=dict(l=50, r=20, t=20, b=40),
                                  xaxis_title="温度 (°C)", yaxis_title="腐蚀速率 (mm/a)",
                                  template="plotly_dark" if dark_mode else "plotly_white")
            st.plotly_chart(fig_co2, width="stretch")
            st.caption("📚 " + r["reference"])

    # --- 冲蚀 ---
    with mech_tab2:
        st.markdown("#### 冲蚀临界流速 (API RP 14E) 与含砂冲蚀速率 (Salama)")
        st.markdown("临界流速 V_crit = C / √ρ_m；Salama 含砂冲蚀 E = 0.182·W·V²·D/(d²·ρ_m)。")
        e1, e2 = st.columns(2)
        with e1:
            ero_mat = st.selectbox("管材", list(EROSION_C.keys()), key="ero_mat")
            ero_rho = st.number_input("混合密度 ρ_m (kg/m³)", min_value=1.0, max_value=2000.0, value=1000.0, step=10.0, key="ero_rho")
            ero_v = st.slider("实际混合流速 (m/s)", 0.0, 20.0, 5.0, step=0.1, key="ero_v")
        with e2:
            ero_d = st.number_input("管径 (mm)", min_value=10.0, max_value=1500.0, value=150.0, step=10.0, key="ero_d")
            ero_w = st.number_input("含砂速率 (kg/天)", min_value=0.0, max_value=5000.0, value=50.0, step=5.0, key="ero_w")
            ero_size = st.number_input("砂粒粒径 (μm)", min_value=1.0, max_value=2000.0, value=200.0, step=10.0, key="ero_size")
        if st.button("🔢 计算冲蚀风险", key="ero_calc", width="stretch"):
            vc = erosion_critical_velocity(ero_rho, ero_mat)
            er = erosion_rate_salama(ero_w, ero_v, ero_d, ero_size, ero_rho)
            v_ok = ero_v <= vc["V_crit_m_s"]
            st.markdown(f"**临界流速 V_crit**: {vc['V_crit_m_s']:.2f} m/s (C={vc['C']})")
            st.markdown(f"**流速判定**: {'✅ 低于临界，冲蚀风险可控' if v_ok else '⚠️ 超过临界，存在冲蚀风险'}")
            st.markdown(f"**Salama 含砂冲蚀速率**: {er['rate_mm_yr']:.4f} mm/a — {er['verdict']}")
            st.caption("📚 " + er["reference"])

    # --- H2S 开裂 ---
    with mech_tab3:
        st.markdown("#### H₂S 环境开裂筛查 (NACE MR0175 / ISO 15156)")
        st.markdown("酸性服役判定：pH2S ≥ 0.0003 bar 须按 MR0175 选材；SSC 硬度上限 ≤ 22 HRC。本筛查为保守定性，非 ISO 15156-2 图1 精确曲线。")
        h1, h2, h3 = st.columns(3)
        with h1:
            h2s_p = st.number_input("H₂S 分压 (bar)", min_value=0.0, max_value=100.0, value=0.5, step=0.01, key="h2s_p")
        with h2:
            h2s_ph = st.number_input("原位 pH", min_value=0.0, max_value=14.0, value=4.0, step=0.1, key="h2s_ph")
        with h3:
            h2s_hrc = st.number_input("材料硬度 HRC (可选)", min_value=0.0, max_value=60.0, value=24.0, step=1.0, key="h2s_hrc")
        if st.button("🔢 筛查 H₂S 开裂", key="h2s_calc", width="stretch"):
            hr = h2s_ssc_screening(h2s_p, h2s_ph, h2s_hrc)
            st.markdown(f"**酸性服役**: {'是 (须按 MR0175 选材)' if hr['sour'] else '否 (豁免)'}")
            st.markdown(f"**区域**: {hr['region']}")
            st.markdown(f"**严苛度**: {hr['severity']}")
            if hr['hardness_hrc'] is not None:
                st.markdown(f"**硬度 {hr['hardness_hrc']:.0f} HRC**: {'✅ 满足 SSC 上限 ≤22 HRC' if hr['hardness_ok'] else '⚠️ 超过 SSC 硬度上限 ≤22 HRC'}")
            with st.expander("📋 控制措施"):
                for c in hr["controls"]:
                    st.markdown(f"- {c}")
            st.caption("📚 " + hr["reference"])

    # --- SCC 敏感性 ---
    with mech_tab4:
        st.markdown("#### 应力腐蚀开裂 (SCC) 敏感性筛查 (NACE SP0204)")
        st.markdown("外部 SCC 两类机理：高 pH (晶间, pH 9–11, 碳酸盐) 与近中性 pH (穿晶, pH 6–7.5, 富 CO2 地下水)。评分 0–100，越高越敏感。")
        s1, s2 = st.columns(2)
        with s1:
            s_coat = st.selectbox("涂层类型", ["旧涂层(煤焦油/沥青)", "现代涂层(FBE/PE)", "未知"], key="s_coat")
            s_stress = st.slider("操作应力 (%SMYS)", 0, 100, 70, step=1, key="s_stress")
            s_age = st.number_input("管龄 (年)", min_value=0, max_value=80, value=20, step=1, key="s_age")
        with s2:
            s_temp = st.slider("运行温度 (°C)", -10, 80, 45, step=1, key="s_temp")
            s_cp = st.checkbox("阴极保护被屏蔽 (CP shielded)", value=True, key="s_cp")
            s_terr = st.selectbox("土壤地形", ["排水良好砂土", "黏土/保水", "未知"], key="s_terr")
        if st.button("🔢 筛查 SCC 敏感性", key="scc_calc", width="stretch"):
            sr = scc_susceptibility(s_coat, s_stress, s_age, s_temp, s_cp, s_terr)
            hp = min(sr["high_pH_score"], 100)
            nn = min(sr["near_neutral_score"], 100)
            fig_scc = go.Figure(data=go.Bar(
                x=[hp, nn], y=["高pH SCC", "近中性pH SCC"], orientation="h",
                marker=dict(color=[ "#e74c3c" if hp>=60 else ("#f39c12" if hp>=30 else "#27ae60"),
                                     "#e74c3c" if nn>=60 else ("#f39c12" if nn>=30 else "#27ae60")]),
                text=[f"{sr['high_pH_level']} ({hp})", f"{sr['near_neutral_level']} ({nn})"],
                textposition="auto"))
            fig_scc.update_layout(height=240, margin=dict(l=90, r=20, t=20, b=30),
                                 xaxis_title="敏感性评分 (0–100)",
                                 template="plotly_dark" if dark_mode else "plotly_white")
            st.plotly_chart(fig_scc, width="stretch")
            with st.expander("📋 高 pH SCC 驱动因素"):
                for d in sr["drivers_high_pH"]:
                    st.markdown(f"- {d}")
            with st.expander("📋 近中性 pH SCC 驱动因素"):
                for d in sr["drivers_near_neutral"]:
                    st.markdown(f"- {d}")
            st.caption("📚 " + sr["reference"])

            st.divider()
            st.markdown("##### 🛠️ SCC-1 开挖验证优先级 (ECDA)")
            s_hca = st.checkbox("位于高后果区 (HCA)", value=False, key="s_hca")
            s_ili = st.checkbox("内检测(ILI)异常提示", value=False, key="s_ili")
            if st.button("🔢 生成开挖优先级", key="scc_exc", width="stretch"):
                ep = scc_excavation_priority(s_coat, s_stress, s_age, s_temp, s_cp, s_terr, hca=s_hca, ili_anomaly=s_ili)
                st.markdown(f"**开挖优先级**: {ep['priority']} ｜ 优先级评分 {ep['priority_score']} ｜ 置信度 {ep['confidence']}")
                st.markdown(f"**判定依据**: {ep['priority_reason']}")
                st.markdown("**ECDA 四步流程**:")
                for step in ep["edcda_steps"]:
                    st.markdown(f"- {step}")
                st.caption("📚 " + ep["reference"])

            st.divider()
            st.markdown("##### 📉 SCC-2 裂纹扩展与剩余寿命")
            c1, c2, c3 = st.columns(3)
            with c1:
                sl_a0 = st.number_input("初始裂纹深度 a₀ (mm)", min_value=0.1, max_value=20.0, value=2.0, step=0.1, key="sl_a0")
                sl_wt = st.number_input("壁厚 t (mm)", min_value=2.0, max_value=40.0, value=12.0, step=0.5, key="sl_wt")
            with c2:
                sl_d = st.number_input("管径 D (mm)", min_value=50.0, max_value=2000.0, value=610.0, step=10.0, key="sl_d")
                sl_y = st.number_input("屈服强度 SMYS (MPa)", min_value=200.0, max_value=800.0, value=450.0, step=10.0, key="sl_y")
            with c3:
                sl_stress2 = st.number_input("操作应力 %SMYS", min_value=10.0, max_value=100.0, value=70.0, step=1.0, key="sl_stress2")
                sl_type = st.selectbox("SCC 类型", ["near_neutral (近中性pH)", "high_pH (高pH)"], key="sl_type")
                sl_len = st.number_input("裂纹长度 L (mm)", min_value=5.0, max_value=500.0, value=50.0, step=5.0, key="sl_len")
            if st.button("🔢 计算剩余寿命", key="scc_life", width="stretch"):
                life = scc_crack_life(sl_a0, sl_wt, sl_d, sl_y, sl_stress2,
                                      scc_type="near_neutral" if sl_type.startswith("near") else "high_pH",
                                      crack_length_mm=sl_len)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("临界裂纹深度 a_c", f"{life['a_c_mm']:.2f} mm")
                m2.metric("Folias 因子 M", f"{life['folias_M']:.2f}")
                m3.metric("扩展速率", f"{life['growth_rate_mm_yr']:.2f} mm/a")
                m4.metric("剩余寿命", f"{life['life_years']:.1f} 年")
                st.markdown(f"**处置建议**: {life['verdict']}")
                st.caption("📚 " + life["reference"])

            st.divider()
            st.markdown("##### 🌳 SCC-3 缓解决策树")
            st.markdown("基于敏感性评分给出缓解组合建议：CP优化/涂层修复/降压运行/裂纹检测型ILI。")
            if st.button("🔢 生成缓解决策", key="scc_mit", width="stretch"):
                mt = scc_mitigation_tree(s_coat, s_stress, s_age, s_temp, s_cp, s_terr)
                st.markdown(f"**{mt['summary']}**")
                for m in mt["mitigations"]:
                    st.markdown(f"- **{m['action']}**（优先级 {m['priority']}，{m['applicability']}）：{m['detail']}")
                st.caption("📚 " + mt["reference"])

            st.divider()
            st.markdown("##### 🗺️ SCC-4 风险矩阵叠加")
            st.markdown("将 SCC 敏感性映射为失效概率维度，与管径/压力/位置（后果）叠加，输出综合风险等级。复用项目 5×5 风险矩阵。")
            s41, s42 = st.columns(2)
            with s41:
                s4_dia = st.number_input("管径 D (mm)", min_value=50.0, max_value=2000.0, value=610.0, step=10.0, key="s4_dia")
                s4_pres = st.number_input("操作压力 (MPa)", min_value=0.1, max_value=20.0, value=8.0, step=0.5, key="s4_pres")
            with s42:
                s4_loc = st.selectbox("位置类型", ["人口密集区", "一般区域", "荒野"], key="s4_loc")
                s4_hca = st.checkbox("高后果区(HCA)", value=False, key="s4_hca")
            if st.button("🔢 计算 SCC 综合风险", key="scc_risk", width="stretch"):
                ro = scc_risk_overlay(s_coat, s_stress, s_age, s_temp, s_cp, s_terr,
                                      s4_dia, s4_pres, s4_loc, hca=s4_hca)
                st.markdown(f"**SCC 敏感性(max)**: {ro['scc_max_score']} → 失效概率等级 **{ro['prob_level']}**")
                st.markdown(f"**失效后果等级**: {ro['cons_level']}（HCA 已按「{ro['hca_treated_as']}」处理）")
                st.markdown(f"**综合风险**: <span style='color:{ro['risk_color']};font-weight:bold;font-size:1.2em'>{ro['risk_level']}</span>",
                            unsafe_allow_html=True)
                st.caption("📚 " + ro["reference"])

    # --- PREN ---
    with mech_tab5:
        st.markdown("#### 点蚀抗力当量 PREN 对比")
        st.markdown("PREN = %Cr + 3.3×%Mo + 16×%N，用于不锈钢/双相钢/镍基合金抗氯离子点蚀能力对比（钛依赖氧化膜，不适用 PREN）。")
        pren_rows = pren_all()
        names = [r["material"].split(" (")[0] for r in pren_rows]
        vals = [r["PREN"] if r["PREN"] is not None else 0 for r in pren_rows]
        cols = ["#27ae60", "#27ae60", "#f39c12", "#f39c12", "#f39c12",
                "#2980b9", "#2980b9", "#8e44ad", "#8e44ad", "#7f8c8d"]
        fig_pren = go.Figure(data=go.Bar(
            x=names, y=vals, marker=dict(color=cols),
            text=[f"{v:.1f}" if v else "N/A" for v in vals], textposition="auto"))
        fig_pren.update_layout(height=380, margin=dict(l=50, r=20, t=20, b=90),
                               yaxis_title="PREN", template="plotly_dark" if dark_mode else "plotly_white")
        st.plotly_chart(fig_pren, width="stretch")
        st.caption("📚 PREN 为工程经验指标；实际点蚀行为还受温度、微生物、缝隙等因素影响。")

    # --- SCC 裂纹形貌蒙特卡洛模拟 ---
    with mech_tab6:
        st.markdown("#### SCC 裂纹形貌与扩展蒙特卡洛模拟 (P3)")
        st.markdown("对一组裂纹种群（初始深度 + 年扩展速率均服从对数正态分布）进行蒙特卡洛抽样，"
                    "给出 T 年后的深度分布、超概率(POD)曲线与深度-长度形貌(分叉着色)，"
                    "并统计穿壁失效比例。扩展速率区间与 SCC-2 经验值一致。")
        m1, m2, m3 = st.columns(3)
        with m1:
            mc_n = st.slider("裂纹数量 n", 50, 1000, 300, step=50, key="mc_n")
            mc_years = st.slider("模拟年限 (年)", 1, 50, 25, step=1, key="mc_years")
        with m2:
            mc_wt = st.number_input("壁厚/临界深度 a_c (mm)", min_value=3.0, max_value=40.0, value=12.0, step=0.5, key="mc_wt")
            mc_a0 = st.number_input("初始深度中位数 (mm)", min_value=0.2, max_value=5.0, value=1.0, step=0.1, key="mc_a0")
        with m3:
            mc_g = st.number_input("年扩展速率中位数 (mm/a)", min_value=0.01, max_value=0.5, value=0.06, step=0.01, key="mc_g")
            mc_branch = st.slider("分叉裂纹比例", 0.0, 1.0, 0.4, step=0.05, key="mc_branch")
        if st.button("🎲 运行蒙特卡洛模拟", key="mc_run", width="stretch"):
            sim = simulate_crack_population(
                n_cracks=mc_n, years=mc_years, a0_mean=mc_a0, growth_mean=mc_g,
                wall_thickness=mc_wt, branch_prob=mc_branch, seed=42,
            )
            figs = build_morphology_figures(sim, dark_mode=dark_mode)
            s = sim["summary"]
            a1, a2, a3, a4, a5, a6 = st.columns(6)
            a1.metric("平均深度", f"{s['mean_depth']:.2f} mm")
            a2.metric("P90 深度", f"{s['p90_depth']:.2f} mm")
            a3.metric("P99 深度", f"{s['p99_depth']:.2f} mm")
            a4.metric("最大深度", f"{s['max_depth']:.2f} mm")
            a5.metric("穿壁失效数", f"{s['failures']} ({s['failure_prob']*100:.1f}%)")
            a6.metric("近临界数", f"{s['near_critical']} ({s['near_critical_prob']*100:.1f}%)")
            st.plotly_chart(figs["fig_hist"], width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(figs["fig_pod"], width="stretch")
            with c2:
                st.plotly_chart(figs["fig_scatter"], width="stretch")
            st.caption("📚 参考: NACE SP0204 (SCC 直接评估); API 579 / RSTRENG (裂纹评定); "
                       "Battelle NG-18。注: 对数正态参数固定 σ=0.5；可据 ILI 统计标定。")

# ======================
# Tab 8: 实测数据标定 (P3)
# ======================
with tab8:
    st.markdown("### 📡 实测数据标定 (P3)")
    st.markdown("上传真实腐蚀/检测 CSV，自动识别目标列（数值→回归 / 类别→分类），用随机森林重新标定模型，"
                "并对比基线给出精度提升与特征重要性。无数据时可用『合成 demo』或『模板下载』先验证流程。")
    st.warning("⚠️ 标定仅在本会话临时生效；可下载指标/特征重要性归档。正式部署需将标定结果纳入版本管理。")

    tab8_opt = st.radio("数据来源", ["上传 CSV", "载入合成 demo"], key="tab8_opt", horizontal=True)
    df = None
    if tab8_opt == "上传 CSV":
        up = st.file_uploader("选择 CSV 文件", type=["csv"], key="tab8_up")
        if up is not None:
            try:
                df = parse_uploaded_csv(up)
                st.success(f"已读取 {df.shape[0]} 行 × {df.shape[1]} 列")
            except Exception as e:
                st.error(f"读取失败: {e}")
    else:
        if st.button("载入合成 demo 数据集", key="tab8_demo", width="stretch"):
            df = demo_synthetic_df()
            st.success(f"已生成合成 demo {df.shape[0]} 行")

    if df is not None:
        st.dataframe(df.head(10), width="stretch")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target = st.selectbox("选择目标列（待预测）", num_cols, index=max(0, len(num_cols) - 1), key="tab8_target")
        auto_feats = [c for c in num_cols if c != target]
        feats_sel = st.multiselect("选择特征列", auto_feats, default=auto_feats, key="tab8_feats")
        task_opt = st.radio("任务类型(可选覆盖)", ["自动", "回归", "分类"], key="tab8_task", horizontal=True)
        if st.button("🚀 运行标定", key="tab8_run", width="stretch"):
            try:
                task_map = {"自动": None, "回归": "regression", "分类": "classification"}
                res = calibrate_with_data(df, target, feature_cols=feats_sel, task=task_map[task_opt])
                m = res["metrics"]
                if m["task"] == "regression":
                    st.markdown(f"**任务: 回归** ｜ 样本 {res['n_samples']}")
                    ca, cb = st.columns(2)
                    ca.metric("R² (模型)", f"{m['r2']:.3f}")
                    cb.metric("R² (基线/均值)", f"{m['r2_baseline']:.3f}")
                    ca.metric("MAE (模型)", f"{m['mae']:.4f}")
                    cb.metric("MAE (基线)", f"{m['mae_baseline']:.4f}")
                else:
                    st.markdown(f"**任务: 分类** ｜ 样本 {res['n_samples']}")
                    ca, cb = st.columns(2)
                    ca.metric("准确率 (模型)", f"{m['accuracy']*100:.1f}%")
                    cb.metric("准确率 (基线)", f"{m['accuracy_baseline']*100:.1f}%")
                    ca.metric("宏F1 (模型)", f"{m['f1_macro']:.3f}")
                    cb.metric("宏F1 (基线)", f"{m['f1_baseline']:.3f}")
                fig_imp = go.Figure(go.Bar(
                    x=[v for _, v in res["importances"]][::-1],
                    y=[k for k, _ in res["importances"]][::-1],
                    orientation="h", marker_color="#8e44ad"))
                fig_imp.update_layout(height=360, margin=dict(l=140, r=20, t=20, b=30),
                                     xaxis_title="特征重要性", template="plotly_dark" if dark_mode else "plotly_white")
                st.plotly_chart(fig_imp, width="stretch")
                st.caption("📚 模型: RandomForest；基线: 回归=均值预测器 / 分类=众数预测器。")
            except Exception as e:
                st.error(f"标定失败: {e}")

    st.divider()
    st.markdown("#### 📥 模板下载")
    tpl = sample_template_df()
    st.markdown("CSV 需含数值特征列与一列目标（数值=腐蚀速率回归；低基数类别=风险分类）。示例如下：")
    st.dataframe(tpl, width="stretch")
    csv_tpl = tpl.to_csv(index=False).encode("utf-8")
    st.download_button("下载模板 CSV", csv_tpl, "mic_calibration_template.csv", "text/csv", key="tab8_dl")

# ======================
# Tab 9: 无损检测 NDT (P4 增强)
# ======================
with tab9:
    st.markdown("### 🔍 无损检测 (NDT) 与内检测 (ILI) 知识库")
    st.markdown("""
    管道完整性管理的**检测闭环**：建设期焊接检测(PAUT/RT) → 在役内检测(MFL/UT-CD/EMAT/Caliper)
    → 不可检段直接评估(ECDA/ICDA/SCCDA) → 开挖验证(MT/PT/UT)。本页依据威胁类型给出 ILI 工具选型，
    并提供 11 种 NDT 方法的工程知识卡片与 API 1163 性能量化概念。
    """)

    # --- 1. ILI 工具选型 ---
    st.markdown("#### 🛰️ 内检测 (ILI) 工具选型建议")
    ndt_options = list(ILI_THREAT_MAP.keys()) + ["金属损失/腐蚀", "裂纹/开裂", "几何变形"]
    ndt_threat = st.selectbox("选择主导威胁 / 缺陷类型", ndt_options, key="ndt_threat")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        piggable = st.toggle("管线可内检测（有收发球筒/可通过）", value=True, key="ndt_pig")
    with col_p2:
        prio = st.selectbox("策略侧重", ["balanced", "metal_loss", "crack"],
                            format_func=lambda x: {"balanced": "均衡", "metal_loss": "偏金属损失", "crack": "偏裂纹"}[x],
                            key="ndt_prio")

    rec_threat = ndt_threat if ndt_threat in ILI_THREAT_MAP else None
    rec_defects = [ndt_threat] if ndt_threat not in ILI_THREAT_MAP else None
    rec = recommend_ndt(threat=rec_threat, defect_types=rec_defects, piggable=piggable, priority=prio)

    st.markdown("**推荐检测工具（按优先级）：**")
    for m in rec["methods"]:
        st.markdown(f"{m['rank']}. **{m['name']}** — {m['reason']}")
    st.divider()
    st.markdown("**直接评估 / 验证路径：**")
    for d in rec["direct_assessment"]:
        st.markdown(f"- {d}")
    st.info("💡 " + rec["note"])

    # --- 2. NDT 方法知识库 ---
    st.markdown("#### 📚 NDT 方法知识库（点击展开）")
    for code, m in NDT_METHODS.items():
        with st.expander(m["name"]):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**原理：** {m['principle']}")
                st.markdown(f"**可检缺陷：** {('、'.join(m['detects']))}")
            with c2:
                st.markdown(f"**灵敏度/性能：** {m['sensitivity']}")
                st.markdown(f"**局限：** {m['limitations']}")
            st.markdown(f"**典型用途：** {m['use']}")
            st.caption(f"📚 标准：{m['standard']}")

    # --- 3. API 1163 性能量化 ---
    st.markdown("#### 📏 ILI 性能量化（API 1163 第 3 版, 2021）")
    st.markdown("""
    API 1163 要求 ILI 系统以三类指标**量化性能**，避免"单一工具盲区"误判：
    - **POD（检出概率）**：如金属损失深度阈值 10%t、POD=90%（即能检出 90% 达阈值的缺陷）。
    - **POI（识别概率）**：如孤立裂纹深度阈值 1 mm、POI=90%（正确识别为裂纹类）。
    - **尺寸精度**：如深度 ±10%t @ 80% 置信（80% 情况下报告深度与真值偏差 ≤ 10% 壁厚）。

    ⚠️ **最佳实践**：MFL 对金属损失灵敏但对轴向裂纹几乎盲区；UT-CD 对裂纹灵敏但常漏全面腐蚀。
    对威胁谱宽的管线，应在同一窗口**组合 run（MFL + UT-CD）**，成本翻倍、威胁覆盖率提升数倍
    （PHMSA §192.937 明确允许）。验证(verification)与确认(validation)为 API 1163 两个独立步骤。
    """)

# ======================
