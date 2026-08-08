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
)
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
    - [关于](#tab4)
    - [完整性工具](#tab5)
    - [腐蚀环境分析](#tab6)
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
        "tabs": ["📊 腐蚀预测", "💬 标准问答", "📈 数据探索", "ℹ️ 关于", "🔧 完整性工具", "🌍 腐蚀环境分析"],
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
        "tabs": ["📊 Prediction", "💬 Q&A", "📈 Data Explorer", "ℹ️ About", "🔧 Integrity Tools", "🌍 Env. Analysis"],
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(T["tabs"])

# ======================
# Tab 1: 腐蚀预测 (增强版)
# ======================
with tab1:
    col_input, col_result = st.columns(2)

    with col_input:
        st.markdown(f"### {T['input_params']}")

        material_label = st.selectbox("管材类型 / Material", options=list(MATERIAL_CHOICES.keys()))
        material = MATERIAL_CHOICES[material_label]

        temperature = st.slider("温度 Temperature (°C)", 0, 150, 80, step=1)
        ph = st.slider("pH 值", 3.0, 10.0, 6.0, step=0.1)
        co2_pressure = st.slider("CO2 分压 (MPa)", 0.0, 10.0, 1.0, step=0.1)
        h2s_concentration = st.slider("H2S 浓度 (ppm)", 0, 1000, 50, step=10)
        flow_rate = st.slider("流速 Flow Rate (m/s)", 0.0, 10.0, 3.0, step=0.1)
        chloride_content = st.slider("氯离子 Cl- (ppm)", 0, 100000, 5000, step=500)

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
    ]
    cols = st.columns(len(example_questions))
    for i, (col, q) in enumerate(zip(cols, example_questions)):
        if col.button(q, key=f"example_{i}", width="stretch"):
            st.session_state["pending_question"] = q

    if hasattr(rag, "_cache") and len(rag._cache._cache) > 0:
        st.caption(f"📦 缓存: {len(rag._cache._cache)} 条常见问答（重复问题秒回）")

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
        st.caption("训练 4 种算法并对比性能指标，展示模型选型决策过程")

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
        **📌 选型决策**：GradientBoosting 在 R² 和 MAE 上均表现最优，选择作为生产模型。
        RandomForest 紧随其后，LinearRegression 表现最差（腐蚀速率与特征间为非线性关系）。
        """)

# ======================
# Tab 4: 关于 (更新)
# ======================
with tab4:
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
    - Streaming 流式响应（首字 2-3 秒）+ LRU 缓存
    - 三级降级策略：Dify API → 本地向量检索 → 基础模式

    **5. 数据探索模块**
    - 数据集统计概览与预览
    - 分布分析（直方图 + 箱线图）
    - 相关性热力图 + 特征重要性
    - 多模型对比（GBR / RF / DT / LR）

    **6. 腐蚀环境分析模块**
    - 多环境腐蚀模型：土壤（DIN 50929 思路）/海水/微生物(MIC)/电偶腐蚀的简化速率估算
    - 腐蚀成本估算：基于管径/长度/壁厚/腐蚀速率 → 年度金属损失、检测、停产与维修成本
    - 腐蚀失效案例库：CO₂腐蚀/MIC/氯离子SCC/土壤外腐蚀/电偶腐蚀等典型失效案例

    ### 技术栈

    | 组件 | 技术 |
    |------|------|
    | Web 界面 | Streamlit + Plotly |
    | 预测模型 | scikit-learn (GradientBoosting / RandomForest / DecisionTree / LinearRegression) |
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
# Tab 5: 完整性工具 (新增 P2)
# ======================
with tab5:
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
# Tab 6: 腐蚀环境分析 (新增 P3)
# ======================
with tab6:
    st.markdown("### 🌍 多环境腐蚀分析与成本评估")

    env_tab1, env_tab2, env_tab3 = st.tabs([
        "🌱 多环境腐蚀模型", "💰 腐蚀成本估算", "📚 腐蚀案例库"
    ])

    # --- 多环境腐蚀模型 ---
    with env_tab1:
        st.markdown("#### 多环境腐蚀速率估算")
        st.markdown("选择腐蚀环境类型，输入关键参数估算腐蚀速率与严重程度。公式为工程简化估算，正式评估以现场检测为准。")

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
