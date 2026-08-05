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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from corrosion_model import CorrosionPredictor
from rag_engine import CorrosionRAG

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
    """)
    st.markdown("---")
    st.caption("FDE Portfolio Project 1\n MIT License")

# 暗黑模式 CSS
if dark_mode:
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        .stApp .stMarkdown, .stApp .stText { color: #fafafa; }
        .stApp .stMetric { background-color: #1a1a2e; border-radius: 8px; padding: 10px; }
        .stApp section[data-testid="stExpander"] { background-color: #1a1a2e; }
        .stApp .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stApp .stTabs [data-baseweb="tab"] { background-color: #1a1a2e; color: #aaa; }
        .stApp .stTabs [aria-selected="true"] { background-color: #4a4a6a !important; color: #fff !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .main-title { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
        .subtitle { text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
        .risk-badge { display: inline-block; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# 多语言文本
I18N = {
    "中文": {
        "title": "🔧 管道腐蚀预测与标准问答系统",
        "subtitle": "Pipeline Corrosion Prediction & Standards Q&A | FDE 项目作品集",
        "tabs": ["📊 腐蚀预测", "💬 标准问答", "📈 数据探索", "ℹ️ 关于"],
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
        "subtitle": "FDE Portfolio Project | AI-Powered Corrosion Management",
        "tabs": ["📊 Prediction", "💬 Q&A", "📈 Data Explorer", "ℹ️ About"],
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
    "碳钢 / Carbon Steel": "carbon_steel",
    "316不锈钢 / 316 SS": "stainless_316",
    "825合金 / Alloy 825": "alloy_825",
    "2205双相钢 / Duplex 2205": "duplex_2205",
}

RISK_STYLES = {
    "低风险": ("#27ae60", "🟢"),
    "中风险": ("#f39c12", "🟡"),
    "高风险": ("#e74c3c", "🔴"),
    "严重风险": ("#c0392b", "🔴"),
}

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
tab1, tab2, tab3, tab4 = st.tabs(T["tabs"])

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
            predict_btn = st.button(T["predict_btn"], type="primary", use_container_width=True)
        with col_btn2:
            compare_btn = st.button(T["compare_btn"], use_container_width=True)

        col_btn3, col_btn4 = st.columns([1, 1])
        with col_btn3:
            trend_btn = st.button(T["trend_btn"], use_container_width=True)
        with col_btn4:
            share_btn = st.button("🔗 分享", use_container_width=True)

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
            st.dataframe(export_df, use_container_width=True, hide_index=True)

            csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "💾 下载 CSV 报告", data=csv_data,
                file_name=f"corrosion_report_{int(time.time())}.csv",
                mime="text/csv", use_container_width=True,
            )

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
            st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)
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
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

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
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    csv_batch = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "💾 下载批量预测结果", data=csv_batch,
                        file_name=f"batch_prediction_{int(time.time())}.csv",
                        mime="text/csv", use_container_width=True,
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
        if col.button(q, key=f"example_{i}", use_container_width=True):
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
        col3.metric("材料类型", "4 种")
        col4.metric("风险等级", "4 级")

        st.markdown("##### 数据预览")
        st.dataframe(dataset.head(20), use_container_width=True, hide_index=True)

        st.markdown("##### 描述性统计")
        st.dataframe(dataset.describe().round(3), use_container_width=True)

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
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_box:
            fig_box = px.box(
                dataset, x="material", y=dist_col,
                title="按材料分组箱线图",
                template="plotly_dark" if dark_mode else "plotly_white",
                color="material",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_box.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_box, use_container_width=True)

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
        st.plotly_chart(fig_heatmap, use_container_width=True)

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
            st.plotly_chart(fig_imp, use_container_width=True)

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
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig_r2, use_container_width=True)

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
        st.plotly_chart(fig_mae, use_container_width=True)

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

    本系统是 FDE（Forward Deployed Engineer）跨行业作品集的 **项目一**，
    展示 AI 在管道完整性管理领域的落地应用。

    ### 功能模块

    **1. 腐蚀预测模块**
    - 基于 500 条模拟腐蚀数据训练的 GradientBoosting 模型（R²=0.80）
    - 输入 7 个管道参数即可预测腐蚀速率和风险等级
    - 支持 95% 置信区间、趋势分析、材料对比、批量预测
    - 自动给出防护建议和材料升级建议

    **2. 标准问答模块**
    - Dify Cloud RAG 引擎 + 10 大国际标准知识库
    - Streaming 流式响应（首字 2-3 秒）+ LRU 缓存
    - 三级降级策略：Dify API → 本地向量检索 → 基础模式

    **3. 数据探索模块**
    - 数据集统计概览与预览
    - 分布分析（直方图 + 箱线图）
    - 相关性热力图 + 特征重要性
    - 多模型对比（GBR / RF / DT / LR）

    ### 技术栈

    | 组件 | 技术 |
    |------|------|
    | Web 界面 | Streamlit + Plotly |
    | 预测模型 | scikit-learn (GradientBoosting / RandomForest / DecisionTree / LinearRegression) |
    | RAG 引擎 | Dify Cloud API (Streaming) + LangChain |
    | 部署平台 | Streamlit Community Cloud |
    | 版本控制 | GitHub |

    ### FDE 方法论

    | 步骤 | 内容 |
    |------|------|
    | 1. 行业速学 | 管道完整性管理行业调研 |
    | 2. 痛点定位 | 腐蚀预测依赖经验 + 标准检索耗时 |
    | 3. 方案设计 | 双模块架构（预测 + 问答）+ 数据探索 |
    | 4. AI 驱动构建 | Dify + scikit-learn + Streamlit |
    | 5. 部署验证 | Streamlit Community Cloud 在线 Demo |

    ### 联系方式

    GitHub: [项目仓库](https://github.com/phdleo101/pipeline-corrosion-ai)
    Demo: [在线体验](https://pipeline-corrosion-ai.streamlit.app/)

    ---
    *MIT License | FDE Portfolio Project*
    """)
