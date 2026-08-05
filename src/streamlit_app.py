"""
streamlit_app.py
管道腐蚀预测与标准问答系统 - Streamlit Web 界面
Streamlit Community Cloud 部署入口

启动方式: streamlit run src/streamlit_app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from corrosion_model import CorrosionPredictor
from rag_engine import CorrosionRAG

# ----------------------
# 页面配置
# ----------------------
st.set_page_config(
    page_title="管道腐蚀预测与标准问答系统",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------
# 从 Streamlit Secrets 读取 Dify 配置
# ----------------------
try:
    if "dify" in st.secrets:
        os.environ["DIFY_API_URL"] = st.secrets["dify"]["api_url"]
        os.environ["DIFY_API_KEY"] = st.secrets["dify"]["api_key"]
except Exception:
    pass  # Secrets 未配置时静默跳过，使用 fallback 模式

# 自定义样式
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------
# 模型初始化（缓存）
# ----------------------
@st.cache_resource
def get_predictor():
    return CorrosionPredictor()

@st.cache_resource
def get_rag():
    return CorrosionRAG()

predictor = get_predictor()
rag = get_rag()

MATERIAL_CHOICES = {
    "碳钢": "carbon_steel",
    "316不锈钢": "stainless_316",
    "825合金": "alloy_825",
    "2205双相不锈钢": "duplex_2205",
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
st.markdown('<div class="main-title">🔧 管道腐蚀预测与标准问答系统</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Pipeline Corrosion Prediction & Standards Q&A | FDE 项目作品集</div>', unsafe_allow_html=True)

# ----------------------
# 选项卡
# ----------------------
tab1, tab2, tab3 = st.tabs(["📊 腐蚀预测", "💬 标准问答", "ℹ️ 关于"])

# ======================
# Tab 1: 腐蚀预测
# ======================
with tab1:
    col_input, col_result = st.columns(2)

    with col_input:
        st.markdown("### 输入管道参数")

        material_label = st.selectbox("管材类型", options=list(MATERIAL_CHOICES.keys()))
        material = MATERIAL_CHOICES[material_label]

        temperature = st.slider("温度 (°C)", 0, 150, 80, step=1)
        ph = st.slider("pH 值", 3.0, 10.0, 6.0, step=0.1)
        co2_pressure = st.slider("CO2 分压 (MPa)", 0.0, 10.0, 1.0, step=0.1)
        h2s_concentration = st.slider("H2S 浓度 (ppm)", 0, 1000, 50, step=10)
        flow_rate = st.slider("流速 (m/s)", 0.0, 10.0, 3.0, step=0.1)
        chloride_content = st.slider("氯离子含量 (ppm)", 0, 100000, 5000, step=500)

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            predict_btn = st.button("🔍 预测腐蚀速率", type="primary", use_container_width=True)
        with col_btn2:
            compare_btn = st.button("📊 对比材料", use_container_width=True)

    with col_result:
        st.markdown("### 预测结果")

        if predict_btn:
            result = predictor.predict(
                material=material,
                temperature=float(temperature),
                ph=float(ph),
                co2_pressure=float(co2_pressure),
                h2s_concentration=float(h2s_concentration),
                flow_rate=float(flow_rate),
                chloride_content=float(chloride_content),
            )

            color, icon = RISK_STYLES.get(result["risk_level"], ("#333", "⚪"))

            # 腐蚀速率仪表盘（HTML 可视化）
            rate = result["corrosion_rate"]
            gauge_pct = min(rate / 2.0 * 100, 100)  # 0-2 mm/a 映射到 0-100%
            st.markdown(f"""
            <div style="background: #f0f2f6; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                <div style="text-align: center; margin-bottom: 10px;">
                    <span style="font-size: 1.1rem; color: #666;">腐蚀速率</span>
                </div>
                <div style="background: #ddd; border-radius: 10px; height: 30px; overflow: hidden; position: relative;">
                    <div style="background: linear-gradient(90deg, #27ae60 0%, #f39c12 40%, #e74c3c 70%, #c0392b 100%); height: 100%; width: {gauge_pct}%; border-radius: 10px; transition: width 0.5s;"></div>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: 700; color: #333; font-size: 1.2rem;">{rate:.2f} mm/a</div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.75rem; color: #999;">
                    <span>0 (安全)</span><span>0.5</span><span>1.0</span><span>2.0+ (严重)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 风险等级 + 数据卡片
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
                st.metric("管材", result["material_label"])

            st.markdown("#### 📋 建议")
            st.info(result["suggestion"])
            st.markdown(f"**🔧 材料建议**: {result['material_advice']}")

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
            # 材料对比模式：相同工况下不同材料的腐蚀速率
            st.markdown("#### 📊 不同材料对比（相同工况）")
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
                compare_results.append((mat_label, r["corrosion_rate"], r["risk_level"]))

            # 用柱状图展示
            for mat_label, rate, risk in compare_results:
                c, icon = RISK_STYLES.get(risk, ("#333", "⚪"))
                pct = min(rate / 2.0 * 100, 100)
                st.markdown(
                    f'<div style="margin-bottom: 8px;">'
                    f'<div style="display: flex; justify-content: space-between; margin-bottom: 2px;">'
                    f'<span style="font-weight: 600;">{icon} {mat_label}</span>'
                    f'<span style="color: {c}; font-weight: 700;">{rate:.2f} mm/a</span>'
                    f'</div>'
                    f'<div style="background: #ddd; border-radius: 6px; height: 18px; overflow: hidden;">'
                    f'<div style="background: {c}; height: 100%; width: {pct}%; border-radius: 6px;"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("👈 调整左侧参数后，点击「预测腐蚀速率」或「对比材料」按钮查看结果。")

    # 预测历史
    if "prediction_history" in st.session_state and st.session_state["prediction_history"]:
        with st.expander(f"📋 预测历史（{len(st.session_state['prediction_history'])} 条）", expanded=False):
            for i, h in enumerate(reversed(st.session_state["prediction_history"][-5:])):
                c, icon = RISK_STYLES.get(h["risk"], ("#333", "⚪"))
                st.markdown(
                    f'<span style="color: {c};">{icon} {h["material"]}</span> | '
                    f'速率: **{h["rate"]:.2f} mm/a** | '
                    f'温度: {h["temp"]}°C | pH: {h["ph"]} | CO2: {h["co2"]} MPa',
                    unsafe_allow_html=True,
                )

    st.markdown("""
    ---
    **📌 典型工况参考**:
    - 🔴 高腐蚀风险：碳钢 + 80°C + pH 5.5 + CO2 1.5 MPa + H2S 100 ppm
    - 🟢 低腐蚀风险：316不锈钢 + 25°C + pH 7.5 + CO2 0.1 MPa + H2S 0 ppm
    """)

# ======================
# Tab 2: 标准问答
# ======================
with tab2:
    st.markdown("### 💬 腐蚀标准知识问答")

    # RAG 模式状态指示
    mode_labels = {
        "dify": ("🟢 Dify API 智能问答模式", "已连接 Dify Cloud，支持自然语言智能问答"),
        "local": ("🟡 本地向量检索模式", "使用 LangChain + ChromaDB 本地检索"),
        "fallback": ("⚪ 基础模式", "配置 Dify API 后可获得更强问答能力"),
    }
    mode_label, mode_desc = mode_labels.get(rag.mode, mode_labels["fallback"])
    st.caption(f"{mode_label} — {mode_desc}")

    # 示例问题
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

    # 缓存状态提示
    if hasattr(rag, "_cache") and len(rag._cache._cache) > 0:
        st.caption(f"📦 缓存: {len(rag._cache._cache)} 条常见问答（重复问题秒回）")

    st.markdown("---")

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 用户输入
    question = st.chat_input("输入你的问题，例如：在什么条件下需要使用抗硫材料？")

    # 处理示例问题点击
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        # 显示用户消息
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

        # 获取回答（流式模式）
        with chat_container:
            with st.chat_message("assistant"):
                if rag.mode == "dify":
                    # Dify 模式：使用流式显示
                    status_placeholder = st.empty()
                    status_placeholder.info("🔍 正在检索知识库...")
                    try:
                        # 切换状态到"生成中"
                        status_placeholder.info("✍️ 正在生成回答...")
                        # 流式输出
                        full_answer = st.write_stream(rag.query_stream(question))
                        status_placeholder.empty()
                    except Exception as e:
                        status_placeholder.empty()
                        full_answer = f"⚠️ 回答生成出错: {e}"
                        st.markdown(full_answer)
                else:
                    # 本地/降级模式：同步显示
                    with st.spinner("正在检索知识库..."):
                        full_answer = rag.query(question)
                    st.markdown(full_answer)

        # 保存到历史
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": full_answer})

    # 清空按钮
    col_clear, col_info = st.columns([1, 5])
    with col_clear:
        if st.session_state.messages and st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("""
    ---
    **📚 知识库涵盖**:
    - NACE MR0175/ISO 15156（抗硫材料选用）
    - API 571（损伤机理辨识）
    - ASME B31.8S（输气管道完整性管理）
    - ASME B31G（腐蚀缺陷评估）
    - NACE SP0162（阴极保护）
    - API 1163（管道内检测）
    - NACE SP0775（腐蚀速率分类与缓蚀剂）
    - NACE SP0185（管道涂层系统）
    - de Waard-Milliams CO2 腐蚀预测模型
    - 管道完整性管理最佳实践

    **🔧 增强模式**: 已配置 Dify Cloud API，支持自然语言智能问答。
    """)

# ======================
# Tab 3: 关于
# ======================
with tab3:
    st.markdown("""
    ## 关于本系统

    本系统是 FDE（Forward Deployed Engineer）跨行业作品集的 **项目一**，
    展示 AI 在管道完整性管理领域的落地应用。

    ### 功能模块

    **1. 腐蚀预测模块**
    - 基于 500 条模拟腐蚀数据训练的 GradientBoosting 模型
    - 输入 7 个管道参数即可预测腐蚀速率和风险等级
    - 自动给出防护建议和材料升级建议

    **2. 标准问答模块**
    - 内置 NACE/API/ASME 标准知识库（8 大标准领域）
    - 支持 RAG 检索增强生成
    - 三级降级策略：Dify API → 本地向量检索 → 基础模式
    - Dify Cloud 集成：自然语言智能问答 + 知识库检索

    ### 技术栈

    | 组件 | 技术 |
    |------|------|
    | Web 界面 | Streamlit |
    | 预测模型 | scikit-learn GradientBoosting |
    | RAG 引擎 | LangChain + ChromaDB |
    | 部署平台 | Streamlit Community Cloud |

    ### FDE 方法论

    本项目遵循五步 FDE 方法论：

    | 步骤 | 内容 |
    |------|------|
    | 1. 行业速学 | 管道完整性管理行业调研 |
    | 2. 痛点定位 | 腐蚀预测依赖经验 + 标准检索耗时 |
    | 3. 方案设计 | 双模块架构（预测 + 问答） |
    | 4. AI 驱动构建 | LangChain + scikit-learn |
    | 5. 部署验证 | Streamlit Community Cloud 在线 Demo |

    ### 联系方式

    GitHub: [项目仓库](https://github.com/phdleo101/pipeline-corrosion-ai)

    ---
    *MIT License | FDE Portfolio Project*
    """)
