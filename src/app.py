"""
app.py
管道腐蚀预测与标准问答系统 - Gradio Web 界面
HuggingFace Spaces 部署入口

启动方式: python src/app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from corrosion_model import CorrosionPredictor, format_prediction
from rag_engine import CorrosionRAG

predictor = CorrosionPredictor()
rag = CorrosionRAG()

MATERIAL_CHOICES = [
    ("碳钢", "carbon_steel"),
    ("316不锈钢", "stainless_316"),
    ("825合金", "alloy_825"),
    ("2205双相不锈钢", "duplex_2205"),
]


def predict_corrosion(material, temperature, ph, co2_pressure,
                      h2s_concentration, flow_rate, chloride_content):
    result = predictor.predict(
        material=material,
        temperature=float(temperature),
        ph=float(ph),
        co2_pressure=float(co2_pressure),
        h2s_concentration=float(h2s_concentration),
        flow_rate=float(flow_rate),
        chloride_content=float(chloride_content),
    )

    risk_colors = {
        "低风险": "#27ae60",
        "中风险": "#f39c12",
        "高风险": "#e74c3c",
        "严重风险": "#c0392b",
    }
    color = risk_colors.get(result["risk_level"], "#333")

    output = f"""## 预测结果

| 指标 | 数值 |
|------|------|
| 管材 | {result['material_label']} |
| 腐蚀速率 | **{result['corrosion_rate']} mm/a** |
| 风险等级 | <span style="color:{color};font-weight:bold;font-size:18px">{result['risk_level']}</span> |

### 建议

{result['suggestion']}

**材料建议**: {result['material_advice']}
"""
    return output


def answer_question(question, history):
    if not question.strip():
        return history, ""
    answer = rag.query(question)
    history.append((question, answer))
    return history, ""


def clear_chat():
    return [], ""


with gr.Blocks(
    title="管道腐蚀预测与标准问答系统",
    theme=gr.themes.Soft(),
    css="""
    .main-title { text-align: center; margin-bottom: 20px; }
    .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
    """,
) as demo:

    gr.HTML("""
    <div class="main-title">
        <h1>管道腐蚀预测与标准问答系统</h1>
    </div>
    <div class="subtitle">
        <p>Pipeline Corrosion Prediction & Standards Q&A | FDE 项目作品集</p>
    </div>
    """)

    with gr.Tab("腐蚀预测"):

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入管道参数")
                material = gr.Dropdown(
                    choices=MATERIAL_CHOICES,
                    value="carbon_steel",
                    label="管材类型",
                )
                temperature = gr.Slider(
                    minimum=0, maximum=150, value=80, step=1,
                    label="温度 (°C)",
                )
                ph = gr.Slider(
                    minimum=3, maximum=10, value=6.0, step=0.1,
                    label="pH 值",
                )
                co2_pressure = gr.Slider(
                    minimum=0, maximum=10, value=1.0, step=0.1,
                    label="CO2 分压 (MPa)",
                )
                h2s_concentration = gr.Slider(
                    minimum=0, maximum=1000, value=50, step=10,
                    label="H2S 浓度 (ppm)",
                )
                flow_rate = gr.Slider(
                    minimum=0, maximum=10, value=3.0, step=0.1,
                    label="流速 (m/s)",
                )
                chloride_content = gr.Slider(
                    minimum=0, maximum=100000, value=5000, step=500,
                    label="氯离子含量 (ppm)",
                )
                predict_btn = gr.Button("预测腐蚀速率", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 预测结果")
                prediction_output = gr.Markdown(
                    value="点击「预测腐蚀速率」按钮查看结果。"
                )

        predict_btn.click(
            fn=predict_corrosion,
            inputs=[
                material, temperature, ph, co2_pressure,
                h2s_concentration, flow_rate, chloride_content,
            ],
            outputs=prediction_output,
        )

        gr.Markdown("""
        ---
        **典型工况参考**:
        - 高腐蚀风险：碳钢 + 80°C + pH 5.5 + CO2 1.5 MPa + H2S 100 ppm
        - 低腐蚀风险：316不锈钢 + 25°C + pH 7.5 + CO2 0.1 MPa + H2S 0 ppm
        """)

    with gr.Tab("标准问答"):

        chatbot = gr.Chatbot(
            label="腐蚀标准知识问答",
            height=450,
            show_label=True,
        )

        with gr.Row():
            question_input = gr.Textbox(
                placeholder="输入你的问题，例如：在什么条件下需要使用抗硫材料？",
                label="提问",
                scale=4,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)

        clear_btn = gr.Button("清空对话")

        gr.Markdown("""
        ---
        **内置知识库涵盖**:
        - NACE MR0175/ISO 15156（抗硫材料选用）
        - API 571（损伤机理辨识）
        - ASME B31.8S（输气管道完整性管理）
        - ASME B31G（腐蚀缺陷评估）
        - NACE SP0162（阴极保护）
        - API 1163（管道内检测）

        **增强模式**: 配置 Dify API 或 LLM API Key 后，可获得更精准的智能问答能力。
        """)

        send_btn.click(
            fn=answer_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input],
        )
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input],
        )
        clear_btn.click(fn=clear_chat, outputs=[chatbot, question_input])

    with gr.Tab("关于"):

        gr.Markdown("""
        ## 关于本系统

        本系统是 FDE（Forward Deployed Engineer）跨行业作品集的项目一，
        展示 AI 在管道完整性管理领域的落地应用。

        ### 功能模块

        1. **腐蚀预测模块**
           - 基于 500 条模拟腐蚀数据训练的 GradientBoosting 模型
           - 输入 7 个管道参数即可预测腐蚀速率和风险等级
           - 自动给出防护建议和材料升级建议

        2. **标准问答模块**
           - 内置 NACE/API/ASME 标准知识库
           - 支持 RAG 检索增强生成
           - 可对接 Dify Cloud 获取更强的问答能力

        ### 技术栈

        | 组件 | 技术 |
        |------|------|
        | Web 界面 | Gradio |
        | 预测模型 | scikit-learn GradientBoosting |
        | RAG 引擎 | LangChain + ChromaDB |
        | 部署平台 | HuggingFace Spaces |

        ### FDE 方法论

        本项目遵循五步 FDE 方法论：
        1. 行业速学 -> 管道完整性管理行业调研
        2. 痛点定位 -> 腐蚀预测依赖经验 + 标准检索耗时
        3. 方案设计 -> 双模块架构（预测 + 问答）
        4. AI 驱动构建 -> LangChain + scikit-learn
        5. 部署验证 -> HuggingFace Spaces 在线 Demo

        ### 联系方式

        GitHub: [项目仓库](https://github.com/your-username/pipeline-corrosion-ai)
        """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
