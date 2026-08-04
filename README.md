---
title: Pipeline Corrosion AI
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: src/app.py
pinned: true
---

# 管道腐蚀预测与标准问答系统

> FDE 跨行业作品集 | 项目一 | 能源行业
> 用 AI 技术解决管道完整性管理中的腐蚀预测和标准检索问题。

## 在线 Demo

部署到 HuggingFace Spaces 后，在此处添加 Demo 链接。

## 问题背景

- **行业**: 油气管道完整性管理
- **痛点 1**: 腐蚀风险预测依赖工程师经验，未充分利用历史数据
- **痛点 2**: NACE/API/ASME 标准文档检索耗时（15-30 分钟/次）
- **AI 解决方案**: ML 预测模型 + RAG 智能问答

## 功能

### Tab 1: 腐蚀预测
输入管道参数（管材、温度、pH、CO2 分压、H2S 浓度、流速、氯离子含量）→ 预测腐蚀速率和风险等级 → 给出防护建议

### Tab 2: 标准问答
对 NACE MR0175、API 571、ASME B31.8S 等标准进行智能问答，支持自然语言提问

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 界面 | Gradio |
| 预测模型 | scikit-learn GradientBoosting |
| RAG 引擎 | LangChain + ChromaDB |
| 部署 | HuggingFace Spaces |

## 本地运行

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/pipeline-corrosion-ai.git
cd pipeline-corrosion-ai

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 生成数据并训练模型
python src/data_processor.py
python src/corrosion_model.py

# 5. 启动应用
python src/app.py
```

访问 http://localhost:7860

## 配置（可选）

编辑 `app_config.yaml` 配置 LLM API Key 或 Dify Cloud API，启用增强问答功能。

## FDE 案例文档

- [行业调研报告](docs/01-industry-analysis.md)
- [方案设计文档](docs/02-solution-design.md)
- [部署报告](docs/03-deployment-report.md)

## License

MIT
