---
title: Pipeline Corrosion AI
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: src/streamlit_app.py
pinned: true
---

# 管道腐蚀预测与标准问答系统

> [English](README_EN.md) | 中文

> 用 AI 技术解决管道完整性管理中的腐蚀预测和标准检索问题。

## 在线 Demo

🔗 **https://pipeline-corrosion-ai.streamlit.app/**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pipeline-corrosion-ai.streamlit.app/)

## 问题背景

- **行业**: 油气管道完整性管理
- **痛点 1**: 腐蚀风险预测依赖工程师经验，未充分利用历史数据
- **痛点 2**: NACE/API/ASME 标准文档检索耗时（15-30 分钟/次）
- **AI 解决方案**: ML 预测模型 + RAG 智能问答

## 功能

### Tab 1: 腐蚀预测
- 输入管道参数（管材、温度、pH、CO₂ 分压、H₂S 浓度、流速、氯离子含量）→ 预测腐蚀速率和风险等级
- 支持 10 种工业管材：碳钢、316SS、13Cr、超级13Cr、2205双相、2507超级双相、825合金、625合金、C-276合金、钛合金
- 95% 置信区间、材料对比、趋势分析、批量预测（CSV）、报告导出
- **剩余寿命预测**：基于壁厚参数 + 腐蚀速率计算管道剩余服役寿命
- **检测周期推荐**：依据 NACE SP0775 风险分类推荐内检测/外检测/阴保监测周期

### Tab 2: 标准问答
- 对 NACE MR0175、API 571、ASME B31.8S 等国际标准和 GB/T 23258、SY/T 0087 等中国标准进行智能问答
- Dify Cloud RAG 流式响应 + LRU 缓存

### Tab 3: 数据探索
- 数据分布分析、相关性热力图、特征重要性、多模型对比（GBR/RF/DT/LR）

### Tab 4: 关于
- 系统功能模块说明、技术栈与联系方式

### Tab 5: 完整性工具
- **B31G 剩余强度计算**：输入管径/壁厚/缺陷尺寸/屈服强度 → 计算 Folias 因子、失效压力、剩余强度率
- **缓蚀剂推荐**：基于温度/流速/CO₂/H₂S/介质类型 → 推荐缓蚀剂类型、注入浓度、预期缓蚀率
- **风险矩阵评估**：5×5 概率×后果矩阵热力图，定位管道综合风险等级

### Tab 6: 腐蚀环境分析（P3）
- **多环境腐蚀模型**：土壤腐蚀（DIN 50929 思路）、海水腐蚀、微生物腐蚀 MIC、电偶腐蚀的简化速率估算与严重程度分级
- **腐蚀成本估算**：基于管径/长度/壁厚/腐蚀速率 → 年度金属损失、检测、停产与维修成本（饼图构成分解）
- **腐蚀失效案例库**：CO₂ 腐蚀 / MIC / 氯离子 SCC / 土壤外腐蚀 / 电偶腐蚀等 7 类典型失效案例与速查表

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 界面 | Streamlit |
| 预测模型 | scikit-learn GradientBoosting (R²=0.89) |
| RAG 引擎 | Dify Cloud API (Streaming) + LangChain |
| 部署 | Streamlit Community Cloud + GitHub |

## 本地运行

```bash
# 1. 克隆仓库
git clone https://github.com/phdleo101/pipeline-corrosion-ai.git
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
streamlit run src/streamlit_app.py
```

访问 http://localhost:8501

## 配置（可选）

编辑 `app_config.yaml` 配置 LLM API Key 或 Dify Cloud API，启用增强问答功能。

## 项目文档

- [行业调研报告](docs/01-industry-analysis.md)
- [方案设计文档](docs/02-solution-design.md)
- [部署报告](docs/03-deployment-report.md)

## License

MIT
