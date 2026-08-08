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
- 知识库新增**条款级细分**（`china_standards_clauses.md`：防腐层厚度、阴极保护电位、评价等级、设计系数等数值阈值）
- 新增**文献与公开数据覆盖**（`research_references.md`：PHMSA/PRCI/EGIG 等数据库链接 + 关键论文与标准引用）
- Dify Cloud RAG 流式响应 + LRU 缓存；本地模式自动收录 `data/standards/` 下全部 `.md`

### Tab 3: 数据探索
- 数据分布分析、相关性热力图、特征重要性、多模型对比（GBR/RF/DT/LR）

### Tab 4: 完整性工具
- **B31G 剩余强度计算**：输入管径/壁厚/缺陷尺寸/屈服强度 → 计算 Folias 因子、失效压力、剩余强度率
- **缓蚀剂推荐**：基于温度/流速/CO₂/H₂S/介质类型 → 推荐缓蚀剂类型、注入浓度、预期缓蚀率
- **风险矩阵评估**：5×5 概率×后果矩阵热力图，定位管道综合风险等级

### Tab 5: 腐蚀环境分析（P3）
- **多环境腐蚀模型**：土壤腐蚀（DIN 50929 思路）、海水腐蚀、微生物腐蚀 MIC、电偶腐蚀的简化速率估算与严重程度分级
- **MIC 深入子功能**：
  - **MIC-1 多菌属与生物膜热点**：在 SRB 基础上纳入 APB(产酸)/IRB(铁氧化)/SOB(硫氧化)，定位低流速/死管/焊缝 under-deposit 热点（NACE SP0192 / TM0194）
  - **MIC-2 杀菌剂方案设计**：按风险等级给出戊二醛/THPS/DBNPA/氧化性杀菌剂的投加方式、剂量、季度轮换与杀灭率监测（NACE TM0212）
  - **MIC-3 材料升级决策**：基于 MIC 风险等级 + 当前材料，结合氯离子/温度给出升级阶梯（碳钢→316→双相→镍基/钛），复用 PREN 思路（NACE SP0192 / MR0175）
  - **MIC-4 监测与再评估计划**：按风险等级给出腐蚀挂片/生物膜探针/ILI 监测手段与再筛查周期（高/极高 3–6 月，中 1–2 年）（NACE SP0192 / API 570）
  - **P3 · MIC 机器学习预测**：基于物理约束合成数据集训练随机森林，从 12 项环境/材料/运行特征预测 MIC 风险等级与腐蚀速率，并展示特征重要性（模型为合成数据，见「实测数据标定」接入真实数据）
- **腐蚀成本估算**：基于管径/长度/壁厚/腐蚀速率 → 年度金属损失、检测、停产与维修成本（饼图构成分解）
- **腐蚀失效案例库**：CO₂ 腐蚀 / MIC / 氯离子 SCC / 土壤外腐蚀 / 电偶腐蚀等 7 类典型失效案例与速查表

### Tab 6: 机理与工程模型（新增）
基于公开文献与行业标准（de Waard-Milliams / NORSOK M-506、API RP 14E、NACE MR0175 / ISO 15156、NACE SP0204）的**工程筛选与估算**模型：
- **CO₂ 腐蚀**：温度 / CO₂ 分压 / pH 耦合的基础腐蚀速率与 pH 修正，附温度-速率曲线
- **冲蚀**：API RP 14E 临界流速判定 + Salama 含砂冲蚀速率估算
- **H₂S 开裂筛查**：酸性服役判定、SSC 严苛度区域、≤22 HRC 硬度上限与 HIC 控制
- **SCC 敏感性筛查 + 深入子功能**：
  - 高 pH / 近中性 pH 两类机理评分与驱动因素（NACE SP0204）
  - **SCC-1 开挖验证优先级**：叠加 HCA 与 ILI 异常 → P1/P2/P3 开挖决策 + ECDA 四步流程（NACE SP0204 / API RP 1176）
  - **SCC-2 裂纹扩展与剩余寿命**：B31G Folias 因子求临界裂纹深度 + 经验扩展速率 → 剩余寿命与处置建议
  - **SCC-3 缓解决策树**：基于敏感性评分给出 CP 优化 / 涂层修复 / 降压运行 / 裂纹检测型 ILI 的缓解组合与优先级（NACE SP0204 / PRCI）
  - **SCC-4 风险矩阵叠加**：将 SCC 敏感性映射为失效概率，与管径/压力/位置（后果）叠加 → 综合风险等级，复用项目 5×5 风险矩阵
  - **P3 · SCC 裂纹形貌与扩展蒙特卡洛模拟**：对裂纹种群（初始深度 + 年扩展速率均服从对数正态）抽样，给出 T 年后深度分布、超概率(POD)曲线与深度-长度形貌(分叉着色)，并统计穿壁失效比例（NACE SP0204 / API 579 / RSTRENG）
- **PREN 点蚀抗力**：10 种管材抗氯离子点蚀能力对比

### Tab 7: 关于
- 系统功能模块说明、技术栈与联系方式
- 研究资料与公开数据源见 `data/standards/research_references.md`
- 中国标准条款级细分见 `data/standards/china_standards_clauses.md`

### Tab 8: 实测数据标定（P3 新增）
- **CSV 上传 / 合成 demo / 模板下载**：用户上传真实腐蚀-检测 CSV，或先用合成 demo 验证流程
- **自动任务识别**：数值目标→回归（R² / MAE）、低基数或类别目标→分类（准确率 / 宏 F1），并与基线（均值/众数预测器）对比
- **特征重要性**：用随机森林重新标定后输出，可下载归档（标定仅本会话临时生效）
- 用途：将 MIC 机器学习模型与经验模型从合成数据切换到真实现场数据，形成「数据驱动」闭环

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
