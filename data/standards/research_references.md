# 管道腐蚀研究资料与公开数据源

> 本文件为「管道腐蚀预测与标准问答系统」的扩展参考资料，面向**研究者、管道完整性工程师、学生与运维人员**四类用户，汇总当前管线腐蚀的主要痛点、可公开获取的数据库/报告，以及支撑本系统工程模型的关键文献。
>
> ⚠️ 本资料用于学习与筛查参考；工程设计、合规则与失效分析请以最新版标准原文及现场检测为准。

---

## 一、管道腐蚀的核心痛点（行业现状）

1. **CO₂ 腐蚀（甜腐蚀）**：油气田湿气/产出水中 CO₂ 形成碳酸，碳钢均匀腐蚀速率可达数十至数百 mpy（mm/a 量级），是内部腐蚀最主要机理之一。裸钢模型在高 pH / 高温下因 FeCO₃ 保护膜失效而严重高估。
   - 应对：de Waard-Milliams / NORSOK M-506 模型、pH 修正、缓蚀剂、CRA 选材。
2. **H₂S 与酸性环境开裂（SSC / HIC / SOHIC）**：湿 H₂S 环境中原子氢致开裂，是高强度钢（>22 HRC）的致命威胁；选材须按 NACE MR0175 / ISO 15156。
3. **外部应力腐蚀开裂（SCC）**：高 pH（晶间，pH 9–11，碳酸盐电解质）与近中性 pH（穿晶，pH 6–7.5，富 CO₂ 地下水）两类，曾造成北美多条长输管线重大事故。
4. **微生物腐蚀（MIC）**：SRB 产 H₂S、形成垢下微环境，表现为 6 点钟位置离散点蚀；低流速、死区、20–60°C 最易感。
5. **冲蚀-腐蚀（砂）**：产砂井/多相流中砂粒磨损破坏保护膜，局部减薄快；API RP 14E 临界流速与 Salama 含砂模型是行业基准。
6. **外腐蚀 / 土壤腐蚀**：土壤电阻率、pH、含氧/氯/硫酸根、剥离涂层+CP 屏蔽共同决定外腐蚀速率。
7. **数据驱动评估缺口**：运营商积累了大量检测（ILI/CP/开挖）与失效数据，但缺乏低门槛的**工程筛选工具**与**标准知识检索**，本系统即针对此缺口。

---

## 二、公开数据库与权威报告（免费 / 公开）

| 来源 | 内容 | 链接 |
|------|------|------|
| **PHMSA（美国管道安全局）** | 1970 年至今管线事故/事件数据（天然气、危险液体、LNG），强制报告，公开下载（TXT/Excel），含失效原因、后果、地理位置 | https://www.phmsa.dot.gov/data-and-statistics/pipeline |
| **PHMSA Data Hub（Socrata）** | 上述数据的机器可读接口（OData/API），可直接接入 Excel / Tableau | https://datahub.transportation.gov/d/27nc-rsge |
| **PRCI（管道研究委员会国际）** | 管线完整性、检测、材料、SCC 等联合研究项目报告与最佳实践 | https://www.prci.org/ |
| **EGIG（欧洲输气管道事故组）** | 欧洲天然气管道事故统计年报（第三方破坏、腐蚀、材料缺陷等根因占比） | https://www.egig.nl/ |
| **CONCAWE** | 欧洲液体管道（油品）事故数据库与趋势报告 | https://www.concawe.eu/ |
| **NTSB（美国运输安全委员会）** | 重大管道事故独立调查报告（含根因与改进建议） | https://www.ntsb.gov/ |
| **NIST（美国标准技术院）** | 材料腐蚀动力学数据、热力学/动力学数据库（MPD、Kinetics） | https://www.nist.gov/materials |
| **NETL（国家能源技术实验室）** | 碳捕集、H₂、CO₂ 管道与腐蚀相关研究报告 | https://netl.doe.gov/ |
| **API / ASME / AMPP(NACE)** | 设计、完整性管理、腐蚀与材料标准正文（部分公开，部分购买） | https://www.api.org/ · https://www.asme.org/ · https://www.amp.org/ |
| **中国标准（公开检索）** | GB/T 23258、SY/T 0087、GB 50251、SY/T 6648、GB/T 21447、GB/T 30582、SY/T 0036 等 | 全国标准信息公共服务平台 https://std.samr.gov.cn/ |

> 使用建议：研究者可用 PHMSA/EGIG 数据做失效根因统计与趋势建模；工程师可用 PRCI/NTSB 报告对标本企业的完整性管理短板；学生可据此理解真实失效机理。

---

## 三、支撑本系统模型的关键文献与标准

### 3.1 CO₂ 腐蚀（Tab「机理与模型 → CO₂腐蚀」）
- de Waard, C. & Milliams, D. E. (1975). *Carbonic acid corrosion of steel*. **Corrosion**, 31(5), 177–181. （基础式）
- de Waard, C., Lotz, U. & Milliams, D. E. (1991). *Predictive model for CO₂ corrosion engineering in wet natural gas pipelines*. Corrosion/91, Paper 577. （pH 与逸度修正）
- de Waard, C. & Lotz, U. (1993). *Prediction of CO₂ corrosion of carbon steel*. Corrosion/93, Paper 69. （膜因子）
- **NORSOK M-506** (Rev. 3, 2017). *CO₂ Corrosion Rate Calculation Model*. （模块化温度/pH/剪切/膜/缓蚀剂修正）
- NACE SP0106 — 管道内腐蚀直接评估（ICDA）。

### 3.2 冲蚀（Tab「冲蚀」）
- **API RP 14E** — *Design and Installation of Offshore Production Platform Piping Systems*（临界流速 `V = C/√ρ`）。
- Salama, M. M. & Venkatesh, E. S. (1983). *Evaluation of API RP 14E erosional velocity limitations for offshore gas wells*. OTC 4485. （含砂冲蚀速率）
- Salama, M. M. (1993). 多相流含砂冲蚀修正（引入粒径与混合密度）。

### 3.3 H₂S 开裂（Tab「H₂S开裂」）
- **NACE MR0175 / ISO 15156** (Parts 1–3) — *Petroleum and natural gas industries — Materials for use in H₂S-containing environments*. （酸性服役定义、材料适用条件、硬度上限 ≤ 22 HRC、HIC 试验要求）
- NACE TM0177 — SSC 试验方法；NACE TM0284 — HIC 试验（CLR/CTR/CSR 验收）。

### 3.4 应力腐蚀开裂 SCC（Tab「SCC敏感性」）
- **NACE SP0204** — *Stress Corrosion Cracking Direct Assessment (SCCDA)* Methodology. （外部 SCC 直接评估）
- API RP 1176 — *Assessment and Management of Cracking in Pipelines*.
- Kiefner, J. F. & Vieth, P. H. (Battelle) — SCC 检测与开挖验证研究；**NEB RH-2-2008**（加拿大国家能源局 SCC 安全管理报告）。
- Parkins, R. N. 等 — 高 pH 与近中性 pH SCC 机理（阳极溶解 vs 氢脆）。
- 综述：*A review of crack growth models for near-neutral pH SCC* (PMC8591668)。

### 3.5 点蚀抗力（Tab「PREN」）
- PREN = %Cr + 3.3×%Mo + 16×%N（不锈钢/双相钢/镍基合金经验指标）。
- 相关：ASTM G48（点蚀临界温度 CPT 试验）、NACE MR0175 氯离子与温度选材边界。

### 3.6 其他已落地模型（本系统既有模块）
- ASME B31G（Level 1 剩余强度）、NACE SP0775（检测周期）、DIN 50929（土壤腐蚀性分级思路）、MIC/SRB 评估、电偶序。

---

## 四、本系统功能 ↔ 用户痛点 映射

| 用户类型 | 关注点 | 对应功能 |
|----------|--------|----------|
| 研究者 | 文献溯源、数据获取、机理建模 | 本资料 + 机理与模型 Tab（5 类工程模型）+ 数据探索（ML 对比） |
| 完整性工程师 | 剩余强度、裂纹/开裂风险、材料升级 | B31G、SCC 筛查、H₂S 开裂筛查、PREN、缓蚀剂推荐 |
| 运维人员 | 日常腐蚀速率、检测周期、成本 | 腐蚀预测、剩余寿命、检测周期推荐、腐蚀成本估算 |
| 学生/新人 | 标准检索、入门知识 | 标准问答（RAG）、案例库、PREN 对比、各模型说明与引用 |

---

## 五、延伸阅读建议
1. 先用 PHMSA/EGIG 数据做「失效根因分布」统计，识别本企业主导威胁。
2. 对高 pH / 近中性 pH SCC 高风险管段，按 NACE SP0204 启动 SCCDA（无需 ILI）。
3. 酸性环境选材严格走 NACE MR0175 / ISO 15156 的流程，本 Tab 仅作初筛。
4. 将所有工程模型的输出与 ILI/开挖实测比对，校准本企业经验系数。
