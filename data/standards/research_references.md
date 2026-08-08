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

### 2.1 增补数据库与平台（2026 扩充）

| 来源 | 内容 | 链接 |
|------|------|------|
| **CER（加拿大能源监管局）** | 2008 至今 CER 监管管线事故数据（含原因），CSV 开放下载，季度更新 | https://www.cer-rec.gc.ca/open/incident/pipeline-incidents-comprehensive-data.csv |
| **CEPA（加拿大能源管道协会）** | 金属损失 ILI 工具验证指南（10 点评分法）+ 完整性行业报告 | https://www.cepa.com/ |
| **EPRG（欧洲管道研究集团）** | 裂纹/凹陷/止裂等联合研究项目与选型图 | https://www.eprg.org/ |
| **DNV / DNV Pipeline Failure Statistics** | 海底与陆上管道失效统计、可靠性与风险评估方法 | https://www.dnv.com/ |
| **Pipeline Safety Trust** | 汇总的 PHMSA 数据入口、事故地图与统计简报（便于公众检索） | https://pstrust.org/stats |
| **NIST MPD / WebSCD / WebHTS** | 材料腐蚀性能、热力学与高温结构陶瓷/超导体数据库（在线检索+评估） | https://www.nist.gov/materials |
| **MatWeb** | 19,000+ 材料物性数据库（金属/塑料/陶瓷/复合材料，含部分腐蚀数据） | https://www.matweb.com/ |
| **DECHEMA** | 德国化工材料腐蚀数据库（数据量最大、最权威的化工腐蚀数据源之一） | https://www.dechema.de/ |
| **NIMS（日本国立材料科学研究所）** | 在线材料数据库（金属/聚合物/无机/扩散，含腐蚀与无损评估） | https://mits.nims.go.jp/index_en.html |
| **国家材料腐蚀与防护科学数据中心 (corrdata)** | 国内材料腐蚀数据库（免费） | http://www.ecorr.org/ |
| **NACE IMPACT 腐蚀成本研究** | 各国腐蚀经济损失量化研究（支撑腐蚀成本估算模块） | https://www.nace.org/ |
| **GB/T 29780 管道内检测技术规范** | 中国内检测技术标准（与 API 1163 / ISO 13847 对应） | 全国标准信息公共服务平台 https://std.samr.gov.cn/ |

### 2.2 无损检测与完整性管理标准（支撑 NDT 模块）

| 标准 / 机构 | 内容 | 备注 |
|------|------|------|
| **API 1163（第3版, 2021）** | 内检测(ILI)系统资格认证：POD/POI/尺寸精度量化，验证与确认两独立步骤 | ILI 工具选型核心依据 |
| **ASME B31.8S** | 气体管道系统完整性管理 | 直接评估与响应框架 |
| **API 1160** | 危险液体管道完整性管理（第4版, 2021） | 与 B31.8S 对应 |
| **NACE SP0502** | 外腐蚀直接评估(ECDA) 四步法 | 不可内检测管线首选 |
| **NACE SP0102** | 管道内检测 | ILI 实施 |
| **NACE SP0204** | 应力腐蚀开裂直接评估(SCCDA) | SCC 威胁 |
| **API 1104 §11/Annex A** | 管道环焊缝 PAUT/TOFD 自动超声 | 建设期焊接检测 |
| **API 579-1 / ASME FFS-1** | 合于使用(FFS)工程临界评估 | 缺陷剩余强度 |
| **BS 7910** | 金属结构缺陷验收（ECA） | 裂纹评定 |
| **PRCI** | API 1163 验证指南 + 电子表格工具（L1/L2/L3 验证） | 与 CEPA 指南协同 |

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

## 三之补、系统内置管线类型与主导威胁预设

为覆盖不同工况，系统「腐蚀预测」与「腐蚀环境分析」联动 **10 类管线类型** 预设（详见 `src/pipeline_types.py`）：

| 管线类型 | 常用管材 | 主导腐蚀威胁 |
|----------|----------|--------------|
| 天然气长输 | 碳钢 | CO₂内腐蚀、H₂S开裂、外部腐蚀、SCC(近中性pH) |
| 原油/成品油长输 | 碳钢 | 外部腐蚀、内腐蚀(沉积水)、冲蚀(含砂)、MIC |
| 油气集输 | 碳钢 | CO₂内腐蚀、H₂S开裂、MIC、冲蚀(含砂) |
| 注水/注气 | 碳钢 | MIC、CO₂内腐蚀、冲蚀(含氧) |
| 海底管道 | 双相2205 | 海水外腐蚀、CO₂内腐蚀、H₂S开裂、电偶腐蚀、SCC |
| 城市燃气 | 碳钢/PE | 外部腐蚀(杂散电流)、应力腐蚀、第三方破坏 |
| 化工工艺 | 316SS | 点蚀(Cl⁻)、缝隙腐蚀、MIC、应力腐蚀 |
| 酸性气田(H₂S/CO₂) | 超级13Cr | H₂S开裂(SSC/HIC/SOHIC)、CO₂内腐蚀、SCC |
| 输水/给排水 | 碳钢 | 外部土壤腐蚀、MIC、内结垢 |
| 氢气/掺氢 | 碳钢 | 氢致开裂(HAC)、外部腐蚀、疲劳/氢脆 |

> 用户在「腐蚀预测」页点选管线类型并"套用典型工况"，即可一键载入该类型的典型 CO₂/H₂S/Cl⁻/流速/温度与推荐管材，并联动后果分析与维护建议。

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
