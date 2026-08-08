"""
industrial_scenarios.py
工业应用场景库 —— 面向「完整性工具」与「腐蚀环境分析」两个板块的扩展内容。

每个场景给出：
- label / description：场景概述
- typical_params：用于机器学习模型预测的输入（material/temperature/ph/co2_pressure/
  h2s_concentration/flow_rate/chloride_content）
- dominant_threats：主导腐蚀威胁（对应 consequence_remediation.REMEDIATION_KB 键）
- ndt_threat：主导威胁（对应 ndt_knowledge.ILI_THREAT_MAP 键，用于推荐 NDT/ILI）
- factors：腐蚀因素分析（列表）
- solutions：解决措施概述（列表，详尽的工程建议由 recommend_remediation 给出）

覆盖行业：炼油 / 海底 / LNG / 化工 / 市政供热 / 注水 / 掺氢 / 电厂 / 造纸 / 矿浆 /
酸性气田 / 长输天然气。
"""

INDUSTRIAL_SCENARIOS = {
    "refinery_cdu": {
        "label": "炼油厂常减压/馏分管线",
        "description": "炼油厂常减压装置及下游馏分管线，介质含硫化物、环烷酸与少量 CO₂，"
                       "并长期受保温层下腐蚀(CUI)威胁。高温段（>240°C）以环烷酸/高温硫腐蚀为主，"
                       "低温段以 CO₂/H₂S 酸性腐蚀与 CUI 为主。",
        "typical_params": {"material": "carbon_steel", "temperature": 120, "ph": 6.5,
                            "co2_pressure": 0.3, "h2s_concentration": 30, "flow_rate": 1.5,
                            "chloride_content": 2000},
        "dominant_threats": ["CO₂内腐蚀", "外部腐蚀"],
        "ndt_threat": "CO₂内腐蚀",
        "factors": [
            "高温硫/环烷酸腐蚀（>240°C 硫+环烷酸协同，速率可达数 mm/a）",
            "保温层下腐蚀(CUI)：保温层破损进水后形成闭塞腐蚀电池",
            "CO₂/H₂S 酸性腐蚀（低温段与注水/冷凝段）",
            "焊缝与热影响区材料劣化",
        ],
        "solutions": [
            "高温段选用 13Cr/316 或渗铝钢；环烷酸环境考虑 316L/825",
            "CUI 重点段采用防水保温 + 定期红外/UT 厚度监测",
            "低温段注缓蚀剂 + 脱水，控制 CO₂ 分压",
            "建立基于风险的检验(RBI, API 580/581)计划",
        ],
    },
    "offshore_riser": {
        "label": "海底平台立管/海底管道",
        "description": "海上油气平台立管与海底管线，外部受海水腐蚀与海生物污损，内部受 CO₂/H₂S"
                       "酸性腐蚀。阴极保护(CP)与涂层是外部防护主线，内部以缓蚀剂与材料升级为主。",
        "typical_params": {"material": "duplex_2205", "temperature": 15, "ph": 7.5,
                            "co2_pressure": 0.8, "h2s_concentration": 5, "flow_rate": 2.0,
                            "chloride_content": 15000},
        "dominant_threats": ["外部腐蚀", "CO₂内腐蚀"],
        "ndt_threat": "外部腐蚀",
        "factors": [
            "海水氯离子点蚀（Cl⁻≈15000–20000 ppm）",
            "海生物污损导致局部缺氧/浓差电池",
            "阴极保护屏蔽（管卡/阳极块遮挡）",
            "内部 CO₂ 甜腐蚀与多相流冲蚀",
        ],
        "solutions": [
            "选用 2205 双相钢（PREN≈35）提升抗点蚀能力",
            "涂层 + 牺牲阳极/外加电流 CP 联合防护",
            "立管段采用 UT 壁厚高频监测 + 阴极保护电位遥测",
            "内部加缓蚀剂并定期清管(pigging)",
        ],
    },
    "lng_terminal": {
        "label": "LNG 接收站低温管线",
        "description": "LNG 接收站低温（-162°C）输送与蒸发气(BOG)管线。低温下腐蚀速率本身较低，"
                       "主要风险来自保冷层下外部腐蚀(CUI)与保冷破损部位的凝水/微生物腐蚀，以及"
                       "BOG 管线常温段的 CO₂ 腐蚀。",
        "typical_params": {"material": "stainless_316", "temperature": -10, "ph": 7.0,
                            "co2_pressure": 0.1, "h2s_concentration": 1, "flow_rate": 1.0,
                            "chloride_content": 500},
        "dominant_threats": ["外部腐蚀", "MIC"],
        "ndt_threat": "外部腐蚀",
        "factors": [
            "保冷层下腐蚀(CUI)：冷热交变 + 凝水",
            "BOG 常温段 CO₂/H₂O 弱酸性腐蚀",
            "保冷支撑部位的应力腐蚀与疲劳",
            "不锈钢在含 Cl⁻ 凝水中的点蚀风险",
        ],
        "solutions": [
            "保冷系统设计防水排水，关键节点红外热成像巡检",
            "不锈钢选用 316/316L 并控制 Cl⁻ 与温度",
            "保冷破损部位及时修复 + 局部 UT/PT 检测",
            "定期排查保冷层下隐蔽腐蚀",
        ],
    },
    "chemical_acid": {
        "label": "化工厂酸性水管线",
        "description": "化工装置酸性水(含 H₂S、NH₃、酚、CN⁻)管线，pH 低、含硫含氨，"
                       "腐蚀以 CO₂/H₂S 酸性腐蚀与微生物腐蚀(MIC)为主，垢下与死区为高发部位。",
        "typical_params": {"material": "alloy_825", "temperature": 60, "ph": 4.0,
                            "co2_pressure": 0.2, "h2s_concentration": 50, "flow_rate": 1.0,
                            "chloride_content": 8000},
        "dominant_threats": ["CO₂内腐蚀", "MIC"],
        "ndt_threat": "MIC",
        "factors": [
            "低 pH 酸性水加速全面腐蚀",
            "SRB/APB 主导的微生物腐蚀（垢下离散点蚀）",
            "死区/低流速段的沉积与生物膜富集",
            "含硫含氨的应力腐蚀倾向",
        ],
        "solutions": [
            "材料升级至 825/625 合金抗酸抗 MIC",
            "冲击式杀菌剂（戊二醛/THPS/DBNPA）轮换投加",
            "提高流速消除死区，定期清管/冲洗",
            "ER 腐蚀探针 + 生物膜监测闭环",
        ],
    },
    "municipal_heating": {
        "label": "市政供热管网",
        "description": "城市集中供热一次/二次管网，介质为热水或蒸汽，主要风险为溶解氧腐蚀、"
                       "氧去极化腐蚀与水垢下腐蚀，以及补偿器/阀门等异种金属接头处的电偶腐蚀。",
        "typical_params": {"material": "carbon_steel", "temperature": 90, "ph": 8.0,
                            "co2_pressure": 0.0, "h2s_concentration": 0, "flow_rate": 1.0,
                            "chloride_content": 300},
        "dominant_threats": ["外部腐蚀", "电偶腐蚀"],
        "ndt_threat": "外部腐蚀",
        "factors": [
            "溶解氧腐蚀（补水除氧不彻底）",
            "水垢/沉积下的局部腐蚀",
            "补偿器、法兰等异种金属接头的电偶腐蚀",
            "直埋段土壤外腐蚀与阴极保护缺失",
        ],
        "solutions": [
            "严格除氧 + 缓蚀阻垢剂，控制 pH 与碱度",
            "异种金属接头加装绝缘法兰/接头",
            "直埋段采用外防腐层 + 阴极保护",
            "管网定期壁厚检测与阀门井巡检",
        ],
    },
    "water_injection": {
        "label": "油田注水/注水管线",
        "description": "油田注水系统将处理后的水或地层水高压回注，含溶解氧、SRB 与地层矿物质，"
                       "腐蚀以微生物腐蚀(MIC)与 CO₂ 腐蚀为主， injector 井口与弯头为高发部位。",
        "typical_params": {"material": "carbon_steel", "temperature": 40, "ph": 6.5,
                            "co2_pressure": 0.0, "h2s_concentration": 0, "flow_rate": 1.5,
                            "chloride_content": 1000},
        "dominant_threats": ["MIC", "CO₂内腐蚀"],
        "ndt_threat": "MIC",
        "factors": [
            "SRB 主导的微生物腐蚀（注水中常见）",
            "溶解氧未除尽导致的氧去极化腐蚀",
            "垢下/死水段的生物膜富集",
            "高压回注流速导致的冲蚀-腐蚀",
        ],
        "solutions": [
            "脱氧 + 杀菌剂（氧化性/非氧化性轮换）",
            "沿程加缓蚀剂成膜保护",
            "注水井井口与弯头高频 UT 壁厚监测",
            "定期清管与生物膜治理",
        ],
    },
    "hydrogen_blending": {
        "label": "掺氢输送管线（H₂-blending）",
        "description": "在天然气管道中掺入一定比例氢气（如 ≤20% vol）以实现低碳输送。氢会加剧"
                       "氢致开裂(HIC)/氢脆与焊缝失效风险，同时原有外部腐蚀与 SCC 威胁依然存在。",
        "typical_params": {"material": "carbon_steel", "temperature": 30, "ph": 7.0,
                            "co2_pressure": 0.0, "h2s_concentration": 0, "flow_rate": 5.0,
                            "chloride_content": 200},
        "dominant_threats": ["H₂S开裂", "外部腐蚀"],
        "ndt_threat": "裂纹/开裂",
        "factors": [
            "氢致开裂(HIC)/氢脆（特别是焊缝与热影响区）",
            "掺氢后材料硬度敏感的 SSC 风险",
            "原有外部腐蚀与近中性 pH SCC",
            "高压高流速下焊缝疲劳",
        ],
        "solutions": [
            "控制材料硬度 ≤ 22 HRC，按 NACE MR0175/ISO 15156 选材",
            "焊后热处理(PWHT)降低残余氢敏感",
            "裂纹型 ILI(UT-CD) + PAUT 焊缝检测",
            "掺氢比例与运行压力分步验证（试点运行）",
        ],
    },
    "power_cooling": {
        "label": "电厂凝汽器冷却水管",
        "description": "沿海电厂凝汽器与循环冷却水管，介质为海水或淡水，管材多为钛/不锈钢/铜合金，"
                       "风险来自海水点蚀、异种金属接头电偶腐蚀与沉积物下腐蚀。",
        "typical_params": {"material": "titanium", "temperature": 35, "ph": 8.0,
                            "co2_pressure": 0.0, "h2s_concentration": 0, "flow_rate": 2.0,
                            "chloride_content": 20000},
        "dominant_threats": ["外部腐蚀", "电偶腐蚀"],
        "ndt_threat": "外部腐蚀",
        "factors": [
            "海水高 Cl⁻ 点蚀（尤其不锈钢）",
            "钛/不锈钢与铜合金接头的电偶腐蚀",
            "沉积物/海生物下的局部腐蚀",
            "流速过低导致的污损与过高导致的冲蚀",
        ],
        "solutions": [
            "主材选用钛或高 PREN 双相钢，异种接头绝缘隔离",
            "控制流速在合理区间（防污损又防冲蚀）",
            "阴极保护 + 杀生剂控制海生物",
            "定期涡流(ET)/UT 管壁检测",
        ],
    },
    "pulp_paper": {
        "label": "造纸厂漂白工段管线",
        "description": "造纸厂漂白与化学回收工段管线，介质含二氧化氯、次氯酸盐与低 pH 酸性液，"
                       "腐蚀以低 pH 酸腐蚀与微生物腐蚀(MIC)为主，316/双相钢在极端段仍可能点蚀。",
        "typical_params": {"material": "alloy_625", "temperature": 70, "ph": 3.5,
                            "co2_pressure": 0.0, "h2s_concentration": 0, "flow_rate": 1.5,
                            "chloride_content": 1000},
        "dominant_threats": ["MIC", "CO₂内腐蚀"],
        "ndt_threat": "MIC",
        "factors": [
            "二氧化氯/次氯酸盐强氧化性酸腐蚀",
            "低 pH 环境下的全面与点蚀",
            "有机垢下微生物腐蚀(MIC)",
            "高温段的材料应力腐蚀",
        ],
        "solutions": [
            "极端段选用 625/C-276 等镍基合金",
            "控制 pH 与氧化剂浓度，优化缓蚀",
            "定期杀菌与清垢，监测生物膜",
            "关键节点 UT/PT 表面与壁厚检测",
        ],
    },
    "mining_slurry": {
        "label": "矿浆/尾矿输送管线",
        "description": "选矿厂矿浆（含固体颗粒）长距离输送管线，腐蚀与磨损协同（冲蚀-腐蚀），"
                       "弯头、三通与泵出口为磨损失效高发部位。",
        "typical_params": {"material": "carbon_steel", "temperature": 25, "ph": 7.5,
                            "co2_pressure": 0.0, "h2s_concentration": 0, "flow_rate": 3.0,
                            "chloride_content": 500},
        "dominant_threats": ["冲蚀", "外部腐蚀"],
        "ndt_threat": "冲蚀(含砂)",
        "factors": [
            "含固颗粒导致的冲蚀-腐蚀协同",
            "弯头/三通/泵出口局部高速磨损",
            "矿浆 pH 与含氧导致的腐蚀",
            "外部土壤腐蚀（埋地段）",
        ],
        "solutions": [
            "弯头/三通采用耐磨衬里或加厚设计",
            "优化流速与含固量平衡（API RP 14E）",
            "加缓蚀剂成膜抗冲蚀",
            "弯头/阀门 UT 壁厚高频监测",
        ],
    },
    "sour_gas_plant": {
        "label": "酸性气田集气干线",
        "description": "高含 H₂S/CO₂ 的酸性气田集气与处理干线，材料须满足 NACE MR0175/ISO 15156"
                       "酸性服役要求，SSC/HIC 与甜/酸腐蚀是核心威胁。",
        "typical_params": {"material": "13cr", "temperature": 65, "ph": 5.5,
                            "co2_pressure": 3.0, "h2s_concentration": 200, "flow_rate": 4.0,
                            "chloride_content": 5000},
        "dominant_threats": ["H₂S开裂", "CO₂内腐蚀"],
        "ndt_threat": "H₂S开裂(SSC/HIC)",
        "factors": [
            "高 H₂S 分压下的 SSC/HIC/SOHIC 开裂",
            "高 CO₂ 分压甜腐蚀",
            "酸性环境下的局部点蚀",
            "焊缝与热影响区硬度敏感",
        ],
        "solutions": [
            "按 MR0175/ISO 15156 选用合格 CRA（13Cr/超级13Cr/825）",
            "严格控制硬度 ≤ 22 HRC 与焊后热处理",
            "脱水降低 H₂S/CO₂ 分压，注缓蚀剂",
            "裂纹型 ILI + 硬度普查 + 焊缝 PAUT",
        ],
    },
    "crosscountry_gas": {
        "label": "长输天然气干线",
        "description": "跨区域长输天然气管道，主要风险为埋地外部腐蚀（涂层失效+CP 屏蔽）与近中性 pH"
                       "应力腐蚀开裂(SCC)，高后果区(HCA)段还需关注第三方破坏与疲劳。",
        "typical_params": {"material": "carbon_steel", "temperature": 40, "ph": 6.5,
                            "co2_pressure": 0.5, "h2s_concentration": 20, "flow_rate": 6.0,
                            "chloride_content": 100},
        "dominant_threats": ["外部腐蚀", "SCC"],
        "ndt_threat": "SCC(近中性pH)",
        "factors": [
            "埋地外腐蚀（涂层剥离 + 阴极保护屏蔽）",
            "近中性 pH SCC（富 CO₂ 地下水，pH 6–7.5）",
            "高后果区(HCA)的泄漏后果放大",
            "第三方破坏与应力集中",
        ],
        "solutions": [
            "外防腐层完整性管理 + 阴极保护电位年度测量",
            "ECDA(NACE SP0502) 间接检测 + 开挖验证",
            "高 SCC 风险段采用裂纹型 ILI(UT-CD)",
            "HCA 段 SCCDA 直接评估（NACE SP0204）",
        ],
    },
}


def get_industrial_scenarios():
    """返回工业场景字典（key -> 场景定义）。"""
    return INDUSTRIAL_SCENARIOS


def scenario_keys_in_order():
    """返回场景 key 的有序列表，供 selectbox 使用。"""
    return list(INDUSTRIAL_SCENARIOS.keys())
