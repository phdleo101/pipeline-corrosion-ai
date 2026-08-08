"""
pipeline_types.py
管线类型预设 —— 面向"4 大腐蚀痛点"（内腐蚀 CO₂/H₂S、外腐蚀/土壤、SCC、MIC）
覆盖更多管线类型与典型工况，自动填充预测/环境分析的典型参数。

每类管线给出：
- default_material: 常用管材
- dominant_threats: 主导腐蚀威胁（用于联动后果与维护建议）
- env: 典型环境参数范围（CO₂ 分压 / H₂S / 氯离子 / 流速 / 温度 / pH）
- notes: 工程说明与典型失效案例
"""

# 管线类型预设字典
# env 字段单位：co2_pressure(MPa), h2s(ppm), chloride(ppm), flow(m/s), temp(°C), ph
PIPELINE_PRESETS = {
    "gas_transmission": {
        "label": "天然气长输管道",
        "default_material": "carbon_steel",
        "product_hazard": "高（易燃/爆炸）",
        "dominant_threats": ["CO₂内腐蚀", "H₂S开裂(SSC/HIC)", "外部腐蚀", "SCC(近中性pH)"],
        "env": {
            "co2_pressure": 1.2, "h2s": 30, "chloride": 500,
            "flow": 6.0, "temp": 40, "ph": 6.8,
        },
        "typical": "管径 DN500–1200，压力 6–12 MPa，干气或湿气；干气段内腐蚀轻，湿气段 CO₂ 甜腐蚀突出。",
        "notes": "北美多条长输管线因近中性 pH SCC 失效；酸性气田来气须按 NACE MR0175/ISO 15156 选材。",
    },
    "oil_transmission": {
        "label": "原油/成品油长输管道",
        "default_material": "carbon_steel",
        "product_hazard": "中（环境污染）",
        "dominant_threats": ["外部腐蚀", "内腐蚀(CO₂+沉积水)", "冲蚀(含砂)", "MIC"],
        "env": {
            "co2_pressure": 0.3, "h2s": 10, "chloride": 3000,
            "flow": 2.0, "temp": 55, "ph": 6.5,
        },
        "typical": "管径 DN400–1000，压力 4–10 MPa；含地层水段底部 6 点钟位置内腐蚀 + MIC。",
        "notes": "低流速段沉积水诱发 MIC 与垢下腐蚀；高含砂原油冲蚀弯头/阀门。",
    },
    "gathering": {
        "label": "油气集输管道",
        "default_material": "carbon_steel",
        "product_hazard": "高（含 H₂S/易燃）",
        "dominant_threats": ["CO₂内腐蚀", "H₂S开裂", "MIC", "冲蚀(含砂)"],
        "env": {
            "co2_pressure": 2.5, "h2s": 200, "chloride": 15000,
            "flow": 3.5, "temp": 70, "ph": 5.5,
        },
        "typical": "井场→处理厂，多变工况、高 CO₂/H₂S、高 Cl⁻、含砂、高含水；腐蚀最复杂。",
        "notes": "酸性气田集输是腐蚀最严峻场景，常采用 CRA(13Cr/双相)或缓蚀剂批处理。",
    },
    "water_injection": {
        "label": "注水/注气管道",
        "default_material": "carbon_steel",
        "product_hazard": "低（水，环境局部）",
        "dominant_threats": ["MIC", "CO₂内腐蚀", "冲蚀(含氧)"],
        "env": {
            "co2_pressure": 0.5, "h2s": 5, "chloride": 30000,
            "flow": 2.5, "temp": 45, "ph": 7.0,
        },
        "typical": "回注污水高矿化度、含氧/SRB；注入井附近高流速冲蚀 + 全面 MIC。",
        "notes": "SRB 在 20–60°C、低流速死区最活跃；须脱氧 + 杀菌剂 + 缓蚀剂联用。",
    },
    "subsea": {
        "label": "海底管道",
        "default_material": "duplex_2205",
        "product_hazard": "高（海洋环境/难以抢修）",
        "dominant_threats": ["外部海水腐蚀", "CO₂内腐蚀", "H₂S开裂", "电偶腐蚀", "SCC"],
        "env": {
            "co2_pressure": 3.0, "h2s": 50, "chloride": 35000,
            "flow": 5.0, "temp": 90, "ph": 6.2,
        },
        "typical": "海水全浸 + 高温高压(HPHT)；外防腐靠涂层+阴极保护，内腐蚀靠 CRA 或缓蚀剂。",
        "notes": "修复成本高、窗口短；选材以双相/超级双相或 825 为主，PREN 要求高。",
    },
    "city_gas": {
        "label": "城市燃气管道",
        "default_material": "carbon_steel",
        "product_hazard": "极高（人口密集/爆炸）",
        "dominant_threats": ["外部腐蚀(杂散电流)", "应力腐蚀(PE 管与钢接头)", "第三方破坏"],
        "env": {
            "co2_pressure": 0.05, "h2s": 2, "chloride": 2000,
            "flow": 8.0, "temp": 20, "ph": 7.2,
        },
        "typical": "中低压(0.4 MPa 以下)，钢质老旧管网外腐蚀为主；PE 管逐步替代。",
        "notes": "杂散电流与阴极保护失效是城市钢质管网外腐蚀主因；高后果区(HCA)密集。",
    },
    "chemical": {
        "label": "化工工艺管道",
        "default_material": "316ss",
        "product_hazard": "高（有毒/腐蚀介质）",
        "dominant_threats": ["点蚀(Cl⁻)", "缝隙腐蚀", "MIC", "应力腐蚀"],
        "env": {
            "co2_pressure": 0.1, "h2s": 0, "chloride": 8000,
            "flow": 2.0, "temp": 60, "ph": 4.0,
        },
        "typical": "酸性/含氯工艺介质，温度与浓度波动大；点蚀与缝隙腐蚀主导。",
        "notes": "316 在高 Cl⁻+高温下点蚀；关键部位用 2205/625；注意焊后热处理。",
    },
    "sour_gas": {
        "label": "酸性气田管道(H₂S/CO₂)",
        "default_material": "super_13cr",
        "product_hazard": "极高（H₂S 剧毒/爆炸）",
        "dominant_threats": ["H₂S开裂(SSC/HIC/SOHIC)", "CO₂内腐蚀", "SCC"],
        "env": {
            "co2_pressure": 4.0, "h2s": 500, "chloride": 10000,
            "flow": 4.0, "temp": 80, "ph": 5.0,
        },
        "typical": "高 H₂S(>50 ppm 即酸性服役)、高 CO₂；按 NACE MR0175/ISO 15156 严格选材。",
        "notes": "硬度控制 ≤ 22 HRC、HIC 试验(CLR/CTR/CSR)、焊后热处理(PWHT)缺一不可。",
    },
    "water_supply": {
        "label": "输水/给排水管道",
        "default_material": "carbon_steel",
        "product_hazard": "低（民生/局部）",
        "dominant_threats": ["外部土壤腐蚀", "MIC", "内结垢/局部腐蚀"],
        "env": {
            "co2_pressure": 0.0, "h2s": 0, "chloride": 500,
            "flow": 1.5, "temp": 15, "ph": 7.5,
        },
        "typical": "埋地钢管外腐蚀 + 饮用水管内壁铁细菌/硫酸盐还原菌；混凝土管碱集料。",
        "notes": "饮用水管内壁须控制消毒副产物，缓蚀剂选择受限；外防腐靠涂层+CP。",
    },
    "hydrogen": {
        "label": "氢气/掺氢管道",
        "default_material": "carbon_steel",
        "product_hazard": "高（氢脆/易燃）",
        "dominant_threats": ["氢致开裂(HAC)", "外部腐蚀", "疲劳/氢脆"],
        "env": {
            "co2_pressure": 0.0, "h2s": 0, "chloride": 1000,
            "flow": 7.0, "temp": 30, "ph": 7.0,
        },
        "typical": "纯氢或天然气掺氢(≤20%)；氢进入金属致氢脆/台阶状开裂，硬度敏感。",
        "notes": "新兴场景；选材须控制硬度、进行氢脆评估，参考 ASME B31.12 / NACE TM0177。",
    },
}


def get_pipeline_presets():
    """返回 {key: label} 用于下拉框"""
    return {k: v["label"] for k, v in PIPELINE_PRESETS.items()}


def get_preset(key):
    """返回某管线类型的完整预设字典"""
    return PIPELINE_PRESETS.get(key, PIPELINE_PRESETS["gas_transmission"])


def apply_preset_to_inputs(key, base_inputs):
    """
    将管线类型预设的环境参数合并到预测输入，仅对"未显式改变"的参数给默认值。
    这里直接返回预设环境值（UI 端调用时作为 slider 默认值）。
    """
    p = get_preset(key)
    merged = dict(base_inputs)
    for k, v in p["env"].items():
        merged[k] = v
    merged["default_material"] = p["default_material"]
    merged["dominant_threats"] = p["dominant_threats"]
    merged["product_hazard"] = p["product_hazard"]
    return merged
