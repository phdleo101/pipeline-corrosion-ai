"""
environment_models.py
多环境腐蚀模型与腐蚀成本估算模块（P3 增强）

- 土壤腐蚀（参考 DIN 50929 土壤腐蚀性分级思路）
- 海水腐蚀（溶解氧 / 盐度 / 温度 / 流速 / 生物污损）
- 微生物腐蚀 MIC（SRB 硫酸盐还原菌驱动）
- 电偶腐蚀（电偶序电位差 + 面积比）
- 腐蚀成本估算（金属损失 / 检测 / 停产 / 维修）

说明：以下公式为工程简化估算，用于方案比选与风险初筛，
正式评估应以现场挂片、在线监测及标准方法为准。
"""


# ----------------------------------------------------------------------
# 通用：腐蚀速率 → 严重程度
# ----------------------------------------------------------------------
def _severity_from_rate(rate):
    """根据均匀腐蚀速率(mm/a)给出严重程度标签与颜色。"""
    if rate < 0.05:
        return "极轻微", "#27ae60"
    elif rate < 0.1:
        return "轻微", "#2ecc71"
    elif rate < 0.25:
        return "中等", "#f39c12"
    elif rate < 0.5:
        return "严重", "#e74c3c"
    else:
        return "极严重", "#c0392b"


# 各材料在土壤/海水中的基础耐蚀系数（相对碳钢，越小越耐蚀）
_MATERIAL_FACTOR = {
    "carbon_steel": 1.00,
    "stainless_316": 0.35,
    "13cr": 0.55,
    "super_13cr": 0.45,
    "duplex_2205": 0.30,
    "duplex_2507": 0.25,
    "alloy_825": 0.20,
    "alloy_625": 0.12,
    "alloy_c276": 0.08,
    "titanium": 0.05,
}
_MATERIAL_NAME = {
    "carbon_steel": "碳钢",
    "stainless_316": "316不锈钢",
    "13cr": "13Cr马氏体不锈钢",
    "super_13cr": "超级13Cr",
    "duplex_2205": "2205双相不锈钢",
    "duplex_2507": "2507超级双相不锈钢",
    "alloy_825": "825合金",
    "alloy_625": "625合金",
    "alloy_c276": "C-276合金",
    "titanium": "钛合金",
}


# ----------------------------------------------------------------------
# 1. 土壤腐蚀
# ----------------------------------------------------------------------
def soil_corrosion(resistivity, ph, moisture, chloride, sulfate, material="carbon_steel"):
    """
    土壤腐蚀速率估算（参考 DIN 50929 分级思路）

    参数:
        resistivity: 土壤电阻率 (Ω·cm)
        ph: 土壤 pH
        moisture: 含水率 (%)
        chloride: 氯离子含量 (ppm)
        sulfate: 硫酸根含量 (ppm)
        material: 管材代码

    返回: 估算结果字典
    """
    # 电阻率评分（电阻率越低越腐蚀）
    if resistivity > 5000:
        s_res = 0
    elif resistivity > 2000:
        s_res = 1
    elif resistivity > 1000:
        s_res = 2
    elif resistivity > 500:
        s_res = 3
    else:
        s_res = 4

    # pH 评分（酸性/强碱性更腐蚀）
    if 6.5 <= ph <= 8.5:
        s_ph = 0
    elif (5.5 <= ph < 6.5) or (8.5 < ph <= 9.5):
        s_ph = 1
    elif (4.5 <= ph < 5.5) or (9.5 < ph <= 11):
        s_ph = 2
    elif (3.5 <= ph < 4.5) or (11 < ph <= 12.5):
        s_ph = 3
    else:
        s_ph = 4

    # 含水率评分（过干过湿都不利，中等含水+透气最危险）
    if moisture < 10:
        s_moist = 1
    elif moisture <= 25:
        s_moist = 3
    elif moisture <= 40:
        s_moist = 2
    else:
        s_moist = 1

    # 氯离子评分
    if chloride < 100:
        s_cl = 0
    elif chloride < 500:
        s_cl = 1
    elif chloride < 2000:
        s_cl = 2
    else:
        s_cl = 3

    # 硫酸根评分（促进硫酸盐还原菌 MIC）
    if sulfate < 200:
        s_so4 = 0
    elif sulfate < 1000:
        s_so4 = 1
    elif sulfate < 5000:
        s_so4 = 2
    else:
        s_so4 = 3

    score = s_res + s_ph + s_moist + s_cl + s_so4  # 0 ~ 17

    # 土壤腐蚀性分级
    if score <= 4:
        corrosivity = "极低"
        base_rate = 0.02
    elif score <= 7:
        corrosivity = "低"
        base_rate = 0.05
    elif score <= 10:
        corrosivity = "中"
        base_rate = 0.10
    elif score <= 13:
        corrosivity = "高"
        base_rate = 0.20
    else:
        corrosivity = "极高"
        base_rate = 0.40

    factor = _MATERIAL_FACTOR.get(material, 1.0)
    rate = round(base_rate * factor, 4)
    severity, color = _severity_from_rate(rate)

    return {
        "score": score,
        "corrosivity": corrosivity,
        "base_rate": base_rate,
        "rate": rate,
        "material": _MATERIAL_NAME.get(material, material),
        "material_factor": factor,
        "severity": severity,
        "color": color,
        "detail": {
            "电阻率评分": s_res,
            "pH评分": s_ph,
            "含水率评分": s_moist,
            "氯离子评分": s_cl,
            "硫酸根评分": s_so4,
        },
        "advice": _soil_advice(corrosivity),
    }


def _soil_advice(corrosivity):
    base = "建议采用外防腐层（3PE/FBE）+ 强制电流/牺牲阳极阴极保护组合方案。"
    if corrosivity == "极高":
        return f"土壤腐蚀性强：{base} 并加密阴极保护测试桩，重点关注穿跨越段与低洼积水段。"
    elif corrosivity == "高":
        return f"土壤腐蚀性较高：{base} 加强绝缘接头与排流设施。"
    elif corrosivity == "中":
        return f"土壤腐蚀性中等：{base} 定期开展阴极保护电位检测。"
    elif corrosivity == "低":
        return f"土壤腐蚀性较低：{base} 常规监测即可。"
    else:
        return f"土壤腐蚀性极低：{base} 可适当延长检测周期。"


# ----------------------------------------------------------------------
# 2. 海水腐蚀
# ----------------------------------------------------------------------
def seawater_corrosion(oxygen, salinity, temperature, flow_rate, material="carbon_steel"):
    """
    海水腐蚀速率估算（碳钢均匀腐蚀 + 不锈钢点蚀风险提示）

    参数:
        oxygen: 溶解氧 (mg/L)
        salinity: 盐度 (‰)
        temperature: 温度 (°C)
        flow_rate: 流速 (m/s)
        material: 管材代码

    返回: 估算结果字典
    """
    # 碳钢海水均匀腐蚀基础速率（mg/L O2、35‰盐度、20°C、静水）
    base_rate = 0.10

    # 溶解氧影响（含氧越高腐蚀越快，0~10 mg/L）
    f_ox = 0.6 + oxygen / 10.0 * 0.8  # 0.6 ~ 1.4

    # 温度影响（每升高10°C速率约翻倍，Arrhenius简化）
    f_temp = 2.0 ** ((temperature - 20) / 20.0)  # 20°C=1.0, 40°C=2.0, 0°C=0.5

    # 流速影响（冲刷腐蚀）：<1 轻微，1-3 正常，>3 加剧
    if flow_rate < 1:
        f_flow = 0.85
    elif flow_rate <= 3:
        f_flow = 1.0
    elif flow_rate <= 6:
        f_flow = 1.5
    else:
        f_flow = 2.2  # 高流速冲刷+磨损腐蚀

    # 盐度影响（淡化/高盐略有差异，35‰为基准）
    f_sal = 0.8 + (salinity / 35.0) * 0.4  # 0.8 ~ 1.2

    rate_steel = base_rate * f_ox * f_temp * f_flow * f_sal
    factor = _MATERIAL_FACTOR.get(material, 1.0)
    rate = round(rate_steel * factor, 4)
    severity, color = _severity_from_rate(rate)

    # 不锈钢/双相钢点蚀风险提示（PREN）
    pren = _pren(material)
    pitting_risk = None
    if pren is not None:
        # 海水氯离子约 19000 ppm，温度越高点蚀风险越大
        if pren < 32:
            pitting_risk = "高风险（PREN<32，易发生点蚀/缝隙腐蚀）"
        elif pren < 40:
            pitting_risk = "中风险（PREN 32-40，需控制温度与流速）"
        else:
            pitting_risk = "低风险（PREN>40，耐海水点蚀良好）"

    return {
        "rate": rate,
        "material": _MATERIAL_NAME.get(material, material),
        "severity": severity,
        "color": color,
        "factors": {
            "溶解氧因子": round(f_ox, 2),
            "温度因子": round(f_temp, 2),
            "流速因子": round(f_flow, 2),
            "盐度因子": round(f_sal, 2),
        },
        "pren": pren,
        "pitting_risk": pitting_risk,
        "advice": _seawater_advice(severity, pitting_risk),
    }


def _pren(material):
    """常见材料点蚀当量 PREN 近似（仅用于风险提示）。"""
    pren_map = {
        "carbon_steel": None,
        "stainless_316": 25,
        "13cr": 13,
        "super_13cr": 16,
        "duplex_2205": 34,
        "duplex_2507": 42,
        "alloy_825": 30,
        "alloy_625": 48,
        "alloy_c276": 65,
        "titanium": 99,
    }
    return pren_map.get(material)


def _seawater_advice(severity, pitting_risk):
    adv = "碳钢海水管线建议采用涂层+阴极保护，并关注海生物污损与冲刷腐蚀。"
    if pitting_risk and "高" in pitting_risk:
        adv += " 当前材料耐点蚀不足，海水环境建议升级为双相钢(2205/2507)或镍基合金。"
    elif pitting_risk and "中" in pitting_risk:
        adv += " 当前材料点蚀风险中等，需控制海水温度(<40°C)并避免缝隙结构。"
    return adv


# ----------------------------------------------------------------------
# 3. 微生物腐蚀 MIC
# ----------------------------------------------------------------------
def mic_corrosion(srb_count, nutrient, temperature, oxygen, material="carbon_steel"):
    """
    微生物腐蚀（MIC）风险评估，以硫酸盐还原菌 SRB 为主要驱动

    参数:
        srb_count: SRB 数量 (MPN/mL，对数级，常见 10^2 ~ 10^6)
        nutrient: 营养物水平 (低/中/高)
        temperature: 温度 (°C)
        oxygen: 溶解氧 (mg/L，高氧抑制严格厌氧菌但促进好氧菌)
        material: 管材代码

    返回: 估算结果字典
    """
    # SRB 数量评分（对数）
    import math
    try:
        log_srb = math.log10(max(srb_count, 1))
    except (ValueError, TypeError):
        log_srb = 2
    if log_srb < 3:
        s_srb = 1
    elif log_srb < 4:
        s_srb = 2
    elif log_srb < 5:
        s_srb = 3
    else:
        s_srb = 4

    nutrient_score = {"低": 0, "中": 1, "高": 2}.get(nutrient, 1)

    # 温度评分（最适 25-40°C）
    if 25 <= temperature <= 45:
        s_temp = 2
    elif 10 <= temperature < 25 or 45 < temperature <= 60:
        s_temp = 1
    else:
        s_temp = 0

    # 氧评分（严格厌氧菌偏好低氧，但微氧促进生物膜）
    if oxygen < 0.5:
        s_ox = 1
    elif oxygen <= 2:
        s_ox = 2  # 微氧环境最易形成生物膜
    else:
        s_ox = 1

    score = s_srb + nutrient_score + s_temp + s_ox  # 0 ~ 9

    if score <= 2:
        risk = "极低"
        rate = 0.02
        color = "#27ae60"
    elif score <= 4:
        risk = "中等"
        rate = 0.08
        color = "#f39c12"
    elif score <= 6:
        risk = "高"
        rate = 0.20
        color = "#e74c3c"
    else:
        risk = "极高"
        rate = 0.45
        color = "#c0392b"

    factor = _MATERIAL_FACTOR.get(material, 1.0)
    rate = round(rate * factor, 4)

    return {
        "score": score,
        "risk": risk,
        "rate": rate,
        "material": _MATERIAL_NAME.get(material, material),
        "color": color,
        "detail": {
            "SRB评分": s_srb,
            "营养物评分": nutrient_score,
            "温度评分": s_temp,
            "氧评分": s_ox,
        },
        "advice": _mic_advice(risk),
    }


def _mic_advice(risk):
    base = "建议投加杀菌剂（氧化性+非氧化性交替）、定期生物清扫、控制沉积与死水区。"
    if risk == "极高":
        return f"MIC 风险极高：{base} 增加 SRB 监测频次，必要时进行管段更换。"
    elif risk == "高":
        return f"MIC 风险高：{base} 开展微生物检测与生物膜评估。"
    elif risk == "中等":
        return f"MIC 风险中等：{base} 纳入例行监测。"
    else:
        return f"MIC 风险极低：{base} 维持现状。"


# ----------------------------------------------------------------------
# 3b. MIC 多菌属评估与生物膜热点 (NACE SP0192 / TM0194)
# ----------------------------------------------------------------------

def mic_multi_organism(srb, apb, irb, sob, nutrient, temperature, oxygen,
                       flow_regime="正常", dead_leg=False, material="carbon_steel"):
    """
    多菌属 MIC 风险评估（在 SRB 基础上扩展到 APB/IRB/SOB，并定位生物膜热点）

    SRB 主导经典的硫酸盐还原产 H2S 腐蚀；APB（产酸菌）使局部 pH 下降；
    IRB（铁氧化菌）形成结瘤与氧浓差电池；SOB（硫氧化菌）把硫化物氧化为硫酸，
    造成极强局部酸化。生物膜与沉积物下方(under-deposit)是 MIC 高发区。

    参数:
        srb/apb/irb/sob: 各菌属数量 (MPN/mL，对数级常见 10^2~10^6)
        nutrient: 营养物水平 (低/中/高)
        temperature: 温度 (°C)
        oxygen: 溶解氧 (mg/L)
        flow_regime: 流速状态 (低流速/正常/高流速)
        dead_leg: 是否存在死管/滞留段
        material: 管材代码
    返回: 字典
    """
    import math

    def _log_score(v):
        try:
            lg = math.log10(max(v, 1))
        except (ValueError, TypeError):
            lg = 2
        if lg < 3:
            return 1
        elif lg < 4:
            return 2
        elif lg < 5:
            return 3
        return 4

    s_srb = _log_score(srb)
    s_apb = _log_score(apb)
    s_irb = _log_score(irb)
    s_sob = _log_score(sob)
    nutrient_score = {"低": 0, "中": 1, "高": 2}.get(nutrient, 1)
    if 25 <= temperature <= 45:
        s_temp = 2
    elif 10 <= temperature < 25 or 45 < temperature <= 60:
        s_temp = 1
    else:
        s_temp = 0
    if oxygen < 0.5:
        s_ox = 1
    elif oxygen <= 2:
        s_ox = 2
    else:
        s_ox = 1

    # 生物膜热点（低流速/死管/积水促进沉积与厌氧微区）
    hotspots = []
    if flow_regime == "低流速" or dead_leg:
        hotspots.append("低流速/死管段：沉积与滞留，生物膜易富集")
    if flow_regime == "高流速":
        hotspots.append("高流速段：冲刷抑制生物膜，但弯头/缩径处仍可能结瘤")
    else:
        hotspots.append("焊缝/垫片/缝隙：under-deposit 微环境利于 MIC")

    # 主导菌属
    scores = {"SRB": s_srb, "APB(产酸)": s_apb, "IRB(铁氧化)": s_irb, "SOB(硫氧化)": s_sob}
    dominant = max(scores, key=scores.get)
    org_sum = s_srb + s_apb + s_irb + s_sob

    mic_index = org_sum + nutrient_score + s_temp + s_ox  # 0 ~ 19
    if mic_index <= 4:
        risk = "极低"
        rate = 0.02
        color = "#27ae60"
    elif mic_index <= 8:
        risk = "中等"
        rate = 0.08
        color = "#f39c12"
    elif mic_index <= 13:
        risk = "高"
        rate = 0.20
        color = "#e74c3c"
    else:
        risk = "极高"
        rate = 0.45
        color = "#c0392b"

    factor = _MATERIAL_FACTOR.get(material, 1.0)
    rate = round(rate * factor, 4)

    return {
        "mic_index": mic_index,
        "risk": risk,
        "rate": rate,
        "color": color,
        "dominant": dominant,
        "organism_scores": scores,
        "hotspots": hotspots,
        "advice": _mic_advice(risk),
        "reference": "NACE SP0192 (MIC 控制); NACE TM0194 (现场微生物检测); API RP 38",
    }


# ----------------------------------------------------------------------
# 3c. MIC 杀菌剂方案设计 (NACE SP0192 / TM0212)
# ----------------------------------------------------------------------

def mic_biocide_program(risk_level, system_type="间歇系统", water_temp=30.0):
    """
    微生物腐蚀杀菌剂方案设计（基于 MIC 风险等级给出可操作投加方案）

    非氧化性杀菌剂（戊二醛 glutaraldehyde、THPS 四羟甲基硫酸磷、DBNPA）对
    SRB 高效且不受硫化物消耗；氧化性（次氯酸钠/二氧化氯）便宜但易被消耗。
    为防止抗药性，建议两类轮换投加。

    参数:
        risk_level: MIC 风险等级 (极低/中等/高/极高)
        system_type: 系统类型 (间歇系统/连续系统)
        water_temp: 水温 (°C)
    返回: 字典
    """
    plan = {
        "极低": {
            "type": "氧化性为主（次氯酸钠/二氧化氯）",
            "mode": "冲击投加",
            "dose_ppm": "20–50 ppm，每季度 1 次",
            "rotation": "单一氧化性即可",
            "monitor": "半年一次 SRB 检测",
        },
        "中等": {
            "type": "非氧化性(戊二醛) + 氧化性轮换",
            "mode": "冲击投加",
            "dose_ppm": "戊二醛 100–150 ppm，每 1–2 月 1 次",
            "rotation": "戊二醛 ↔ THPS 季度轮换",
            "monitor": "每季度 SRB + 挂片失重",
        },
        "高": {
            "type": "非氧化性(THPS / DBNPA) + 氧化性",
            "mode": "冲击 + 必要时连续",
            "dose_ppm": "THPS 150–200 ppm 冲击；连续 10–20 ppm",
            "rotation": "THPS ↔ 戊二醛 ↔ DBNPA 季度轮换",
            "monitor": "每月 SRB 杀灭率(目标≥99%) + 生物膜探针",
        },
        "极高": {
            "type": "非氧化性(戊二醛/THPS/DBNPA 组合) + 氧化性强化",
            "mode": "连续 + 高频冲击",
            "dose_ppm": "连续 15–25 ppm；冲击 200 ppm 每 2 周",
            "rotation": "三种非氧化性季度轮换 + 氧化性辅助",
            "monitor": "每月 SRB 杀灭率 + 在线腐蚀探针 + ILI 复核",
        },
    }
    p = plan.get(risk_level, plan["中等"])
    # 水温提示
    temp_note = ""
    if water_temp > 50:
        temp_note = "水温偏高：优先选热稳定好的 THPS/DBNPA，戊二醛高温易降解。"
    elif water_temp < 15:
        temp_note = "水温偏低：微生物活性下降，但仍需维持投加以防越冬群落复苏。"

    return {
        "risk_level": risk_level,
        "biocide_type": p["type"],
        "dosing_mode": p["mode"],
        "dose": p["dose_ppm"],
        "rotation": p["rotation"],
        "monitoring": p["monitor"],
        "temp_note": temp_note,
        "reference": "NACE SP0192 (MIC 控制); NACE TM0212 (杀菌剂评价); NACE TM0194",
    }


# ----------------------------------------------------------------------
# 3d. MIC 材料升级决策 (复用 PREN 思路，NACE SP0192 / MR0175)
# ----------------------------------------------------------------------

# 材料升级阶梯（按耐蚀性递增），含近似 PREN 用于点蚀抗力对比
_MATERIAL_LADDER = [
    ("carbon_steel", "碳钢", 0),
    ("stainless_316", "316不锈钢", 25),
    ("duplex_2205", "2205双相不锈钢", 34),
    ("duplex_2507", "2507超级双相不锈钢", 42),
    ("alloy_825", "825合金", 30),
    ("alloy_625", "625合金", 48),
    ("alloy_c276", "C-276合金", 65),
    ("titanium", "钛合金", 99),
]


def mic_material_upgrade(risk_level, current_material="carbon_steel",
                         chloride_ppm=0, temperature=30.0):
    """
    MIC 材料升级决策（基于 MIC 风险等级 + 当前材料，给出升级建议）

    碳钢高风险时升级路径：
      316 SS(PREN≈25, 有限) → 2205 双相(PREN≈34) → 2507(C-276 级耐全面腐蚀)。
    结合工况（温度/氯离子）给出「最低合理可行」材料。

    参数:
        risk_level: MIC 风险等级 (极低/中等/高/极高)
        current_material: 当前管材代码（MATERIAL_CHOICES 键）
        chloride_ppm: 氯离子浓度 (ppm)
        temperature: 温度 (°C)
    返回: 字典
    """
    cur_name = _MATERIAL_NAME.get(current_material, current_material)
    cur_pren = _pren(current_material) or 0

    # 按风险等级给出目标 PREN 下限
    if risk_level in ("高", "极高"):
        target_pren = 34 if chloride_ppm >= 10000 else 25
    elif risk_level == "中等":
        target_pren = 25
    else:
        target_pren = 0  # 极低：维持现状

    # 高氯 + 高温 进一步要求更高 PREN
    if chloride_ppm >= 10000 and temperature >= 60:
        target_pren = max(target_pren, 42)

    # 当前已满足？则无需升级
    if cur_pren >= target_pren:
        return {
            "risk_level": risk_level,
            "current_material": cur_name,
            "current_pren": cur_pren,
            "recommended_material": cur_name,
            "recommended_pren": cur_pren,
            "escalation": [],
            "verdict": "当前材料 PREN 已满足该工况 MIC 防护要求，维持并配合杀菌剂与监测。",
            "target_pren": target_pren,
            "reference": "NACE SP0192 (MIC 控制); NACE MR0175 / ISO 15156（材料选型）",
        }

    # 给出升级阶梯（从当前材料之后、且 PREN 达标的首个材料起）
    idx = next((i for i, (c, _, _) in enumerate(_MATERIAL_LADDER) if c == current_material), 0)
    ladder = []
    rec = cur_name
    rec_pren = cur_pren
    for code, name, pren in _MATERIAL_LADDER[idx + 1:]:
        ladder.append({"material": name, "PREN": pren})
        if pren >= target_pren and rec == cur_name:
            rec = name
            rec_pren = pren
    if rec == cur_name and ladder:
        rec = ladder[-1]["material"]
        rec_pren = ladder[-1]["PREN"]

    if risk_level == "极低":
        verdict = "MIC 风险极低：当前 %s 可维持，重点以杀菌剂与监测控制，无需材料升级。" % cur_name
    else:
        verdict = ("MIC 风险%s：当前 %s(PREN≈%s) 不足，建议升级至 %s(PREN≈%s)；"
                   "高氯/高温工况优先选用双相钢或镍基合金。" % (
                       risk_level, cur_name, cur_pren, rec, rec_pren))

    return {
        "risk_level": risk_level,
        "current_material": cur_name,
        "current_pren": cur_pren,
        "recommended_material": rec,
        "recommended_pren": rec_pren,
        "escalation": ladder,
        "verdict": verdict,
        "target_pren": target_pren,
        "reference": "NACE SP0192 (MIC 控制); NACE MR0175 / ISO 15156（材料选型）",
    }


# ----------------------------------------------------------------------
# 3e. MIC 监测与再评估计划 (NACE SP0192 / API 570)
# ----------------------------------------------------------------------

def mic_monitoring_plan(risk_level, system_type="间歇系统"):
    """
    MIC 监测与再评估计划（基于 MIC 风险等级给出监测方案与再筛查周期）

    监测手段：
      在线/离线：腐蚀挂片 + 生物膜探针(NACE SP0192) + MFL/UT ILI 金属损失。
    再筛查周期：高/极高 6–12 月，中 1–2 年，低 2–3 年。

    参数:
        risk_level: MIC 风险等级 (极低/中等/高/极高)
        system_type: 系统类型 (间歇系统/连续系统)
    返回: 字典
    """
    plan = {
        "极低": {
            "re_screen": "2–3 年",
            "methods": [
                "腐蚀挂片（离线，年度取出评估失重）",
                "年度 SRB 平板计数(MPN) 趋势监测",
            ],
            "online": "暂不需在线探针；常规巡检电位与挂片",
            "note": "维持常规监测即可，重点关注死管与低流速段。",
        },
        "中等": {
            "re_screen": "1–2 年",
            "methods": [
                "腐蚀挂片 + 生物膜探针（under-deposit 微环境评估）",
                "半年度 SRB/APB/IRB 群落检测(NACE TM0194)",
                "定期生物清扫与清管(Pigging)",
            ],
            "online": "可选在线腐蚀探针，关注局部速率突变",
            "note": "纳入例行监测，建立微生物基线以便趋势比对。",
        },
        "高": {
            "re_screen": "6–12 月",
            "methods": [
                "在线腐蚀探针 + 生物膜探针（实时 SRB 活性）",
                "季度 SRB 杀灭率评估(目标 ≥ 99%)",
                "MFL/UT ILI 金属损失复核 + 高风险段开挖",
            ],
            "online": "在线腐蚀 + 微生物活性双探针组合",
            "note": "缩短周期并加密取样，杀菌剂效果须用挂片失重验证。",
        },
        "极高": {
            "re_screen": "3–6 月",
            "methods": [
                "在线腐蚀 + biofilm 探针 + 水质多参数连续监测",
                "月度 SRB 杀灭率 + 群落结构(分子生物学)检测",
                "高频 ILI + 关键段更换/内衬修复",
            ],
            "online": "在线腐蚀 + 微生物 + 流量多参数监控",
            "note": "最高频监测，必要时管段更换；间歇系统须排干滞留水防越冬群落。",
        },
    }
    p = plan.get(risk_level, plan["中等"])
    if system_type == "间歇系统":
        sys_note = "间歇系统：停用阶段须排干滞留水/死管段，防止微生物越冬与生物膜累积。"
    else:
        sys_note = "连续系统：保持流动与药剂连续投加，关注流速异常导致的沉积热点。"

    return {
        "risk_level": risk_level,
        "system_type": system_type,
        "re_screen_interval": p["re_screen"],
        "methods": p["methods"],
        "online_recommend": p["online"],
        "note": p["note"],
        "system_note": sys_note,
        "reference": "NACE SP0192 (MIC 控制); API 570 (在役检验); NACE TM0194",
    }


# ----------------------------------------------------------------------
# 4. 电偶腐蚀
# ----------------------------------------------------------------------
def galvanic_corrosion(noble_material, active_material, area_ratio, electrolyte="海水"):
    """
    电偶腐蚀严重性评估

    参数:
        noble_material: 阴极性（贵金属）材料代码
        active_material: 阳极性（活泼金属）材料代码
        area_ratio: 阴极面积 / 阳极面积（>1 时阳极更严重）
        electrolyte: 介质（海水/土壤/淡水）

    返回: 估算结果字典
    """
    # 简化电偶序电位（V vs SHE，相对值，仅用于排序）
    galv_series = {
        "titanium": 0.10,
        "alloy_c276": 0.05,
        "alloy_625": 0.02,
        "alloy_825": -0.05,
        "duplex_2507": -0.10,
        "duplex_2205": -0.15,
        "super_13cr": -0.20,
        "13cr": -0.25,
        "stainless_316": -0.20,
        "carbon_steel": -0.50,
    }
    e_noble = galv_series.get(noble_material, 0.0)
    e_active = galv_series.get(active_material, -0.5)
    delta_e = e_noble - e_active  # 电位差，越大越严重

    # 面积比影响（大阴极小阳极最危险）
    if area_ratio <= 1:
        f_area = 1.0
    elif area_ratio <= 5:
        f_area = 1.5
    elif area_ratio <= 20:
        f_area = 2.5
    else:
        f_area = 4.0

    # 介质电导率影响
    cond_factor = {"海水": 1.5, "淡水": 0.8, "土壤": 1.0}.get(electrolyte, 1.0)

    severity_index = delta_e * f_area * cond_factor  # 综合严重度

    if severity_index < 0.2:
        level, color, base_rate = "轻微", "#27ae60", 0.05
    elif severity_index < 0.5:
        level, color, base_rate = "中等", "#f39c12", 0.15
    elif severity_index < 1.0:
        level, color, base_rate = "严重", "#e74c3c", 0.35
    else:
        level, color, base_rate = "极严重", "#c0392b", 0.70

    return {
        "delta_e": round(delta_e, 3),
        "area_ratio": area_ratio,
        "electrolyte": electrolyte,
        "severity_index": round(severity_index, 3),
        "level": level,
        "color": color,
        "rate": round(base_rate, 4),
        "noble": _MATERIAL_NAME.get(noble_material, noble_material),
        "active": _MATERIAL_NAME.get(active_material, active_material),
        "advice": _galvanic_advice(level, area_ratio),
    }


def _galvanic_advice(level, area_ratio):
    base = "建议采用绝缘法兰/垫片隔离异种金属，或使阳极面积大于阴极面积。"
    if level in ("严重", "极严重"):
        return f"电偶腐蚀{level}：{base} 高面积比({area_ratio:.0f}:1)下阳极将快速失效，务必隔离或加牺牲阳极。"
    elif level == "中等":
        return f"电偶腐蚀中等：{base} 可考虑涂装阴极区域以减小有效面积比。"
    else:
        return f"电偶腐蚀轻微：{base} 常规设计即可。"


# ----------------------------------------------------------------------
# 5. 腐蚀成本估算
# ----------------------------------------------------------------------
def corrosion_cost_estimate(diameter_mm, length_km, wall_thickness_mm, corrosion_rate,
                            material_unit_price, inspection_cost, downtime_loss_per_day,
                            annual_inspection_freq=1.0, remedial_cost_per_m=0.0):
    """
    腐蚀成本估算

    参数:
        diameter_mm: 管径 (mm)
        length_km: 管线长度 (km)
        wall_thickness_mm: 壁厚 (mm)
        corrosion_rate: 腐蚀速率 (mm/a)
        material_unit_price: 管材单价 (¥/kg)
        inspection_cost: 单次检测费用 (¥)
        downtime_loss_per_day: 单日停产损失 (¥/天)
        annual_inspection_freq: 年检测频次 (次/年)
        remedial_cost_per_m: 每米维修成本 (¥/m，可选)

    返回: 成本估算字典
    """
    D = diameter_mm / 1000.0           # m
    L = length_km * 1000.0             # m
    rate_m = corrosion_rate / 1000.0   # m/a
    density = 7850.0                   # 钢密度 kg/m³

    # 年度金属损失体积 (m³/a) = 外表面面积 × 穿透深度
    surface_area = 3.1415926 * D * L   # m²
    annual_volume = surface_area * rate_m  # m³/a
    annual_mass = annual_volume * density  # kg/a

    # 年度金属价值损失
    metal_value_loss = annual_mass * material_unit_price

    # 年度检测成本
    inspection_total = inspection_cost * annual_inspection_freq

    # 年度维修成本（按长度比例，简化）
    remedial_total = remedial_cost_per_m * L if remedial_cost_per_m > 0 else 0.0

    # 停产损失：估算因腐蚀穿孔导致的年停产天数（速率越高、壁厚越薄风险越大）
    # 简化：当剩余壁厚余量按腐蚀速率折算寿命 < 10 年时，计入风险性停产
    remaining_wall = max(wall_thickness_mm - 4.0, 0.1)  # 预留4mm余量
    life_years = remaining_wall / corrosion_rate if corrosion_rate > 0 else 999
    if life_years < 5:
        downtime_days = 5.0
    elif life_years < 10:
        downtime_days = 2.0
    elif life_years < 20:
        downtime_days = 0.5
    else:
        downtime_days = 0.1
    downtime_total = downtime_days * downtime_loss_per_day

    total = metal_value_loss + inspection_total + remedial_total + downtime_total

    return {
        "annual_volume_m3": round(annual_volume, 3),
        "annual_mass_kg": round(annual_mass, 1),
        "metal_value_loss": round(metal_value_loss, 0),
        "inspection_total": round(inspection_total, 0),
        "remedial_total": round(remedial_total, 0),
        "downtime_days": downtime_days,
        "downtime_total": round(downtime_total, 0),
        "total_cost": round(total, 0),
        "life_years": round(life_years, 1),
        "breakdown": {
            "金属损失价值": round(metal_value_loss, 0),
            "检测成本": round(inspection_total, 0),
            "维修成本": round(remedial_total, 0),
            "停产损失": round(downtime_total, 0),
        },
    }
