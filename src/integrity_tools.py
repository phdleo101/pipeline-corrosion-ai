"""
integrity_tools.py
管道完整性管理工具模块
- B31G 剩余强度评估（ASME B31G Level 1 简化公式）
- 缓蚀剂推荐（基于工况条件）
- 风险矩阵评估（失效概率 × 失效后果）
"""


def b31g_calculate(D, t, d, L, sigma_y, P_oper=None):
    """
    ASME B31G Level 1 剩余强度评估

    参数:
        D: 管径 (mm)
        t: 壁厚 (mm)
        d: 缺陷深度 (mm)
        L: 缺陷长度 (mm)
        sigma_y: 材料屈服强度 (MPa)
        P_oper: 操作压力 (MPa)，可选

    返回: 计算结果的字典
    """
    results = {}

    # 缺陷深度比
    dtr = d / t
    results["dtr"] = dtr

    # 流动应力 = 1.1 × 屈服强度
    sigma_f = 1.1 * sigma_y
    results["sigma_f"] = sigma_f

    # Folias 因子
    M = (1 + (0.8 * L ** 2) / (D * t)) ** 0.5
    results["M"] = M

    # 失效压力 (B31G Level 1 公式)
    # Pf = 2*t*σf*(1 - d/t) / [D * (1 - 0.85*d/(t*M))]
    denom = D * (1 - (0.85 * d) / (t * M))
    if denom <= 0:
        Pf = float("inf")
    else:
        Pf = (2 * t * sigma_f * (1 - dtr)) / denom
    results["Pf"] = Pf

    # 安全判定
    if dtr > 0.8:
        results["verdict"] = "危险"
        results["verdict_color"] = "#c0392b"
        results["verdict_msg"] = "缺陷深度超过壁厚80%，需立即维修或更换管段"
    elif dtr > 0.5:
        results["verdict"] = "警戒"
        results["verdict_color"] = "#e74c3c"
        results["verdict_msg"] = "缺陷深度超过壁厚50%，需尽快制定维修计划"
    elif dtr > 0.3:
        results["verdict"] = "关注"
        results["verdict_color"] = "#f39c12"
        results["verdict_msg"] = "缺陷深度超过壁厚30%，建议加强监测"
    else:
        results["verdict"] = "安全"
        results["verdict_color"] = "#27ae60"
        results["verdict_msg"] = "缺陷深度在可接受范围内，按常规周期监测"

    # 剩余强度率 (RSF) - 与操作压力对比
    if P_oper and P_oper > 0:
        results["RSF"] = Pf / P_oper
        if results["RSF"] >= 1.1:
            results["rsf_status"] = "满足安全要求（RSF ≥ 1.1）"
        elif results["RSF"] >= 1.0:
            results["rsf_status"] = "临界状态（RSF 1.0-1.1），建议降压运行"
        else:
            results["rsf_status"] = "不满足安全要求（RSF < 1.0），需降压或维修"

    return results


def recommend_inhibitor(temperature, flow_rate, co2_pressure, h2s_concentration, medium="湿气"):
    """
    缓蚀剂推荐

    参数:
        temperature: 温度 (°C)
        flow_rate: 流速 (m/s)
        co2_pressure: CO2分压 (MPa)
        h2s_concentration: H2S浓度 (ppm)
        medium: 介质类型（湿气/干气/产出水/原油）

    返回: 推荐结果的字典
    """
    rec = {}

    # 缓蚀剂类型选择
    if h2s_concentration > 50 and co2_pressure > 0.5:
        # H2S + CO2 共存，需抗硫缓蚀剂
        inhib_type = "抗硫型成膜胺缓蚀剂（咪唑啉改性）"
        rec["type_reason"] = "H2S与CO2共存环境，常规胺类易与H2S反应生成沉淀，需使用抗硫改性配方"
    elif co2_pressure > 0.5 and temperature < 60:
        inhib_type = "成膜型胺类缓蚀剂（咪唑啉）"
        rec["type_reason"] = "低温CO2腐蚀环境，咪唑啉在金属表面形成致密吸附膜，效果最佳"
    elif co2_pressure > 0.5 and temperature >= 60:
        inhib_type = "高温型缓蚀剂（复配咪唑啉+硫脲衍生物）"
        rec["type_reason"] = "高温下FeCO3膜不稳定，需添加高温稳定剂和协同组分"
    elif h2s_concentration > 50:
        inhib_type = "抗硫型缓蚀剂（避免含氮量过高配方）"
        rec["type_reason"] = "高H2S环境需控制缓蚀剂含氮量，防止与H2S生成硫化物沉淀"
    else:
        inhib_type = "通用型成膜胺缓蚀剂"
        rec["type_reason"] = "腐蚀环境较温和，通用配方即可满足要求"

    rec["type"] = inhib_type

    # 注入浓度推荐
    base_ppm = 15  # 连续注入基础浓度
    if flow_rate > 5:
        base_ppm += 10  # 高流速冲刷，需增加浓度
    if temperature > 80:
        base_ppm += 5   # 高温加速消耗
    if co2_pressure > 2:
        base_ppm += 5   # 高CO2分压，腐蚀驱动力大

    rec["injection_ppm"] = base_ppm
    rec["batch_ppm"] = base_ppm * 10  # 批处理浓度约为连续注入的10倍

    # 预期缓蚀率
    if flow_rate < 2:
        expected_eff = "88%-92%"
    elif flow_rate < 5:
        expected_eff = "85%-90%"
    else:
        expected_eff = "80%-85%"
    rec["expected_efficiency"] = expected_eff

    # 介质特定建议
    medium_advice = {
        "湿气": "在气液分离段和段塞流区域重点加注，关注持液率高的管段",
        "干气": "保持管线干燥，控制露点腐蚀，关注开机/停机阶段的凝析液",
        "产出水": "产出水矿化度高，需配合杀菌剂控制SRB引起的MIC腐蚀",
        "原油": "关注含水率和伴生气CO2，高含水阶段腐蚀加剧",
    }
    rec["medium_advice"] = medium_advice.get(medium, "根据介质特性调整加注方案")

    # 注意事项
    rec["notes"] = [
        "缓蚀剂效果需通过腐蚀挂片或在线探针验证，目标缓蚀率 > 85%",
        "注入点应设在腐蚀最严重管段上游，确保药剂充分分散",
        "定期检测药剂浓度（如荧光示踪法），避免过量或不足",
        "高H2S环境需使用抗硫配方，防止硫化物沉淀堵塞管线和设备",
    ]

    return rec


def risk_matrix(corrosion_rate, diameter, pressure, location_type):
    """
    风险矩阵评估

    参数:
        corrosion_rate: 腐蚀速率 (mm/a)
        diameter: 管径 (mm)
        pressure: 操作压力 (MPa)
        location_type: 位置类型（人口密集区/一般区域/荒野）

    返回: 包含概率等级、后果等级、风险等级的字典
    """
    # 失效概率等级（基于腐蚀速率）
    if corrosion_rate < 0.1:
        prob_level = "极低"
        prob_idx = 0
    elif corrosion_rate < 0.5:
        prob_level = "低"
        prob_idx = 1
    elif corrosion_rate < 1.0:
        prob_level = "中"
        prob_idx = 2
    elif corrosion_rate < 2.0:
        prob_level = "高"
        prob_idx = 3
    else:
        prob_level = "极高"
        prob_idx = 4

    # 失效后果等级（基于管径、压力、位置）
    # 后果评分
    score = 0
    # 管径评分
    if diameter >= 1000:
        score += 2
    elif diameter >= 500:
        score += 1
    # 压力评分
    if pressure >= 10:
        score += 2
    elif pressure >= 4:
        score += 1
    # 位置评分
    location_scores = {"人口密集区": 2, "一般区域": 1, "荒野": 0}
    score += location_scores.get(location_type, 1)

    if score >= 5:
        cons_level = "极高"
        cons_idx = 4
    elif score >= 4:
        cons_level = "高"
        cons_idx = 3
    elif score >= 2:
        cons_level = "中"
        cons_idx = 2
    elif score >= 1:
        cons_level = "低"
        cons_idx = 1
    else:
        cons_level = "极低"
        cons_idx = 0

    # 5×5 风险矩阵
    # 风险 = 概率等级 + 后果等级（0-4）
    risk_score = prob_idx + cons_idx
    if risk_score >= 7:
        risk_level = "极高风险"
        risk_color = "#c0392b"
    elif risk_score >= 5:
        risk_level = "高风险"
        risk_color = "#e74c3c"
    elif risk_score >= 3:
        risk_level = "中风险"
        risk_color = "#f39c12"
    else:
        risk_level = "低风险"
        risk_color = "#27ae60"

    return {
        "prob_level": prob_level,
        "prob_idx": prob_idx,
        "cons_level": cons_level,
        "cons_idx": cons_idx,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_score": risk_score,
    }


# 风险矩阵颜色映射（5×5）
RISK_MATRIX_COLORS = [
    # 后果: 极低, 低, 中, 高, 极高 (行=概率从低到高)
    ["#27ae60", "#27ae60", "#f39c12", "#e74c3c", "#e74c3c"],  # 概率极低
    ["#27ae60", "#f39c12", "#f39c12", "#e74c3c", "#c0392b"],  # 概率低
    ["#f39c12", "#f39c12", "#e74c3c", "#e74c3c", "#c0392b"],  # 概率中
    ["#e74c3c", "#e74c3c", "#e74c3c", "#c0392b", "#c0392b"],  # 概率高
    ["#e74c3c", "#c0392b", "#c0392b", "#c0392b", "#c0392b"],  # 概率极高
]
