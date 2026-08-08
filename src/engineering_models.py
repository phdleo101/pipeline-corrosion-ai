# -*- coding: utf-8 -*-
"""
engineering_models.py
基于公开文献与行业标准的工程腐蚀模型（筛选/估算用途，非设计依据）

涵盖:
  1. CO2 腐蚀 (de Waard-Milliams 1975 基础式 + pH 修正, NORSOK M-506 思路)
  2. 冲蚀临界流速 (API RP 14E) 与 Salama 含砂冲蚀速率
  3. H2S 环境开裂筛查 (NACE MR0175 / ISO 15156)
  4. 应力腐蚀开裂 SCC 敏感性筛查 (NACE SP0204)
  5. 点蚀抗力当量 PREN (不锈钢/双相钢/镍基合金)

所有模型均为工程简化估算，正式设计与合规则以现场检测及最新版标准为准。
"""

import math

# ----------------------------------------------------------------------
# 1. CO2 腐蚀 — de Waard-Milliams / NORSOK M-506 思路
# ----------------------------------------------------------------------

def co2_corrosion_base(T_C, pCO2_bar):
    """
    de Waard-Milliams (1975) 基础腐蚀速率（无保护膜、裸钢估算）

    log10(Vcorr[mm/a]) = 5.8 - 1710/(T+273.15) + 0.67 * log10(pCO2[bar])

    参数:
        T_C: 温度 (°C)
        pCO2_bar: CO2 分压 (bar)
    返回: 基础腐蚀速率 (mm/a)
    """
    if pCO2_bar <= 0:
        return 0.0
    Tk = T_C + 273.15
    log_v = 5.8 - 1710.0 / Tk + 0.67 * math.log10(pCO2_bar)
    return 10.0 ** log_v


def co2_saturated_ph(pCO2_bar):
    """
    CO2 饱和水溶液的近似饱和 pH（~25°C 经验近似，非精确化学平衡计算）
    常用近似: pH_sat ≈ 3.9 - 0.5 * log10(pCO2[bar])
    """
    if pCO2_bar <= 0:
        return None
    return 3.9 - 0.5 * math.log10(pCO2_bar)


def co2_corrosion(T_C, pCO2_bar, pH_actual=None):
    """
    CO2 腐蚀综合估算

    基础速率 (de Waard 1975) × pH 修正因子 (de Waard 1993 / NORSOK M-506 思路):
        f_pH = 10^(0.32 * (pH_sat - pH_actual))

    当温度 > ~60–80°C 时 FeCO3 保护膜可显著削弱实际腐蚀（此处仅作提示）。

    参数:
        T_C: 温度 (°C)
        pCO2_bar: CO2 分压 (bar)
        pH_actual: 实测/计算 pH（可选）
    返回: 字典
    """
    rate_base = co2_corrosion_base(T_C, pCO2_bar)
    pH_sat = co2_saturated_ph(pCO2_bar)
    f_pH = 1.0
    rate_corr = rate_base
    if pH_actual is not None and pH_sat is not None:
        f_pH = 10.0 ** (0.32 * (pH_sat - pH_actual))
        rate_corr = rate_base * f_pH

    # 温度区间与保护膜提示
    if T_C < 60:
        regime = "低温区：动力学受控，速率随温度指数上升；通常无明显 FeCO3 保护膜"
    elif T_C <= 80:
        regime = "过渡区：FeCO3 保护膜可能开始形成，实际腐蚀率可能低于裸钢估算"
    else:
        regime = "高温区：致密 FeCO3 膜易形成，实际速率常显著低于裸钢基础值（需叠加膜因子修正）"

    return {
        "rate_base": round(rate_base, 4),
        "rate_corrected": round(rate_corr, 4),
        "pH_sat": round(pH_sat, 2) if pH_sat else None,
        "f_pH": round(f_pH, 3) if pH_actual is not None else None,
        "regime": regime,
        "reference": "de Waard & Milliams (1975) Corrosion 31(5):177; de Waard & Lotz (1993); NORSOK M-506 (2017)",
    }


def co2_corrosion_curve(pCO2_bar, T_range=None, pH_actual=None):
    """返回温度-腐蚀速率曲线，用于绘图。"""
    if T_range is None:
        T_range = list(range(20, 121, 5))
    xs, ys = [], []
    for T in T_range:
        r = co2_corrosion(T, pCO2_bar, pH_actual)["rate_corrected"]
        xs.append(T)
        ys.append(r)
    return xs, ys


# ----------------------------------------------------------------------
# 2. 冲蚀 — API RP 14E 临界流速 + Salama 含砂冲蚀速率
# ----------------------------------------------------------------------

# API RP 14E 经验常数 C (单位: (ft/s)·(lb/ft³)^0.5)，连续服役取保守值
EROSION_C = {
    "碳钢 (Carbon Steel)": 100,
    "低合金钢 (Low-Alloy)": 125,
    "13Cr (410/420)": 130,
    "双相钢 2205 (Duplex 2205)": 200,
    "超级双相 2507 (Super Duplex)": 350,
    "镍基合金 625 (Inconel 625)": 400,
}


def erosion_critical_velocity(rho_m_kgm3, material="碳钢 (Carbon Steel)"):
    """
    API RP 14E 冲蚀临界流速:
        V_crit = C / sqrt(rho_m)
    C 单位 (ft/s)·(lb/ft³)^0.5，rho_m 需转换为 lb/ft³。
    1 kg/m³ = 0.062428 lb/ft³

    返回: 字典 (m/s, ft/s, C, material)
    """
    C = EROSION_C.get(material, 100)
    rho_lbft3 = rho_m_kgm3 * 0.062428
    if rho_lbft3 <= 0:
        return {"V_crit_m_s": None, "V_crit_ft_s": None, "C": C, "material": material}
    V_ft_s = C / math.sqrt(rho_lbft3)
    V_m_s = V_ft_s * 0.3048
    return {
        "V_crit_m_s": round(V_m_s, 3),
        "V_crit_ft_s": round(V_ft_s, 3),
        "C": C,
        "material": material,
    }


def erosion_rate_salama(sand_rate_kg_day, velocity_m_s, pipe_id_mm,
                        sand_size_um=200.0, rho_m_kgm3=1000.0):
    """
    Salama 含砂冲蚀速率估算 (筛选用):
        E[mm/a] = 0.182 * W * V^2 * D / (d^2 * rho_m)
        W: 含砂速率 (kg/day); V: 混合流速 (m/s);
        D: 砂粒粒径 (μm); d: 管径 (mm); rho_m: 混合密度 (kg/m³)

    返回: 字典
    """
    if pipe_id_mm <= 0 or rho_m_kgm3 <= 0:
        return {"rate_mm_yr": 0.0, "verdict": "输入无效", "reference": "Salama & Venkatesh (1983) OTC 4485"}
    E = 0.182 * sand_rate_kg_day * (velocity_m_s ** 2) * sand_size_um / (
        (pipe_id_mm ** 2) * rho_m_kgm3)
    if E < 0.1:
        verdict = "低速冲蚀风险低（通常指 E < 0.1 mm/a 可接受）"
    elif E < 0.5:
        verdict = "中等冲蚀风险，建议监测与限产"
    else:
        verdict = "高冲蚀风险，需限制流速/含砂或采用耐冲蚀合金(CRA)"
    return {
        "rate_mm_yr": round(E, 4),
        "verdict": verdict,
        "reference": "Salama & Venkatesh (1983) OTC 4485; API RP 14E",
    }


# ----------------------------------------------------------------------
# 3. H2S 环境开裂筛查 — NACE MR0175 / ISO 15156
# ----------------------------------------------------------------------

def h2s_ssc_screening(pH2S_bar, pH_in_situ, hardness_hrc=None):
    """
    H2S 环境开裂（SSC/HIC）筛查 — 依据 NACE MR0175 / ISO 15156 精神

    酸性服役判定: pH2S >= 0.0003 bar (≈0.0345 kPa) 视为酸性环境(需按 MR0175 选材)
    严苛度: 基于 pH2S 量级与 pH 的保守筛查（非 ISO 15156-2 图1 精确曲线）
    SSC 硬度上限: 碳钢/低合金钢 ≤ 22 HRC (≈248 HV)
    HIC 抗力: 需洁净低硫、钙处理(形状控制)钢，并按 NACE TM0284 试验(CLR/CTR/CSR)

    参数:
        pH2S_bar: H2S 分压 (bar)
        pH_in_situ: 原位 pH
        hardness_hrc: 材料硬度 HRC（可选）
    返回: 字典
    """
    sour = pH2S_bar >= 0.0003
    if not sour:
        region = "A 区（非酸性，豁免 MR0175 选材限制）"
        severity = "无限制（仍需控制一般腐蚀）"
    else:
        # 保守筛查：pH2S 越高、pH 越低 → 越严苛
        if pH2S_bar >= 0.1 and pH_in_situ <= 4.5:
            region = "C 区（重度酸性，类似 ISO 15156-2 图1 严苛区）"
            severity = "重度：须严格限制硬度并选用合格抗硫材料/CRA"
        elif pH2S_bar >= 0.01:
            region = "B/C 过渡（中度酸性）"
            severity = "中度：按 MR0175 选材，关注硬度与 HIC 抗力"
        else:
            region = "B 区（轻度酸性）"
            severity = "轻度：按 MR0175 最低要求控制硬度与洁净度"

    controls = [
        "碳钢/低合金钢 SSC 硬度上限 ≤ 22 HRC (≈248 HV)",
        "HIC 抗力由钢洁净度与显微组织决定，须按 NACE TM0284 试验(CLR/CTR/CSR)认定",
        "优先选用 Ca 处理(形状控制)的低硫抗 HIC 钢；必要时升级 CRA(如 13Cr/双相/镍基)",
        "NACE MR0175 / ISO 15156 规定了各材料在酸性环境的适用条件与试验要求",
    ]
    hardness_ok = None
    if hardness_hrc is not None:
        hardness_ok = hardness_hrc <= 22

    return {
        "sour": sour,
        "region": region,
        "severity": severity,
        "controls": controls,
        "hardness_hrc": hardness_hrc,
        "hardness_ok": hardness_ok,
        "reference": "NACE MR0175 / ISO 15156 (Parts 1–3); NACE TM0177; NACE TM0284",
    }


# ----------------------------------------------------------------------
# 4. 应力腐蚀开裂 SCC 敏感性筛查 — NACE SP0204 (SCCDA)
# ----------------------------------------------------------------------

def scc_susceptibility(coating_type, operating_stress_pct, age_years,
                       temperature_C, cp_shielded, terrain):
    """
    外部 SCC 敏感性筛查（NACE SP0204 外部 SCC 直接评估思路）

    两类机理:
      高 pH SCC:  碳酸盐-碳酸氢盐电解质(pH 9–11)，晶间裂纹；受高温(压气站下游)、
                  旧涂层(煤焦油/沥青)剥离、operating stress > 60% SMYS、管龄 > 10–15 年驱动
      近中性 pH SCC: 稀释地下水(pH 6–7.5, 富 CO2)，穿晶裂纹；受剥离涂层+CP 屏蔽、
                  排水良好砂土、operating stress > 60% SMYS、管龄驱动

    参数:
        coating_type: "旧涂层(煤焦油/沥青)" / "现代涂层(FBE/PE)" / "未知"
        operating_stress_pct: 操作应力占 SMYS 百分比
        age_years: 管龄(年)
        temperature_C: 运行温度
        cp_shielded: 阴极保护是否被屏蔽(True/False)
        terrain: "排水良好砂土" / "黏土/保水" / "未知"
    返回: 字典（含两类评分与等级）
    """
    # 高 pH SCC 评分
    hp = 0
    if coating_type == "旧涂层(煤焦油/沥青)":
        hp += 35
    elif coating_type == "未知":
        hp += 15
    if operating_stress_pct >= 60:
        hp += 25
    elif operating_stress_pct >= 40:
        hp += 10
    if age_years >= 15:
        hp += 20
    elif age_years >= 10:
        hp += 10
    if temperature_C >= 40:
        hp += 20
    elif temperature_C >= 20:
        hp += 8

    # 近中性 pH SCC 评分
    nn = 0
    if coating_type in ("旧涂层(煤焦油/沥青)", "未知"):
        nn += 30
    if cp_shielded:
        nn += 30
    if terrain == "排水良好砂土":
        nn += 25
    elif terrain == "未知":
        nn += 10
    if operating_stress_pct >= 60:
        nn += 15
    if age_years >= 10:
        nn += 10
    if temperature_C < 40:
        nn += 5  # 近中性 SCC 在较低温更常见

    def level(s):
        if s >= 60:
            return "高"
        elif s >= 30:
            return "中"
        return "低"

    drivers_hp = []
    if coating_type == "旧涂层(煤焦油/沥青)":
        drivers_hp.append("旧涂层易剥离，形成浓碳酸盐电解质")
    if operating_stress_pct >= 60:
        drivers_hp.append("操作应力 > 60% SMYS")
    if age_years >= 10:
        drivers_hp.append("管龄 ≥ 10 年")
    if temperature_C >= 40:
        drivers_hp.append("高温(压气站下游)促进晶间 SCC")

    drivers_nn = []
    if cp_shielded:
        drivers_nn.append("CP 被剥离涂层屏蔽，管体未受保护")
    if terrain == "排水良好砂土":
        drivers_nn.append("排水良好砂土富 CO2 地下水")
    if operating_stress_pct >= 60:
        drivers_nn.append("操作应力 > 60% SMYS")

    return {
        "high_pH_score": hp,
        "high_pH_level": level(hp),
        "near_neutral_score": nn,
        "near_neutral_level": level(nn),
        "drivers_high_pH": drivers_hp,
        "drivers_near_neutral": drivers_nn,
        "reference": "NACE SP0204 (SCCDA); API RP 1176; Kiefner & Vieth (Battelle); NEB RH-2-2008",
    }


# ----------------------------------------------------------------------
# 5. 点蚀抗力当量 PREN
# ----------------------------------------------------------------------

# 10 种管材的典型成分(质量%)，用于 PREN 估算
PREN_COMP = {
    "碳钢 (Carbon Steel)": {"Cr": 0.1, "Mo": 0.05, "N": 0.005},
    "316不锈钢 (316 SS)": {"Cr": 17.0, "Mo": 2.1, "N": 0.08},
    "13Cr马氏体不锈钢 (13Cr)": {"Cr": 13.0, "Mo": 0.0, "N": 0.02},
    "超级13Cr (Super 13Cr)": {"Cr": 13.0, "Mo": 1.5, "N": 0.02},
    "2205双相不锈钢 (Duplex 2205)": {"Cr": 22.0, "Mo": 3.1, "N": 0.18},
    "2507超级双相不锈钢 (Super Duplex 2507)": {"Cr": 25.0, "Mo": 4.0, "N": 0.27},
    "825合金 (Alloy 825)": {"Cr": 21.5, "Mo": 3.0, "N": 0.0},
    "625合金 (Inconel 625)": {"Cr": 21.5, "Mo": 9.0, "N": 0.0},
    "C-276合金 (Hastelloy C-276)": {"Cr": 15.5, "Mo": 16.0, "N": 0.0},
    "钛合金 (Titanium Gr.2)": {"Cr": 0.0, "Mo": 0.0, "N": 0.0, "note": "钛依赖稳定氧化膜，非 PREN 体系"},
}


def pren(material):
    """
    点蚀抗力当量 PREN = %Cr + 3.3×%Mo + 16×%N
    用于不锈钢/双相钢/镍基合金抗氯离子点蚀能力对比。
    返回: 字典
    """
    comp = PREN_COMP.get(material)
    if not comp:
        return {"material": material, "PREN": None, "rating": "未知", "note": "无成分数据"}
    if material.startswith("钛合金"):
        return {"material": material, "PREN": None, "rating": "N/A",
                "note": "钛依赖稳定氧化膜，不适用 PREN 体系"}
    p = comp["Cr"] + 3.3 * comp["Mo"] + 16 * comp["N"]
    if p < 18:
        rating = "差（一般不适于含氯Point环境）"
    elif p < 24:
        rating = "中等（限低温低氯）"
    elif p < 32:
        rating = "良好"
    else:
        rating = "优异（高氯/高温海水可用）"
    return {"material": material, "PREN": round(p, 1), "rating": rating,
            "Cr": comp["Cr"], "Mo": comp["Mo"], "N": comp["N"]}


def pren_all():
    """返回全部材料 PREN 列表（用于对比图表）。"""
    out = []
    for m in PREN_COMP:
        r = pren(m)
        out.append({"material": m, "PREN": r["PREN"], "rating": r["rating"], "note": r.get("note", "")})
    return out


# ----------------------------------------------------------------------
# 6. SCC 开挖验证优先级 + ECDA 流程 (NACE SP0204 / API RP 1176)
# ----------------------------------------------------------------------

def scc_excavation_priority(coating_type, operating_stress_pct, age_years,
                             temperature_C, cp_shielded, terrain,
                             hca=False, ili_anomaly=False):
    """
    SCC 开挖验证优先级与 ECDA 四步流程（基于 scc_susceptibility 输出再决策一层）

    在敏感性筛查之上，叠加高后果区(HCA)与内检测(ILI)异常，给出开挖优先级与
    标准 ECDA（外部腐蚀直接评估）四步流程。用于把"敏感性分数"翻译成"行动清单"。

    参数:
        hca: 是否位于高后果区（人口/环境敏感）
        ili_anomaly: 内检测(ILI)是否提示金属损失/几何异常
    返回: 字典
    """
    sr = scc_susceptibility(coating_type, operating_stress_pct, age_years,
                            temperature_C, cp_shielded, terrain)
    max_score = min(max(sr["high_pH_score"], sr["near_neutral_score"]), 100)

    p = max_score
    if hca:
        p += 15
    if ili_anomaly:
        p += 20
    p = min(p, 100)

    if p >= 75:
        priority = "P1 — 立即开挖验证"
        reason = "高敏感性叠加高后果区或内检测异常，存在裂纹泄漏风险"
        conf = "高"
    elif p >= 55:
        priority = "P2 — 12 个月内计划开挖"
        reason = "敏感性较高，建议近期安排直接检查确认"
        conf = "中"
    else:
        priority = "P3 — 纳入周期性监测"
        reason = "敏感性较低，按常规 ECDA 周期监测即可"
        conf = "低"

    edcda_steps = [
        "① 预评估 (Pre-assessment)：收集历史数据、涂层年代、应力与温度，圈定 SCC 潜在区域",
        "② 间接检测 (Indirect Inspection)：CIS 密间距电位 + DCVG/Pearson 定位涂层缺陷与异常电位",
        "③ 直接检查 (Direct Examination)：开挖测厚 + 磁粉/超声检测裂纹，取样确认 SCC 类型",
        "④ 后评估 (Post-assessment)：评定缺陷、计算剩余强度(B31G/RSTRENG)，制定修复或监测",
    ]
    return {
        "priority": priority,
        "priority_score": p,
        "priority_reason": reason,
        "confidence": conf,
        "susceptibility": sr,
        "edcda_steps": edcda_steps,
        "reference": "NACE SP0204 (SCCDA); API RP 1176 (SCC 管理); NACE SP0502 (ECDA)",
    }


# ----------------------------------------------------------------------
# 7. SCC 裂纹扩展速率与剩余寿命 (B31G Folias + 经验扩展速率)
# ----------------------------------------------------------------------

def scc_crack_life(a0_mm, wall_t_mm, diameter_mm, yield_strength_mpa,
                   oper_stress_pct, scc_type="near_neutral",
                   crack_length_mm=50.0, growth_rate=None):
    """
    SCC 裂纹扩展与剩余寿命估算

    临界裂纹深度 a_c 采用 B31G/RSTRENG 思路的 Folias（鼓胀）因子：
        σ_f = 1.1·σ_y (流变应力)；σ_oper = (%SMYS/100)·σ_y
        M = √(1 + 0.6275·(L²/Dt) − 0.003375·(L⁴/D²t²))   (轴向缺陷, L²/Dt ≤ 50)
        a_c = t·σ_f / (σ_f + M·σ_oper)
    剩余寿命 = (a_c − a₀) / 经验扩展速率。

    参数:
        a0_mm: 初始裂纹深度 (mm)
        wall_t_mm: 壁厚 (mm)
        diameter_mm: 管径 (mm)
        yield_strength_mpa: 屈服强度 (MPa)
        oper_stress_pct: 操作应力占 SMYS 百分比
        scc_type: "near_neutral" / "high_pH"
        crack_length_mm: 轴向裂纹长度 (mm)
        growth_rate: 经验扩展速率 mm/yr（缺省按类型取默认）
    返回: 字典
    """
    sigma_y = yield_strength_mpa
    sigma_f = 1.1 * sigma_y
    sigma_oper = oper_stress_pct / 100.0 * sigma_y

    D, t, L = diameter_mm, wall_t_mm, crack_length_mm
    ratio = (L * L) / (D * t)
    if ratio <= 50:
        M = (1 + 0.6275 * ratio - 0.003375 * ratio * ratio) ** 0.5
    else:
        M = 0.032 * ratio + 3.3  # 大缺陷近似
    a_c = t * sigma_f / (sigma_f + M * sigma_oper)

    if growth_rate is None:
        growth_rate = 0.30 if scc_type == "near_neutral" else 0.80  # mm/yr

    if a_c <= a0_mm:
        life = 0.0
        verdict = "⚠️ 已达/超过临界深度，须立即评估修复或更换"
    else:
        life = (a_c - a0_mm) / growth_rate
        if life <= 2:
            verdict = "🔴 剩余寿命 ≤2 年，优先开挖/修复"
        elif life <= 5:
            verdict = "🟠 剩余寿命 2–5 年，加密监测并排期修复"
        elif life <= 10:
            verdict = "🟡 剩余寿命 5–10 年，纳入常规监测"
        else:
            verdict = "🟢 剩余寿命 >10 年，常规监测"

    return {
        "a0_mm": a0_mm,
        "a_c_mm": round(a_c, 3),
        "folias_M": round(M, 3),
        "growth_rate_mm_yr": growth_rate,
        "life_years": round(life, 1),
        "verdict": verdict,
        "reference": "B31G / RSTRENG (Folias 因子); Kiefner & Vieth (Battelle); API 579 裂纹评定; NACE SP0204",
    }


# ----------------------------------------------------------------------
# 8. SCC 缓解决策树 (NACE SP0204 / PRCI 缓解指南)
# ----------------------------------------------------------------------

def scc_mitigation_tree(coating_type, operating_stress_pct, age_years,
                        temperature_C, cp_shielded, terrain):
    """
    SCC 缓解决策树（基于敏感性评分给出缓解组合建议）

    近中性 pH SCC：阴极保护(CP)有效，需将管地电位负移到 −0.95 V CSE 以下抑制；
                   剥离涂层下的 CP 屏蔽仍是难点（需改进 CP 设计/排流）。
    高 pH SCC：对 CP 不敏感，主要靠涂层修复 + 降压运行 + 裂纹检测型 ILI。

    参数: (同 scc_susceptibility)
    返回: 字典
    """
    sr = scc_susceptibility(coating_type, operating_stress_pct, age_years,
                            temperature_C, cp_shielded, terrain)
    hp = min(sr["high_pH_score"], 100)
    nn = min(sr["near_neutral_score"], 100)
    dominant = "high_pH" if hp >= nn else "near_neutral"
    max_score = max(hp, nn)

    mitigations = []

    # 1) 阴极保护优化
    if nn >= 30:
        mitigations.append({
            "action": "阴极保护(CP)优化",
            "applicability": "对近中性 pH SCC 有效",
            "detail": "将管地电位负移至 −0.95 V CSE 以下抑制开裂；对 CP 屏蔽段改进阳极布置/排流，消除剥离涂层下的保护盲区",
            "priority": "高" if nn >= 60 else "中",
        })
    if hp >= 30:
        mitigations.append({
            "action": "阴极保护(CP)优化",
            "applicability": "对高 pH SCC 作用有限",
            "detail": "高 pH SCC 对 CP 不敏感，CP 仅作辅助腐蚀控制，不应依赖其抑制开裂",
            "priority": "低",
        })

    # 2) 涂层修复（两类 SCC 的共同诱因是剥离旧涂层）
    if coating_type in ("旧涂层(煤焦油/沥青)", "未知") or max_score >= 30:
        mitigations.append({
            "action": "涂层修复 / 更换",
            "applicability": "两类 SCC 均适用（剥离涂层是共同诱因）",
            "detail": "更换剥离旧涂层为 FBE/PE，并加强补口与焊缝涂装，消除浓电解质微环境",
            "priority": "高" if max_score >= 60 else "中",
        })

    # 3) 降压运行
    if operating_stress_pct >= 60:
        mitigations.append({
            "action": "降压运行",
            "applicability": "两类 SCC 均适用",
            "detail": "当前操作应力 %d%% SMYS ≥ 60%%，建议降至 < 60%% SMYS 以显著减缓裂纹扩展" % operating_stress_pct,
            "priority": "高",
        })
    else:
        mitigations.append({
            "action": "降压运行",
            "applicability": "两类 SCC 均适用",
            "detail": "当前操作应力 %d%% SMYS 已 < 60%%，维持并监控应力波动" % operating_stress_pct,
            "priority": "低",
        })

    # 4) 裂纹检测型内检测 ILI
    if max_score >= 30:
        mitigations.append({
            "action": "裂纹检测型内检测(ILI)",
            "applicability": "两类 SCC 均适用",
            "detail": "采用 EMAT/UT（而非 MFL）裂纹检测型 ILI 识别轴向裂纹，结合开挖验证标定",
            "priority": "高" if max_score >= 60 else "中",
        })

    if dominant == "high_pH":
        core = "涂层修复 + 裂纹检测ILI + 降压运行"
    else:
        core = "CP优化 + 涂层修复 + 裂纹检测ILI"
    summary = ("主导机理: %s；推荐以 %s 为核心缓解组合。"
               % ("高 pH SCC（晶间）" if dominant == "high_pH" else "近中性 pH SCC（穿晶）", core))

    return {
        "dominant": dominant,
        "max_score": max_score,
        "mitigations": mitigations,
        "summary": summary,
        "reference": "NACE SP0204 (SCCDA); PRCI SCC 缓解指南; API RP 1176",
    }


# ----------------------------------------------------------------------
# 9. SCC 风险矩阵叠加 (复用 integrity_tools.risk_matrix)
# ----------------------------------------------------------------------

def scc_risk_overlay(coating_type, operating_stress_pct, age_years,
                     temperature_C, cp_shielded, terrain,
                     diameter_mm, pressure_mpa, location_type, hca=False):
    """
    将 SCC 敏感性评分映射到失效概率维度，与管径/压力/位置（后果）叠加，
    输出综合风险等级与在 5×5 风险矩阵中的位置。复用 integrity_tools.risk_matrix()。

    参数:
        (前 6 项同 scc_susceptibility)
        diameter_mm: 管径 (mm)
        pressure_mpa: 操作压力 (MPa)
        location_type: 位置类型（人口密集区/一般区域/荒野）
        hca: 是否高后果区（若为 True，按高后果位置处理以抬升后果等级）
    返回: 字典
    """
    from integrity_tools import risk_matrix

    sr = scc_susceptibility(coating_type, operating_stress_pct, age_years,
                            temperature_C, cp_shielded, terrain)
    max_score = max(min(sr["high_pH_score"], 100), min(sr["near_neutral_score"], 100))

    # SCC 评分 → 失效概率等级(0–4)
    if max_score >= 80:
        prob_idx = 4
    elif max_score >= 60:
        prob_idx = 3
    elif max_score >= 40:
        prob_idx = 2
    elif max_score >= 20:
        prob_idx = 1
    else:
        prob_idx = 0

    # HCA 视为高后果位置，复用 risk_matrix 的后果与综合评级逻辑
    loc = "人口密集区" if hca else location_type
    rm = risk_matrix(corrosion_rate=0.0, diameter=diameter_mm,
                     pressure=pressure_mpa, location_type=loc,
                     prob_idx_override=prob_idx)

    return {
        "scc_max_score": max_score,
        "prob_idx": prob_idx,
        "prob_level": ["极低", "低", "中", "高", "极高"][prob_idx],
        "cons_level": rm["cons_level"],
        "risk_level": rm["risk_level"],
        "risk_color": rm["risk_color"],
        "risk_score": rm["risk_score"],
        "hca_treated_as": loc,
        "reference": "NACE SP0204 (SCCDA); ASME B31G 风险矩阵思路; API RP 1176",
    }
