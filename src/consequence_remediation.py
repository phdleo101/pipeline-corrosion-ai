"""
consequence_remediation.py
后果分析 + 分场景详细维护建议引擎

- consequence_analysis(): 依据管线类型/管径/压力/介质/位置评估泄漏后果等级
- recommend_remediation(): 依据威胁类型 + 严重度 + 管线类型 + 壁损% 给出
  分场景的多步详细建议（立即措施 / 工程修复 / 监测 / 标准依据 / 优先级 / 周期）

覆盖 7 类主导威胁：CO₂内腐蚀 / H₂S开裂 / 外部腐蚀 / SCC / MIC / 冲蚀 / 电偶腐蚀
"""

from pipeline_types import get_preset


# ---------------- 后果分析 ----------------
def consequence_analysis(pipeline_type, diameter_mm, pressure_mpa,
                         product="天然气", location_type="一般区域", wall_loss_pct=10.0):
    """
    后果（Consequence）定性评估。

    输入:
        pipeline_type: pipeline_types 的 key（如 gas_transmission）
        diameter_mm, pressure_mpa: 几何与工况
        product: 介质
        location_type: 一般区域 / 人口密集区(HCA) / 荒野 / 水体
        wall_loss_pct: 当前最大壁损百分比

    返回: dict
    """
    p = get_preset(pipeline_type)
    hazard = p.get("product_hazard", "中")

    # 介质危险基数
    hazard_base = {
        "极高（人口密集/爆炸）": 4, "极高（H₂S 剧毒/爆炸）": 4,
        "高（易燃/爆炸）": 3, "高（海洋环境/难以抢修）": 3,
        "高（有毒/腐蚀介质）": 3, "高（含 H₂S/易燃）": 3,
        "中（环境污染）": 2, "低（水，环境局部）": 1,
        "低（民生/局部）": 1,
    }.get(hazard, 2)

    # 管径/压力放大
    size_factor = 1 + (diameter_mm - 500) / 1500.0  # 越大越严重
    size_factor = max(0.7, min(2.0, size_factor))
    pres_factor = 1 + (pressure_mpa - 5) / 15.0
    pres_factor = max(0.6, min(2.0, pres_factor))

    # 位置放大
    loc_factor = {
        "人口密集区(HCA)": 2.0, "水体/环境敏感区": 1.6,
        "一般区域": 1.0, "荒野": 0.7,
    }.get(location_type, 1.0)

    # 壁损放大（接近临界更危险）
    loss_factor = 1 + max(0, (wall_loss_pct - 20)) / 40.0
    loss_factor = max(1.0, min(2.5, loss_factor))

    score = hazard_base * size_factor * pres_factor * loc_factor * loss_factor

    if score >= 8.0:
        level, color = "极高", "#c0392b"
    elif score >= 4.5:
        level, color = "高", "#e74c3c"
    elif score >= 2.5:
        level, color = "中", "#f39c12"
    else:
        level, color = "低", "#27ae60"

    # 近似释放量（全断/大孔）：库存近似 = π/4·D²·L·ρ，取单位长度(L=1km)示意
    D = diameter_mm / 1000.0
    release_per_km = 3.1416 / 4 * D ** 2 * 1000  # m³ 管道容积/km
    if product in ("天然气",) :
        release_note = f"天然气泄漏以爆炸性为主，1 km 管段容积约 {release_per_km:.1f} m³（标况），扩散+燃爆风险主导。"
    elif product in ("原油", "成品油", "含水原油"):
        release_note = f"液体泄漏以环境污染为主，1 km 管段约 {release_per_km:.1f} m³，可能进入土壤/水体。"
    elif product in ("海水", "污水", "注水"):
        release_note = f"水相泄漏以局部环境为主，1 km 管段约 {release_per_km:.1f} m³。"
    else:
        release_note = f"介质相关释放，1 km 管段容积约 {release_per_km:.1f} m³。"

    return {
        "level": level, "color": color,
        "score": round(score, 2),
        "hazard_class": hazard,
        "release_per_km_m3": round(release_per_km, 1),
        "release_note": release_note,
        "location_factor": loc_factor,
        "summary": (
            f"后果等级【{level}】（综合评分 {score:.1f}）：介质危险「{hazard}」"
            f"× 管径压力放大 × 位置系数 {loc_factor} × 壁损系数 {loss_factor:.1f}。"
        ),
        "reference": "ASME B31.8S / API 1160 完整性管理；PHMSA 49 CFR 192.933/195.452 响应准则；风险矩阵见本系统「完整性工具」。",
    }


# ---------------- 分场景维护建议知识库 ----------------
# 每威胁: 给出 立即措施 / 工程修复选项(按严重度) / 监测 / 标准依据
REMEDIATION_KB = {
    "CO₂内腐蚀": {
        "threat": "CO₂ 甜腐蚀（内部均匀/局部腐蚀）",
        "immediate": [
            "核算当前工况 CO₂ 分压与温度，确认是否进入严重腐蚀区（de Waard-Milliams / NORSOK M-506）",
            "提高缓蚀剂连续/批处理注入量并验证膜覆盖率（底部 6 点钟位置）",
            "排查底部沉积水，必要时增设排水/清管（pigging）频次",
        ],
        "repair": {
            "低": "常规监测 + 缓蚀剂优化，无需立即修复",
            "中": "加密壁厚监测；对腐蚀热点补口/补强；评估剩余强度(B31G)",
            "高": "对超限点补焊/复合材料补强（CFRP）；降压运行；安排换管窗口",
            "严重": "立即降压或停输；缺陷管段更换；全段复查内腐蚀(ICDA)",
        },
        "monitor": "ILI(MFL) 周期复检；在线腐蚀挂片/ER 探针；缓蚀剂残余浓度与 Fe²⁺ 监测",
        "standard": "NORSOK M-506; NACE SP0106(ICDA); API 571; ASME B31G",
    },
    "H₂S开裂": {
        "threat": "湿 H₂S 环境开裂（SSC/HIC/SOHIC）",
        "immediate": [
            "确认 H₂S 浓度与 pH，凡 H₂S>50 ppm 即进入酸性服役（NACE MR0175/ISO 15156）",
            "核查材料硬度 ≤ 22 HRC（焊缝/热影响区重点）",
            "控制焊接与焊后热处理(PWHT)，避免未回火马氏体",
        ],
        "repair": {
            "低": "硬度复测 + 材料证书核查；保持 pH≥6",
            "中": "对敏感焊口进行 MT/PT 复检；降低 H₂S 分压（脱硫/注入）",
            "高": "更换为合格 CRA（13Cr/825/625）；缺陷部位切除重焊并 PWHT",
            "严重": "立即停用该段；全段 HIC 试验(CLR/CTR/CSR)与 UT 复查；换管",
        },
        "monitor": "硬度普查；UT 体积检测；硫化氢/pH 在线；裂纹敏感段 PAUT/TOFD",
        "standard": "NACE MR0175/ISO 15156; NACE TM0177(SSC); NACE TM0284(HIC); API 579",
    },
    "外部腐蚀": {
        "threat": "外腐蚀（土壤/海水 + 涂层失效 + CP 屏蔽）",
        "immediate": [
            "测量管地电位(ON/OFF)与 IR 降，评估阴极保护(CP)有效性",
            "排查涂层剥离、屏蔽（套管/绝缘接头）与杂散电流",
            "对高风险段安排 ECDA 间接检测（Pearson/DCVG/CIPS）",
        ],
        "repair": {
            "低": "修复涂层局部破损；优化 CP 输出",
            "中": "开挖直接检查 + 涂层更换 + 阴极保护恢复；安装试片/参比电极",
            "高": "缺陷点补强/换管；升级 CP（深井阳极/牺牲阳极）；重点段 ILI(MFL)",
            "严重": "立即修复或降压；失效涂层段整体更换；杂散电流排流",
        },
        "monitor": "ECDA(NACE SP0502) 周期评估；CP 电位年度测量；ILI 金属损失复检",
        "standard": "NACE SP0502(ECDA); NACE SP0169(CP); ISO 15589-1; DIN 50929",
    },
    "SCC": {
        "threat": "外部应力腐蚀开裂（高 pH 晶间 / 近中性 pH 穿晶）",
        "immediate": [
            "识别高 pH（碳酸盐，pH 9–11）或近中性 pH（富 CO₂ 地下水，pH 6–7.5）环境",
            "在 HCA 与历史异常段启动 SCCDA（NACE SP0204），无需依赖 ILI",
            "核查应力水平（操作应力/残余应力）与敏感涂层（旧煤焦油/沥青）",
        ],
        "repair": {
            "低": "SCCDA 间接检测 + 开挖验证；记录裂纹形貌",
            "中": "裂纹部位打磨/补焊（验证无延展）；裂纹型 ILI(UT-CD) 复检",
            "高": "切除含裂纹管段重焊；优化 CP（近中性 pH 对 CP 敏感）；降压",
            "严重": "立即更换开裂段；全段 SCC 风险再评级(见本系统 SCC 模块)；ECDA 四步",
        },
        "monitor": "UT-CD 裂纹型 ILI；开挖验证（ECDA）；应变/应力监测；涂层状况巡检",
        "standard": "NACE SP0204(SCCDA); API RP 1176(裂纹管理); API 579/RSTRENG; Battelle NG-18",
    },
    "MIC": {
        "threat": "微生物腐蚀（SRB 主导，垢下离散点蚀）",
        "immediate": [
            "采集管壁 biofilm/沉积样做 SRB/APB/IRB 计数（NACE TM0194）",
            "脱氧 + 冲击式杀菌剂（戊二醛/THPS/DBNPA）并轮换防耐药",
            "提高流速消除死区/低流速沉积（20–60°C 最易感）",
        ],
        "repair": {
            "低": "杀菌剂方案优化 + 监测；无需立即修复",
            "中": "局部点蚀补强；清洗 biofilm；升级材料(316/双相，参考本系统 MIC-3)",
            "高": "更换为耐 MIC 材料(2205/825)；加强杀菌 + 清管；缺陷点修复",
            "严重": "切除严重点蚀段；全段生物膜治理；材料升级 + 监测闭环",
        },
        "monitor": "生物膜探针/挂片；SRB 计数季度；腐蚀速率 ER 探针；ILI 点蚀复检",
        "standard": "NACE SP0192(MIC 控制); NACE TM0212(杀菌剂); NACE TM0194",
    },
    "冲蚀": {
        "threat": "冲蚀-腐蚀（含砂/多相流磨损保护膜）",
        "immediate": [
            "核算操作流速是否超过 API RP 14E 临界流速 V=C/√ρ",
            "降低含砂量或提高携砂流速（平衡）；弯头/阀门增设耐磨衬里",
            "底部沉积控制（除砂/清管）",
        ],
        "repair": {
            "低": "流速优化；监测弯头壁厚",
            "中": "更换耐磨弯头/三通；增设缓蚀剂（成膜抗冲蚀）",
            "高": "局部换管 + 耐磨内衬；流程改造降砂/降速",
            "严重": "立即降压；冲蚀穿孔段更换；全段流速重核算",
        },
        "monitor": "弯头/阀门 UT 壁厚高频监测；含砂量在线；流速/压降趋势",
        "standard": "API RP 14E; Salama 含砂冲蚀模型; API RP 14C",
    },
    "电偶腐蚀": {
        "threat": "电偶腐蚀（异种金属接触，电位差驱动）",
        "immediate": [
            "识别异种金属接头（碳钢-不锈钢/铜合金/双相）与电解质",
            "绝缘隔离（绝缘接头/法兰）切断电偶回路",
            "涂覆低电位侧（阳极）或加牺牲阳极",
        ],
        "repair": {
            "低": "绝缘件检查；阳极侧涂层修补",
            "中": "加装绝缘接头；牺牲阳极；消除电解质积留",
            "高": "更换为同电位材料；电偶部位整体隔离 + 修复",
            "严重": "切除严重电偶腐蚀段；系统级电位平衡改造",
        },
        "monitor": "电偶电位差测量；接头腐蚀目视/UT；CP 电位复核",
        "standard": "NACE SP0169; 电偶序(EMF) 参考; ISO 8044",
    },
}


def recommend_remediation(threat, severity="中", pipeline_type="gas_transmission",
                          wall_loss_pct=10.0, material="carbon_steel"):
    """
    分场景详细维护建议。

    参数:
        threat: REMEDIATION_KB 的键（如 "CO₂内腐蚀"），或 threats 中的别名
        severity: 低/中/高/严重（腐蚀速率风险或壁损等级）
        pipeline_type: pipeline_types key
        wall_loss_pct: 最大壁损百分比（用于 B31G 临界判断提示）
        material: 当前材料

    返回: dict
    """
    # 别名归一
    alias = {
        "co2": "CO₂内腐蚀", "co2腐蚀": "CO₂内腐蚀", "甜腐蚀": "CO₂内腐蚀",
        "h2s": "H₂S开裂", "ssc": "H₂S开裂", "hic": "H₂S开裂",
        "外腐蚀": "外部腐蚀", "土壤腐蚀": "外部腐蚀", "海水腐蚀": "外部腐蚀",
        "scc": "SCC", "应力腐蚀": "SCC",
        "mic": "MIC", "微生物腐蚀": "MIC",
        "冲蚀": "冲蚀", "砂": "冲蚀", "erosion": "冲蚀",
        "电偶": "电偶腐蚀", "galvanic": "电偶腐蚀",
    }
    tkey = alias.get(threat.strip().lower() if isinstance(threat, str) else threat, threat)
    kb = REMEDIATION_KB.get(tkey)
    if kb is None:
        # 兜底：用通用建议
        kb = REMEDIATION_KB["外部腐蚀"]
        tkey = "外部腐蚀(通用)"

    sev = severity if severity in ("低", "中", "高", "严重") else "中"

    # B31G 临界壁损提示
    if wall_loss_pct >= 80:
        b31g_note = "⚠️ 壁损 ≥ 80% 壁厚：已达 ASME B31G 失效临界，须立即降压/修复/更换！"
    elif wall_loss_pct >= 50:
        b31g_note = "壁损 50–80%：进入 B31G 剩余强度评估区，优先安排修复或降压运行。"
    elif wall_loss_pct >= 20:
        b31g_note = "壁损 20–50%：按计划修复，重点监测扩展速率。"
    else:
        b31g_note = "壁损 < 20%：常规监测即可。"

    # 优先级
    priority = {"低": "P3（计划内）", "中": "P2（本年度）", "高": "P1（季度内）", "严重": "P0（立即）"}[sev]

    p = get_preset(pipeline_type)
    # 结合管线类型的主导威胁给出额外提示
    extra = ""
    if tkey in p.get("dominant_threats", []):
        extra = f"该管线类型「{p['label']}」将本威胁列为主导威胁，建议纳入完整性管理主线。"
    else:
        extra = f"该管线类型「{p['label']}」主导威胁为 {('、'.join(p.get('dominant_threats', [])))}，本威胁作为次要威胁管理。"

    return {
        "threat": kb["threat"],
        "severity": sev,
        "priority": priority,
        "immediate": kb["immediate"],
        "repair": kb["repair"][sev],
        "monitor": kb["monitor"],
        "standard": kb["standard"],
        "b31g_note": b31g_note,
        "pipeline_context": extra,
        "reference": f"综合标准：{kb['standard']}；后果分析见 ASME B31.8S/API 1160。",
    }
