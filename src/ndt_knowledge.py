"""
ndt_knowledge.py
无损检测(NDT)行业知识库 —— 面向管道完整性管理(ILI/直接评估/建设期焊接检测)

提供：
- NDT_METHODS: 各 NDT 方法的结构化知识（原理/可检缺陷/灵敏度/局限/用途/标准）
- ILI_TOOLS: 内检测(ILI)工具（MFL/UT/CD/EMAT/Caliper）能力对比
- recommend_ndt(): 依据缺陷类型/威胁/管线可检性推荐检测方案
- POD 与尺寸精度参考（API 1163 概念）
"""

# ---------------- 常规 NDT 方法 ----------------
NDT_METHODS = {
    "MFL": {
        "name": "漏磁检测 (Magnetic Flux Leakage, MFL)",
        "principle": "对管壁磁化，缺陷处磁通泄漏被传感器拾取；金属损失(体积型)首选。",
        "detects": ["金属损失(腐蚀坑/沟槽)", "壁厚减薄", "焊缝异常(体积型)"],
        "sensitivity": "金属损失深度阈值约 10%t，POD≈90%（API 1163）",
        "limitations": "对轴向裂纹/狭长缺陷灵敏度差；剩磁、壁厚突变影响；需清管条件。",
        "use": "在役管道腐蚀普查（ILI），长输/油气干线主力工具。",
        "standard": "API 1163; ASME B31G; NACE SP0102",
    },
    "UT_CD": {
        "name": "超声裂纹检测 (UT Crack Detection, 压电/EMAT)",
        "principle": "超声波在裂纹界面反射；对裂纹型缺陷(轴向/环向)灵敏度高。",
        "detects": ["应力腐蚀开裂(SCC)", "焊缝裂纹", "选择性缝腐蚀", "氢致开裂(HIC)"],
        "sensitivity": "裂纹深度阈值约 1 mm，POI≈90%（API 1163 表 3）",
        "limitations": "对均匀腐蚀不敏感；耦合要求高；EMAT 可免耦合但分辨率略低。",
        "use": "裂纹威胁主导管段（SCC 高/近中性 pH、LF-ERW 缝腐蚀）。",
        "standard": "API 1163; API 579-1/ASME FFS-1; BS 7910",
    },
    "EMAT": {
        "name": "电磁声换能 (EMAT)",
        "principle": "电磁感应在金属中激发/接收超声波，无需耦合剂。",
        "detects": ["顶部腐蚀(TOL)", "SCC", "壁厚", "焊缝"],
        "sensitivity": "可同时覆盖金属损失与裂纹，适合干管/含蜡管线。",
        "limitations": "信噪比低于压电 UT；复杂管况分辨率受限。",
        "use": "免耦合 ILI（含蜡、不能充液的管线）。",
        "standard": "API 1163; NACE SP0102",
    },
    "Caliper": {
        "name": "几何/变形检测 (Caliper)",
        "principle": "机械臂/电磁测径，记录椭圆度、凹坑、弯曲、变形。",
        "detects": ["几何变形(凹陷/椭圆)", "屈曲", "内径变化", "清管障碍"],
        "sensitivity": "变形量毫米级；为裂纹/腐蚀工具提供通过性判断。",
        "limitations": "不测壁厚/裂纹；仅几何尺寸。",
        "use": "ILI 前通过性评估、凹陷/变形识别。",
        "standard": "API 1163; ASME B31.8S",
    },
    "PAUT": {
        "name": "相控阵超声 (Phased Array UT, PAUT/TOFD)",
        "principle": "多晶片延时聚焦，电子扫查；建设期环焊缝主力。",
        "detects": ["环焊缝未熔合/裂纹", "夹渣", "气孔", "热影响区缺陷"],
        "sensitivity": "焊缝体积缺陷与裂纹高灵敏度；TOFD 测高准确。",
        "limitations": "需扫查面平整；人员资质要求高。",
        "use": "新建管道环焊缝 AUT（替代 RT），速度更快。",
        "standard": "API 1104 §11/Annex A; ISO 10863",
    },
    "RT": {
        "name": "射线检测 (Radiographic Testing, RT)",
        "principle": "X/γ 射线穿透，胶片/数字化成像。",
        "detects": ["焊缝体积缺陷(气孔/夹渣/未熔合)"],
        "sensitivity": "对体积型缺陷好，对面积型裂纹较差。",
        "limitations": "辐射安全、夜间作业、天气敏感；速度慢于 AUT。",
        "use": "传统环焊缝检测（<200 道焊口经济性差）。",
        "standard": "API 1104 §9; ISO 17636",
    },
    "MT": {
        "name": "磁粉检测 (Magnetic Particle Testing, MT)",
        "principle": "磁化表面，磁粉聚集显示表面/近表面缺陷。",
        "detects": ["表面裂纹(焊缝/开口)", "近表面缺陷"],
        "sensitivity": "表面开口缺陷高灵敏。",
        "limitations": "仅铁磁材料；需通电/磁化设备。",
        "use": "开挖验证、阀门/管件表面裂纹。",
        "standard": "ASTM E709; ISO 9934",
    },
    "PT": {
        "name": "渗透检测 (Liquid Penetrant Testing, PT)",
        "principle": "渗透液渗入表面开口缺陷，显像剂显现。",
        "detects": ["表面开口裂纹/气孔"],
        "sensitivity": "表面开口缺陷高灵敏；非铁磁也能用。",
        "limitations": "仅表面开口；需清洁表面。",
        "use": "非铁磁材料(奥氏体/双相)表面裂纹。",
        "standard": "ASTM E165; ISO 3452",
    },
    "ET": {
        "name": "涡流检测 (Eddy Current Testing, ET)",
        "principle": "交变磁场感应涡流，缺陷改变阻抗。",
        "detects": ["表面/近表面裂纹", "涂层下腐蚀(远场涡流)"],
        "sensitivity": "快速表面筛查；远场可测内壁。",
        "limitations": "穿透浅；铁磁材料需特殊探头。",
        "use": "换热管、小径管、涂层下点蚀。",
        "standard": "ASTM E309; ISO 15548",
    },
    "GW": {
        "name": "超声导波 (Guided Wave, LRUT)",
        "principle": "低频导波沿管壁长距离传播，整圈筛查。",
        "detects": ["长距离金属损失(支撑/埋地出露段)", "环向缺陷"],
        "sensitivity": "单点可扫数十米；适合难以进入管段。",
        "limitations": "定位精度低；需局部复检确认。",
        "use": "架空/穿跨越/海底悬空段快速筛查。",
        "standard": "ASTM E2775; ISO 18211",
    },
    "AE": {
        "name": "声发射 (Acoustic Emission, AE)",
        "principle": "缺陷活动(开裂/泄漏)释放弹性波实时监测。",
        "detects": ["活动裂纹扩展", "泄漏", "应力释放"],
        "sensitivity": "在线、动态监测；不定位静态缺陷。",
        "limitations": "背景噪声干扰；需活动源。",
        "use": "加压试验/运行期实时监测、泄漏报警。",
        "standard": "ASTM E1139; ISO 22096",
    },
}

# ---------------- ILI 工具选型矩阵 ----------------
# threat -> 推荐 ILI 工具（按威胁优先级）
ILI_THREAT_MAP = {
    "CO₂内腐蚀": ["MFL", "UT_CD", "Caliper"],
    "H₂S开裂(SSC/HIC)": ["UT_CD", "MFL", "Caliper"],
    "外部腐蚀": ["MFL", "Caliper", "UT_CD"],
    "SCC(近中性pH)": ["UT_CD", "MFL"],
    "SCC(高pH)": ["UT_CD", "MFL"],
    "MIC": ["MFL", "UT_CD", "Caliper"],
    "冲蚀(含砂)": ["MFL", "Caliper"],
    "电偶腐蚀": ["MFL", "UT_CD"],
    "金属损失/腐蚀": ["MFL", "UT_CD", "Caliper"],
    "裂纹/开裂": ["UT_CD", "EMAT", "MFL"],
    "几何变形": ["Caliper"],
}


def recommend_ndt(threat=None, defect_types=None, piggable=True, priority="balanced"):
    """
    推荐 NDT/ILI 方案。

    参数:
        threat: 主导威胁（对应 ILI_THREAT_MAP 键）
        defect_types: 缺陷类型列表，如 ["金属损失","裂纹"]
        piggable: 是否可内检测（不可则推荐直接评估）
        priority: "balanced"(均衡) / "metal_loss"(偏腐蚀) / "crack"(偏裂纹)

    返回:
        dict: {methods: [{code,name,reason,rank}], direct_assessment: [...], note: str}
    """
    methods = []
    seen = set()

    def add(code, reason):
        if code in seen:
            return
        seen.add(code)
        methods.append({
            "code": code,
            "name": NDT_METHODS[code]["name"],
            "reason": reason,
        })

    # 1) 由威胁映射
    threat_list = []
    if threat:
        threat_list = ILI_THREAT_MAP.get(threat, [])
        for c in threat_list:
            add(c, f"针对威胁「{threat}」的推荐工具")

    # 2) 由缺陷类型
    if defect_types:
        for dt in defect_types:
            dt = dt.lower()
            if "裂" in dt or "crack" in dt:
                for c in ["UT_CD", "EMAT"]:
                    add(c, "裂纹型缺陷首选超声/EMAT")
            elif "金属损失" in dt or "腐蚀" in dt or "metal" in dt or "loss" in dt:
                for c in ["MFL", "UT_CD"]:
                    add(c, "体积型金属损失首选 MFL")
            elif "变形" in dt or "凹" in dt or "geometry" in dt:
                add("Caliper", "几何变形首选 Caliper")

    # 3) 优先级补充
    if priority == "crack" and "UT_CD" not in seen:
        add("UT_CD", "裂纹优先策略：超声裂纹检测")
    if priority == "metal_loss" and "MFL" not in seen:
        add("MFL", "腐蚀优先策略：漏磁检测")

    # 4) 兜底
    if not methods:
        for c in ["MFL", "UT_CD", "Caliper"]:
            add(c, "通用在役普查组合")

    # 给排序
    for i, m in enumerate(methods, 1):
        m["rank"] = i

    # 不可内检测 → 直接评估
    da = []
    if not piggable:
        da = [
            "ECDA (NACE SP0502) —— 外腐蚀直接评估（预评估→间接检测→直接检查→后评估）",
            "ICDA (NACE SP0206) —— 内腐蚀直接评估（湿气/干气边界）",
            "SCCDA (NACE SP0204) —— 应力腐蚀开裂直接评估",
        ]
    else:
        da = [
            "ILI 后建议对 I 级异常开挖验证（MT/PT/UT 复测）",
            "高风险段采用 MFL + UT-CD 组合 run 提升威胁覆盖（PHMSA §192.937 允许）",
        ]

    note = (
        "API 1163 第 3 版(2021)要求 ILI 性能以 POD(检出概率)/POI(识别概率)/尺寸精度"
        "(如 ±10%t @ 80%) 量化；单一技术工具会留盲区（MFL 漏裂纹、UT-CD 漏全面腐蚀），"
        "组合 run 是提升威胁覆盖率的最佳实践。"
    )

    return {"methods": methods, "direct_assessment": da, "note": note}
