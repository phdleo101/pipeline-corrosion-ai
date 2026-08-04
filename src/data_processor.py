"""
data_processor.py
管道腐蚀数据处理模块
- 生成模拟腐蚀数据集
- 数据预处理与特征工程
- 数据验证
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

MATERIALS = ["carbon_steel", "stainless_316", "alloy_825", "duplex_2205"]
MATERIAL_LABELS = {
    "carbon_steel": "碳钢",
    "stainless_316": "316不锈钢",
    "alloy_825": "825合金",
    "duplex_2205": "2205双相不锈钢",
}

MATERIAL_BASE_RATE = {
    "carbon_steel": 1.0,
    "stainless_316": 0.15,
    "alloy_825": 0.05,
    "duplex_2205": 0.08,
}


def generate_corrosion_data(n_samples=500, random_state=42):
    """
    基于腐蚀工程原理生成模拟数据集
    融合 de Waard-Milliams CO2腐蚀模型 + 经验修正
    """
    rng = np.random.RandomState(random_state)

    materials = rng.choice(MATERIALS, size=n_samples, p=[0.5, 0.25, 0.15, 0.10])
    temperature = rng.uniform(20, 120, n_samples)
    ph = rng.uniform(3.5, 9.0, n_samples)
    co2_pressure = rng.uniform(0.01, 5.0, n_samples)
    h2s_concentration = rng.uniform(0, 500, n_samples)
    flow_rate = rng.uniform(0.5, 8.0, n_samples)
    chloride_content = rng.uniform(0, 50000, n_samples)

    corrosion_rates = []
    for i in range(n_samples):
        mat = materials[i]
        T = temperature[i]
        pH = ph[i]
        pCO2 = co2_pressure[i]
        H2S = h2s_concentration[i]
        v = flow_rate[i]
        Cl = chloride_content[i]

        base = MATERIAL_BASE_RATE[mat]

        # de Waard-Milliams 简化模型: log(CR) = 5.8 - 1710/(T+273) + 0.67*log(pCO2)
        if pCO2 > 0.01:
            dewaard = 10 ** (5.8 - 1710 / (T + 273) + 0.67 * np.log10(pCO2))
        else:
            dewaard = 0.01

        # pH 修正: pH < 6 时加速腐蚀
        ph_factor = 1.0 + max(0, (6.0 - pH)) * 0.8

        # 温度修正: 高温加速
        temp_factor = 1.0 + max(0, (T - 60)) * 0.015

        # H2S 修正: 硫腐蚀叠加
        h2s_factor = 1.0 + (H2S / 500) * 0.3

        # 流速修正: 冲刷腐蚀
        flow_factor = 1.0 + max(0, (v - 3)) * 0.15

        # 氯离子修正: 点蚀风险
        cl_factor = 1.0 + (Cl / 50000) * 0.2

        # 综合腐蚀速率
        cr = base * dewaard * ph_factor * temp_factor * h2s_factor * flow_factor * cl_factor

        # 添加随机噪声 (±15%)
        cr *= rng.uniform(0.85, 1.15)
        cr = max(0.001, cr)
        corrosion_rates.append(cr)

    corrosion_rates = np.array(corrosion_rates)

    # 风险等级分类
    risk_levels = []
    for cr in corrosion_rates:
        if cr < 0.1:
            risk_levels.append("low")
        elif cr < 0.5:
            risk_levels.append("medium")
        elif cr < 1.0:
            risk_levels.append("high")
        else:
            risk_levels.append("severe")

    df = pd.DataFrame({
        "material": materials,
        "temperature": np.round(temperature, 1),
        "ph": np.round(ph, 2),
        "co2_pressure": np.round(co2_pressure, 3),
        "h2s_concentration": np.round(h2s_concentration, 1),
        "flow_rate": np.round(flow_rate, 2),
        "chloride_content": np.round(chloride_content, 0),
        "corrosion_rate": np.round(corrosion_rates, 4),
        "risk_level": risk_levels,
    })

    return df


def preprocess_data(df):
    """数据预处理：编码 + 缩放"""
    df_processed = df.copy()

    le = LabelEncoder()
    df_processed["material_encoded"] = le.fit_transform(df_processed["material"])

    feature_cols = [
        "material_encoded", "temperature", "ph", "co2_pressure",
        "h2s_concentration", "flow_rate", "chloride_content"
    ]

    X = df_processed[feature_cols].values
    y = df_processed["corrosion_rate"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le, feature_cols


def get_risk_level(corrosion_rate):
    """根据腐蚀速率判定风险等级"""
    if corrosion_rate < 0.1:
        return "低风险", "腐蚀速率在可接受范围内，建议常规监测。"
    elif corrosion_rate < 0.5:
        return "中风险", "需加强监测频率，考虑增加缓蚀剂注入。"
    elif corrosion_rate < 1.0:
        return "高风险", "建议立即评估剩余强度，制定维修计划，增加缓蚀剂用量。"
    else:
        return "严重风险", "需立即降压运行或维修，评估管道完整性，考虑更换管段。"


def get_material_recommendation(risk_level, material):
    """根据风险等级和管材给出材料升级建议"""
    upgrades = {
        "carbon_steel": "考虑升级为 316 不锈钢或内衬防腐涂层",
        "stainless_316": "考虑升级为 825 合金或双相不锈钢 2205",
        "alloy_825": "当前材料等级较高，重点关注焊接质量",
        "duplex_2205": "当前材料等级较高，关注氯离子应力腐蚀开裂",
    }
    if risk_level in ["高风险", "严重风险"]:
        return upgrades.get(material, "建议咨询材料工程师进行升级评估。")
    return "当前材料适用于当前工况。"


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    print("正在生成模拟腐蚀数据集...")
    df = generate_corrosion_data(n_samples=500)
    csv_path = os.path.join(data_dir, "corrosion_dataset.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"数据集已保存: {csv_path}")
    print(f"数据集大小: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"\n风险等级分布:\n{df['risk_level'].value_counts()}")
    print(f"\n腐蚀速率统计:\n{df['corrosion_rate'].describe()}")
