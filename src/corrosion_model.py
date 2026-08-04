"""
corrosion_model.py
管道腐蚀预测模型
- 训练 GradientBoosting 回归模型
- 模型评估与保存
- 加载模型进行预测
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from data_processor import (
    generate_corrosion_data,
    preprocess_data,
    get_risk_level,
    get_material_recommendation,
    MATERIAL_LABELS,
)


class CorrosionPredictor:
    """管道腐蚀预测器"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_cols = None
        self.is_trained = False

    def train(self, df=None):
        """训练模型"""
        if df is None:
            df = generate_corrosion_data(n_samples=500)

        X, y, scaler, le, feature_cols = preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        self.scaler = scaler
        self.label_encoder = le
        self.feature_cols = feature_cols
        self.is_trained = True

        print(f"模型训练完成")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae:.4f} mm/a")
        return {"r2": r2, "mae": mae}

    def save(self, model_dir=None):
        """保存模型"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 train()")

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "corrosion_predictor.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "feature_cols": self.feature_cols,
            }, f)
        print(f"模型已保存: {model_path}")

    def load(self, model_path=None):
        """加载模型"""
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "corrosion_predictor.pkl"
            )

        if not os.path.exists(model_path):
            print("模型文件不存在，开始训练新模型...")
            self.train()
            self.save(os.path.dirname(model_path))
            return

        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = True
        print("模型加载完成")

    def predict(self, material, temperature, ph, co2_pressure,
                h2s_concentration, flow_rate, chloride_content):
        """
        预测腐蚀速率和风险等级

        参数:
            material: 材料代码 (carbon_steel, stainless_316, alloy_825, duplex_2205)
            temperature: 温度 (°C)
            ph: pH 值
            co2_pressure: CO2 分压 (MPa)
            h2s_concentration: H2S 浓度 (ppm)
            flow_rate: 流速 (m/s)
            chloride_content: 氯离子含量 (ppm)

        返回:
            dict: 包含腐蚀速率、风险等级、建议措施
        """
        if not self.is_trained:
            self.load()

        material_encoded = self.label_encoder.transform([material])[0]

        features = np.array([[
            material_encoded, temperature, ph, co2_pressure,
            h2s_concentration, flow_rate, chloride_content
        ]])

        features_scaled = self.scaler.transform(features)
        corrosion_rate = self.model.predict(features_scaled)[0]
        corrosion_rate = max(0.001, corrosion_rate)

        risk_level, suggestion = get_risk_level(corrosion_rate)
        material_advice = get_material_recommendation(risk_level, material)

        return {
            "corrosion_rate": round(corrosion_rate, 4),
            "risk_level": risk_level,
            "suggestion": suggestion,
            "material_advice": material_advice,
            "material_label": MATERIAL_LABELS.get(material, material),
            "inputs": {
                "material": material,
                "temperature": temperature,
                "ph": ph,
                "co2_pressure": co2_pressure,
                "h2s_concentration": h2s_concentration,
                "flow_rate": flow_rate,
                "chloride_content": chloride_content,
            },
        }


def format_prediction(result):
    """格式化预测结果为可读文本"""
    text = f"""
=== 管道腐蚀预测结果 ===

管材类型: {result['material_label']}
温度: {result['inputs']['temperature']} °C
pH 值: {result['inputs']['ph']}
CO2 分压: {result['inputs']['co2_pressure']} MPa
H2S 浓度: {result['inputs']['h2s_concentration']} ppm
流速: {result['inputs']['flow_rate']} m/s
氯离子含量: {result['inputs']['chloride_content']} ppm

--- 预测结果 ---
腐蚀速率: {result['corrosion_rate']} mm/a
风险等级: {result['risk_level']}

--- 建议 ---
{result['suggestion']}
材料建议: {result['material_advice']}
"""
    return text


if __name__ == "__main__":
    predictor = CorrosionPredictor()

    print("=" * 50)
    print("训练腐蚀预测模型")
    print("=" * 50)
    metrics = predictor.train()
    predictor.save()

    print("\n" + "=" * 50)
    print("测试预测")
    print("=" * 50)
    result = predictor.predict(
        material="carbon_steel",
        temperature=80,
        ph=5.5,
        co2_pressure=1.5,
        h2s_concentration=100,
        flow_rate=4.0,
        chloride_content=10000,
    )
    print(format_prediction(result))
