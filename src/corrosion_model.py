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
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
        self.mae = None  # 用于置信区间估算
        self.metrics = {}  # 存储训练指标

    def train(self, df=None):
        """训练模型"""
        if df is None:
            csv_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "corrosion_dataset.csv"
            )
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                print("从 CSV 加载训练数据")
            else:
                df = generate_corrosion_data(n_samples=500)
                print("生成新的模拟训练数据")

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
        self.mae = mae
        self.metrics = {"r2": r2, "mae": mae, "rmse": np.sqrt(mean_squared_error(y_test, y_pred))}

        print(f"模型训练完成")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae:.4f} mm/a")
        return {"r2": r2, "mae": mae}

    def save(self, model_dir=None):
        """保存模型（部署在只读文件系统时自动跳过）"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 train()")

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        try:
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
        except (OSError, PermissionError) as e:
            print(f"模型保存跳过（文件系统只读）: {e}")

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

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.label_encoder = data["label_encoder"]
            self.feature_cols = data["feature_cols"]
            self.is_trained = True
            print("模型加载完成")
        except Exception as e:
            # 兼容旧版/跨环境：pickle 反序列化失败（如依赖缺失）时重新训练
            print(f"模型加载失败（{e}），重新训练以兼容当前环境...")
            self.train()
            try:
                self.save(os.path.dirname(model_path))
            except Exception:
                pass

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

    def predict_with_confidence(self, material, temperature, ph, co2_pressure,
                                h2s_concentration, flow_rate, chloride_content):
        """
        预测腐蚀速率并返回置信区间

        返回:
            dict: 包含预测值、置信区间下限和上限
        """
        if not self.is_trained:
            self.load()

        base_result = self.predict(
            material, temperature, ph, co2_pressure,
            h2s_concentration, flow_rate, chloride_content
        )

        rate = base_result["corrosion_rate"]
        mae = self.mae if self.mae else 0.3

        # 95% 置信区间 ≈ 预测值 ± 2*MAE
        lower = max(0.001, rate - 2 * mae)
        upper = rate + 2 * mae

        base_result["confidence_lower"] = round(lower, 4)
        base_result["confidence_upper"] = round(upper, 4)
        base_result["confidence_range"] = round(upper - lower, 4)
        return base_result

    @staticmethod
    def build_model_dict():
        """构造多种回归模型字典（含可选 XGBoost 与投票集成），供多模型对比/逐样本预测复用。"""
        models = {
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42
            ),
            "DecisionTree": DecisionTreeRegressor(
                max_depth=8, random_state=42
            ),
            "LinearRegression": LinearRegression(),
            "MLP(神经网络)": MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=2000,
                alpha=1e-4, random_state=42, early_stopping=True,
            ),
            "SVR(支持向量)": SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1),
        }

        # 可选 XGBoost（部署环境可能未安装，缺失时跳过）
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=0,
            )
        except Exception:
            pass

        # 投票集成（GBR + RF + MLP）：降低方差、提升稳健性
        ensemble = VotingRegressor(
            estimators=[
                ("gbr", models["GradientBoosting"]),
                ("rf", models["RandomForest"]),
                ("mlp", models["MLP(神经网络)"]),
            ],
            n_jobs=-1,
        )
        models["VotingEnsemble"] = ensemble
        return models

    def _train_split(self, df=None):
        """公共：加载/生成数据并切分训练集（返回 X_scaled 供后续使用）。"""
        if df is None:
            csv_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "corrosion_dataset.csv"
            )
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                df = generate_corrosion_data(n_samples=500)

        X, y, scaler, le, feature_cols = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, scaler, le, feature_cols

    def train_multiple_models(self, df=None):
        """
        训练多种回归模型并返回对比结果（仅指标，供『模型对比』展示）。

        返回:
            dict: 各模型的 R², MAE, RMSE 指标（含已训练的 model 对象）
        """
        X_train, X_test, y_train, y_test, scaler, le, feature_cols = self._train_split(df)

        results = {}
        for name, model in self.build_model_dict().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[name] = {
                "r2": round(r2, 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "model": model,
            }

        return results

    def train_all_full(self, df=None):
        """
        训练全部模型并返回 (results, scaler, le, feature_cols)，
        供『多模型逐样本对比预测』使用（需要统一的缩放器与编码器对新输入做变换）。

        返回:
            tuple: (results{name:{r2,mae,rmse,model}}, scaler, label_encoder, feature_cols)
        """
        X_train, X_test, y_train, y_test, scaler, le, feature_cols = self._train_split(df)

        results = {}
        for name, model in self.build_model_dict().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[name] = {
                "r2": round(r2, 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "model": model,
            }

        return results, scaler, le, feature_cols

    def get_feature_importance(self):
        """返回特征重要性排序"""
        if not self.is_trained:
            self.load()

        if not hasattr(self.model, "feature_importances_"):
            return None

        feature_names = [
            "材料类型", "温度", "pH值", "CO2分压",
            "H2S浓度", "流速", "氯离子含量"
        ]

        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        return [
            {"feature": feature_names[i], "importance": round(importances[i], 4)}
            for i in sorted_idx
        ]

    def get_trend_data(self, param, material, temperature, ph, co2_pressure,
                       h2s_concentration, flow_rate, chloride_content):
        """
        生成趋势分析数据：固定其他参数，变化目标参数，返回腐蚀速率曲线

        参数:
            param: 要变化的参数名 ('temperature', 'ph', 'co2_pressure', 'flow_rate')
        返回:
            dict: {param_values: [...], corrosion_rates: [...]}
        """
        if not self.is_trained:
            self.load()

        param_ranges = {
            "temperature": np.linspace(0, 150, 50),
            "ph": np.linspace(3.0, 10.0, 50),
            "co2_pressure": np.linspace(0.0, 10.0, 50),
            "flow_rate": np.linspace(0.0, 10.0, 50),
        }

        if param not in param_ranges:
            return None

        values = param_ranges[param]
        rates = []

        for v in values:
            kwargs = {
                "material": material,
                "temperature": temperature,
                "ph": ph,
                "co2_pressure": co2_pressure,
                "h2s_concentration": h2s_concentration,
                "flow_rate": flow_rate,
                "chloride_content": chloride_content,
            }
            kwargs[param] = float(v)
            result = self.predict(**kwargs)
            rates.append(result["corrosion_rate"])

        param_labels = {
            "temperature": "温度 (°C)",
            "ph": "pH 值",
            "co2_pressure": "CO2 分压 (MPa)",
            "flow_rate": "流速 (m/s)",
        }

        return {
            "param_name": param,
            "param_label": param_labels.get(param, param),
            "param_values": values.tolist(),
            "corrosion_rates": rates,
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
