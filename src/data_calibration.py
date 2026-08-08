# -*- coding: utf-8 -*-
"""
实测数据标定模块 (P3 深化)

允许用户上传真实腐蚀/检测 CSV，自动识别目标列（数值=回归 / 类别=分类），
用随机森林重新标定模型，并对比基线（均值/众数预测器）给出精度提升与特征重要性。
在用户尚无数据时，提供「合成 demo 数据集」与「模板下载」以便立即验证流程。

说明：标定仅在本会话内生效（Streamlit Cloud 为临时文件系统）；可下载
标定后的指标与特征重要性用于归档。正式模型部署需将标定结果纳入版本管理。
"""

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyRegressor, DummyClassifier
    from sklearn.metrics import (
        r2_score, mean_absolute_error,
        accuracy_score, f1_score,
    )
    _SKLEARN_OK = True
except Exception:  # pragma: no cover
    _SKLEARN_OK = False


def sample_template_df():
    """返回示例模板 DataFrame（数值目标 corrosion_rate 回归示例）。"""
    rows = [
        {"pH": 6.5, "chloride_ppm": 20000, "SRB_log": 4.0, "temperature": 40,
         "flow_velocity": 0.3, "water_cut": 70, "pren": 22, "corrosion_rate": 0.85},
        {"pH": 7.2, "chloride_ppm": 500, "SRB_log": 1.0, "temperature": 25,
         "flow_velocity": 1.2, "water_cut": 10, "pren": 35, "corrosion_rate": 0.12},
        {"pH": 5.5, "chloride_ppm": 45000, "SRB_log": 5.5, "temperature": 55,
         "flow_velocity": 0.1, "water_cut": 90, "pren": 19, "corrosion_rate": 1.35},
        {"pH": 8.0, "chloride_ppm": 100, "SRB_log": 0.5, "temperature": 18,
         "flow_velocity": 2.0, "water_cut": 5, "pren": 40, "corrosion_rate": 0.05},
    ]
    return pd.DataFrame(rows)


def demo_synthetic_df(n=400, seed=7):
    """生成合成回归数据集，便于在无真实数据时验证标定流程。"""
    rng = np.random.default_rng(seed)
    pH = rng.uniform(4, 9, n)
    chloride = rng.lognormal(np.log(2000), 1.4, n)
    SRB = rng.uniform(0, 7, n)
    temp = rng.uniform(5, 80, n)
    flow = rng.uniform(0, 3, n)
    water = rng.uniform(0, 100, n)
    pren = rng.uniform(18, 45, n)
    # 目标：受微生物活动、低流速、低 PREN、高氯驱动的腐蚀速率
    rate = (
        0.10 * _sig(SRB / 7 * 4)
        + 0.08 * (1 - _sig((flow - 0.5) / 0.3))
        + 0.10 * (1 - _sig((pren - 30) / 8))
        + 0.06 * _sig((chloride / 60000) * 4)
        + 0.05 * _sig(((temp - 42) ** 2) / (2 * 15 ** 2) * -1)
        + rng.normal(0, 0.04, n)
    )
    rate = np.clip(rate * 2.0, 0.01, 2.0)
    return pd.DataFrame({
        "pH": pH, "chloride_ppm": chloride, "SRB_log": SRB,
        "temperature": temp, "flow_velocity": flow,
        "water_cut": water, "pren": pren, "corrosion_rate": rate,
    })


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_uploaded_csv(uploaded_file):
    """解析上传的 CSV 文件为 DataFrame。"""
    return pd.read_csv(uploaded_file)


def calibrate_with_data(df, target_col, feature_cols=None, task=None, seed=42):
    """
    用 df 标定模型。

    参数:
        df: 含特征与目标列的 DataFrame
        target_col: 目标列名
        feature_cols: 特征列（None=除目标外全部数值列）
        task: 强制任务类型 'regression'/'classification'，None=自动识别

    返回: 指标 + 特征重要性 字典
    """
    if not _SKLEARN_OK:
        raise RuntimeError("scikit-learn 不可用，无法标定")
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在")

    if feature_cols is None:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in num if c != target_col]
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("未找到可用数值特征列")
    if len(df) < 5:
        raise ValueError("样本量过少（<5），无法标定")

    data = df[[*feature_cols, target_col]].dropna()
    X = data[feature_cols].values
    y = data[target_col].values

    # 任务识别：非数值→分类；浮点→回归；整数低基数→分类，否则回归
    if task in ("regression", "classification"):
        is_regression = (task == "regression")
    elif not pd.api.types.is_numeric_dtype(df[target_col]):
        is_regression = False
    elif pd.api.types.is_float_dtype(df[target_col]):
        is_regression = True
    else:
        is_regression = np.unique(y).size > 10

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed)

    if is_regression:
        model = RandomForestRegressor(n_estimators=300, max_depth=14,
                                      random_state=seed, n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        baseline = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
        base_pred = baseline.predict(X_te)
        metrics = {
            "task": "regression",
            "r2": float(r2_score(y_te, pred)),
            "r2_baseline": float(r2_score(y_te, base_pred)),
            "mae": float(mean_absolute_error(y_te, pred)),
            "mae_baseline": float(mean_absolute_error(y_te, base_pred)),
        }
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=14,
                                       random_state=seed, n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        base_pred = baseline.predict(X_te)
        metrics = {
            "task": "classification",
            "accuracy": float(accuracy_score(y_te, pred)),
            "accuracy_baseline": float(accuracy_score(y_te, base_pred)),
            "f1_macro": float(f1_score(y_te, pred, average="macro")),
            "f1_baseline": float(f1_score(y_te, base_pred, average="macro")),
        }

    importances = sorted(zip(feature_cols, model.feature_importances_),
                         key=lambda t: t[1], reverse=True)
    return {
        "metrics": metrics,
        "importances": importances,
        "feature_cols": feature_cols,
        "n_samples": int(len(data)),
    }
