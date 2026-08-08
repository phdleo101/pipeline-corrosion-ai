# -*- coding: utf-8 -*-
"""
MIC 机器学习预测 (P3 深化)

用「物理约束合成数据集」训练随机森林，从环境/材料/运行特征预测 MIC 风险等级
与腐蚀速率，并给出特征重要性。数据先按目标风险等级分布分配样本（保证 4 类
均衡可学），再按类别条件生成特征：风险越高，微生物活性(SRB/APB/IRB/O2)、
营养(硫酸盐/含水)、不利环境(低流速/高氯/低PREN/中温)特征越显著。用于演示
端到端 ML 流程与特征重要性。**模型基于合成数据，正式评估须以现场检测为准。**

特征 (12):
  pH, chloride_ppm, SRB_log, APB_log, IRB_log, O2_ppm,
  H2S_ppm, temperature, flow_velocity, water_cut, sulfate_ppm, pren
标签:
  risk_class 0-3 -> MIC_RISK_LABELS ['低','中','高','极高']
  mic_rate (mm/a) 连续腐蚀速率
"""

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, f1_score, r2_score, mean_absolute_error,
    )
    _SKLEARN_OK = True
except Exception:  # pragma: no cover
    _SKLEARN_OK = False

MIC_FEATURES = [
    "pH", "chloride_ppm", "SRB_log", "APB_log", "IRB_log", "O2_ppm",
    "H2S_ppm", "temperature", "flow_velocity", "water_cut", "sulfate_ppm", "pren",
]
MIC_RISK_LABELS = ["低", "中", "高", "极高"]

# 模块级缓存，避免每次 rerun 重复训练（Streamlit 进程内模块状态持久）
_MODELS = None


def generate_synthetic_dataset(n=3000, seed=42):
    """
    生成物理约束的合成数据集（类别条件生成，4 类均衡）。
    返回 (X: DataFrame[features], y_class: ndarray, y_rate: ndarray)
    """
    rng = np.random.default_rng(seed)

    # 先按目标风险等级分布分配样本（保证 4 类均衡可学），再按类别条件生成特征
    classes = np.array([0, 1, 2, 3])                       # 低/中/高/极高
    class_prob = np.array([0.20, 0.35, 0.35, 0.10])
    y_class = rng.choice(classes, size=n, p=class_prob)

    def per_class(mu, sd, lo, hi):
        # mu 为长度 4 的数组，按样本类别取均值，叠加噪声并裁剪
        return np.clip(rng.normal(np.array(mu)[y_class], sd, n), lo, hi)

    pH = per_class([7.2, 7.0, 6.7, 6.4], 0.35, 4.0, 9.0)
    chloride = np.exp(per_class([6.9, 8.2, 9.2, 10.3], 0.45, 2.3, 11.0))   # log(ppm)
    SRB = per_class([0.4, 2.5, 4.5, 6.2], 1.2, 0.0, 7.0)                   # log cells/mL
    APB = per_class([0.2, 1.4, 3.0, 4.6], 1.0, 0.0, 6.0)
    IRB = per_class([0.2, 1.4, 3.0, 4.6], 1.0, 0.0, 6.0)
    O2 = per_class([0.3, 1.4, 2.8, 4.0], 0.9, 0.0, 5.0)
    H2S = np.exp(per_class([0.2, 1.6, 2.4, 3.3], 0.4, -1.0, 4.0))          # log(ppm)
    temp = per_class([24, 34, 45, 55], 7.0, 5.0, 80.0)
    flow = per_class([1.7, 1.0, 0.6, 0.3], 0.35, 0.0, 3.0)
    water = per_class([12, 40, 65, 85], 12.0, 0.0, 100.0)
    sulfate = per_class([120, 500, 1200, 2200], 350.0, 0.0, 3000.0)
    pren = per_class([39, 32, 26, 21], 2.5, 18.0, 45.0)

    # 连续腐蚀速率标签：随风险等级升高
    y_rate = np.clip(np.array([0.05, 0.25, 0.55, 0.95])[y_class]
                     + rng.normal(0.0, 0.06, n), 0.01, 2.0)

    X = pd.DataFrame({
        "pH": pH, "chloride_ppm": chloride, "SRB_log": SRB, "APB_log": APB,
        "IRB_log": IRB, "O2_ppm": O2, "H2S_ppm": H2S, "temperature": temp,
        "flow_velocity": flow, "water_cut": water, "sulfate_ppm": sulfate,
        "pren": pren,
    })
    return X, y_class, y_rate


def train_models(X, y_class, y_rate, seed=42):
    """训练分类器(风险等级)与回归器(腐蚀速率)，返回指标与特征重要性。"""
    if not _SKLEARN_OK:
        raise RuntimeError("scikit-learn 不可用，无法训练 MIC 模型")

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X, y_class, test_size=0.2, random_state=seed, stratify=y_class)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y_rate, test_size=0.2, random_state=seed)

    clf = RandomForestClassifier(n_estimators=300, max_depth=14,
                                 random_state=seed, n_jobs=-1)
    clf.fit(Xc_tr, yc_tr)
    yc_pred = clf.predict(Xc_te)

    reg = RandomForestRegressor(n_estimators=300, max_depth=14,
                                random_state=seed, n_jobs=-1)
    reg.fit(Xr_tr, yr_tr)
    yr_pred = reg.predict(Xr_te)

    imp_clf = sorted(zip(MIC_FEATURES, clf.feature_importances_),
                     key=lambda t: t[1], reverse=True)

    metrics = {
        "accuracy": float(accuracy_score(yc_te, yc_pred)),
        "f1_macro": float(f1_score(yc_te, yc_pred, average="macro")),
        "r2": float(r2_score(yr_te, yr_pred)),
        "mae": float(mean_absolute_error(yr_te, yr_pred)),
        "n_train": int(len(Xc_tr)),
        "n_test": int(len(Xc_te)),
    }
    return {"clf": clf, "reg": reg, "metrics": metrics,
            "importances": imp_clf}


def get_trained_models(seed=42):
    """模块级缓存训练结果（进程内只训练一次）。"""
    global _MODELS
    if _MODELS is None:
        X, yc, yr = generate_synthetic_dataset(seed=seed)
        _MODELS = train_models(X, yc, yr, seed=seed)
    return _MODELS


def predict_mic_risk(features, seed=42):
    """
    单条预测。
    features: dict，键为 MIC_FEATURES。
    返回 {risk_class, risk_label, probabilities, predicted_rate}
    probabilities 为对齐 4 类的完整向量（缺失类记为 0），与 MIC_RISK_LABELS 对齐。
    """
    m = get_trained_models(seed=seed)
    x = pd.DataFrame(
        [[float(features.get(f, 0.0)) for f in MIC_FEATURES]], columns=MIC_FEATURES)
    proba = m["clf"].predict_proba(x)[0]
    classes_ = m["clf"].classes_
    idx = int(np.argmax(proba))
    cls = int(classes_[idx])

    full = [0.0] * len(MIC_RISK_LABELS)
    for i, c in enumerate(classes_):
        full[int(c)] = float(proba[i])

    rate = float(m["reg"].predict(x)[0])
    return {
        "risk_class": cls,
        "risk_label": MIC_RISK_LABELS[cls],
        "probabilities": full,
        "predicted_rate": rate,
    }


DEFAULT_FEATURES = {
    "pH": 6.5, "chloride_ppm": 20000.0, "SRB_log": 4.0, "APB_log": 2.0,
    "IRB_log": 1.5, "O2_ppm": 0.5, "H2S_ppm": 8.0, "temperature": 40.0,
    "flow_velocity": 0.3, "water_cut": 70.0, "sulfate_ppm": 800.0, "pren": 22.0,
}
