# -*- coding: utf-8 -*-
"""
SCC 裂纹形貌与扩展蒙特卡洛模拟 (P3 深化)

基于初始裂纹深度与扩展速率的概率分布，对一组裂纹种群在 T 年内的演化进行
蒙特卡洛抽样，输出：
  - 深度分布直方图（含临界深度 a_c 标线）
  - 超概率曲线 POD（Probability of Exceedance）
  - 深度-长度形貌散点（按分叉着色）
并给出穿壁失效比例等统计。扩展速率与 SCC-2 (scc_crack_life) 经验区间一致。

参考：NACE SP0204 (SCC 直接评估)、API 579 / RSTRENG (裂纹评定)、
      Battelle NG-18 / Kiefner & Vieth。
"""

import numpy as np
import plotly.graph_objects as go


def simulate_crack_population(
    n_cracks=300,
    years=25,
    a0_mean=1.0,
    a0_sigma=0.5,
    growth_mean=0.06,
    growth_sigma=0.5,
    wall_thickness=12.0,
    crack_aspect=15.0,
    branch_prob=0.4,
    seed=42,
):
    """
    蒙特卡洛模拟裂纹种群扩展。

    参数:
        n_cracks: 裂纹数量
        years: 模拟年限
        a0_mean: 初始裂纹深度中位数 (mm)
        a0_sigma: 初始深度对数正态标准差
        growth_mean: 年扩展速率中位数 (mm/a)
        growth_sigma: 扩展速率对数正态标准差
        wall_thickness: 壁厚 (mm)，用于判定穿壁临界深度 a_c
        crack_aspect: 裂纹深宽比 a/c 的倒数近似 (c ≈ a * aspect)
        branch_prob: 分叉裂纹比例
        seed: 随机种子

    返回: 含数组与统计的字典
    """
    rng = np.random.default_rng(seed)

    # 初始深度：对数正态（保证正值）
    a0 = rng.lognormal(mean=np.log(max(a0_mean, 0.05)), sigma=a0_sigma, size=n_cracks)
    # 年扩展速率：对数正态
    g = rng.lognormal(mean=np.log(max(growth_mean, 0.002)), sigma=growth_sigma, size=n_cracks)

    depth_T = a0 + g * years
    depth_T = np.clip(depth_T, 0.0, None)

    # 裂纹半长 c（深宽比近似）
    c = depth_T * crack_aspect * rng.uniform(0.7, 1.3, size=n_cracks)

    # 分叉（穿晶 SCC 常见支化）
    branch = (rng.random(n_cracks) < branch_prob).astype(int)

    a_c = float(wall_thickness)
    failures = int(np.sum(depth_T >= a_c))
    near_critical = int(np.sum((depth_T >= 0.8 * a_c) & (depth_T < a_c)))

    summary = {
        "n": n_cracks,
        "years": years,
        "a_c": a_c,
        "mean_depth": float(np.mean(depth_T)),
        "p50_depth": float(np.percentile(depth_T, 50)),
        "p90_depth": float(np.percentile(depth_T, 90)),
        "p99_depth": float(np.percentile(depth_T, 99)),
        "max_depth": float(np.max(depth_T)),
        "failures": failures,
        "near_critical": near_critical,
        "failure_prob": float(failures / n_cracks),
        "near_critical_prob": float(near_critical / n_cracks),
    }

    return {
        "a0": a0,
        "growth": g,
        "depth_T": depth_T,
        "length": c,
        "branch": branch,
        "summary": summary,
    }


def _pod_curve(depths, n_bins=60):
    """超概率 (Probability of Exceedance) 曲线。"""
    x = np.linspace(0.0, float(np.max(depths)) * 1.02, n_bins)
    n = len(depths)
    y = np.array([np.mean(depths >= xi) for xi in x])
    return x, y


def build_morphology_figures(sim, dark_mode=True):
    """
    构建三张图：深度直方图 / POD 曲线 / 形貌散点。
    返回 dict(fig_hist, fig_pod, fig_scatter)
    """
    template = "plotly_dark" if dark_mode else "plotly_white"
    s = sim["summary"]
    a_c = s["a_c"]

    # 1) 深度直方图
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=sim["depth_T"], nbinsx=40, marker_color="#3498db",
        name="裂纹深度", opacity=0.85,
    ))
    fig_hist.add_vline(x=a_c, line_color="#e74c3c", line_dash="dash",
                       annotation_text=f"穿壁临界 a_c={a_c:.1f}mm",
                       annotation_position="top right")
    fig_hist.update_layout(
        title=f"T={s['years']}年后裂纹深度分布 (n={s['n']})",
        xaxis_title="裂纹深度 (mm)", yaxis_title="频数",
        height=320, margin=dict(l=50, r=20, t=40, b=40), template=template,
    )

    # 2) POD 曲线
    x, y = _pod_curve(sim["depth_T"])
    fig_pod = go.Figure()
    fig_pod.add_trace(go.Scatter(
        x=x, y=y * 100, mode="lines", fill="tozeroy",
        line=dict(color="#e67e22", width=2), name="POD",
    ))
    fig_pod.add_vline(x=a_c, line_color="#e74c3c", line_dash="dash")
    fig_pod.update_layout(
        title="超概率曲线 POD (深度 ≥ x 的裂纹比例)",
        xaxis_title="裂纹深度 (mm)", yaxis_title="比例 (%)",
        height=320, margin=dict(l=50, r=20, t=40, b=40), template=template,
    )

    # 3) 形貌散点 (深度 vs 长度, 按分叉着色)
    branched = sim["branch"] == 1
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=sim["length"][~branched], y=sim["depth_T"][~branched],
        mode="markers", name="无分叉",
        marker=dict(color="#2980b9", size=5, opacity=0.6),
    ))
    fig_scatter.add_trace(go.Scatter(
        x=sim["length"][branched], y=sim["depth_T"][branched],
        mode="markers", name="分叉裂纹",
        marker=dict(color="#c0392b", size=6, opacity=0.75, symbol="diamond"),
    ))
    fig_scatter.add_hline(y=a_c, line_color="#e74c3c", line_dash="dash",
                          annotation_text="a_c 穿壁临界")
    fig_scatter.update_layout(
        title="裂纹形貌: 深度-长度分布 (分叉着色)",
        xaxis_title="裂纹半长 c (mm)", yaxis_title="裂纹深度 a (mm)",
        height=360, margin=dict(l=50, r=20, t=40, b=40), template=template,
    )

    return {"fig_hist": fig_hist, "fig_pod": fig_pod, "fig_scatter": fig_scatter}
