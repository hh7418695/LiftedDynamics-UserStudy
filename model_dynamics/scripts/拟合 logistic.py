import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 读数据
df = pd.read_csv("behaviour_results\participant_HUANGHAO_stretch_behaviour.csv")

# 计算 Δks
df["delta_ks"] = df["k_stretch_comp"] - df["k_stretch_ref"]

# 按 Δks 聚合
psych = (
    df.groupby("delta_ks")
      .agg(
          p_correct=("is_correct", "mean"),
          n=("is_correct", "size")
      )
      .reset_index()
      .sort_values("delta_ks")
)

print(psych)

x = psych["delta_ks"].values
y = psych["p_correct"].values

# 2. 定义 2AFC 的 logistic 心理物理函数
def psych_func(x, alpha, beta):
    """
    2AFC psychometric:
    p(x) = 0.5 + 0.5 * sigmoid( (x - alpha) / beta )
    其中 alpha 是 75% 正确的阈值
    """
    return 0.5 + 0.5 * (1.0 / (1.0 + np.exp(-(x - alpha) / beta)))

# 初始猜测：阈值在中间，斜率随便给个大致范围
alpha0 = np.median(x)
beta0 = (x.max() - x.min()) / 4

p0 = [alpha0, beta0]

# 3. 拟合参数
params, cov = curve_fit(
    psych_func,
    x, y,
    p0=p0,
    bounds=([0, 0], [np.inf, np.inf])  # alpha、beta 都限制为非负
)

alpha_hat, beta_hat = params
print("拟合得到的阈值（Δks at 75% correct）:", alpha_hat)
print("拟合得到的斜率参数 beta:", beta_hat)

# 4. 画拟合曲线 + 原始点
x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = psych_func(x_fit, alpha_hat, beta_hat)

plt.figure()
plt.scatter(x, y, label="Data (p_correct)")
plt.plot(x_fit, y_fit, label="Fitted psychometric")

# 画出 75% 正确的水平线和阈值竖线
plt.axhline(0.75, linestyle="--")
plt.axvline(alpha_hat, linestyle="--")

plt.xlabel("Δks (k_stretch_comp - k_stretch_ref)")
plt.ylabel("Proportion correct")
plt.ylim(0.4, 1.05)
plt.title("Psychometric curve with fitted logistic (stretching)")
plt.grid(True)
plt.legend()
plt.show()
