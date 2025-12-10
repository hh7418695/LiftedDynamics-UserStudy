import pandas as pd
import matplotlib.pyplot as plt

# 1. 读入你的 CSV 文件
df = pd.read_csv("behaviour_results\participant_HUANGHAO_stretch_behaviour.csv")

# 看一下前几行，确认列名（可选）
print(df.head())

# 2. 计算 Δks（比较物体 - 参考物体）
# 按我们之前的设计，comp 的 ks 应该总是 >= ref 的 ks
df["delta_ks"] = df["k_stretch_comp"] - df["k_stretch_ref"]

# 3. 按 Δks 分组，计算正确率（这里用 is_correct 列）
psych = (
    df.groupby("delta_ks")
      .agg(
          p_correct=("is_correct", "mean"),  # 这一档 Δks 下的平均正确率
          n=("is_correct", "size")           # 这一档下有多少 trial
      )
      .reset_index()
      .sort_values("delta_ks")
)

print("按 Δks 聚合后的结果：")
print(psych)

# 4. 画 psychometric curve
plt.figure()
plt.plot(psych["delta_ks"], psych["p_correct"], marker="o")
plt.xlabel("Δks (k_stretch_comp - k_stretch_ref)")
plt.ylabel("Proportion correct")
plt.ylim(0, 1)
plt.title("Psychometric curve – stretching block")
plt.grid(True)
plt.show()

def chose_comp(row):
    # 把被试说“第一个/第二个”翻译成“选的是哪个 object_id”
    if row["chosen_object"] == "first":
        chosen_id = row["first_object"]
    elif row["chosen_object"] == "second":
        chosen_id = row["second_object"]
    else:
        return None  # 万一有奇怪的值

    # 如果选中的那个就是 comp_object_id，则记为 1，否则 0
    return 1 if chosen_id == row["comp_object_id"] else 0

df["choose_comp"] = df.apply(chose_comp, axis=1)

psych_choice = (
    df.groupby("delta_ks")
      .agg(
          p_choose_comp=("choose_comp", "mean"),
          n=("choose_comp", "size")
      )
      .reset_index()
      .sort_values("delta_ks")
)

print(psych_choice)

plt.figure()
plt.plot(psych_choice["delta_ks"], psych_choice["p_choose_comp"], marker="o")
plt.xlabel("Δks (k_stretch_comp - k_stretch_ref)")
plt.ylabel("P(choose comparison)")
plt.ylim(0, 1)
plt.title("Psychometric curve – choose comparison (stretching)")
plt.grid(True)
plt.show()
