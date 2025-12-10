import itertools
import random

def build_stretch_trials():
    """先给拉伸 block 随便造几条 trial 列表：object 1–5，全组合，每对重复 4 次。"""
    objs = [1, 2, 3, 4, 5]
    pairs = list(itertools.combinations(objs, 2))  # [(1,2), (1,3), ...]
    trials = []

    for (a, b) in pairs:
        for _ in range(4):  # 每个 pair 重复 4 次
            # 随机决定谁先谁后
            if random.random() < 0.5:
                first, second = a, b
            else:
                first, second = b, a

            trials.append({
                "block": "stretch",
                "obj_first": first,
                "obj_second": second,
                "response": None,  # 先留空，后面再填 1 或 2
            })

    random.shuffle(trials)  # 打乱 trial 顺序
    return trials

def run_trials_keyboard_only(trials, question_text):
    """只用键盘问：哪一个更硬？记录 1 / 2。"""
    for t_idx, t in enumerate(trials, start=1):
        print(f"\n=== Trial {t_idx}/{len(trials)} ===")
        print(f"First object : {t['obj_first']}")
        print(f"Second object: {t['obj_second']}")

        # 这里本来是“让被试摸第一个物体”的阶段
        input("现在是第一个物体：按 Enter 继续（这里先当占位，不连设备）...")
        # 这里本来是“让被试摸第二个物体”
        input("现在是第二个物体：按 Enter 继续...")

        # 让被试或实验者按 1 / 2
        while True:
            resp = input(
                f"{question_text} (按 1 = 第一个, 2 = 第二个): "
            ).strip()
            if resp in ("1", "2"):
                t["response"] = int(resp)
                break
            else:
                print("只能输入 1 或 2 哦，再试一次。")

    return trials

if __name__ == "__main__":
    trials = build_stretch_trials()
    # 拉伸 block 的问题
    question = "哪一个更不容易被拉长？"
    finished_trials = run_trials_keyboard_only(trials, question)

    print("\n实验结束，前 3 条记录示例：")
    for t in finished_trials[:3]:
        print(t)
