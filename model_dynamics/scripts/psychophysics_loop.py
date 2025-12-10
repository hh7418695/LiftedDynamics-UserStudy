import itertools
import random
import csv
from pathlib import Path
from datetime import datetime
import json

# 你现在代码中使用的刚度列表
STRETCH_VALUES = [50.0, 287.5, 525.0, 762.5, 1000.0]   # ks 对应 object 1–5
BEND_VALUES    = [0.0, 0.025, 0.05, 0.075, 0.1]        # kb 对应 object 6–10


def get_stiffness_for_object(object_id: int):
    """
    根据 object_id 计算对应的拉伸刚度 ks 和弯曲刚度 kb。
    规则和你原来的 main.py 保持一致：
    - object 1–5：ks 变化，kb 固定为 BEND_VALUES[2] = 0.05
    - object 6–10：kb 变化，ks 固定为 STRETCH_VALUES[2] = 525
    """
    if 1 <= object_id <= 5:
        ks = STRETCH_VALUES[object_id - 1]
        kb = BEND_VALUES[2]
    elif 6 <= object_id <= 10:
        ks = STRETCH_VALUES[2]
        kb = BEND_VALUES[object_id - 6]
    else:
        raise ValueError(f"Invalid object_id {object_id}, must be 1–10.")
    return ks, kb


def build_block_trials(participant_id: str, block_type: str, reps_per_pair: int = 4):
    """
    构造一个 block（'stretch' 或 'bend'）的 trial 列表。
    先不加被试的选择，只生成 trial 信息。
    """
    if block_type not in ("stretch", "bend"):
        raise ValueError("block_type must be 'stretch' or 'bend'")

    if block_type == "stretch":
        objs = [1, 2, 3, 4, 5]
    else:
        objs = [6, 7, 8, 9, 10]

    pairs = list(itertools.combinations(objs, 2))  # e.g. [(1,2), (1,3), ...]
    trials = []

    for (a, b) in pairs:
        # 约定：ref = 较小 id，comp = 较大 id，方便读
        ref_id, comp_id = sorted([a, b])

        ks_ref, kb_ref = get_stiffness_for_object(ref_id)
        ks_comp, kb_comp = get_stiffness_for_object(comp_id)

        # block 不同，看“物理上更硬”的定义不同：
        # - stretch：看 ks 谁大
        # - bend：   看 kb 谁大
        if block_type == "stretch":
            target_id = ref_id if ks_ref > ks_comp else comp_id
        else:
            target_id = ref_id if kb_ref > kb_comp else comp_id

        for _ in range(reps_per_pair):
            # 随机决定谁是 first / second
            if random.random() < 0.5:
                first, second = ref_id, comp_id
            else:
                first, second = comp_id, ref_id

            trial = {
                "participant_id": participant_id,
                "trial_index": None,        # 稍后统一编号
                "block_type": block_type,   # 'stretch' or 'bend'
                "ref_object_id": ref_id,
                "comp_object_id": comp_id,
                "k_stretch_ref": ks_ref,
                "k_stretch_comp": ks_comp,
                "k_bend_ref": kb_ref,
                "k_bend_comp": kb_comp,
                "first_object": first,
                "second_object": second,
                "target_object": target_id,  # 正确答案对应的 object_id
                "chosen_object": None,       # 之后填 'first' or 'second'
                "is_correct": None,          # 之后填 1/0
                "notes": "",
            }
            trials.append(trial)

    random.shuffle(trials)
    return trials

def append_trial_row(csv_path: Path, trial: dict):
    """
    把一个 trial 的信息追加写入 CSV。
    如果文件不存在，先写表头。
    """
    fieldnames = [
        "participant_id",
        "trial_index",
        "block_type",
        "ref_object_id",
        "comp_object_id",
        "k_stretch_ref",
        "k_stretch_comp",
        "k_bend_ref",
        "k_bend_comp",
        "first_object",
        "second_object",
        "chosen_object",
        "is_correct",
        "notes",
    ]
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {k: trial.get(k, "") for k in fieldnames}
        writer.writerow(row)

def run_2afc_experiment_cli():
    """
    终端驱动单个 2AFC block（stretch 或 bend）：
      - 输入 participant_id
      - 选择只做 stretching 或 bending
      - 自动生成 40 个 trial，逐一运行
      - 每个 trial 结束按 1/2，程序自动算 is_correct 并写入 CSV
      - 中间在 20/40 处提示休息
    """
    # 1) participant_id (字母标识)
    while True:
        pid_raw = input("请输入 participant_id（字母，例如 A）：").strip()
        if len(pid_raw) == 0:
            print("输入不能为空，请输入一个字母。")
            continue
        # 只接受字母作为被试 id（允许多字符，但只保留字母，规范为大写）
        pid_filtered = ''.join(ch for ch in pid_raw if ch.isalpha())
        if len(pid_filtered) == 0:
            print("输入必须包含字母，请重新输入。")
            continue
        participant_id = pid_filtered.upper()
        break

    # 2) 选择要测试的 block 类型
    print("\n请选择要测试的 block：")
    print("  1 = stretching（拉伸刚度 ks）")
    print("  2 = bending   （弯曲刚度 kb）")
    choice = input("你的选择（1 或 2）：").strip()
    if choice == "1":
        block_type = "stretch"
    else:
        block_type = "bend"

    # 3) 准备结果目录与 order 文件（用于保证重跑时顺序一致）
    results_dir = Path("behaviour_results")
    results_dir.mkdir(exist_ok=True)
    order_path = results_dir / f"participant_{participant_id}_{block_type}_order.json"

    # 4) 构造这个 block 的 40 个 trials（如果已有 order 文件则按文件恢复）
    if order_path.exists():
        with order_path.open("r", encoding="utf-8") as f:
            trials = json.load(f)
        print(f"从已保存的顺序文件恢复试次（{order_path.name}）。")
    else:
        trials = build_block_trials(participant_id, block_type, reps_per_pair=4)
        with order_path.open("w", encoding="utf-8") as f:
            json.dump(trials, f, ensure_ascii=False, indent=2)
        print(f"已生成新的试次顺序并保存到 {order_path.name}。")

    # 统一编号 1..N（应该是 40）
    for idx, t in enumerate(trials, start=1):
        t["trial_index"] = idx

    total_trials = len(trials)
    print(f"\n本次将进行 {total_trials} 个 trial（单个 block：{block_type}）。")

    # 5) 准备 CSV 路径（按 block 类型分文件）
    csv_path = results_dir / f"participant_{participant_id}_{block_type}_behaviour.csv"
    print(f"数据将保存到: {csv_path.resolve()}")

    # 如果已有文件，允许从上次继续 / 备份并重跑 / 取消
    if csv_path.exists():
        # 读取已有行数（排除 header）
        with csv_path.open("r", newline="") as f:
            existing_rows = list(csv.reader(f))
        completed = max(0, len(existing_rows) - 1)  # header 一行
        if completed > 0:
            while True:
                ans = input(
                    f"检测到已有 {completed} 条已保存记录。选择： (r) 继续 (s) 备份并重跑 (c) 取消："
                ).strip().lower()
                if ans.startswith("r"):
                    # 跳过已完成的 trial
                    trials = [t for t in trials if t["trial_index"] > completed]
                    print(f"将从第 {completed+1} 条开始继续，共剩 {len(trials)} 条。")
                    break
                if ans.startswith("s"):
                    # 备份旧文件并开始重跑
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bak = csv_path.parent / f"{csv_path.name}.bak_{ts}"
                    csv_path.rename(bak)
                    print(f"已备份旧文件到 {bak.name}，开始重跑。")
                    break
                if ans.startswith("c"):
                    print("操作已取消。")
                    return
                print("请输入 r / s / c。")

    # 5) 主循环
    for t in trials:
        i = t["trial_index"]

        # 在中间提示一次休息（20/40 之后）
        if i == total_trials // 2 + 1:  # 即 trial 21 之前提示
            input(
                f"\n你已经完成一半（{total_trials//2}/{total_trials}），"
                "可以休息一下，准备好后按 Enter 继续..."
            )

        block = t["block_type"]
        print("\n" + "=" * 40)
        print(f"Trial {i}/{total_trials} | Block = {block}")
        print(f"  Pair: ref={t['ref_object_id']}  comp={t['comp_object_id']}")
        print(f"  First object : {t['first_object']}")
        print(f"  Second object: {t['second_object']}")

        # 这里将来会换成真正的 haptic 呈现，目前先占位
        input("\n现在让被试接触【第一个物体】，操作完按 Enter 继续（占位）...")
        input("现在让被试接触【第二个物体】，操作完按 Enter 继续（占位）...")

        # 根据 block 决定提问内容
        if block == "stretch":
            question = "哪一个更不容易被拉长？"
        else:
            question = "哪一个在弯的时候感觉更硬？"

        # 让被试按 1 或 2
        while True:
            resp = input(f"{question} (1 = 第一个, 2 = 第二个)：").strip()
            if resp in ("1", "2"):
                t["chosen_object"] = "first" if resp == "1" else "second"
                chosen_id = t["first_object"] if resp == "1" else t["second_object"]
                # target_object 在 build_block_trials 里已经算好
                t["is_correct"] = 1 if chosen_id == t["target_object"] else 0
                break
            else:
                print("只能输入 1 或 2，请重新输入。")

        t["notes"] = ""  # 先留空

        # ✅ 每个 trial 结束立刻写一行 CSV
        append_trial_row(csv_path, t)

    print("\n本 block 实验完成！所有行为数据已写入 CSV。")



def build_all_trials(participant_id: str, reps_per_pair: int = 4):
    """生成两个 block 的所有 trial，并统一给 trial_index 编号。"""
    all_trials = []

    # 先拉伸再弯曲（顺序之后可以改成让用户选，这里先固定）
    for block_type in ("stretch", "bend"):
        block_trials = build_block_trials(participant_id, block_type, reps_per_pair)
        all_trials.extend(block_trials)

    # 给所有 trial 编号 1..N
    for idx, t in enumerate(all_trials, start=1):
        t["trial_index"] = idx

    return all_trials


if __name__ == "__main__":
    run_2afc_experiment_cli()
