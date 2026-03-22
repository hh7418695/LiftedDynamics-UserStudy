import itertools
import random
import csv
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

# 你现在代码中使用的刚度列表
STRETCH_VALUES = [50.0, 287.5, 525.0, 762.5, 1000.0]   # ks 对应 object 1–5
BEND_VALUES    = [0.0, 0.025, 0.05, 0.075, 0.1]        # kb 对应 object 6–10

# CSV 字段名定义（避免重复定义）
CSV_FIELDNAMES = [
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


def get_stiffness_for_object(object_id: int) -> Tuple[float, float]:
    """
    根据 object_id 计算对应的拉伸刚度 ks 和弯曲刚度 kb。
    规则和你原来的 main.py 保持一致：
    - object 1–5：ks 变化，kb 固定为 BEND_VALUES[2] = 0.05
    - object 6–10：kb 变化，ks 固定为 STRETCH_VALUES[2] = 525

    Args:
        object_id: Object identifier (1-10)

    Returns:
        Tuple of (ks, kb) stiffness values

    Raises:
        ValueError: If object_id is not in range 1-10
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


def _determine_target_object(ref_id: int, comp_id: int, block_type: str) -> int:
    """
    确定哪个物体是"更硬"的目标物体。

    Args:
        ref_id: Reference object ID
        comp_id: Comparison object ID
        block_type: 'stretch' or 'bend'

    Returns:
        Target object ID (the stiffer one)
    """
    ks_ref, kb_ref = get_stiffness_for_object(ref_id)
    ks_comp, kb_comp = get_stiffness_for_object(comp_id)

    if block_type == "stretch":
        return ref_id if ks_ref > ks_comp else comp_id
    else:
        return ref_id if kb_ref > kb_comp else comp_id


def build_block_trials(participant_id: str, block_type: str, reps_per_pair: int = 4) -> List[Dict]:
    """
    构造一个 block（'stretch' 或 'bend'）的 trial 列表。
    先不加被试的选择，只生成 trial 信息。

    Args:
        participant_id: Participant identifier
        block_type: 'stretch' or 'bend'
        reps_per_pair: Number of repetitions per object pair

    Returns:
        List of trial dictionaries (shuffled)

    Raises:
        ValueError: If block_type is invalid
    """
    if block_type not in ("stretch", "bend"):
        raise ValueError("block_type must be 'stretch' or 'bend'")

    # 根据 block 类型选择对象
    objs = list(range(1, 6)) if block_type == "stretch" else list(range(6, 11))
    pairs = list(itertools.combinations(objs, 2))
    trials = []

    for ref_id, comp_id in pairs:
        # 获取刚度值
        ks_ref, kb_ref = get_stiffness_for_object(ref_id)
        ks_comp, kb_comp = get_stiffness_for_object(comp_id)

        # 确定目标物体
        target_id = _determine_target_object(ref_id, comp_id, block_type)

        for _ in range(reps_per_pair):
            # 随机决定呈现顺序
            first, second = (ref_id, comp_id) if random.random() < 0.5 else (comp_id, ref_id)

            trial = {
                "participant_id": participant_id,
                "trial_index": None,
                "block_type": block_type,
                "ref_object_id": ref_id,
                "comp_object_id": comp_id,
                "k_stretch_ref": ks_ref,
                "k_stretch_comp": ks_comp,
                "k_bend_ref": kb_ref,
                "k_bend_comp": kb_comp,
                "first_object": first,
                "second_object": second,
                "target_object": target_id,
                "chosen_object": None,
                "is_correct": None,
                "notes": "",
            }
            trials.append(trial)

    random.shuffle(trials)
    return trials

def append_trial_row(csv_path: Path, trial: Dict) -> None:
    """
    把一个 trial 的信息追加写入 CSV。
    如果文件不存在，先写表头。

    Args:
        csv_path: Path to CSV file
        trial: Trial dictionary containing data to write

    Raises:
        IOError: If file cannot be written
    """
    file_exists = csv_path.exists()
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            row = {k: trial.get(k, "") for k in CSV_FIELDNAMES}
            writer.writerow(row)
    except IOError as e:
        raise IOError(f"Failed to write trial data to {csv_path}: {e}")

def _get_participant_id() -> str:
    """
    获取并验证 participant ID。

    Returns:
        Validated participant ID (uppercase letters only)
    """
    while True:
        pid_raw = input("请输入 participant_id（字母，例如 A）/ Enter participant ID (letter, e.g., A): ").strip()
        if not pid_raw:
            print("输入不能为空，请输入一个字母。/ Input cannot be empty, please enter a letter.")
            continue

        pid_filtered = ''.join(ch for ch in pid_raw if ch.isalpha())
        if not pid_filtered:
            print("输入必须包含字母，请重新输入。/ Input must contain letters, please try again.")
            continue

        return pid_filtered.upper()


def _get_block_type() -> str:
    """
    获取用户选择的 block 类型。

    Returns:
        'stretch' or 'bend'
    """
    print("\n请选择要测试的 block / Select block to test:")
    print("  1 = stretching（拉伸刚度 ks / Stretching stiffness ks）")
    print("  2 = bending   （弯曲刚度 kb / Bending stiffness kb）")

    while True:
        choice = input("你的选择（1 或 2）/ Your choice (1 or 2): ").strip()
        if choice == "1":
            return "stretch"
        elif choice == "2":
            return "bend"
        else:
            print("无效输入，请输入 1 或 2。/ Invalid input, please enter 1 or 2.")

def _load_or_create_trials(order_path: Path, participant_id: str, block_type: str,
                           reps_per_pair: int = 4) -> List[Dict]:
    """
    加载已有的试次顺序或创建新的试次列表。

    Args:
        order_path: Path to order JSON file
        participant_id: Participant identifier
        block_type: 'stretch' or 'bend'
        reps_per_pair: Number of repetitions per pair

    Returns:
        List of trial dictionaries
    """
    if order_path.exists():
        try:
            with order_path.open("r", encoding="utf-8") as f:
                trials = json.load(f)
            print(f"✓ 从已保存的顺序文件恢复试次 / Restored trials from saved order file ({order_path.name})")
            return trials
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠ 警告：无法读取顺序文件 / Warning: Cannot read order file {order_path.name}: {e}")
            print("将创建新的试次顺序... / Creating new trial order...")

    # 创建新的试次列表
    trials = build_block_trials(participant_id, block_type, reps_per_pair)
    try:
        with order_path.open("w", encoding="utf-8") as f:
            json.dump(trials, f, ensure_ascii=False, indent=2)
        print(f"✓ 已生成新的试次顺序并保存 / Generated and saved new trial order to {order_path.name}")
    except IOError as e:
        print(f"⚠ 警告：无法保存顺序文件 / Warning: Cannot save order file: {e}")

    return trials


def _handle_existing_data(csv_path: Path, trials: List[Dict]) -> List[Dict]:
    """
    处理已存在的数据文件，允许继续或重新开始。

    Args:
        csv_path: Path to CSV data file
        trials: List of all trials

    Returns:
        List of remaining trials to complete
    """
    if not csv_path.exists():
        return trials

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.reader(f))
        completed = max(0, len(existing_rows) - 1)  # 减去 header
    except IOError as e:
        print(f"⚠ 警告：无法读取现有数据文件 / Warning: Cannot read existing data file: {e}")
        return trials

    if completed == 0:
        return trials

    while True:
        ans = input(
            f"\n检测到已有 {completed} 条已保存记录。/ Detected {completed} saved records.\n"
            f"  (r) 继续从第 {completed + 1} 条开始 / Resume from trial {completed + 1}\n"
            f"  (s) 备份并重新开始 / Backup and restart\n"
            f"  (c) 取消 / Cancel\n"
            f"你的选择 / Your choice: "
        ).strip().lower()

        if ans.startswith("r"):
            remaining = [t for t in trials if t["trial_index"] > completed]
            print(f"✓ 将从第 {completed + 1} 条开始继续，共剩 {len(remaining)} 条 / Resuming from trial {completed + 1}, {len(remaining)} remaining")
            return remaining

        if ans.startswith("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = csv_path.parent / f"{csv_path.stem}_bak_{ts}{csv_path.suffix}"
            try:
                csv_path.rename(bak)
                print(f"✓ 已备份旧文件到 / Backed up old file to {bak.name}")
                return trials
            except IOError as e:
                print(f"✗ 备份失败 / Backup failed: {e}")
                continue

        if ans.startswith("c"):
            print("操作已取消 / Operation cancelled")
            exit(0)

        print("⚠ 无效输入，请输入 r、s 或 c / Invalid input, please enter r, s, or c")


def _get_question_text(block_type: str) -> str:
    """
    根据 block 类型返回对应的问题文本。

    Args:
        block_type: 'stretch' or 'bend'

    Returns:
        Question text for the participant
    """
    if block_type == "stretch":
        return "哪一个更不容易被拉长？/ Which one is harder to stretch?"
    else:
        return "哪一个在弯的时候感觉更硬？/ Which one feels stiffer when bending?"


def _collect_response(trial: Dict) -> None:
    """
    收集被试的响应并更新 trial 数据。

    Args:
        trial: Trial dictionary to update with response
    """
    question = _get_question_text(trial["block_type"])

    while True:
        resp = input(f"\n{question}\n  (1) 第一个 / First\n  (2) 第二个 / Second\n你的选择 / Your choice: ").strip()
        if resp in ("1", "2"):
            trial["chosen_object"] = "first" if resp == "1" else "second"
            chosen_id = trial["first_object"] if resp == "1" else trial["second_object"]
            trial["is_correct"] = 1 if chosen_id == trial["target_object"] else 0
            return
        print("⚠ 无效输入，请输入 1 或 2 / Invalid input, please enter 1 or 2")


def _show_progress(current: int, total: int, block_type: str) -> None:
    """
    显示进度条和当前状态。

    Args:
        current: Current trial number
        total: Total number of trials
        block_type: 'stretch' or 'bend'
    """
    progress = current / total
    bar_length = 30
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\n{'='*60}")
    print(f"进度 Progress: [{bar}] {current}/{total} ({progress*100:.1f}%)")
    print(f"Block 类型 Type: {block_type}")
    print(f"{'='*60}")


def run_2afc_experiment_cli() -> None:
    """
    终端驱动单个 2AFC block（stretch 或 bend）：
      - 输入 participant_id
      - 选择只做 stretching 或 bending
      - 自动生成 40 个 trial，逐一运行
      - 每个 trial 结束按 1/2，程序自动算 is_correct 并写入 CSV
      - 中间在 20/40 处提示休息
    """
    # 1) 获取 participant_id
    participant_id = _get_participant_id()

    # 2) 选择 block 类型
    block_type = _get_block_type()

    # 3) 准备结果目录与文件路径
    results_dir = Path("behaviour_results")
    results_dir.mkdir(exist_ok=True)
    order_path = results_dir / f"participant_{participant_id}_{block_type}_order.json"
    csv_path = results_dir / f"participant_{participant_id}_{block_type}_behaviour.csv"

    # 4) 加载或创建试次列表
    trials = _load_or_create_trials(order_path, participant_id, block_type, reps_per_pair=4)

    # 统一编号
    for idx, t in enumerate(trials, start=1):
        t["trial_index"] = idx

    total_trials = len(trials)
    print(f"\n本次将进行 {total_trials} 个 trial / Will conduct {total_trials} trials (Block: {block_type})")
    print(f"数据将保存到 / Data will be saved to: {csv_path.resolve()}\n")

    # 5) 处理已有数据
    trials = _handle_existing_data(csv_path, trials)

    if not trials:
        print("\n✓ 所有试次已完成！/ All trials completed!")
        return

    # 6) 主循环
    print("\n开始实验... / Starting experiment...")
    input("准备好后按 Enter 开始... / Press Enter when ready to start...")

    for t in trials:
        i = t["trial_index"]

        # 中间休息提示
        if i == total_trials // 2 + 1:
            print(f"\n{'='*60}")
            print(f"🎉 你已经完成一半了！/ You're halfway done! ({total_trials//2}/{total_trials})")
            print(f"{'='*60}")
            input("可以休息一下，准备好后按 Enter 继续... / Take a break, press Enter to continue...")

        # 显示进度
        _show_progress(i, total_trials, t["block_type"])

        print(f"\nTrial {i}/{total_trials}")
        print(f"  对比对象 / Comparing: Object {t['ref_object_id']} vs Object {t['comp_object_id']}")
        print(f"  呈现顺序 / Presentation order: 第一个 First = Object {t['first_object']}, 第二个 Second = Object {t['second_object']}")

        # 占位：将来替换为真实的 haptic 呈现
        input("\n→ 现在让被试接触【第一个物体】/ Let participant touch [FIRST object], press Enter when done...")
        input("→ 现在让被试接触【第二个物体】/ Let participant touch [SECOND object], press Enter when done...")

        # 收集响应
        _collect_response(t)

        # 立即保存数据
        try:
            append_trial_row(csv_path, t)
            print(f"✓ Trial {i} 数据已保存 / Data saved")
        except IOError as e:
            print(f"✗ 警告：保存失败 / Warning: Save failed - {e}")
            retry = input("是否重试保存？/ Retry save? (y/n): ").strip().lower()
            if retry.startswith("y"):
                try:
                    append_trial_row(csv_path, t)
                    print("✓ 重试成功 / Retry successful")
                except IOError:
                    print("✗ 重试失败，数据可能丢失 / Retry failed, data may be lost")

    print(f"\n{'='*60}")
    print("🎉 本 block 实验完成！/ Block experiment completed! 所有行为数据已写入 CSV / All behavioral data saved to CSV.")
    print(f"{'='*60}")
    print(f"数据文件 / Data file: {csv_path.resolve()}")



def build_all_trials(participant_id: str, reps_per_pair: int = 4) -> List[Dict]:
    """
    生成两个 block 的所有 trial，并统一给 trial_index 编号。

    Args:
        participant_id: Participant identifier
        reps_per_pair: Number of repetitions per pair

    Returns:
        List of all trials from both blocks with sequential indices
    """
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
