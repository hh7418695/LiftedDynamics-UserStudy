"""
experiment_runner.py - Integrated 2AFC psychophysics experiment runner.

Combines psychophysics_loop.py (trial management) with main.py (haptic rendering).
Shares a single UDP socket across all trials.

Usage:
    cd model_dynamics/scripts
    python experiment_runner.py

Terminal layout (two windows recommended):
    Window 1: python mock_cpp.py      # simulates C++ haptic device
    Window 2: python experiment_runner.py

Control hierarchy:
    Participant : stylus button (b in mock_cpp) -> start/stop rendering
    Experimenter: Enter -> advance trial steps
    You (PI)    : s -> skip current trial | q -> save and quit
"""

import socket
import sys
import select
from pathlib import Path

# Allow imports from this directory
sys.path.insert(0, str(Path(__file__).parent))

from psychophysics_loop import (
    build_block_trials,
    append_trial_row,
    _get_participant_id,
    _get_block_type,
    _load_or_create_trials,
    _handle_existing_data,
    _get_question_text,
    _collect_response,
    _show_progress,
)

# ── UDP config ────────────────────────────────────────────────────────────────
SERVER_IP = "0.0.0.0"
SERVER_PORT = 12312
BUFFER_SIZE = 512

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("behaviour_results")


# ── PI emergency controls ─────────────────────────────────────────────────────

def _check_pi_input() -> str:
    """
    Non-blocking check for PI keyboard input.
    Returns 's' (skip), 'q' (quit), or '' (nothing pressed).
    Only works on Unix; on Windows falls back to no-op.
    """
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip().lower()
    except Exception:
        pass
    return ""


# ── Haptic rendering for one object ──────────────────────────────────────────

def run_haptic_trial(object_id: int, sock: socket.socket, participant_id: str) -> str:
    """
    Run haptic rendering for one object presentation.

    Imports and calls main.py's rendering logic with the shared socket.
    Returns 'ok', 'skip', or 'quit' based on PI input during rendering.

    The rendering stops when:
      - C++ sends timestamp < 0 (stylus button stop signal), OR
      - PI presses s/q
    """
    # Lazy import to avoid circular deps and keep startup fast
    import main as _main_module

    print(f"\n  [触觉渲染 / Haptic] 物体 Object {object_id} — 等待触笔按钮按下开始 / Waiting for stylus button press to start...")
    print(  "  [主持人控制 / PI controls: s=跳过试次 skip trial | q=退出实验 quit experiment]")

    # run_haptic_object is a thin wrapper we add below that accepts sock + object_id
    result = _main_module.run_single_object(
        sock=sock,
        object_id=object_id,
        participant_id=participant_id,
        pi_input_fn=_check_pi_input,
    )
    return result  # 'ok' | 'skip' | 'quit'


# ── Experimenter gate ─────────────────────────────────────────────────────────

def experimenter_advance(prompt: str) -> str:
    """
    Block until experimenter presses Enter.
    PI can type s/q here too.
    Returns 'ok', 'skip', or 'quit'.
    """
    print(f"\n  [实验者 / Experimenter] {prompt}")
    print(  "  (按 Enter 继续 / Press Enter to continue | s = 跳过 skip | q = 退出 quit)")
    val = input("  > ").strip().lower()
    if val == "q":
        return "quit"
    if val == "s":
        return "skip"
    return "ok"


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(trial: dict, sock: socket.socket, participant_id: str,
              csv_path: Path, total: int) -> str:
    """
    Run one 2AFC trial: present two objects, collect response, save data.
    Returns 'ok', 'skip', or 'quit'.
    """
    i = trial["trial_index"]
    _show_progress(i, total, trial["block_type"])
    print(f"\n  试次 Trial {i}/{total}")
    print(f"  对比 Comparing: 物体 Object {trial['ref_object_id']} vs 物体 Object {trial['comp_object_id']}")
    print(f"  顺序 Order: 第一个 FIRST = 物体 Object {trial['first_object']} | 第二个 SECOND = 物体 Object {trial['second_object']}")

    # ── Present first object ──────────────────────────────────────────────────
    action = experimenter_advance(f"准备呈现第一个物体 / Ready to present FIRST object (物体 Object {trial['first_object']})?")
    if action != "ok":
        return action

    result = run_haptic_trial(trial["first_object"], sock, participant_id)
    if result != "ok":
        return result

    # ── Present second object ─────────────────────────────────────────────────
    action = experimenter_advance(f"准备呈现第二个物体 / Ready to present SECOND object (物体 Object {trial['second_object']})?")
    if action != "ok":
        return action

    result = run_haptic_trial(trial["second_object"], sock, participant_id)
    if result != "ok":
        return result

    # ── Collect response ──────────────────────────────────────────────────────
    action = experimenter_advance("两个物体已呈现，准备收集响应 / Both objects presented. Ready to collect response?")
    if action != "ok":
        return action

    _collect_response(trial)

    # ── Save immediately ──────────────────────────────────────────────────────
    try:
        append_trial_row(csv_path, trial)
        print(f"  ✓ 试次 Trial {i} 已保存 / saved.")
    except IOError as e:
        print(f"  ✗ 保存失败 / Save failed: {e}")

    return "ok"


# ── Main experiment loop ──────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Participant setup ─────────────────────────────────────────────────────
    participant_id = _get_participant_id()
    block_type = _get_block_type()

    order_path = RESULTS_DIR / f"participant_{participant_id}_{block_type}_order.json"
    csv_path   = RESULTS_DIR / f"participant_{participant_id}_{block_type}_behaviour.csv"

    trials = _load_or_create_trials(order_path, participant_id, block_type, reps_per_pair=4)
    for idx, t in enumerate(trials, start=1):
        t["trial_index"] = idx

    total_trials = len(trials)
    print(f"\n  本次将进行 {total_trials} 个试次 / {total_trials} trials to run (block: {block_type})")
    print(f"  数据保存至 / Data → {csv_path.resolve()}\n")

    trials = _handle_existing_data(csv_path, trials)
    if not trials:
        print("✓ 所有试次已完成 / All trials already completed.")
        return

    # ── Open shared UDP socket ────────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((SERVER_IP, SERVER_PORT))
    print(f"  UDP socket 已绑定 / bound to {SERVER_IP}:{SERVER_PORT}")

    input("\n  准备好后按 Enter 开始实验 / Press Enter when ready to start the experiment...")

    # ── Trial loop ────────────────────────────────────────────────────────────
    try:
        for trial in trials:
            i = trial["trial_index"]

            # Mid-block rest
            if i == total_trials // 2 + 1:
                print(f"\n{'='*60}")
                print(f"  已完成一半！/ Halfway! ({total_trials//2}/{total_trials})")
                print(f"{'='*60}")
                input("  可以休息一下，准备好后按 Enter 继续 / Take a break. Press Enter to continue...")

            result = run_trial(trial, sock, participant_id, csv_path, total_trials)

            if result == "skip":
                trial["notes"] = "skipped_by_PI"
                try:
                    append_trial_row(csv_path, trial)
                except IOError:
                    pass
                print(f"  ⚠ 试次 Trial {i} 已跳过 / skipped.")
                continue

            if result == "quit":
                print("\n  主持人请求退出，所有已完成数据已保存 / PI requested quit. All completed data saved.")
                break

        else:
            print(f"\n{'='*60}")
            print("  实验完成！所有数据已保存 / Experiment complete! All data saved.")
            print(f"{'='*60}")

    finally:
        sock.close()
        print("  UDP socket 已关闭 / closed.")


if __name__ == "__main__":
    main()
