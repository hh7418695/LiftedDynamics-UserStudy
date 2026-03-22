"""
experiment_runner_gui.py - GUI version of the 2AFC psychophysics experiment.

Usage:
    cd model_dynamics/scripts
    python experiment_runner_gui.py

Requirements:
    - tkinter (included in standard Python)
    - mock_cpp.py running in another terminal (or real C++ client)
"""

import socket
import sys
import threading
import queue
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

sys.path.insert(0, str(Path(__file__).parent))

from psychophysics_loop import (
    build_block_trials,
    append_trial_row,
    get_stiffness_for_object,
    _load_or_create_trials,
    _handle_existing_data,
    _get_question_text,
)

# ── UDP config ────────────────────────────────────────────────────────────────
SERVER_IP = "0.0.0.0"
SERVER_PORT = 12312
BUFFER_SIZE = 512

RESULTS_DIR = Path("behaviour_results")


# ══════════════════════════════════════════════════════════════════════════════
# GUI Application
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("触觉渲染实验 / Haptic Rendering Experiment")
        self.root.geometry("900x700")

        # State
        self.participant_id: Optional[str] = None
        self.block_type: Optional[str] = None
        self.trials: list = []
        self.current_trial_idx: int = 0
        self.sock: Optional[socket.socket] = None
        self.csv_path: Optional[Path] = None
        self.order_path: Optional[Path] = None

        # Threading
        self.rendering_thread: Optional[threading.Thread] = None
        self.stop_rendering = threading.Event()
        self.rendering_result_queue = queue.Queue()

        # UI state
        self.phase = "setup"  # setup | trial_first | trial_second | response | done

        self._build_ui()

    def _build_ui(self):
        # ── Top frame: Setup ──────────────────────────────────────────────────
        setup_frame = ttk.LabelFrame(self.root, text="实验设置 / Experiment Setup", padding=10)
        setup_frame.pack(fill="x", padx=10, pady=5)

        # Participant ID
        ttk.Label(setup_frame, text="被试编号 / Participant ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.entry_pid = ttk.Entry(setup_frame, width=20)
        self.entry_pid.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Block type
        ttk.Label(setup_frame, text="Block 类型 / Type:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.block_var = tk.StringVar(value="stretch")
        ttk.Radiobutton(setup_frame, text="拉伸 Stretching (ks)", variable=self.block_var, value="stretch").grid(row=1, column=1, sticky="w", padx=5)
        ttk.Radiobutton(setup_frame, text="弯曲 Bending (kb)", variable=self.block_var, value="bend").grid(row=1, column=2, sticky="w", padx=5)

        # Start button
        self.btn_start = ttk.Button(setup_frame, text="开始实验 / Start Experiment", command=self._start_experiment)
        self.btn_start.grid(row=2, column=0, columnspan=3, pady=10)

        # ── Middle frame: Trial info ──────────────────────────────────────────
        trial_frame = ttk.LabelFrame(self.root, text="当前试次 / Current Trial", padding=10)
        trial_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Progress
        self.lbl_progress = ttk.Label(trial_frame, text="进度 Progress: 0/0", font=("Arial", 14, "bold"))
        self.lbl_progress.pack(pady=5)

        self.progress_bar = ttk.Progressbar(trial_frame, mode="determinate", length=400)
        self.progress_bar.pack(pady=5)

        # Trial details
        self.lbl_trial_info = ttk.Label(trial_frame, text="等待开始... / Waiting to start...", font=("Arial", 12), justify="left")
        self.lbl_trial_info.pack(pady=10)

        # Status log
        ttk.Label(trial_frame, text="状态日志 / Status Log:").pack(anchor="w", padx=5)
        self.log_text = scrolledtext.ScrolledText(trial_frame, height=10, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # ── Bottom frame: Controls ────────────────────────────────────────────
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.btn_next = ttk.Button(control_frame, text="下一步 / Next", command=self._next_step, state="disabled")
        self.btn_next.pack(side="left", padx=5)

        self.btn_skip = ttk.Button(control_frame, text="跳过试次 / Skip Trial", command=self._skip_trial, state="disabled")
        self.btn_skip.pack(side="left", padx=5)

        self.btn_quit = ttk.Button(control_frame, text="退出实验 / Quit", command=self._quit_experiment, state="disabled")
        self.btn_quit.pack(side="left", padx=5)

        # Response buttons (hidden initially)
        self.response_frame = ttk.LabelFrame(control_frame, text="被试响应 / Participant Response", padding=10)
        self.btn_response_1 = ttk.Button(self.response_frame, text="第一个更硬 / First is stiffer",
                                         command=lambda: self._record_response("first"), state="disabled")
        self.btn_response_1.pack(side="left", padx=5)
        self.btn_response_2 = ttk.Button(self.response_frame, text="第二个更硬 / Second is stiffer",
                                         command=lambda: self._record_response("second"), state="disabled")
        self.btn_response_2.pack(side="left", padx=5)

    def _log(self, message: str):
        """Append message to log."""
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _start_experiment(self):
        """Initialize experiment with participant ID and block type."""
        pid_raw = self.entry_pid.get().strip()
        if not pid_raw:
            messagebox.showerror("错误 / Error", "请输入被试编号 / Please enter participant ID")
            return

        self.participant_id = ''.join(ch for ch in pid_raw if ch.isalpha()).upper()
        if not self.participant_id:
            messagebox.showerror("错误 / Error", "被试编号必须包含字母 / ID must contain letters")
            return

        self.block_type = self.block_var.get()

        # Prepare paths
        RESULTS_DIR.mkdir(exist_ok=True)
        self.order_path = RESULTS_DIR / f"participant_{self.participant_id}_{self.block_type}_order.json"
        self.csv_path = RESULTS_DIR / f"participant_{self.participant_id}_{self.block_type}_behaviour.csv"

        # Load or create trials
        self.trials = _load_or_create_trials(self.order_path, self.participant_id, self.block_type, reps_per_pair=4)
        for idx, t in enumerate(self.trials, start=1):
            t["trial_index"] = idx

        total = len(self.trials)
        self._log(f"✓ 加载了 {total} 个试次 / Loaded {total} trials")

        # Handle existing data
        self.trials = _handle_existing_data(self.csv_path, self.trials)
        if not self.trials:
            messagebox.showinfo("完成 / Done", "所有试次已完成 / All trials completed")
            return

        # Open UDP socket
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((SERVER_IP, SERVER_PORT))
            self._log(f"✓ UDP socket 绑定到 / bound to {SERVER_IP}:{SERVER_PORT}")
        except Exception as e:
            messagebox.showerror("错误 / Error", f"无法绑定 UDP socket / Cannot bind socket:\n{e}")
            return

        # Update UI
        self.btn_start.config(state="disabled")
        self.entry_pid.config(state="disabled")
        self.btn_next.config(state="normal")
        self.btn_skip.config(state="normal")
        self.btn_quit.config(state="normal")

        self.current_trial_idx = 0
        self.phase = "trial_first"
        self._update_trial_display()
        self._log('实验已开始 / Experiment started. 点击"下一步"开始第一个试次 / Click "Next" to begin first trial.')

    def _update_trial_display(self):
        """Update progress bar and trial info."""
        if not self.trials:
            return

        total = len(self.trials)
        current = self.current_trial_idx + 1

        self.progress_bar["maximum"] = total
        self.progress_bar["value"] = self.current_trial_idx
        self.lbl_progress.config(text=f"进度 Progress: {current}/{total} ({current/total*100:.1f}%)")

        if self.current_trial_idx < len(self.trials):
            trial = self.trials[self.current_trial_idx]
            info = (
                f"试次 Trial {trial['trial_index']}/{total}\n"
                f"对比 Comparing: 物体 Object {trial['ref_object_id']} vs 物体 Object {trial['comp_object_id']}\n"
                f"顺序 Order: 第一个 FIRST = 物体 Object {trial['first_object']} | 第二个 SECOND = 物体 Object {trial['second_object']}\n"
                f"当前阶段 Phase: {self.phase}"
            )
            self.lbl_trial_info.config(text=info)

    def _next_step(self):
        """Advance to next step in trial."""
        if self.phase == "trial_first":
            self._present_object("first")
        elif self.phase == "trial_second":
            self._present_object("second")
        elif self.phase == "response":
            self._show_response_buttons()
        elif self.phase == "done":
            messagebox.showinfo("完成 / Done", "所有试次已完成！/ All trials completed!")
            self._cleanup()

    def _present_object(self, which: str):
        """Start haptic rendering for first or second object."""
        trial = self.trials[self.current_trial_idx]
        object_id = trial["first_object"] if which == "first" else trial["second_object"]

        self._log(f"→ 呈现{which}物体 / Presenting {which} object (物体 Object {object_id})")
        self._log("  等待触笔按钮按下... / Waiting for stylus button press...")

        self.btn_next.config(state="disabled")
        self.btn_skip.config(state="disabled")

        # Start rendering in background thread
        self.stop_rendering.clear()
        self.rendering_thread = threading.Thread(
            target=self._run_haptic_rendering,
            args=(object_id,),
            daemon=True
        )
        self.rendering_thread.start()

        # Poll for completion
        self.root.after(100, self._check_rendering_done)

    def _run_haptic_rendering(self, object_id: int):
        """Run haptic rendering in background thread."""
        try:
            import main as _main_module
            result = _main_module.run_single_object(
                sock=self.sock,
                object_id=object_id,
                participant_id=self.participant_id,
                pi_input_fn=None,  # GUI doesn't use keyboard input
            )
            self.rendering_result_queue.put(("ok", result))
        except Exception as e:
            self.rendering_result_queue.put(("error", str(e)))

    def _check_rendering_done(self):
        """Poll rendering thread for completion."""
        try:
            status, result = self.rendering_result_queue.get_nowait()
            if status == "error":
                self._log(f"✗ 渲染错误 / Rendering error: {result}")
                messagebox.showerror("错误 / Error", f"渲染失败 / Rendering failed:\n{result}")
                self.btn_next.config(state="normal")
                self.btn_skip.config(state="normal")
            else:
                self._log(f"✓ 渲染完成 / Rendering complete")
                self._advance_phase()
        except queue.Empty:
            # Still running, check again
            self.root.after(100, self._check_rendering_done)

    def _advance_phase(self):
        """Move to next phase after rendering."""
        if self.phase == "trial_first":
            self.phase = "trial_second"
            self._update_trial_display()
            self.btn_next.config(state="normal")
            self.btn_skip.config(state="normal")
            self._log('准备呈现第二个物体 / Ready for second object. 点击"下一步" / Click "Next"')
        elif self.phase == "trial_second":
            self.phase = "response"
            self._update_trial_display()
            self._show_response_buttons()

    def _show_response_buttons(self):
        """Show response buttons for participant."""
        trial = self.trials[self.current_trial_idx]
        question = _get_question_text(trial["block_type"])

        self._log(f"→ 收集响应 / Collecting response: {question}")

        self.btn_next.config(state="disabled")
        self.response_frame.pack(side="right", padx=10)
        self.btn_response_1.config(state="normal")
        self.btn_response_2.config(state="normal")

    def _record_response(self, choice: str):
        """Record participant's response and save."""
        trial = self.trials[self.current_trial_idx]
        trial["chosen_object"] = choice

        chosen_id = trial["first_object"] if choice == "first" else trial["second_object"]
        trial["is_correct"] = 1 if chosen_id == trial["target_object"] else 0

        # Save immediately
        try:
            append_trial_row(self.csv_path, trial)
            self._log(f"✓ 试次 Trial {trial['trial_index']} 已保存 / saved")
        except IOError as e:
            self._log(f"✗ 保存失败 / Save failed: {e}")
            messagebox.showerror("错误 / Error", f"保存失败 / Save failed:\n{e}")

        # Hide response buttons
        self.response_frame.pack_forget()
        self.btn_response_1.config(state="disabled")
        self.btn_response_2.config(state="disabled")

        # Move to next trial
        self.current_trial_idx += 1

        # Check if halfway
        total = len(self.trials)
        if self.current_trial_idx == total // 2:
            messagebox.showinfo("休息 / Break",
                              f"已完成一半！/ Halfway done! ({total//2}/{total})\n"
                              "可以休息一下 / Take a break")

        if self.current_trial_idx >= total:
            self.phase = "done"
            self._log("🎉 所有试次已完成！/ All trials completed!")
            messagebox.showinfo("完成 / Done", "实验完成！所有数据已保存 / Experiment complete! All data saved.")
            self._cleanup()
        else:
            self.phase = "trial_first"
            self._update_trial_display()
            self.btn_next.config(state="normal")
            self._log(f'准备下一个试次 / Ready for next trial. 点击"下一步" / Click "Next"')

    def _skip_trial(self):
        """Skip current trial."""
        if messagebox.askyesno("确认 / Confirm", "确定跳过当前试次？/ Skip current trial?"):
            trial = self.trials[self.current_trial_idx]
            trial["notes"] = "skipped_by_PI"
            trial["chosen_object"] = None
            trial["is_correct"] = None

            try:
                append_trial_row(self.csv_path, trial)
                self._log(f"⚠ 试次 Trial {trial['trial_index']} 已跳过 / skipped")
            except IOError:
                pass

            self.current_trial_idx += 1
            if self.current_trial_idx >= len(self.trials):
                self.phase = "done"
                self._cleanup()
            else:
                self.phase = "trial_first"
                self._update_trial_display()

    def _quit_experiment(self):
        """Quit experiment early."""
        if messagebox.askyesno("确认 / Confirm", "确定退出实验？所有已完成数据将保存 / Quit experiment? All completed data will be saved."):
            self._log("主持人请求退出 / PI requested quit")
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        if self.sock:
            self.sock.close()
            self._log("UDP socket 已关闭 / closed")
        self.btn_next.config(state="disabled")
        self.btn_skip.config(state="disabled")
        self.btn_quit.config(state="disabled")


def main():
    root = tk.Tk()
    app = ExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
