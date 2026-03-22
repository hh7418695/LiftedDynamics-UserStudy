"""
experiment_runner_gui.py - GUI version of the 2AFC psychophysics experiment.

Usage:
    cd model_dynamics/scripts
    python experiment_runner_gui.py

Architecture:
    The GUI process handles ONLY experiment flow and UI.
    Haptic rendering runs in a long-lived subprocess (render_worker.py).
    Communication uses simple file-based signals (no pipes, no stdin/stdout).
"""

import subprocess
import sys
import tempfile
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

RESULTS_DIR = Path("behaviour_results")
RENDER_WORKER = str(Path(__file__).parent / "render_worker.py")


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
        self.csv_path: Optional[Path] = None
        self.order_path: Optional[Path] = None

        # Subprocess & file-based signals
        self.rendering_process: Optional[subprocess.Popen] = None
        self.signal_dir: Optional[Path] = None
        self.cmd_file: Optional[Path] = None
        self.result_file: Optional[Path] = None
        self.ready_file: Optional[Path] = None

        # UI state
        self.phase = "setup"

        self._build_ui()

    def _build_ui(self):
        setup_frame = ttk.LabelFrame(self.root, text="实验设置 / Experiment Setup", padding=10)
        setup_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(setup_frame, text="被试编号 / Participant ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.entry_pid = ttk.Entry(setup_frame, width=20)
        self.entry_pid.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(setup_frame, text="Block 类型 / Type:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.block_var = tk.StringVar(value="stretch")
        ttk.Radiobutton(setup_frame, text="拉伸 Stretching (ks)", variable=self.block_var, value="stretch").grid(row=1, column=1, sticky="w", padx=5)
        ttk.Radiobutton(setup_frame, text="弯曲 Bending (kb)", variable=self.block_var, value="bend").grid(row=1, column=2, sticky="w", padx=5)

        self.btn_start = ttk.Button(setup_frame, text="开始实验 / Start Experiment", command=self._start_experiment)
        self.btn_start.grid(row=2, column=0, columnspan=3, pady=10)

        trial_frame = ttk.LabelFrame(self.root, text="当前试次 / Current Trial", padding=10)
        trial_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.lbl_progress = ttk.Label(trial_frame, text="进度 Progress: 0/0", font=("Arial", 14, "bold"))
        self.lbl_progress.pack(pady=5)
        self.progress_bar = ttk.Progressbar(trial_frame, mode="determinate", length=400)
        self.progress_bar.pack(pady=5)
        self.lbl_trial_info = ttk.Label(trial_frame, text="等待开始... / Waiting to start...", font=("Arial", 12), justify="left")
        self.lbl_trial_info.pack(pady=10)

        ttk.Label(trial_frame, text="状态日志 / Status Log:").pack(anchor="w", padx=5)
        self.log_text = scrolledtext.ScrolledText(trial_frame, height=10, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        self.btn_next = ttk.Button(control_frame, text="下一步 / Next", command=self._next_step, state="disabled")
        self.btn_next.pack(side="left", padx=5)
        self.btn_skip = ttk.Button(control_frame, text="跳过试次 / Skip Trial", command=self._skip_trial, state="disabled")
        self.btn_skip.pack(side="left", padx=5)
        self.btn_quit = ttk.Button(control_frame, text="退出实验 / Quit", command=self._quit_experiment, state="disabled")
        self.btn_quit.pack(side="left", padx=5)

        self.response_frame = ttk.LabelFrame(control_frame, text="被试响应 / Participant Response", padding=10)
        self.btn_response_1 = ttk.Button(self.response_frame, text="第一个更硬 / First is stiffer",
                                         command=lambda: self._record_response("first"), state="disabled")
        self.btn_response_1.pack(side="left", padx=5)
        self.btn_response_2 = ttk.Button(self.response_frame, text="第二个更硬 / Second is stiffer",
                                         command=lambda: self._record_response("second"), state="disabled")
        self.btn_response_2.pack(side="left", padx=5)

    def _log(self, message: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _start_experiment(self):
        pid_raw = self.entry_pid.get().strip()
        if not pid_raw:
            messagebox.showerror("错误 / Error", "请输入被试编号 / Please enter participant ID")
            return
        self.participant_id = ''.join(ch for ch in pid_raw if ch.isalpha()).upper()
        if not self.participant_id:
            messagebox.showerror("错误 / Error", "被试编号必须包含字母 / ID must contain letters")
            return

        self.block_type = self.block_var.get()

        RESULTS_DIR.mkdir(exist_ok=True)
        self.order_path = RESULTS_DIR / f"participant_{self.participant_id}_{self.block_type}_order.json"
        self.csv_path = RESULTS_DIR / f"participant_{self.participant_id}_{self.block_type}_behaviour.csv"

        self.trials = _load_or_create_trials(self.order_path, self.participant_id, self.block_type, reps_per_pair=4)
        for idx, t in enumerate(self.trials, start=1):
            t["trial_index"] = idx

        total = len(self.trials)
        self._log(f"Loaded {total} trials / 加载了 {total} 个试次")

        self.trials = _handle_existing_data(self.csv_path, self.trials)
        if not self.trials:
            messagebox.showinfo("完成 / Done", "所有试次已完成 / All trials completed")
            return

        # Create signal directory
        self.signal_dir = Path(tempfile.mkdtemp(prefix="haptic_gui_"))
        self.cmd_file = self.signal_dir / "command.txt"
        self.result_file = self.signal_dir / "result.txt"
        self.ready_file = self.signal_dir / "ready.txt"

        # Launch subprocess
        cmd = [
            sys.executable, RENDER_WORKER,
            "--participant_id", self.participant_id,
            "--signal_dir", str(self.signal_dir),
        ]
        self._log("启动渲染子进程... / Launching rendering subprocess...")
        try:
            self.rendering_process = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
        except Exception as e:
            messagebox.showerror("错误 / Error", f"启动子进程失败 / Failed to launch subprocess:\n{e}")
            return

        # Wait for ready signal
        self.root.after(200, self._wait_for_ready)

    def _wait_for_ready(self):
        if self.rendering_process.poll() is not None:
            self._log("子进程异常退出 / Subprocess died unexpectedly")
            messagebox.showerror("错误 / Error", "子进程启动失败 / Subprocess failed to start")
            return

        if self.ready_file.exists():
            self._log("子进程就绪 / Subprocess ready")
            self.btn_start.config(state="disabled")
            self.entry_pid.config(state="disabled")
            self.btn_next.config(state="normal")
            self.btn_skip.config(state="normal")
            self.btn_quit.config(state="normal")
            self.current_trial_idx = 0
            self.phase = "trial_first"
            self._update_trial_display()
            self._log('实验已开始，点击"下一步"开始第一个试次 / Click "Next" to begin first trial.')
        else:
            self.root.after(200, self._wait_for_ready)

    def _update_trial_display(self):
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
        trial = self.trials[self.current_trial_idx]
        object_id = trial["first_object"] if which == "first" else trial["second_object"]

        self._log(f"呈现{which}物体 / Presenting {which} object (物体 Object {object_id})")
        self._log("  请按触笔按钮开始渲染 / Press stylus button to start rendering...")

        self.btn_next.config(state="disabled")
        self.btn_skip.config(state="disabled")

        # Clean up any previous result file
        self.result_file.unlink(missing_ok=True)

        # Write command
        self.cmd_file.write_text(str(object_id))

        # Poll for result
        self.root.after(500, self._check_rendering_done)

    def _check_rendering_done(self):
        if self.rendering_process is None:
            return
        if self.rendering_process.poll() is not None:
            self._log("子进程异常退出 / Subprocess died")
            messagebox.showerror("错误 / Error", "渲染子进程崩溃 / Rendering subprocess crashed")
            self.btn_next.config(state="normal")
            self.btn_skip.config(state="normal")
            return

        if self.result_file.exists():
            result = self.result_file.read_text().strip()
            self.result_file.unlink(missing_ok=True)
            self._log(f"渲染完成 / Rendering complete (result: {result})")
            self._advance_phase()
        else:
            self.root.after(500, self._check_rendering_done)

    def _advance_phase(self):
        if self.phase == "trial_first":
            self.phase = "trial_second"
            self._update_trial_display()
            self.btn_next.config(state="normal")
            self.btn_skip.config(state="normal")
            self._log('准备呈现第二个物体，点击"下一步" / Ready for second object. Click "Next"')
        elif self.phase == "trial_second":
            self.phase = "response"
            self._update_trial_display()
            self._show_response_buttons()

    def _show_response_buttons(self):
        trial = self.trials[self.current_trial_idx]
        question = _get_question_text(trial["block_type"])
        self._log(f"收集响应 / Collecting response: {question}")
        self.btn_next.config(state="disabled")
        self.response_frame.pack(side="right", padx=10)
        self.btn_response_1.config(state="normal")
        self.btn_response_2.config(state="normal")

    def _record_response(self, choice: str):
        trial = self.trials[self.current_trial_idx]
        trial["chosen_object"] = choice
        chosen_id = trial["first_object"] if choice == "first" else trial["second_object"]
        trial["is_correct"] = 1 if chosen_id == trial["target_object"] else 0

        try:
            append_trial_row(self.csv_path, trial)
            self._log(f"试次 Trial {trial['trial_index']} 已保存 / saved")
        except IOError as e:
            self._log(f"保存失败 / Save failed: {e}")
            messagebox.showerror("错误 / Error", f"保存失败 / Save failed:\n{e}")

        self.response_frame.pack_forget()
        self.btn_response_1.config(state="disabled")
        self.btn_response_2.config(state="disabled")

        self.current_trial_idx += 1
        total = len(self.trials)

        if self.current_trial_idx == total // 2:
            messagebox.showinfo("休息 / Break",
                              f"已完成一半！/ Halfway done! ({total//2}/{total})\n"
                              "可以休息一下 / Take a break if needed")

        if self.current_trial_idx >= total:
            self.phase = "done"
            self._log("所有试次已完成！/ All trials completed!")
            messagebox.showinfo("完成 / Done", "实验完成！所有数据已保存 / Experiment complete! All data saved.")
            self._cleanup()
        else:
            self.phase = "trial_first"
            self._update_trial_display()
            self.btn_next.config(state="normal")
            self._log('准备下一个试次，点击"下一步" / Ready for next trial. Click "Next"')

    def _skip_trial(self):
        if messagebox.askyesno("确认 / Confirm", "确定跳过当前试次？/ Skip current trial?"):
            trial = self.trials[self.current_trial_idx]
            trial["notes"] = "skipped_by_PI"
            trial["chosen_object"] = None
            trial["is_correct"] = None
            try:
                append_trial_row(self.csv_path, trial)
                self._log(f"试次 Trial {trial['trial_index']} 已跳过 / skipped")
            except IOError:
                pass
            self.current_trial_idx += 1
            if self.current_trial_idx >= len(self.trials):
                self.phase = "done"
                self._cleanup()
            else:
                self.phase = "trial_first"
                self._update_trial_display()
                self.btn_next.config(state="normal")
                self.btn_skip.config(state="normal")

    def _quit_experiment(self):
        if messagebox.askyesno("确认 / Confirm", "确定退出实验？所有已完成数据将保存 / Quit? All completed data will be saved."):
            self._log("主持人请求退出 / PI requested quit")
            self._cleanup()

    def _cleanup(self):
        if self.rendering_process and self.rendering_process.poll() is None:
            try:
                self.cmd_file.write_text("quit")
                self.rendering_process.wait(timeout=5)
            except Exception:
                self.rendering_process.terminate()
                try:
                    self.rendering_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.rendering_process.kill()
            self.rendering_process = None
            self._log("子进程已终止 / Subprocess terminated")

        if self.signal_dir and self.signal_dir.exists():
            for f in self.signal_dir.iterdir():
                f.unlink(missing_ok=True)
            try:
                self.signal_dir.rmdir()
            except Exception:
                pass

        self.btn_next.config(state="disabled")
        self.btn_skip.config(state="disabled")
        self.btn_quit.config(state="disabled")


def main():
    root = tk.Tk()
    app = ExperimentGUI(root)
    # When user closes the window (X button), ensure subprocess is killed
    root.protocol("WM_DELETE_WINDOW", lambda: (_force_cleanup(app), root.destroy()))
    root.mainloop()


def _force_cleanup(app):
    """Forcefully kill subprocess when GUI window is closed."""
    if app.rendering_process and app.rendering_process.poll() is None:
        app.rendering_process.kill()
        try:
            app.rendering_process.wait(timeout=2)
        except Exception:
            pass
    # Clean up signal files
    if app.signal_dir and app.signal_dir.exists():
        for f in app.signal_dir.iterdir():
            f.unlink(missing_ok=True)
        try:
            app.signal_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
