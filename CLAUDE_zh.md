# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个用于研究触觉交互中提升物体动力学的触觉渲染研究系统。该系统使用 TouchX 触觉设备，基于柔性物体的物理仿真提供实时力反馈。系统架构由两个通过 UDP 通信的主要组件组成：

1. **C++ UDP 客户端** (`udp_client/`) - 使用 OpenHaptics SDK 与 TouchX 触觉设备交互
2. **Python 动力学引擎** (`model_dynamics/`) - 使用 JAX 和拉格朗日神经网络计算物体动力学和力反馈

## 系统架构

### 通信流程

系统在实时循环中运行：
- C++ 客户端以约 1kHz 的频率读取触觉触笔位置并通过 UDP 发送
- Python 服务器接收位置，使用拉格朗日力学计算力
- Python 将力反馈发送回 C++
- C++ 将力渲染到触觉设备
- 用户点击触笔按钮以启动/停止交互

### 坐标系转换

**关键**：TouchX 设备和 Python 仿真使用不同的坐标系。转换由以下函数处理：
- `getCoordinate_TouchX2Python()` - 将设备坐标转换为仿真坐标：(x, y, z) → (x, -z, y)
- `getCoordinate_Python2TouchX()` - 将仿真坐标转换为设备坐标：(x, y, z) → (x, z, -y)

### 物理仿真

系统使用拉格朗日力学，包含：
- **拉伸刚度**：控制轴向变形 (50-1000 × 10³ N/m)
- **弯曲刚度**：控制角度变形 (0-0.1 × 10³ N/m)
- **虚拟耦合**：连接用户与虚拟物体的弹簧-阻尼器 (750 × 10³ N/m 刚度, 25 × 10³ Ns/m 阻尼)
- **阻尼**：通用速度相关阻尼 (0.1 × object_damping_scale)
- **重力**：默认启用

物体建模为 N=5 个由弹簧连接的节点，具有可配置的质量（每个节点 40g）和长度（每段 5cm）。

## 开发命令

### Python 设置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r ./model_dynamics/requirements.txt
```

### 运行系统

**启动顺序很重要**：始终先启动 C++ 客户端，然后启动 Python GUI。

1. **启动 C++ UDP 客户端**：
   - 在 Visual Studio 2019 中打开 `udp_client/udp_client.sln`
   - 构建并运行项目
   - 触觉设备将初始化并将触笔居中

2. **启动 Python GUI**：
   ```bash
   cd model_dynamics/scripts
   python experiment_runner_gui.py
   ```

3. **交互**：
   - GUI 管理实验流程（参与者 ID、block 类型、试次进度）
   - 点击触笔按钮一次以开始渲染
   - 摇晃/移动触笔以与虚拟物体交互
   - 再次点击按钮以停止
   - GUI 在每对试次后收集 2AFC 响应

## 代码组织

### Python 结构 (`model_dynamics/scripts/`)

- `experiment_runner_gui.py` - **主入口**（Tkinter GUI，2AFC 实验界面）
- `render_worker.py` - 触觉渲染子进程（由 GUI 调用）
- `psychophysics_loop.py` - 试次生成、数据持久化、2AFC 逻辑
- `src/lnn.py` - 拉格朗日神经网络实现，加速度计算
- `src/md.py` - 分子动力学工具，状态预测
- `src/nve.py` - NVE 系综积分器（微正则）
- `src/utils.py` - 卡尔曼滤波器，轨迹绘图，辅助函数
- `src/models.py` - 神经网络模型和前向传播
- `src/io.py` - I/O 工具

### C++ 结构 (`udp_client/`)

- `udp_client/main.cpp` - 触觉设备接口和 UDP 客户端
- `master_interface()` 回调 - 以 1kHz 运行，读取位置，渲染力
- 按钮点击检测管理状态转换（空闲 → 渲染 → 停止）

### 关键 Python 函数

`render_worker.py` 中：
- `render_single_object()` - 执行物理仿真和 UDP 触觉渲染循环

`psychophysics_loop.py` 中：
- `build_block_trials()` - 生成随机试次对
- `get_stiffness_for_object()` - 将物体 ID 映射到刚度参数
- `append_trial_row()` - 将试次响应数据保存到 CSV

## 数据输出

结果保存到：`model_dynamics/results/Participant-{id}/Obj-{id}/{timestamp}/`

生成的文件：
- `render_hist.csv` - 力、用户位置、物体节点位置的时间序列
- `render.gif` - 动画轨迹（如果 `execute_TrajPlot=True`）
- `render_execution_time.png` - 性能分析（如果 `execute_SpeedAnalysis=True`）
- `test_basic.gif` - 动画检查输出
- `test_RF.png` - 共振频率响应曲线

## 重要实现说明

### JAX 配置
- 启用 64 位精度：`jax.config.update("jax_enable_x64", True)`
- 所有物理计算使用双精度以确保准确性

### 卡尔曼滤波
- 默认启用（`enable_KalmanFilter=True`）
- 从噪声位置测量中估计速度
- 使用具有过程/测量噪声调整的 3D 卡尔曼滤波器

### 力限制
- 每个轴的最大力限制为 2.5 N 以防止设备饱和
- 如果幅度超过限制，力按比例缩放
- 对设备安全和稳定触觉至关重要

### 性能考虑
- 目标更新率：约 1000 Hz（每次迭代 1 ms）
- UDP 通信是阻塞的 - Python 在计算前等待 C++ 位置
- JIT 编译在第一次调用时发生 - 预期初始迭代较慢
- 使用 `stride` 参数平衡准确性与计算时间

### 预热阶段
- C++ 在实际数据之前发送 5 条零位置的预热消息
- 允许 JIT 编译在实时渲染前完成
- 预热后用户的第一个位置成为原点

## 依赖项

### Python（参见 requirements.txt）
- JAX 0.4.23 - 自动微分和 JIT 编译
- jax-md 0.2.8 - 分子动力学工具
- jraph 0.0.6.dev0 - 图神经网络
- numpy, scipy, pandas, scikit-learn - 科学计算
- matplotlib - 可视化
- fire - CLI 参数解析

### C++（仅限 Windows）
- OpenHaptics SDK 3.5 - TouchX 设备驱动程序和 API
- Visual Studio 2019 - 构建工具链
- Winsock2 - UDP 网络

## 研究背景

该项目基于拉格朗日图神经网络（LGNN）进行动力学学习。该系统设计用于心理物理学实验，研究：
- 人类如何通过触觉反馈感知物体柔软度
- 弯曲刚度与拉伸刚度在柔软度感知中的作用
- 触觉错觉及其与 VR 中视觉线索的交互

物体参数（刚度、质量、长度）可以变化，为用户研究创造不同的感知体验。
