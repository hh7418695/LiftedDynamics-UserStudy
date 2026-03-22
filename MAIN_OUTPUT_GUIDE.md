# Main.py 输出指南 / Main.py Output Guide

本文档详细说明 `model_dynamics/scripts/main.py` 的输出内容、存储位置及其意义。

---

## 📊 终端输出内容 / Terminal Output

### 1. 启动信息 / Startup Information

```
******************************
Running main
******************************

>> Args
>> KwArgs
seed=42
dt=0.0001
stride=10
dim=3
socket_serverIP=127.0.0.1
socket_serverPort=12312
socket_bufferSize=512
participant_id=1
```

### 2. Trial 设置提示 / Trial Settings Prompt

```
******************************
Starting a new trial
******************************

>> Input trial settings

Format (separated by space):
1) Object ID (default 1)
2) If execute haptic rendering (default 1)
3) If pre-check object animation before rendering (default 0)
4) If pre-test object's resonance frequency before rendering (default 0)
(Example input: 5 1 1 0)

Enter here: [等待用户输入 / Waiting for user input]
```

**输入示例 / Input Examples:**
- `5 1 0 0` - Object 5, 触觉渲染, 无预检查
- `3 0 1 0` - Object 3, 仅动画检查
- `7 0 0 1` - Object 7, 仅共振频率测试

### 3. 物体参数信息 / Object Parameters

```
>> Setting object parameters

N = 5
species = [0 0 0 0 0]
masses (g) = [40. 40. 40. 40. 40.] -> total weight (g) = 200.0
lengths (m) = [0.05 0.05 0.05 0.05] -> total length (m) = 0.2
stretching stiffness (10^3 N/m) = [50. 50. 50. 50. 50.]
bending stiffness (10^3 N/m) = [0.05 0.05 0.05 0.05 0.05]
general damping (10^3 Ns/m) = [1. 1. 1. 1. 1.]
virtual coupling stiffness (10^3 N/m) = 750.0
virtual coupling damping (10^3 N/m) = 25.0
```

**参数说明 / Parameter Explanation:**
- **N**: 节点数量 (5个节点)
- **masses**: 每个节点质量 40g，总重 200g
- **lengths**: 每段长度 5cm，总长 20cm
- **stretching stiffness**: 拉伸刚度 (Object 1-5 变化)
- **bending stiffness**: 弯曲刚度 (Object 6-10 变化)
- **virtual coupling**: 虚拟耦合参数 (连接用户和虚拟物体)

### 4. 触觉渲染提示 / Haptic Rendering Prompt

```
>> Running haptic rendering

Please click the button on the haptic stylus and hold still until you can feel the weight
Click the button again to stop the process
```

### 5. 完成信息 / Completion Information

```
Rendering process stopped, hist_time.shape = (1234,)
Render history data saved!
```

---

## 💾 数据存储位置 / Data Storage Location

### 目录结构 / Directory Structure

```
model_dynamics/results/
└── Participant-{participant_id}/
    └── Obj-{object_id}/
        └── {timestamp}/
            ├── render_hist.csv              # 主要数据文件 / Main data file
            ├── render.gif                   # 轨迹动画 / Trajectory animation (optional)
            ├── render_execution_time.png    # 性能分析 / Performance analysis (optional)
            ├── test_basic.gif               # 动画检查 / Animation check (optional)
            └── test_RF.png                  # 共振频率测试 / Resonance frequency test (optional)
```

### 路径示例 / Path Example

```
model_dynamics/results/
└── Participant-1/
    └── Obj-5/
        └── 20260207_143052/
            ├── render_hist.csv
            ├── render.gif
            └── render_execution_time.png
```

**路径组成 / Path Components:**
- `Participant-{participant_id}`: 被试编号
- `Obj-{object_id}`: 物体编号 (1-10)
- `{timestamp}`: 时间戳 (YYYYMMDD_HHMMSS)

---

## 📁 输出文件详解 / Output Files Explained

### 1. render_hist.csv ⭐⭐⭐⭐⭐

**最重要的数据文件 / Most Important Data File**

#### 内容 / Content
触觉渲染过程中的所有时间序列数据 / All time-series data during haptic rendering

#### 列结构 / Column Structure
```csv
Timestamp (s),
Rendered Force X (N), Rendered Force Y (N), Rendered Force Z (N),
User Position X (m), User Position Y (m), User Position Z (m),
Node 0 Position X (m), Node 0 Position Y (m), Node 0 Position Z (m),
Node 1 Position X (m), Node 1 Position Y (m), Node 1 Position Z (m),
Node 2 Position X (m), Node 2 Position Y (m), Node 2 Position Z (m),
Node 3 Position X (m), Node 3 Position Y (m), Node 3 Position Z (m),
Node 4 Position X (m), Node 4 Position Y (m), Node 4 Position Z (m)
```

**总共 22 列 / Total 22 Columns:**
- 1 列时间戳 / 1 timestamp column
- 3 列力数据 / 3 force columns
- 3 列用户位置 / 3 user position columns
- 15 列物体节点位置 / 15 object node position columns (5 nodes × 3 axes)

#### 采样率 / Sampling Rate
- **200 Hz** (每 5 个通信周期保存一次 / saves every 5 communication cycles)
- 最多 20,000 行 / Maximum 20,000 rows (约 50 秒数据 / ~50 seconds of data)

#### 用处 / Purpose
1. **用户行为分析 / User Behavior Analysis**: 通过 User Position 分析被试操作模式
2. **力反馈验证 / Force Feedback Verification**: 检查系统给用户的力是否合理
3. **物体动力学 / Object Dynamics**: 5个节点的运动反映物体特性
4. **论文数据 / Research Data**: 发表论文的原始数据来源

#### 典型分析 / Typical Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据 / Read data
data = pd.read_csv('render_hist.csv')

# 分析力的时间序列 / Analyze force time series
plt.plot(data['Timestamp (s)'], data['Rendered Force X (N)'])
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force Profile During Interaction')

# 分析用户运动轨迹 / Analyze user trajectory
plt.plot(data['User Position X (m)'], data['User Position Y (m)'])
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('User Movement Trajectory')

# 分析物体变形 / Analyze object deformation
node0_x = data['Node 0 Position X (m)']
node4_x = data['Node 4 Position X (m)']
deformation = node4_x - node0_x
plt.plot(data['Timestamp (s)'], deformation)
plt.title('Object Deformation Over Time')
```

---

### 2. render.gif ⭐⭐⭐⭐

**轨迹可视化 / Trajectory Visualization**

#### 内容 / Content
3D 动画显示物体节点的运动轨迹 / 3D animation showing object node trajectories

#### 生成条件 / Generation Condition
需要修改代码 / Requires code modification:
```python
# main.py:1084
execute_hapticRendering(
    execute_TrajPlot=True  # 设置为 True / Set to True
)
```

#### 包含内容 / Includes
- 5 个节点的 3D 位置轨迹 / 5 nodes' 3D position trajectories
- 用户位置（虚拟耦合点）/ User position (virtual coupling point)
- 力向量可视化 / Force vector visualization

#### 用处 / Purpose
1. **快速检查 / Quick Check**: 直观看到物体运动是否合理
2. **异常检测 / Anomaly Detection**: 发现物理模拟问题（穿透、爆炸等）
3. **演示材料 / Presentation Material**: 用于论文、报告、演讲

#### 诊断示例 / Diagnostic Examples
- 节点突然跳跃 → 数值不稳定 / Sudden jumps → Numerical instability
- 过度振荡 → 阻尼太小 / Excessive oscillation → Damping too low
- 响应迟钝 → 刚度太大 / Sluggish response → Stiffness too high

---

### 3. render_execution_time.png ⭐⭐⭐

**性能分析图 / Performance Analysis**

#### 内容 / Content
4 个子图显示执行时间 / 4 subplots showing execution times

#### 子图 / Subplots
1. **Communication Time**: UDP 通信时间
2. **Force Calculation Time**: 力计算时间
3. **Dynamic Calculation Time**: 动力学计算时间
4. **Overall Rendering Time**: 总渲染时间

#### 生成条件 / Generation Condition
```python
# main.py:1084
execute_hapticRendering(
    execute_SpeedAnalysis=True  # 设置为 True / Set to True
)
```

#### 性能指标 / Performance Metrics

**理想性能 / Ideal Performance:**
```
Communication Time:      0.2-0.5 ms
Force Calculation:       0.05-0.1 ms
Dynamic Calculation:     0.3-0.5 ms
Overall Rendering Time:  < 1 ms (1000 Hz)
```

**如果超过这些值 / If exceeding these values:**
- 需要优化代码 / Need code optimization
- 或降低复杂度 / Or reduce complexity
- 或使用更快的硬件 / Or use faster hardware

#### 用处 / Purpose
1. **性能优化 / Performance Optimization**: 找出计算瓶颈
2. **实时性验证 / Real-time Verification**: 确保达到 1000 Hz
3. **故障排查 / Troubleshooting**: 发现性能突然下降的时刻

---

### 4. test_basic.gif ⭐⭐⭐

**动画预检查 / Animation Pre-check**

#### 内容 / Content
物体在正弦波激励下的运动动画 / Object motion under sinusoidal excitation

#### 生成条件 / Generation Condition
```
Trial 设置输入 / Trial settings input: X 0 1 0
例如 / Example: 5 0 1 0
```

#### 激励参数 / Excitation Parameters
- **频率 / Frequency**: 随机 (0.5-1.5 × 基准周期 / base period)
- **幅度 / Amplitude**: 
  - XY 方向 / XY direction: 5-15 cm
  - Z 方向 / Z direction: 1-3 cm
- **时长 / Duration**: 约 2000 个时间步 / ~2000 time steps

#### 用处 / Purpose
1. **实验前验证 / Pre-experiment Verification**: 确认物理模拟正常
2. **参数调试 / Parameter Tuning**: 快速测试不同参数效果
3. **教学演示 / Teaching Demonstration**: 展示不同物体的动力学行为

#### 使用场景 / Use Cases
```bash
# 测试 Object 1（最软）/ Test Object 1 (softest)
输入 / Input: 1 0 1 0
预期 / Expected: 物体很柔软，大幅度弯曲 / Very flexible, large bending

# 测试 Object 5（最硬）/ Test Object 5 (stiffest)
输入 / Input: 5 0 1 0
预期 / Expected: 物体很硬，几乎不弯曲 / Very stiff, minimal bending
```

---

### 5. test_RF.png ⭐⭐⭐⭐

**共振频率测试 / Resonance Frequency Test**

#### 内容 / Content
频率响应曲线图 / Frequency response curve

#### 生成条件 / Generation Condition
```
Trial 设置输入 / Trial settings input: X 0 0 1
例如 / Example: 5 0 0 1
```

#### 测试参数 / Test Parameters
- **频率范围 / Frequency Range**: 0.5-5.0 Hz
- **扫描点数 / Sweep Points**: 450 个频率点 / 450 frequency points
- **激励幅度 / Excitation Amplitude**: 10 cm (XY 方向 / XY direction)

#### 图表内容 / Chart Content
- **X 轴 / X-axis**: 频率 (Hz) / Frequency (Hz)
- **Y 轴 / Y-axis**: 最大响应幅度 (m) / Maximum response amplitude (m)
- **绿色虚线 / Green dashed line**: 共振频率标记 / Resonance frequency marker

#### 物理意义 / Physical Meaning

**共振频率取决于 / Resonance frequency depends on:**
```
f_resonance ≈ (1/2π) × √(k/m)

其中 / Where:
- k: 刚度 / Stiffness (越大 → 频率越高 / larger → higher frequency)
- m: 质量 / Mass (越大 → 频率越低 / larger → lower frequency)
```

**阻尼的影响 / Damping Effect:**
- 阻尼小 / Low damping → 共振峰尖锐 / Sharp resonance peak
- 阻尼大 / High damping → 共振峰平缓 / Broad resonance peak

#### 用处 / Purpose
1. **物体特性分析 / Object Characterization**: 找到自然频率
2. **实验设计 / Experiment Design**: 避免在共振频率附近操作
3. **物理验证 / Physical Verification**: 验证动力学模型正确性

#### 实际应用 / Practical Application
- 如果共振频率在 2 Hz → 被试在 2 Hz 附近操作时会感觉物体"特别软"
- 如果共振峰很尖锐 → 阻尼太小，物体会持续振荡
- 如果没有明显共振峰 → 阻尼太大，物体响应迟钝

---

## 🎯 整体意义：心理物理学实验的完整闭环

### 实验流程 / Experiment Workflow

```
1. 预检查 / Pre-check (test_basic.gif)
   ↓ 确认物理模拟正常 / Confirm physics simulation works
   
2. 共振测试 / Resonance test (test_RF.png)
   ↓ 了解物体特性 / Understand object characteristics
   
3. 触觉渲染 / Haptic rendering (render_hist.csv + render.gif)
   ↓ 被试进行实际交互 / Participant performs actual interaction
   
4. 性能分析 / Performance analysis (render_execution_time.png)
   ↓ 验证实时性 / Verify real-time performance
   
5. 数据分析 / Data analysis (render_hist.csv)
   ↓ 提取心理物理学指标 / Extract psychophysical metrics
```

### 研究问题示例 / Research Question Examples

#### 问题 1: 人类如何感知物体的软硬？
**Question 1: How do humans perceive object stiffness?**

使用 `render_hist.csv` 分析 / Analyze using `render_hist.csv`:
- 用户施加的力大小 / Force magnitude applied by user
- 物体的变形程度 / Object deformation extent
- 力-变形关系（刚度感知）/ Force-deformation relationship (stiffness perception)

#### 问题 2: 拉伸刚度 vs 弯曲刚度，哪个更重要？
**Question 2: Stretching vs bending stiffness, which is more important?**

对比分析 / Comparative analysis:
- Object 1-5（拉伸变化）/ (stretching variation)
- Object 6-10（弯曲变化）/ (bending variation)
- 结合 `psychophysics_loop.py` 的辨别准确率 / Combined with discrimination accuracy
- 分析 `render_hist.csv` 中的交互策略差异 / Analyze interaction strategy differences

#### 问题 3: 触觉反馈的延迟对感知的影响？
**Question 3: Effect of haptic feedback delay on perception?**

使用 `render_execution_time.png` / Use `render_execution_time.png`:
- 确认延迟大小 / Confirm delay magnitude
- 如果延迟 > 1 ms，可能影响真实感 / If delay > 1 ms, may affect realism
- 分析被试的主观评分 / Analyze subjective ratings

---

## 💡 实际使用建议 / Practical Usage Recommendations

### 日常实验 / Daily Experiments

```bash
# 只需要核心数据 / Only need core data
输入 / Input: 5 1 0 0
输出 / Output: render_hist.csv  # 足够进行数据分析 / Sufficient for analysis
```

### 调试阶段 / Debugging Phase

```python
# 需要完整诊断 / Need full diagnostics
# 修改 main.py:1084 / Modify main.py:1084
execute_hapticRendering(
    execute_SpeedAnalysis=True,   # 性能分析 / Performance analysis
    execute_TrajPlot=True          # 轨迹可视化 / Trajectory visualization
)
# 输出全部文件 / Output all files
```

### 论文准备 / Paper Preparation

```
1. 使用 test_basic.gif 展示物体动力学 / Use test_basic.gif to show object dynamics
2. 使用 test_RF.png 展示物体特性 / Use test_RF.png to show object characteristics
3. 使用 render.gif 展示交互过程 / Use render.gif to show interaction process
4. 使用 render_hist.csv 生成统计图表 / Use render_hist.csv to generate statistical plots
```

---

## 🔧 如何控制输出 / How to Control Output

### 修改代码位置 / Code Modification Location

**文件 / File**: `model_dynamics/scripts/main.py`

**行号 / Line**: 1084

**默认设置 / Default Settings**:
```python
execute_hapticRendering()
# 等价于 / Equivalent to:
execute_hapticRendering(
    execute_MessagePrint=False,      # 不打印详细消息 / No detailed messages
    execute_SpeedAnalysis=False,     # 不生成性能分析 / No performance analysis
    execute_DataSave=True,           # 保存 CSV 数据 / Save CSV data
    execute_TrajPlot=False           # 不生成轨迹动画 / No trajectory animation
)
```

### 完整输出设置 / Full Output Settings

```python
execute_hapticRendering(
    execute_MessagePrint=True,       # 打印位置/速度消息 / Print position/velocity
    execute_SpeedAnalysis=True,      # 生成性能分析图 / Generate performance plot
    execute_DataSave=True,           # 保存 CSV 数据 / Save CSV data
    execute_TrajPlot=True            # 生成轨迹动画 / Generate trajectory animation
)
```

---

## 📈 数据分析代码示例 / Data Analysis Code Examples

### 基础分析 / Basic Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据 / Read data
data = pd.read_csv('render_hist.csv')

# 1. 力的统计特性 / Force statistics
force_x = data['Rendered Force X (N)']
print(f"平均力 / Mean force: {force_x.mean():.3f} N")
print(f"最大力 / Max force: {force_x.max():.3f} N")
print(f"力标准差 / Force std: {force_x.std():.3f} N")

# 2. 用户运动范围 / User movement range
user_x = data['User Position X (m)']
user_y = data['User Position Y (m)']
user_z = data['User Position Z (m)']
print(f"X 范围 / X range: {user_x.max() - user_x.min():.3f} m")
print(f"Y 范围 / Y range: {user_y.max() - user_y.min():.3f} m")
print(f"Z 范围 / Z range: {user_z.max() - user_z.min():.3f} m")

# 3. 物体最大变形 / Maximum object deformation
node0_x = data['Node 0 Position X (m)']
node4_x = data['Node 4 Position X (m)']
max_deformation = (node4_x - node0_x).abs().max()
print(f"最大变形 / Max deformation: {max_deformation:.3f} m")
```

### 高级分析 / Advanced Analysis

```python
# 4. 力-位移关系（刚度估计）/ Force-displacement relationship (stiffness estimation)
displacement = node4_x - node0_x
force_magnitude = np.sqrt(force_x**2 + data['Rendered Force Y (N)']**2)

# 线性拟合 / Linear fit
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(displacement, force_magnitude)
print(f"估计刚度 / Estimated stiffness: {slope:.1f} N/m")
print(f"R² = {r_value**2:.3f}")

# 5. 频谱分析 / Spectral analysis
from scipy.fft import fft, fftfreq
dt = data['Timestamp (s)'].diff().mean()
N = len(force_x)
yf = fft(force_x.values)
xf = fftfreq(N, dt)[:N//2]

plt.figure(figsize=(10, 4))
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Force Frequency Spectrum')
plt.xlim(0, 10)
plt.grid(True)
plt.show()
```

---

## 📝 总结 / Summary

### 核心输出 / Core Output
- **render_hist.csv**: 最重要，包含所有实验数据 / Most important, contains all experimental data

### 辅助输出 / Auxiliary Output
- **render.gif**: 可视化验证 / Visual verification
- **render_execution_time.png**: 性能诊断 / Performance diagnostics
- **test_basic.gif**: 预检查 / Pre-check
- **test_RF.png**: 物体特性分析 / Object characterization

### 数据流程 / Data Flow
```
实验设计 / Experiment Design
    ↓
参数设置 / Parameter Setting
    ↓
触觉渲染 / Haptic Rendering
    ↓
数据采集 / Data Collection (render_hist.csv)
    ↓
数据分析 / Data Analysis
    ↓
研究结论 / Research Conclusions
```

### 关键要点 / Key Points
1. **默认只生成 CSV** / Only CSV by default - 足够进行数据分析 / Sufficient for analysis
2. **需要可视化时修改代码** / Modify code for visualization - 设置 `execute_TrajPlot=True`
3. **性能分析用于调试** / Performance analysis for debugging - 设置 `execute_SpeedAnalysis=True`
4. **预检查避免浪费时间** / Pre-check saves time - 使用 `test_basic.gif` 和 `test_RF.png`

---

**文档版本 / Document Version**: 1.0  
**最后更新 / Last Updated**: 2026-02-07  
**作者 / Author**: Claude Code Optimization Team
