# 项目完整知识库：触觉刚度感知心理物理学实验系统

> 本文档是本项目的完整知识图谱。可以将此文档作为上下文喂给任何AI工具（Claude、GPT、Gemini等），使其成为你的专属项目Agent，能够回答关于研究背景、实验设计、代码架构、数据分析的所有问题。

---

## 目录

1. [一句话说清楚这个项目](#1-一句话说清楚这个项目)
2. [研究大背景：从触觉到心理物理学](#2-研究大背景从触觉到心理物理学)
3. [核心概念词典](#3-核心概念词典)
4. [系统架构全景](#4-系统架构全景)
5. [物理仿真引擎详解](#5-物理仿真引擎详解)
6. [实验设计详解](#6-实验设计详解)
7. [数据输出与分析方法](#7-数据输出与分析方法)
8. [代码文件清单与依赖关系](#8-代码文件清单与依赖关系)
9. [实验操作完整指南](#9-实验操作完整指南)
10. [必读论文分层清单](#10-必读论文分层清单)
11. [研究可延展方向](#11-研究可延展方向)
12. [常见问题FAQ](#12-常见问题faq)

---

## 1. 一句话说清楚这个项目

**用真实物理仿真驱动力反馈触觉设备，通过心理物理学实验（2AFC范式），量化人类对虚拟柔性物体"拉伸刚度"和"弯曲刚度"的感知分辨能力（JND）。**

翻译成大白话：你摸两根虚拟的"弹性棒"，告诉我哪根更硬。我改变它们的硬度差距，看你到底能分辨多细微的差别。

---

## 2. 研究大背景：从触觉到心理物理学

### 2.1 为什么研究触觉感知？

人类通过触觉获取物体物理属性（软硬、粗糙、温度）的能力，是VR/远程手术/机器人交互的核心问题。但我们对触觉感知的理解远不如视觉和听觉。

关键问题：
- 人能"摸出来"多细微的刚度差异？（**感知阈值**问题）
- 拉伸刚度 vs 弯曲刚度，人是通过同一个"通道"感知的吗？（**感知维度**问题）
- 视觉会不会干扰/增强触觉判断？（**多感官整合**问题）

### 2.2 为什么用心理物理学？

心理物理学（Psychophysics）是一套严格的实验方法论，用于量化"物理刺激"和"主观感知"之间的关系。核心工具：

- **Weber定律**：能察觉的最小变化（JND）与基准刺激成正比。ΔI / I = k（Weber fraction）
- **Psychometric curve（心理物理曲线）**：正确率随刺激差异变化的S型曲线
- **2AFC（两选一强迫选择）**：每次给两个刺激，强迫选择，消除判断偏差

### 2.3 为什么用力反馈设备而不是直接摸真东西？

- 可以精确控制物理参数（真实材料很难精确制造连续变化的刚度）
- 可以独立操控拉伸 vs 弯曲刚度（真实物体两者耦合在一起）
- 可重复、可参数化、可大规模实验

### 2.4 这个研究有什么用？

| 应用领域 | 具体价值 |
|---------|---------|
| VR/游戏 | 知道触觉渲染的"够用精度"，避免过度计算 |
| 远程手术 | 医生通过机器人触诊组织时的力反馈精度要求 |
| 触觉设备设计 | 量化设备的感知分辨率指标 |
| 材料科学 | 理解人类对材料柔软度的主观评价机制 |
| 假肢/康复 | 触觉反馈假肢需要知道用户能感知的最小力差 |

---

## 3. 核心概念词典

### 3.1 心理物理学概念

| 概念 | 英文 | 定义 | 在本项目中的体现 |
|------|------|------|----------------|
| JND | Just Noticeable Difference | 能被感知到的最小刺激差异，通常定义为psychometric curve上75%正确率对应的刺激差值 | 最终要提取的核心指标——拉伸/弯曲刚度的JND |
| Weber Fraction | Weber Fraction (WF) | JND / 基准刺激值 = 常数k。触觉刚度的典型WF约0.20~0.30 | 如果WF恒定，说明Weber定律成立 |
| 2AFC | Two-Alternative Forced Choice | 每次给两个刺激，被试必须选一个（不能说"不知道"） | 实验的基本范式：摸两个物体，选更硬的那个 |
| Psychometric Curve | 心理物理曲线 | 正确判断概率 vs 刺激差异的S型函数（通常是logistic或累积正态分布） | 从40个trial的正确率拟合得到 |
| PSE | Point of Subjective Equality | 主观等同点——50%正确率对应的刺激值，此时被试认为两个刺激相同 | 曲线中点 |
| Lapse Rate | 失误率 | 即使刺激差异很大时的错误率（注意力不集中等） | 曲线拟合的上渐近线参数 |
| Guess Rate | 猜测率 | 2AFC中纯猜正确率 = 50% | 曲线下渐近线固定为0.5 |

### 3.2 触觉渲染概念

| 概念 | 英文 | 定义 | 在本项目中的体现 |
|------|------|------|----------------|
| Virtual Coupling | 虚拟耦合 | 连接用户（haptic device）和虚拟物体的弹簧-阻尼器，是触觉渲染的标准方法 | ks=750×10³ N/m, kd=25×10³ Ns/m |
| Haptic Rendering | 触觉渲染 | 实时计算并输出力反馈信号到触觉设备 | 整个系统的1kHz主循环 |
| Update Rate | 更新率 | 力反馈的刷新频率，低于~300Hz人会感觉到"颗粒感" | 目标1000Hz (1kHz) |
| Passivity | 无源性 | 系统不会自发产生能量的条件，是稳定性的充分条件 | 虚拟耦合参数需满足passivity条件 |
| Impedance Rendering | 阻抗渲染 | 读位置→算力→输出力的渲染模式（与导纳渲染相反） | 本系统采用的模式 |
| God Object / Proxy | 代理点 | 虚拟物体上跟踪用户位置的点，不穿透物体表面 | 本系统中object_position[0]（第一个节点） |

### 3.3 物理仿真概念

| 概念 | 英文 | 定义 | 在本项目中的体现 |
|------|------|------|----------------|
| Lagrangian Mechanics | 拉格朗日力学 | L = T - V（动能减势能），通过Euler-Lagrange方程求运动方程 | sim_Lagrangian() = kinetic - potential |
| Stretching Stiffness | 拉伸刚度 (ks) | 抵抗轴向拉伸变形的能力，V = 0.5·ks·(Δl)² | 50~1000 ×10³ N/m（Object 1-5） |
| Bending Stiffness | 弯曲刚度 (kb) | 抵抗弯曲变形的能力，V = 0.5·kb·θ² | 0~0.1 ×10³ N/m（Object 6-10） |
| Velocity Verlet | 速度Verlet积分 | 二阶辛积分器，保持能量守恒 | NVE integrator in nve.py |
| Kalman Filter | 卡尔曼滤波 | 从噪声观测中估计真实状态的最优滤波器 | 从位置估计速度（KalmanFilter3D） |

### 3.4 机器学习概念（扩展方向）

| 概念 | 英文 | 定义 | 与本项目的关系 |
|------|------|------|--------------|
| LNN | Lagrangian Neural Network | 用神经网络参数化拉格朗日量L，通过自动微分得到运动方程 | 项目名包含LNN，但当前系统用解析公式而非学习模型 |
| HNN | Hamiltonian Neural Network | 用神经网络参数化哈密顿量H，保证能量守恒 | LNN的姊妹方法 |
| LGNN | Lagrangian Graph Neural Network | 将LNN与图神经网络结合，处理多粒子系统 | 未来可替换解析公式的方向 |
| Physics-Informed NN | 物理信息神经网络 | 将物理方程作为loss或结构先验注入神经网络 | LNN/HNN都属于此类 |

---

## 4. 系统架构全景

### 4.1 硬件

- **TouchX（原Phantom Omni）**：3自由度力反馈触觉设备
  - 工作空间：约160×120×70mm
  - 力输出：3.3N持续，最大约7.5N（不同说法有别）
  - 接口：USB / FireWire
  - 驱动：OpenHaptics SDK 3.5（仅Windows）

### 4.2 软件架构

```
┌─────────────────────────┐     UDP (JSON)      ┌──────────────────────────┐
│     C++ UDP Client      │ ◄──────────────────► │   Python Physics Engine  │
│  (udp_client/main.cpp)  │     12312 port       │   (render_worker.py)     │
│                         │                      │                          │
│  ┌───────────────────┐  │  position (mm)  →    │  ┌────────────────────┐  │
│  │ OpenHaptics SDK    │  │                      │  │ JAX + Lagrangian   │  │
│  │ - read position    │  │  ← force (N)        │  │ - coord transform  │  │
│  │ - render force     │  │                      │  │ - physics sim      │  │
│  │ - button detect    │  │                      │  │ - Kalman filter    │  │
│  └───────────────────┘  │                      │  │ - force compute    │  │
│                         │                      │  │ - force clamp 2.5N │  │
│  1kHz scheduler callback│                      │  └────────────────────┘  │
└─────────────────────────┘                      └──────────┬───────────────┘
                                                            │
                                                   File IPC │ (command.txt/result.txt)
                                                            │
                                                 ┌──────────▼───────────────┐
                                                 │  Tkinter GUI             │
                                                 │  (experiment_runner_     │
                                                 │   gui.py)               │
                                                 │                          │
                                                 │  - 参与者ID输入          │
                                                 │  - block类型选择         │
                                                 │  - 试次流程控制          │
                                                 │  - 2AFC响应收集          │
                                                 │  - CSV数据记录           │
                                                 └──────────────────────────┘
```

### 4.3 坐标系转换

TouchX设备坐标系和Python仿真坐标系不同：

```
TouchX设备:          Python仿真:
  Y (up)               Z (up)
  |                    |
  |____ X              |____ X
 /                    /
Z (toward user)      Y (toward user, but inverted)

转换: TouchX(x,y,z) → Python(x, -z, y)
逆转: Python(x,y,z) → TouchX(x, z, -y)
```

C++不做任何转换，Python端负责双向转换。

### 4.4 通信协议

| 方向 | 格式 | 示例 |
|------|------|------|
| C++ → Python | `{"position":[x,y,z],"timestamp":t}` | `{"position":[12.5, -3.2, 8.1],"timestamp":1042}` |
| Python → C++ | `{"force":[fx,fy,fz]}` | `{"force":[0.15, -0.08, 0.22]}` |
| C++ → Python (停止) | `{"position":[...],"timestamp":-1.0}` | timestamp为负数表示停止 |

### 4.5 渲染循环时序

```
时间轴(ms):  0    1    2    3    4    5    ...
C++ 1kHz:    P₁   P₂   P₃   P₄   P₅   P₆   ...  (读位置)
UDP发送:     →    →    →    →    →    →   ...  (每次回调发一次)
Python:      F₁   F₂   F₃   F₄   F₅   F₆   ...  (算力)
UDP返回:     ←    ←    ←    ←    ←    ←   ...
C++渲染:     F₁   F₂   F₃   F₄   F₅   F₆   ...  (输出到设备)

整体延迟 ≈ 1-2ms（本地UDP + 计算时间）
```

---

## 5. 物理仿真引擎详解

### 5.1 物体模型

虚拟物体 = 5个质点用弹簧串联的链：

```
    Node 0 (leader, 连接用户)
    │  segment 0 (5cm)
    Node 1
    │  segment 1 (5cm)
    Node 2
    │  segment 2 (5cm)
    Node 3
    │  segment 3 (5cm)
    Node 4 (末端自由)

每个节点质量: 40g
总质量: 200g (5 × 40g)
每段长度: 5cm
总长度: 20cm
```

### 5.2 能量与力

**拉格朗日量 L = T - V**

```
T (动能) = Σᵢ 0.5 · mᵢ · |vᵢ|²

V (势能) = V_stretching + V_bending + V_gravity

V_stretching = Σᵢ 0.5 · ksᵢ · |Δlᵢ|²     (弹簧拉伸能)
V_bending    = Σᵢ 0.5 · kbᵢ · θᵢ²         (弯曲角度能)
V_gravity    = Σᵢ mᵢ · g · zᵢ              (重力势能)
```

**非保守力：**
- 阻尼力：F_damping = -c · v_relative（速度相关耗散）
- 虚拟耦合力：F_vc = -ks_vc · Δx - kd_vc · Δv（弹簧-阻尼器）

**加速度求解：** Euler-Lagrange方程 → M·a = -C·v + ∂L/∂x + F_nc + F_ext

### 5.3 10个物体的参数

**Stretch block（Object 1-5）：变拉伸刚度，固定弯曲刚度**

| Object ID | ks (×10³ N/m) | kb (×10³ N/m) | 直觉描述 |
|-----------|---------------|---------------|---------|
| 1 | 50 | 0.05 | 非常软，像橡皮筋 |
| 2 | 287.5 | 0.05 | 较软 |
| 3 | 525 | 0.05 | 中等（参考值） |
| 4 | 762.5 | 0.05 | 较硬 |
| 5 | 1000 | 0.05 | 非常硬，几乎不拉伸 |

**Bend block（Object 6-10）：变弯曲刚度，固定拉伸刚度**

| Object ID | ks (×10³ N/m) | kb (×10³ N/m) | 直觉描述 |
|-----------|---------------|---------------|---------|
| 6 | 525 | 0 | 完全柔软，像绳子 |
| 7 | 525 | 0.025 | 轻微抵抗弯曲 |
| 8 | 525 | 0.05 | 中等弯曲刚度 |
| 9 | 525 | 0.075 | 较硬，像细杆 |
| 10 | 525 | 0.1 | 最硬，几乎不弯曲 |

### 5.4 力的安全限制

- 每个轴最大力：2.5 N
- 超限处理：等比例缩放（保持方向，降低幅度）
- 力缩放因子：`force *= -1e-3`（仿真内力单位转换到设备力单位）

---

## 6. 实验设计详解

### 6.1 实验范式：2AFC

```
        Trial结构：
        ┌─────────────────┐
        │  显示"第一个物体" │
        │  C++开始发position │
        │  Python渲染力反馈  │  ← 被试摇晃/提升物体，感受软硬
        │  被试点击按钮停止  │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  显示"第二个物体" │
        │  C++开始发position │
        │  Python渲染力反馈  │  ← 同样操作
        │  被试点击按钮停止  │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  GUI显示选择按钮  │
        │  "第一个更硬" OR  │  ← 被试2AFC响应
        │  "第二个更硬"     │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  记录响应到CSV    │
        │  进入下一trial    │
        └─────────────────┘
```

### 6.2 试次生成逻辑

每个block有5个物体，生成 C(5,2)=10 个唯一配对：

**Stretch block配对（Object 1-5）：**
```
(1,2), (1,3), (1,4), (1,5),
(2,3), (2,4), (2,5),
(3,4), (3,5),
(4,5)
```

每对重复4次 → 10×4 = **40 trials per block**

**每个trial的呈现顺序随机化**：哪个物体先呈现是50/50随机的（counterbalancing）。

### 6.3 正确性判定

- **target_object** = 该配对中刚度更高的那个物体
- Stretch block：ks更大的 = target
- Bend block：kb更大的 = target
- `is_correct = 1` 如果被试选择了target_object

### 6.4 数据持久化

- 试次顺序保存为JSON，支持中断恢复
- 每个trial完成后立即追加到CSV（不缓冲）
- 物理渲染数据（力、位置时间序列）保存到单独的 render_hist.csv

---

## 7. 数据输出与分析方法

### 7.1 行为数据格式

**文件**: `behaviour_results/participant_{ID}_{block}_behaviour.csv`

| 字段 | 说明 |
|------|------|
| participant_id | 参与者ID |
| trial_index | 试次编号 (0-39) |
| block_type | "stretch" 或 "bend" |
| ref_object_id | 配对中的第一个物体ID |
| comp_object_id | 配对中的第二个物体ID |
| k_stretch_ref | ref物体的ks值 |
| k_stretch_comp | comp物体的ks值 |
| k_bend_ref | ref物体的kb值 |
| k_bend_comp | comp物体的kb值 |
| first_object | 先呈现的物体ID |
| second_object | 后呈现的物体ID |
| chosen_object | 被试选择的物体ID |
| is_correct | 1=正确, 0=错误 |
| notes | 备注 |

### 7.2 物理渲染数据格式

**文件**: `model_dynamics/results/Participant-{ID}/Obj-{ObjID}/{timestamp}/render_hist.csv`

| 字段 | 说明 |
|------|------|
| Timestamp (s) | 从渲染开始的时间 |
| Rendered Force X/Y/Z (N) | 输出到设备的力（Python坐标系） |
| User Position X/Y/Z (m) | 用户位置（Python坐标系） |
| Node 0-4 Position X/Y/Z (m) | 5个节点的位置 |

### 7.3 数据分析流程

#### Step 1：按配对聚合正确率

```python
# 对每个唯一配对 (obj_A, obj_B)，计算4次重复中的正确率
# 得到: pair → accuracy (0%, 25%, 50%, 75%, 100%)
```

#### Step 2：计算刺激差异

```python
# Stretch block: Δks = |ks_A - ks_B|
# Bend block: Δkb = |kb_A - kb_B|
```

#### Step 3：拟合Psychometric Curve

使用4参数logistic函数：

```
P(correct | Δk) = γ + (1 - γ - λ) · 1/(1 + exp(-β·(Δk - α)))
```

参数：
- **α (threshold/PSE)**：曲线中点，对应~75%正确率的刺激差
- **β (slope)**：曲线斜率，越大表示感知越敏锐
- **γ (guess rate)**：2AFC中固定为0.5
- **λ (lapse rate)**：注意力失误率，通常0.01~0.05

#### Step 4：提取JND

**JND = psychometric curve上75%正确率对应的Δk值**

```python
# JND_stretch = Δks at P=0.75
# JND_bend = Δkb at P=0.75
```

#### Step 5：计算Weber Fraction

```python
# WF = JND / reference_stiffness
# 典型值：触觉刚度WF ≈ 0.20 ~ 0.30
```

### 7.4 推荐Python分析工具

| 工具 | 用途 | 安装 |
|------|------|------|
| psignifit (pypsignifit) | 专业psychometric curve拟合，支持2AFC | `pip install python-psignifit` |
| psychofit (cortex-lab) | 轻量级2AFC拟合 | GitHub: cortex-lab/psychofit |
| scipy.optimize.curve_fit | 通用logistic拟合 | `pip install scipy` |
| pandas | CSV数据处理 | `pip install pandas` |
| matplotlib | 绘制psychometric curve | `pip install matplotlib` |

### 7.5 分析代码示例（伪代码）

```python
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 1. 加载数据
df = pd.read_csv("behaviour_results/participant_XXX_stretch_behaviour.csv")

# 2. 计算每个配对的刺激差和正确率
df["delta_k"] = abs(df["k_stretch_ref"] - df["k_stretch_comp"])
accuracy = df.groupby("delta_k")["is_correct"].mean()

# 3. 定义psychometric function (logistic, 2AFC)
def psychometric(x, alpha, beta, lapse=0.02):
    gamma = 0.5  # guess rate for 2AFC
    return gamma + (1 - gamma - lapse) / (1 + np.exp(-beta * (x - alpha)))

# 4. 拟合
popt, pcov = curve_fit(psychometric, accuracy.index, accuracy.values,
                       p0=[300, 0.01], bounds=([0, 0], [1000, 1]))

# 5. 提取JND (75% correct threshold)
alpha_fit, beta_fit = popt
# JND ≈ alpha (psychometric curve的中点对应75%正确率)

# 6. 绘制
import matplotlib.pyplot as plt
x_fit = np.linspace(0, 1000, 200)
plt.plot(x_fit, psychometric(x_fit, *popt), 'r-')
plt.scatter(accuracy.index, accuracy.values, c='black')
plt.axhline(0.75, linestyle='--', color='gray')
plt.xlabel("Δ Stretching Stiffness (×10³ N/m)")
plt.ylabel("Proportion Correct")
plt.title("Psychometric Curve - Stretch Block")
plt.show()
```

---

## 8. 代码文件清单与依赖关系

### 8.1 文件依赖图

```
experiment_runner_gui.py  (Tkinter GUI, 主入口)
    ├── imports: psychophysics_loop.py (试次逻辑)
    └── subprocess: render_worker.py (触觉渲染子进程)
                        ├── imports: src/md.py (分子动力学)
                        │               └── imports: src/nve.py (NVE积分器)
                        ├── imports: src/lnn.py (拉格朗日力学)
                        │               └── imports: src/models.py (模型架构)
                        └── imports: src/utils.py (Kalman滤波, 绘图)

psychophysics_loop.py (纯Python, 无外部依赖)
    └── 试次生成, 刚度映射, CSV记录

udp_client/main.cpp (C++ 独立程序)
    └── OpenHaptics SDK, Winsock2
```

### 8.2 各文件职责一句话

| 文件 | 职责 |
|------|------|
| `experiment_runner_gui.py` | Tkinter GUI：收集参与者信息，控制试次流程，收集2AFC响应 |
| `render_worker.py` | 长驻子进程：绑定UDP，按指令渲染不同物体，保存力/位置数据 |
| `psychophysics_loop.py` | 纯逻辑：生成试次配对，管理试次顺序，写CSV，判断正确性 |
| `src/lnn.py` | 核心物理：定义accelerationFull()，用JAX自动微分求Euler-Lagrange方程 |
| `src/md.py` | 积分器：predition()函数，调用NVE积分器推进物理状态 |
| `src/nve.py` | NVE积分：Velocity Verlet算法，更新位置和速度 |
| `src/models.py` | 神经网络模板：forward_pass(), loadmodel()（当前未使用，留给LNN扩展） |
| `src/io.py` | 文件IO：pickle读写，轨迹保存 |
| `src/utils.py` | 工具箱：KalmanFilter3D, plot_trajectory, 常量定义 |
| `udp_client/main.cpp` | C++客户端：1kHz读设备位置，发UDP，收力，渲染到设备 |

### 8.3 关键Python依赖

| 包 | 版本 | 用途 |
|----|------|------|
| jax | 0.4.23 | 自动微分、JIT编译、GPU加速 |
| jax-md | 0.2.8 | 分子动力学数据结构 |
| jraph | 0.0.6.dev0 | 图神经网络（预留扩展） |
| numpy | 1.26.4 | 数组操作 |
| scipy | 1.12.0 | 科学计算 |
| matplotlib | | 可视化 |
| fire | | CLI参数解析 |

---

## 9. 实验操作完整指南

### 9.1 实验前准备

1. **硬件检查**
   - TouchX设备已连接USB并供电
   - 触笔可自由移动，无机械卡阻
   - OpenHaptics驱动已安装

2. **软件环境**
   ```bash
   # Python虚拟环境
   cd model_dynamics/scripts
   python -m venv venv
   venv\Scripts\activate      # Windows
   pip install -r ../requirements.txt
   ```

3. **C++编译**
   - Visual Studio 2019 打开 `udp_client/udp_client.sln`
   - 编译 Release x64

### 9.2 运行实验

**步骤1：启动C++客户端**
- 运行编译好的 udp_client.exe
- 看到 "Haptic rendering started :)" 表示就绪
- 触笔会被弹簧力拉向原点（正常行为）

**步骤2：启动Python GUI**
```bash
cd model_dynamics/scripts
python experiment_runner_gui.py
```

**步骤3：GUI操作**
- 输入参与者ID（仅字母）
- 选择block类型（stretch 或 bend）
- 点击"开始实验"

**步骤4：每个trial的操作**
1. GUI显示"正在渲染第一个物体"
2. 在C++窗口中，**点击触笔按钮一次**开始渲染
3. 摇晃/提升触笔，感受物体的软硬
4. **再次点击按钮**停止渲染
5. GUI自动切换到第二个物体，重复步骤2-4
6. GUI弹出选择按钮："第一个更硬" / "第二个更硬"
7. 点击选择，进入下一个trial

**步骤5：实验完成**
- 40个trial全部完成后GUI会提示
- 数据自动保存到 `behaviour_results/` 目录

### 9.3 中断恢复

实验可以安全中断。重新启动时：
- 输入相同的参与者ID和block类型
- 系统自动检测已有数据，从上次中断处继续

### 9.4 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| C++启动失败 "Failed to initialize" | 设备未连接或驱动未安装 | 检查USB连接，安装OpenHaptics |
| Python连接超时 | C++未启动或端口冲突 | 先启动C++，检查端口12312 |
| 力反馈感觉不对 | 坐标系问题或参数异常 | 检查console输出力的方向 |
| JIT编译很慢 | 首次运行JAX需要编译 | 前5次warm-up消息就是为此设计的 |
| GUI无响应 | render_worker崩溃 | 检查stderr输出，重启GUI |

---

## 10. 必读论文分层清单

### 第一层：必读基础（入门级，理解"在做什么"）

| # | 论文 | 为什么必读 |
|---|------|-----------|
| 1 | **Gescheider (1997) "Psychophysics: The Fundamentals"** (教科书) | 心理物理学方法论的圣经，理解2AFC、JND、psychometric curve |
| 2 | **Jones & Hunter (1990) "A perceptual analysis of stiffness"** Exp Brain Res | 触觉刚度感知的经典研究，建立了刚度JND的基准 |
| 3 | **Colgate & Brown (1994) "Factors affecting the Z-width of a haptic display"** ICRA | 虚拟耦合(virtual coupling)的奠基论文，解释你系统中弹簧-阻尼器为什么这么设计 |

### 第二层：核心方法（理解"怎么做的"）

| # | 论文 | 为什么重要 |
|---|------|-----------|
| 4 | **Tan et al. (1994) "Human factors for the design of force-reflecting haptic interfaces"** ASME | 触觉设备的人因工程设计原则 |
| 5 | **Srinivasan & LaMotte (1995) "Tactual discrimination of softness"** J Neurophysiology | 软硬度触觉感知的神经生理学基础 |
| 6 | **Bergmann Tiest & Kappers (2009) "Cues for haptic perception of compliance"** IEEE ToH | 人感知软硬度时使用的触觉线索分析 |
| 7 | **Prins & Kingdom (2018) "Applying the model-comparison approach to test specific research hypotheses in psychophysical research using the Palamedes toolbox"** Frontiers | Psychometric curve拟合方法论（虽然用的Matlab，方法通用） |

### 第三层：技术实现（理解"为什么这么实现"）

| # | 论文 | 为什么相关 |
|---|------|-----------|
| 8 | **Adams & Hannaford (1999) "Stable haptic interaction with virtual environments"** IEEE T-Robotics | 触觉渲染稳定性分析，passivity条件 |
| 9 | **Cranmer et al. (2020) ["Lagrangian Neural Networks"](https://arxiv.org/abs/2003.04630)** ICLR Workshop | LNN原始论文——你项目名字的来源 |
| 10 | **Sanchez-Gonzalez et al. (2022) ["Learning Dynamics of Particle-based Systems with LGNN"](https://arxiv.org/abs/2209.01476)** | 图结构的LGNN，与你的弹簧-质点系统最相关 |
| 11 | **Greydanus et al. (2019) ["Hamiltonian Neural Networks"](https://arxiv.org/abs/1906.01563)** NeurIPS | HNN——LNN的姊妹方法，保证能量守恒 |

### 第四层：最新进展（2024-2025）

| # | 论文 | 链接 |
|---|------|------|
| 12 | **"Stiffness Perception in Haptic Teleoperation with Imperfect Network" (2025)** | [MDPI Electronics](https://www.mdpi.com/2079-9292/14/4/792) |
| 13 | **"Measuring Perception of Bond Stiffness in VR via Gamified Psychophysics" (2024)** | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-71707-9_13) |
| 14 | **"Wearable Vibrotactile Haptic Device for Stiffness Discrimination"** Frontiers | [Frontiers](https://www.frontiersin.org/articles/10.3389/frobt.2020.00042/full) |

---

## 11. 研究可延展方向

### 方向A：视觉-触觉跨模态感知
- 加入VR头显渲染物体变形动画
- 研究问题：看到物体变形 vs 只摸到力反馈，JND有什么差异？视觉会产生触觉错觉吗？
- 关键词：visual-haptic integration, cross-modal perception, haptic illusion

### 方向B：用LNN替换解析物理
- 当前系统用的是手写的弹簧-质点公式
- 可以训练Lagrangian Neural Network从真实材料的运动轨迹学习动力学
- 研究问题：学习到的物理 vs 解析物理，人能摸出差别吗？
- 关键词：physics-informed neural network, learned simulation, sim-to-real

### 方向C：网络延迟对感知的影响
- 在UDP通信中人为添加延迟和抖动
- 研究问题：多大的延迟会让用户无法分辨刚度？
- 关键词：haptic teleoperation, time delay, stability margin

### 方向D：多维刚度感知空间
- 同时改变拉伸和弯曲刚度
- 研究问题：这两个维度是独立感知通道还是相互干扰？
- 关键词：multidimensional scaling, perceptual space, integral vs separable dimensions

### 方向E：个体差异与学习效应
- 跨被试分析JND差异
- 研究问题：反复练习后触觉分辨能力会提高吗？提高到什么程度？
- 关键词：perceptual learning, individual differences, expertise

---

## 12. 常见问题FAQ

### Q: 项目名"LiftedDynamics"是什么意思？
A: "Lifted"指的是用户通过触觉设备"提起"虚拟物体的动作。整个研究关注的是在这个提起/摇晃过程中，物体的动力学响应如何被人感知。

### Q: 为什么用JAX而不是PyTorch？
A: JAX的自动微分可以直接对拉格朗日量L求导得到运动方程，且JIT编译能达到1kHz的实时要求。PyTorch的autograd在这种"对物理方程求导"的场景下不够自然。

### Q: 系统真的能跑到1kHz吗？
A: 第一次调用会慢（JAX JIT编译），之后在现代CPU上通常能达到。warm-up阶段（5条消息）就是为了让JIT编译在正式渲染前完成。

### Q: 为什么弯曲刚度的范围（0~0.1）比拉伸刚度（50~1000）小这么多？
A: 因为物理含义不同。弯曲是角度变形，拉伸是长度变形，量纲和数量级本来就不同。两者的JND也可能完全不同——这正是实验要回答的问题。

### Q: 2AFC中guess rate为什么是50%？
A: 因为只有两个选项。即使完全猜，也有50%概率猜对。所以psychometric curve的下渐近线是0.5而不是0。

### Q: Weber定律在这个实验中一定成立吗？
A: 不一定。Weber定律是经验规律，在极端刺激值时可能失效。本实验可以检验：对不同参考刚度，Weber fraction是否恒定。

### Q: 现有的模型权重文件在哪里？
A: 当前系统**不使用任何预训练模型**。`src/models.py`中的loadmodel()是预留接口。物理仿真完全基于解析的拉格朗日力学方程。

---

*文档版本: 2026-03-22*
*项目仓库: LiftedDynamics-UserStudy*
