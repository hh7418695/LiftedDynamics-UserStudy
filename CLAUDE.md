# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Always use relative paths when referring to files. Do not use absolute paths.

## Project Overview

This is a haptic rendering research system for user studies on the dynamics of lifted objects during haptic interactions. The system uses a TouchX haptic device to provide real-time force feedback based on physics simulations of flexible objects. The architecture consists of two main components that communicate via UDP:

1. **C++ UDP Client** (`udp_client/`) - Interfaces with the TouchX haptic device using OpenHaptics SDK
2. **Python Dynamics Engine** (`model_dynamics/`) - Computes object dynamics and force feedback using JAX and Lagrangian Neural Networks

## System Architecture

### Communication Flow

The system operates in a real-time loop:
- C++ client reads haptic stylus position at ~1kHz and sends via UDP
- Python server receives position, computes forces using Lagrangian mechanics
- Python sends force feedback back to C++
- C++ renders force to the haptic device
- User clicks stylus button to start/stop interaction

### Coordinate System Transformation

**Critical**: The TouchX device and Python simulation use different coordinate systems. Transformations are handled by:
- `getCoordinate_TouchX2Python()` - Converts device coords to simulation coords: (x, y, z) → (x, -z, y)
- `getCoordinate_Python2TouchX()` - Converts simulation coords to device coords: (x, y, z) → (x, z, -y)

### Physics Simulation

The system uses Lagrangian mechanics with:
- **Stretching stiffness**: Controls axial deformation (50-1000 × 10³ N/m)
- **Bending stiffness**: Controls angular deformation (0-0.1 × 10³ N/m)
- **Virtual coupling**: Spring-damper connecting user to virtual object (750 × 10³ N/m stiffness, 25 × 10³ Ns/m damping)
- **Damping**: General velocity-dependent damping (0.1 × object_damping_scale)
- **Gravity**: Enabled by default

Objects are modeled as N=5 nodes connected by springs, with configurable mass (40g per node) and length (5cm per segment).

## Development Commands

### Python Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r ./model_dynamics/requirements.txt
```

### Running the System

**Start order matters**: Always start C++ client first, then Python GUI.

1. **Start C++ UDP Client**:
   - Open `udp_client/udp_client.sln` in Visual Studio 2019
   - Build and run the project
   - The haptic device will initialize and center the stylus

2. **Start Python GUI**:
   ```bash
   cd model_dynamics/scripts
   python experiment_runner_gui.py
   ```

3. **Interaction**:
   - The GUI manages experiment flow (participant ID, block type, trial progression)
   - Click stylus button once to start rendering
   - Shake/move the stylus to interact with virtual object
   - Click button again to stop
   - GUI collects 2AFC responses after each trial pair

## Code Organization

### Python Structure (`model_dynamics/scripts/`)

- `experiment_runner_gui.py` - **Main entry point** (Tkinter GUI for 2AFC experiment)
- `render_worker.py` - Haptic rendering subprocess (spawned by GUI)
- `psychophysics_loop.py` - Trial generation, data persistence, 2AFC logic
- `src/lnn.py` - Lagrangian Neural Network implementation, acceleration computation
- `src/md.py` - Molecular dynamics utilities, state prediction
- `src/nve.py` - NVE ensemble integrator (microcanonical)
- `src/utils.py` - Kalman filter, trajectory plotting, helper functions
- `src/models.py` - Neural network models and forward pass
- `src/io.py` - I/O utilities

### C++ Structure (`udp_client/`)

- `udp_client/main.cpp` - Haptic device interface and UDP client
- `master_interface()` callback - Runs at 1kHz, reads position, renders force
- Button click detection manages state transitions (idle → rendering → stop)

### Key Python Functions

In `render_worker.py`:
- `render_single_object()` - Performs physics simulation and UDP haptic rendering loop

In `psychophysics_loop.py`:
- `build_block_trials()` - Generates randomized trial pairs
- `get_stiffness_for_object()` - Maps object ID to stiffness parameters
- `append_trial_row()` - Saves trial response data to CSV

## Data Output

Results are saved to: `model_dynamics/results/Participant-{id}/Obj-{id}/{timestamp}/`

Files generated:
- `render_hist.csv` - Time series of forces, user position, object node positions
- `render.gif` - Animated trajectory (if `execute_TrajPlot=True`)
- `render_execution_time.png` - Performance analysis (if `execute_SpeedAnalysis=True`)
- `test_basic.gif` - Animation check output
- `test_RF.png` - Resonance frequency response curve

## Important Implementation Notes

### JAX Configuration
- 64-bit precision is enabled: `jax.config.update("jax_enable_x64", True)`
- All physics computations use double precision for accuracy

### Kalman Filtering
- Enabled by default (`enable_KalmanFilter=True`)
- Estimates velocity from noisy position measurements
- Uses 3D Kalman filter with process/measurement noise tuning

### Force Limiting
- Maximum force clamped to 2.5 N per axis to prevent device saturation
- Forces scaled proportionally if magnitude exceeds limit
- Critical for device safety and stable haptics

### Performance Considerations
- Target update rate: ~1000 Hz (1 ms per iteration)
- UDP communication is blocking - Python waits for C++ position before computing
- JIT compilation happens on first call - expect slower initial iteration
- Use `stride` parameter to balance accuracy vs. computation time

### Warm-up Phase
- C++ sends 5 warm-up messages with zero position before actual data
- Allows JIT compilation to complete before real-time rendering
- User's first position after warm-up becomes the origin

## Dependencies

### Python (see requirements.txt)
- JAX 0.4.23 - Automatic differentiation and JIT compilation
- jax-md 0.2.8 - Molecular dynamics utilities
- jraph 0.0.6.dev0 - Graph neural networks
- numpy, scipy, pandas, scikit-learn - Scientific computing
- matplotlib - Visualization
- fire - CLI argument parsing

### C++ (Windows only)
- OpenHaptics SDK 3.5 - TouchX device drivers and API
- Visual Studio 2019 - Build toolchain
- Winsock2 - UDP networking

## Psychophysics Experiment (2AFC)

The behavioral experiment is managed by `experiment_runner_gui.py` (Tkinter GUI) and collects two-alternative forced-choice (2AFC) discrimination data.

### Running the experiment

```bash
cd model_dynamics/scripts
python experiment_runner_gui.py
```

The GUI prompts for participant ID and block type (stretch or bend), then runs 40 trials (10 pairs x 4 reps). Trial order is saved to `behaviour_results/participant_{ID}_{block}_order.json` so sessions can be resumed after interruption.

### Behavioral data

Results land in `behaviour_results/`:
- `participant_{ID}_{block}_order.json` - randomized trial order (persists across sessions)
- `participant_{ID}_{block}_behaviour.csv` - one row per trial with fields: `participant_id`, `trial_index`, `block_type`, `ref_object_id`, `comp_object_id`, `k_stretch_ref/comp`, `k_bend_ref/comp`, `first_object`, `second_object`, `chosen_object`, `is_correct`, `notes`

### Object stiffness mapping

| Object IDs | Varies | Fixed |
|---|---|---|
| 1-5 | `ks` = 50, 287.5, 525, 762.5, 1000 (x10^3 N/m) | `kb` = 0.05 x10^3 N/m |
| 6-10 | `kb` = 0, 0.025, 0.05, 0.075, 0.1 (x10^3 N/m) | `ks` = 525 x10^3 N/m |

`get_stiffness_for_object(object_id)` in `psychophysics_loop.py` is the single source of truth for this mapping.

## Research Context

This project builds on Lagrangian Graph Neural Networks (LGNN) for learning dynamics. The system is designed for psychophysics experiments investigating:
- How humans perceive object softness through haptic feedback
- The role of bending vs. stretching stiffness in softness perception
- Haptic illusions and their interaction with visual cues in VR

Object parameters (stiffness, mass, length) can be varied to create different perceptual experiences for user studies.
