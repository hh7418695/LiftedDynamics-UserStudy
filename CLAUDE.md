# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

#### Single Machine Setup (Windows only)

**Start order matters**: Always start C++ client first, then Python server.

1. **Start C++ UDP Client**:
   - Open `udp_client/udp_client.sln` in Visual Studio 2019
   - Build and run the project
   - The haptic device will initialize and center the stylus

2. **Start Python Dynamics Server**:
   ```bash
   cd model_dynamics/scripts
   python main.py --participant_id=1
   ```

#### Cross-Platform Setup (macOS + Windows)

Since the TouchX haptic device only works on Windows, you can run Python on macOS and C++ on Windows. See `CROSS_PLATFORM_SETUP.md` for detailed instructions.

**Quick setup:**

1. **On macOS - Get IP address**:
   ```bash
   ipconfig getifaddr en0  # or en1 for ethernet
   ```

2. **On Windows - Edit C++ code**:
   - Open `udp_client/udp_client/main.cpp`
   - Change line 19: `#define SERVER "192.168.x.x"` (use your macOS IP)
   - Rebuild the project

3. **On macOS - Start Python server**:
   ```bash
   cd model_dynamics/scripts
   python main.py --participant_id=1
   ```

4. **On Windows - Start C++ client**:
   - Run the compiled program

**Test network connection**:
```bash
# On macOS
python model_dynamics/scripts/test_network.py
```

3. **Trial Settings** (when prompted in Python):
   Format: `<object_id> <enable_haptic> <enable_animation_check> <enable_resonance_test>`
   - Example: `5 1 1 0` - Object 5, haptic on, animation check on, resonance test off
   - Object IDs 1-5: Vary stretching stiffness
   - Object IDs 6-10: Vary bending stiffness

4. **Interaction**:
   - Click stylus button once to start rendering
   - Shake/move the stylus to interact with virtual object
   - Click button again to stop and view animation

### Key Python Parameters

When calling `main.py`, you can override defaults:
- `--seed=42` - Random seed for reproducibility
- `--dt=1.0e-4` - Simulation time step (100 μs)
- `--stride=10` - Integration steps per UDP message
- `--socket_serverIP="0.0.0.0"` - UDP server IP (use "0.0.0.0" for cross-platform, "127.0.0.1" for local)
- `--socket_serverPort=12312` - UDP port
- `--participant_id=1` - Participant identifier for data organization

### Testing and Analysis

The Python program includes built-in testing modes:

**Object Animation Check** (trial setting: `X 0 1 0`):
- Generates sinusoidal leader motion
- Simulates object dynamics without haptic device
- Outputs trajectory animation as GIF
- Useful for verifying physics before user study

**Resonance Frequency Test** (trial setting: `X 0 0 1`):
- Sweeps frequencies from 0.5-5.0 Hz
- Identifies resonance peaks
- Outputs frequency response plot
- Helps understand object's natural frequencies

## Code Organization

### Python Structure (`model_dynamics/scripts/`)

- `main.py` - Main entry point, UDP server, haptic rendering loop
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

### Key Python Functions in main.py

- `sim_nextState()` - Predicts next object state given current state and user input
- `getForce_virtualCoupling()` - Computes spring-damper force between user and object
- `sim_acceleration()` - Computes accelerations using Lagrangian mechanics
- `execute_hapticRendering()` - Main real-time rendering loop
- `execute_objectAnimationCheck()` - Pre-visualization of object dynamics
- `execute_resonanceFrequencyTest()` - Frequency sweep analysis

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

## Research Context

This project builds on Lagrangian Graph Neural Networks (LGNN) for learning dynamics. The system is designed for psychophysics experiments investigating:
- How humans perceive object softness through haptic feedback
- The role of bending vs. stretching stiffness in softness perception
- Haptic illusions and their interaction with visual cues in VR

Object parameters (stiffness, mass, length) can be varied to create different perceptual experiences for user studies.
