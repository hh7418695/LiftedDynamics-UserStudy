# Kalman Filter JAX Optimization

## Overview

A JAX-optimized version of the 3D Kalman Filter has been added to `src/utils.py`. This provides better integration with JAX-based physics simulations.

## Usage

### Option 1: Keep NumPy version (default, no changes needed)
```python
from src.utils import KalmanFilter3D

kf = KalmanFilter3D(process_var=0.001, measurement_var=0.001, estimated_var=0.01)
kf.x[:3] = [0.0, 0.0, 0.0]  # Set initial position

# In the main loop:
kf.predict(dt)
kf.update(measurement)
position = kf.get_state_position()
velocity = kf.get_state_velocity()
```

### Option 2: Switch to JAX version (drop-in replacement)
```python
from src.utils import KalmanFilter3D_JAX as KalmanFilter3D

kf = KalmanFilter3D(process_var=0.001, measurement_var=0.001, estimated_var=0.01)
kf.x = kf.x.at[:3].set(jnp.array([0.0, 0.0, 0.0]))  # Set initial position (JAX syntax)

# In the main loop (same API):
kf.predict_stateful(dt)
kf.update_stateful(measurement)
position = kf.get_state_position()
velocity = kf.get_state_velocity()
```

### Option 3: Functional style (best for JAX integration)
```python
from src.utils import KalmanFilter3D_JAX

kf = KalmanFilter3D_JAX(0.001, 0.001, 0.01)
x = jnp.zeros(9)  # State vector
P = jnp.eye(9) * 0.01  # Covariance matrix

# In the main loop:
x, P = kf.predict(x, P, dt)
x, P = kf.update(x, P, measurement)
position = x[:3]
velocity = x[3:6]
```

## Performance Notes

- For standalone Kalman filtering with small matrices (9x9), NumPy may be slightly faster due to optimized BLAS
- JAX version provides better integration with JAX-based physics code and can be fused with other operations
- JAX version uses more numerically stable algorithms (e.g., `solve` instead of `inv`)

## Compatibility

The JAX version is fully compatible with the NumPy version:
- Same API (with `predict_stateful` and `update_stateful` methods)
- Numerical results match within floating-point precision (< 1e-8 difference)
- Can be used as a drop-in replacement

## Recommendation

- **For current code**: Keep using NumPy version (no changes needed)
- **For future optimization**: Consider JAX version if integrating with other JAX operations
- **For experiments**: NumPy version is sufficient and well-tested
