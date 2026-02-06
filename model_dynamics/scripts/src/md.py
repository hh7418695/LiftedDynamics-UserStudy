"""
Molecular Dynamics utilities for Lagrangian mechanics simulation.

This module provides functions for:
- Dynamics generation and prediction
- Trajectory solving using JAX's scan and fori_loop
- Energy minimization
- Boundary conditions (open, periodic, reflective)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.example_libraries import optimizers

from .nve import nve_DIY


def dynamics_generator(ensemble, force_fn, shift_fn, params, dt, mass):
    """Generate dynamics functions for a given ensemble.

    Args:
        ensemble: Ensemble function (e.g., nve_DIY)
        force_fn: Force function
        shift_fn: Shift function for boundary conditions
        params: System parameters
        dt: Time step
        mass: Particle masses

    Returns:
        init: Initialization function
        f: Dynamics function that solves for multiple steps
    """
    func = partial(force_fn, mass=mass)
    init, apply = ensemble(lambda R, V: func(R, V, params), shift_fn, dt)

    def f(state, runs=100, stride=10):
        return solve_dynamics(
            state, apply, runs=runs, stride=stride)

    return init, f


def predition(R, V, R_lead, V_lead, params, force_fn, shift_fn, dt, mass, runs=1000, stride=10):
    """Predict system trajectory with leader node tracking.

    This is the main prediction function used in haptic rendering. It integrates
    the equations of motion for the object nodes while tracking the leader node
    (user's haptic stylus position).

    Args:
        R: Initial positions of object nodes (N, dim)
        V: Initial velocities of object nodes (N, dim)
        R_lead: Leader node position (1, dim)
        V_lead: Leader node velocity (1, dim)
        params: System parameters
        force_fn: Force function that computes forces on nodes
        shift_fn: Shift function for boundary conditions
        dt: Time step
        mass: Node masses
        runs: Number of trajectory points to return
        stride: Number of integration steps between saved points

    Returns:
        states: Trajectory of NVEState_DIY objects
    """
    func = partial(force_fn, mass=mass)
    init, apply = nve_DIY(lambda R, V, R_lead, V_lead: func(R, V, R_lead, V_lead, params), shift_fn, dt)
    state = init(R, V, R_lead, V_lead, mass)
    states = solve_dynamics(state, apply, runs=runs, stride=stride)
    return states


def solve_dynamics(init_state, apply, runs=100, stride=10):
    """Solve dynamics using JAX's scan for efficient trajectory computation.

    This function uses JAX's scan operation to efficiently compute trajectories
    by JIT-compiling the integration loop. The stride parameter allows multiple
    integration steps between saved trajectory points.

    Args:
        init_state: Initial state (NVEState or NVEState_DIY)
        apply: Apply function from ensemble (performs one integration step)
        runs: Number of trajectory points to save
        stride: Number of integration steps between saved points

    Returns:
        traj: Trajectory as array of states (runs,)
    """
    step = jit(lambda i, state: apply(state))

    def f(state):
        y = jax.lax.fori_loop(0, stride, step, state)
        return y, y

    def func(state, i):
        return f(state)

    @jit
    def scan(init_state):
        return jax.lax.scan(func, init_state, jnp.array(range(runs)))

    final_state, traj = scan(init_state)
    return traj


def minimize(R, params, shift, pot_energy_fn, steps=10, gtol=1.0e-7, lr=1.0e-3):
    """Minimize potential energy using Adam optimizer.

    Args:
        R: Initial positions
        params: System parameters
        shift: Shift function for boundary conditions
        pot_energy_fn: Potential energy function
        steps: Maximum number of optimization steps
        gtol: Gradient tolerance for convergence
        lr: Learning rate

    Returns:
        R: Optimized positions
    """
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(R)

    def gloss2(R):
        return value_and_grad(lambda R: pot_energy_fn(R, params))(R)

    print(f"Step\tPot. Eng.\t\tTolerance")
    for i in range(steps):
        v, grads_ = gloss2(R)
        grads = jnp.clip(jnp.nan_to_num(grads_), a_min=-1.0, a_max=1.0)
        opt_state = opt_update(0, grads, opt_state)
        R_ = get_params(opt_state)
        dR = R_ - R
        R, _ = shift(R, dR, R)
        if i % 100 == 0:
            _tol = jnp.square(grads).sum()
            print(f"{i}\t{v}\t\t{_tol}")
            if _tol < gtol:
                print(f"gtol reached: {_tol} which is < {gtol}")
                break
    return R


# Boundary Conditions
# ============================================================================

def _reflective(R, dR, V, _min=0.0, _max=4.0):
    """Reflective boundary conditions.

    Particles bounce off boundaries with velocity reversal.

    Args:
        R: Current positions
        dR: Displacement
        V: Current velocities
        _min: Minimum boundary
        _max: Maximum boundary

    Returns:
        R_: New positions
        V_: New velocities (reversed if boundary hit)
    """
    V_ = V
    R_ = R
    dR_ = jnp.maximum(jnp.minimum(dR, (_max - _min) / 2), -(_max - _min) / 2)
    V_ = jnp.where(R + dR_ < _min, -V, V)
    V_ = jnp.where(R + dR_ > _max, -V, V_)
    R_ = jnp.where(R + dR_ < _min, 2 * _min - (R + dR_), R + dR_)
    R_ = jnp.where(R + dR_ > _max, 2 * _max - (R + dR_), R_)
    return R_, V_


def _periodic(R, dR, V, _min=0.0, _max=4.0):
    """Periodic boundary conditions.

    Particles wrap around when crossing boundaries.

    Args:
        R: Current positions
        dR: Displacement
        V: Current velocities
        _min: Minimum boundary
        _max: Maximum boundary

    Returns:
        R_: New positions (wrapped)
        V_: New velocities (unchanged)
    """
    V_ = V
    R_ = R
    dR_ = jnp.maximum(jnp.minimum(dR, (_max - _min) / 2), -(_max - _min) / 2)
    R_ = jnp.where(R + dR_ < _min, _max - _min + (R + dR_), R + dR_)
    R_ = jnp.where(R + dR_ > _max, _min - _max + (R + dR_), R_)
    return R_, V_


def _open(R, dR, V):
    """Open boundary conditions (no boundaries).

    Particles move freely without any boundary constraints.

    Args:
        R: Current positions
        dR: Displacement
        V: Current velocities

    Returns:
        R_: New positions (R + dR)
        V_: New velocities (unchanged)
    """
    return R + dR, V


# Default shift function (open boundaries)
shift = _open


def displacement(a, b):
    """Compute displacement vector from b to a.

    Args:
        a: Vector A
        b: Vector B

    Returns:
        Displacement vector (a - b)
    """
    return a - b
