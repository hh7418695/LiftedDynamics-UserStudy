"""
render_worker.py - Long-lived rendering subprocess for the GUI.

Binds one UDP socket, reads object IDs from a command file,
renders each one, and writes result to a status file.

Communication protocol (via files, not pipes):
  - GUI writes object_id to COMMAND_FILE
  - Worker detects it, reads object_id, deletes file, starts rendering
  - Worker writes "ok"/"skip"/"quit" to RESULT_FILE when done
  - GUI polls RESULT_FILE

Usage (called by experiment_runner_gui.py):
    python render_worker.py --participant_id TEST --signal_dir <path>
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path
from datetime import datetime
from functools import partial

sys.path.insert(0, str(Path(__file__).parent))

# Redirect all stdout to stderr so main.py prints don't interfere
sys.stdout = sys.stderr

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

jax.config.update("jax_enable_x64", True)

from src.md import predition
from src.utils import KalmanFilter3D, plot_trajectory, const_gravity_acc
from src import lnn


def render_single_object(sock, object_id, participant_id,
                         seed=42, dt=1.0e-4, stride=10, dim=3,
                         socket_bufferSize=512):
    """
    Corrected version of run_single_object that matches execute_hapticRendering()
    in main.py. Key fixes vs the original run_single_object:
      1. const_gravity_acc uses positive value from src/utils.py (9.81)
      2. force *= -1e-3 scaling applied (matching execute_hapticRendering)
      3. Kalman filter uses actual delta_time instead of fixed dt
      4. Sends final reply on stop signal so C++ doesn't block
    """
    const_numerical = 1e-10

    rng = jax.random.PRNGKey(seed)

    N = 5
    masses = np.array([40.0] * N)
    object_mass = np.sum(masses)
    length = np.array([0.05] * (N - 1))

    stretching_val_list = np.linspace(50, 1000, num=5)
    ks_val = stretching_val_list[object_id - 1] if 1 <= object_id <= 5 else stretching_val_list[2]
    object_stiffness_stretching_sim = np.array([ks_val] * N) * 1e3

    bending_val_list = np.linspace(0, 0.1, num=5)
    kb_val = bending_val_list[object_id - 6] if 6 <= object_id <= 10 else bending_val_list[2]
    object_stiffness_bending_sim = np.array([kb_val] * N) * 1e3

    object_damping_scale = 0.01 * 1e3
    object_damping_sim = np.array([0.1] * N) * object_damping_scale

    VirtualCoupling_stiffness = 750 * 1e3
    VirtualCoupling_damping = 25 * 1e3

    # ── JIT helpers ───────────────────────────────────────────────────────
    @jit
    def get_angle(vec1, vec2):
        v1 = vec1 / (jnp.linalg.norm(vec1) + const_numerical)
        v2 = vec2 / (jnp.linalg.norm(vec2) + const_numerical)
        return jnp.arccos(jnp.clip(jnp.dot(v1, v2), -1.0, 1.0))

    @jit
    def getVector_relativeVelocity(vel_from, pos_from, vel_to, pos_to):
        pos_rel = pos_to - pos_from
        vel_rel = vel_to - vel_from
        dist = jnp.sqrt(jnp.square(pos_rel).sum())
        return vel_rel / (dist + const_numerical)

    @jit
    def get_squaredDistance(x, x_lead, original_length):
        x_diff = x - x_lead
        direction = x_diff / (jnp.linalg.norm(x_diff) + const_numerical)
        return jnp.square(x_diff - original_length * direction).sum()

    @jit
    def getEnergy_stretching(x, x_lead, stiffness, length):
        return 0.5 * stiffness * get_squaredDistance(x, x_lead, length)

    @jit
    def getEnergy_bending(angle, stiffness):
        return 0.5 * stiffness * (angle ** 2)

    @jit
    def getForce_virtualCoupling(x, v, x_lead, v_lead):
        x_diff = x[0, :] - x_lead[0, :]
        v_diff = v[0, :] - v_lead[0, :]
        force = -VirtualCoupling_stiffness * x_diff - VirtualCoupling_damping * v_diff
        force_gravity = jnp.zeros(dim)
        force_gravity = force_gravity.at[2].set(object_mass * const_gravity_acc)
        force += force_gravity
        return force

    @jit
    def getCoordinate_TouchX2Python(array):
        c = array.reshape(dim)
        return jnp.array([c[0], -c[2], c[1]]).reshape(array.shape)

    @jit
    def getCoordinate_Python2TouchX(array):
        c = array.reshape(dim)
        return jnp.array([c[0], c[2], -c[1]]).reshape(array.shape)

    # ── Physics ───────────────────────────────────────────────────────────
    def sim_bendingEnergy(x):
        result = 0.0
        for i in range(2, N):
            angle = get_angle(x[i-1] - x[i-2], x[i] - x[i-1])
            result += getEnergy_bending(angle, object_stiffness_bending_sim[i])
        return result

    def sim_stretchingEnergy(x):
        result = 0.0
        for i in range(1, N):
            result += getEnergy_stretching(x[i], x[i-1], object_stiffness_stretching_sim[i], length[i-1])
        return result

    def sim_gravityEnergy(x):
        return (masses * const_gravity_acc * x[:, 2]).sum()

    def sim_potentialEnergy(x):
        return sim_gravityEnergy(x) + sim_bendingEnergy(x) + sim_stretchingEnergy(x)

    sim_kineticEnergy = partial(lnn._T, mass=masses)

    def sim_Lagrangian(x, v, params):
        return sim_kineticEnergy(v) - sim_potentialEnergy(x)

    def sim_externalForce(x, v, x_lead, v_lead, params):
        result = jnp.zeros((N, dim))
        force_vc = getForce_virtualCoupling(x=x, v=v, x_lead=x_lead, v_lead=v_lead)
        result = result.at[0, :].set(force_vc)
        return result.reshape(-1, 1)

    def sim_damping(x, v, x_lead, v_lead, params):
        vel_rel = jnp.zeros((N, dim))
        for i in range(1, N):
            vel_rel = vel_rel.at[i, :].set(
                getVector_relativeVelocity(v[i-1], x[i-1], v[i], x[i])
            )
        return (-vel_rel * object_damping_sim.reshape((N, 1))).reshape(-1, 1)

    def sim_constraints(x, v, params):
        return jnp.zeros((1, N * dim))

    sim_acceleration = jit(
        lnn.accelerationFull(
            N, dim,
            lagrangian=sim_Lagrangian,
            non_conservative_forces=sim_damping,
            external_force=sim_externalForce,
            constraints=sim_constraints,
        )
    )

    def sim_totalForce(R, V, R_lead, V_lead, params, mass):
        acc = sim_acceleration(R, V, R_lead, V_lead, params)
        return acc if mass is None else acc * mass.reshape(-1, 1)

    R_init = jnp.zeros((N, dim))
    for i in range(N):
        R_init = R_init.at[i, 2].set(-sum(length[:i]))
    V_init = jnp.zeros((N, dim))

    @partial(jax.jit, static_argnames=['runs'])
    def sim_nextState(R, V, R_lead, V_lead, runs):
        return predition(R, V, R_lead, V_lead, None, sim_totalForce,
                         lambda R, dR, V: (R + dR, V), dt, masses, runs, stride)

    # ── Result path ───────────────────────────────────────────────────────
    result_dir = (Path("../results") / f"Participant-{participant_id}" /
                  f"Obj-{object_id}" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── Rendering loop ────────────────────────────────────────────────────
    object_position, object_velocity = R_init, V_init
    kf = KalmanFilter3D(0.001, 0.001, 0.01)
    kf.x[:3] = [0.0, 0.0, 0.0]

    buffer_size = 1000 * 20
    buffer_interval = 5
    buffer_object_position = np.zeros((buffer_size, N, dim))
    buffer_user_position   = np.zeros((buffer_size, 1, dim))
    buffer_force           = np.zeros((buffer_size, 1, dim))
    buffer_time            = np.zeros(buffer_size)

    com_count = -1
    buffer_count = -1
    hist_time = []
    start_time_global = time.time()

    while True:
        com_count += 1

        data_byte, address = sock.recvfrom(socket_bufferSize)
        data_dict = json.loads(data_byte.decode("utf-8"))
        data_position  = data_dict.get("position", [])
        data_timestamp = data_dict.get("timestamp")

        if data_timestamp < 0:
            # Send final reply so C++ doesn't block on recvfrom
            stop_reply = json.dumps(dict(force=[0.0, 0.0, 0.0])).encode("utf-8")
            sock.sendto(stop_reply, address)
            break

        diff_time = time.time() - start_time_global
        hist_time.append(diff_time)

        user_position = np.array([data_position]) * 1e-3
        user_position = getCoordinate_TouchX2Python(user_position)

        if com_count <= 1:
            user_position = np.zeros((1, dim))
            user_velocity = np.zeros((1, dim))
        else:
            hist_time_np = np.array(hist_time).reshape(-1)
            delta_time = hist_time_np[com_count] - hist_time_np[com_count - 1]
            kf.predict(delta_time)
            kf.update(np.array(user_position).reshape(dim))
            state = kf.get_state()
            user_position = state[:3].reshape(1, dim)
            user_velocity = state[3:6].reshape(1, dim)

        force = getForce_virtualCoupling(
            x=object_position, v=object_velocity,
            x_lead=user_position, v_lead=user_velocity,
        )
        force *= -1e-3
        force = np.array(force)

        force_max = np.max(np.abs(force))
        if force_max > 2.5:
            force = force / force_max * 2.5

        predict_traj = sim_nextState(
            R=object_position, V=object_velocity,
            R_lead=user_position, V_lead=user_velocity,
            runs=1,
        )
        object_position = predict_traj.position.reshape(N, dim)
        object_velocity = predict_traj.velocity.reshape(N, dim)

        output_dict = dict(force=getCoordinate_Python2TouchX(force).tolist())
        sock.sendto(json.dumps(output_dict).encode("utf-8"), address)

        if com_count % buffer_interval == 0 and buffer_count + 1 < buffer_size:
            buffer_count += 1
            buffer_object_position[buffer_count] = object_position
            buffer_user_position[buffer_count]   = user_position
            buffer_force[buffer_count]            = force
            buffer_time[buffer_count]             = hist_time[-1]

    # ── Save CSV ──────────────────────────────────────────────────────────
    import csv
    csv_path = result_dir / "render_hist.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["Timestamp (s)",
                  "Rendered Force X (N)", "Rendered Force Y (N)", "Rendered Force Z (N)",
                  "User Position X (m)", "User Position Y (m)", "User Position Z (m)"]
        for n in range(N):
            for ax in ["X", "Y", "Z"]:
                header.append(f"Node {n} Position {ax} (m)")
        writer.writerow(header)
        for k in range(buffer_count + 1):
            row = [buffer_time[k]]
            row += buffer_force[k, 0, :].tolist()
            row += buffer_user_position[k, 0, :].tolist()
            for n in range(N):
                row += buffer_object_position[k, n, :].tolist()
            writer.writerow(row)
    print(f"  Render history saved -> {csv_path}", file=sys.stderr)

    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant_id", default="TEST")
    parser.add_argument("--server_ip", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=12312)
    parser.add_argument("--signal_dir", required=True)
    args = parser.parse_args()

    signal_dir = Path(args.signal_dir)
    cmd_file = signal_dir / "command.txt"
    result_file = signal_dir / "result.txt"
    ready_file = signal_dir / "ready.txt"

    for f in [cmd_file, result_file, ready_file]:
        f.unlink(missing_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.server_ip, args.server_port))

    ready_file.write_text("ready")
    print("render_worker: socket bound, ready", file=sys.stderr, flush=True)

    try:
        while True:
            if cmd_file.exists():
                cmd = cmd_file.read_text().strip()
                cmd_file.unlink(missing_ok=True)

                if cmd == "quit":
                    break

                try:
                    object_id = int(cmd)
                except ValueError:
                    result_file.write_text("error_bad_input")
                    continue

                print(f"render_worker: rendering object {object_id}", file=sys.stderr, flush=True)
                result = render_single_object(
                    sock=sock,
                    object_id=object_id,
                    participant_id=args.participant_id,
                )
                print(f"render_worker: done, result={result}", file=sys.stderr, flush=True)
                result_file.write_text(str(result))
            else:
                time.sleep(0.05)
    finally:
        sock.close()
        for f in [cmd_file, result_file, ready_file]:
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
