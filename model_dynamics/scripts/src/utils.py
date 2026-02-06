import importlib
from functools import partial

import jax
import jax.numpy as jnp
import jax_md
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jax_md import smap
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

from . import lnn, models

const_numerical = 2e-7  # note that linalg also has numerical problem (because of float precision)
const_gravity_acc = 9.81


def colnum(i, j, N):
    """Gives linear index for upper triangle matrix.
    """
    assert (j >= i), "j >= i, Upper Triangle indices."
    assert (i < N) and (j < N), "i<N & j<N where i and \
            j are atom type and N is number of species."
    return int(i * N - i * (i - 1) / 2 + j - i + 1)


def pair2mat(
        fn, displacement_or_metric, species, parameters,
        ignore_unused_parameters=True,
        reduce_axis=None,
        keepdims=False,
        use_onehot=False,
        **kwargs
        ):
    kwargs, param_combinators = smap._split_params_and_combinators(kwargs)

    merge_dicts = partial(
        jax_md.util.merge_dicts,
        ignore_unused_parameters=ignore_unused_parameters
        )
    d = lnn.t1(displacement=displacement_or_metric)
    if species is None:
        def fn_mapped(R: smap.Array, **dynamic_kwargs) -> smap.Array:
            _kwargs = merge_dicts(kwargs, dynamic_kwargs)
            _kwargs = smap._kwargs_to_parameters(
                None, _kwargs, param_combinators
                )
            dr = d(R)
            # NOTE(schsam): Currently we place a diagonal mask no matter what function
            # we are mapping. Should this be an option?
            return smap.high_precision_sum(
                fn(dr, **_kwargs),
                axis=reduce_axis, keepdims=keepdims
                ) * smap.f32(0.5)

    elif jax_md.util.is_array(species):
        species = np.array(species)
        smap._check_species_dtype(species)
        species_count = int(np.max(species) + 1)
        if reduce_axis is not None or keepdims:
            raise ValueError

        def onehot(i, j, N):
            col = colnum(i, j, species_count)
            oneh = jnp.zeros(
                (N, colnum(species_count - 1, species_count - 1, species_count))
                )
            # Updated from deprecated jax.ops.index_update to modern .at[] syntax
            oneh = oneh.at[:, int(col - 1)].set(1)
            return oneh

        def pot_pair_wise():
            if use_onehot:
                def func(i, j, dr, **s_kwargs):
                    dr = jnp.linalg.norm(dr, axis=1, keepdims=True)
                    ONEHOT = onehot(i, j, len(dr))
                    h = vmap(
                        models.forward_pass, in_axes=(
                            None, 0)
                        )(parameters["ONEHOT"], ONEHOT)
                    dr = jnp.concatenate([h, dr], axis=1)
                    return smap.high_precision_sum(
                        fn(dr, params=parameters["PEF"], **s_kwargs)
                        )

                return func
            else:
                def func(i, j, dr, **s_kwargs):
                    return smap.high_precision_sum(
                        fn(dr, **parameters[i][j - i], **s_kwargs)
                        )

                return func

        pot_pair_wise_fn = pot_pair_wise()

        def fn_mapped(R, **dynamic_kwargs):
            U = smap.f32(0.0)
            for i in range(species_count):
                for j in range(i, species_count):
                    _kwargs = merge_dicts(kwargs, dynamic_kwargs)
                    s_kwargs = smap._kwargs_to_parameters(
                        (i, j), _kwargs, param_combinators
                        )
                    Ra = R[species == i]
                    Rb = R[species == j]
                    if j == i:
                        dr = d(Ra)
                        dU = pot_pair_wise_fn(i, j, dr, **s_kwargs)
                        U = U + smap.f32(0.5) * dU
                    else:
                        dr = vmap(
                            vmap(displacement_or_metric, in_axes=(0, None)), in_axes=(
                                None, 0)
                            )(Ra, Rb).reshape(-1, Ra.shape[1])
                        dU = pot_pair_wise_fn(i, j, dr, **s_kwargs)
                        U = U + dU
            return U
    return fn_mapped


def map_parameters(fn, displacement, species, parameters, **kwargs):
    mapped_fn = lnn.MAP(fn)

    def f(x, *args, **kwargs):
        out = mapped_fn(x, *args, **kwargs)
        return out

    return pair2mat(f, displacement, species, parameters, **kwargs)


class VV_unroll():
    def __init__(self, R, dt=1):
        self.R = R
        self.dt = dt

    def get_position(self):
        r = self.R[1:-1]
        return r

    def get_acceleration(self, dt=None):
        r = self.R[1:-1]
        r_minus = self.R[:-2]
        r_plus = self.R[2:]
        if dt is not None:
            return (r_plus + r_minus - 2 * r) / dt ** 2
        else:
            return (r_plus + r_minus - 2 * r) / self.dt ** 2

    def get_velocity(self, dt=None):
        r_minus = self.R[:-2]
        r_plus = self.R[2:]
        if dt is not None:
            return (r_plus - r_minus) / 2 / dt
        else:
            return (r_plus - r_minus) / 2 / self.dt

    def get_kin(self, dt=None):
        return self.get_position(), self.get_velocity(dt=dt), self.get_acceleration(dt=dt)


class States:
    def __init__(self, state=None, const_size=True):
        if state is None:
            self.isarrays = False
            self.const_size = const_size
            self.position = []
            self.velocity = []
            self.force = []
            if self.const_size:
                self.mass = None
            else:
                self.mass = []
        else:
            self.position = [state.position]
            self.velocity = [state.velocity]
            self.force = [state.force]
            if self.const_size:
                self.mass = state.mass
            else:
                self.mass = [state.mass]

    def add(self, state):
        self.position += [state.position]
        self.velocity += [state.velocity]
        self.force += [state.force]
        if self.const_size:
            if self.mass is None:
                self.mass = state.mass
        else:
            self.mass += [state.mass]

    def fromlist(self, states, const_size=True):
        out = States(const_size=const_size)
        for state in states:
            out.add(state)
        return out

    def makearrays(self):
        if not (self.isarrays):
            self.position = jnp.array(self.position)
            self.velocity = jnp.array(self.velocity)
            self.force = jnp.array(self.force)
            self.mass = jnp.array([self.mass])
            self.isarrays = True

    def get_array(self):
        self.makearrays()
        return self.position, self.velocity, self.force

    def get_mass(self):
        self.makearrays()
        return self.mass

    def get_kin(self):
        self.makearrays()
        if self.const_size:
            acceleration = self.force / self.mass.reshape(1, self.mass.shape)
        else:
            acceleration = self.force / self.mass
        return self.position, self.velocity, acceleration


def reload(list_of_modules):
    for module in list_of_modules:
        try:
            print("Reload: ", module.__name__)
            importlib.reload(module)
        except:
            print("Reimports failed.")


def timeit(stmt, setup="", number=5):
    from timeit import timeit
    return timeit(stmt=stmt, setup=setup, number=number)


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def nCk(n, k):
    return factorial(n) // factorial(k) // factorial(n - k)


def plot_trajectory(
        data_traj, N, data_num, data_time, filepath,
        data_compare=None, data_force=None, data_virtual=None,
        show_history=False, show_structure=True, show_skip=20, show_fps=20, show_dataDetail=False,
        multiChain=False, modelReduction=False, connectLength=None, N_chain=None
        ):
    """
    Generate and save a 3D animation visualizing the trajectory of a chain-like articulated object (or multiple
    chains) over time. Supports optional comparison trajectories, virtual reference nodes, applied force
    visualization, and configurable rendering options.

    Notes:
        - Ensures equal scaling across X/Y/Z axes for correct spatial interpretation.
        - If data_virtual is provided, the first edge references the virtual node instead of origin.
        - For multiChain=True, inter-chain connection lines are rendered unless modelReduction=True.
        - Figure is closed after saving to prevent unsolicited interactive display.

    Parameters:
        data_traj (ndarray): Array of shape (T, N, 3), main trajectory over time.
        N (int): Total number of nodes to render.
        data_num (int): Raw number of timestamps before skipping.
        data_time (ndarray): Array of shape (T,), timestamps for each frame.
        filepath (str): Destination path to save the output animation file (required to have ".gif" suffix).
        data_compare (ndarray, optional): Comparison trajectory of the same shape as data_traj (usually used for
            ground truth data).
        data_force (ndarray, optional): Array of shape (T, 3), force vectors applied at the top object node over time.
        data_virtual (ndarray, optional): Virtual leading node positions of shape (T, 1, 3).
        show_history (bool): If True, plots full trajectory history traces per node.
        show_structure (bool): If True, plots frame-by-frame structure (nodes and edges).
        show_skip (int): Frame skipping factor to downsample temporal data.
        show_fps (int): Frames per second for the saved animation.
        show_dataDetail (bool): If True, includes edge length and force info in frame titles.
        multiChain (bool): If True, handles rendering of two connected chains.
        modelReduction (bool): If True, suppresses rendering of the second chain connections.
        connectLength (list or ndarray, optional): Offset(s) for second chain positioning.
        N_chain (int, optional): Number of nodes in each chain (defaults to N when None).
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    if N_chain is None:
        N_chain = N
    if multiChain:
        if connectLength is None:
            raise ValueError("Need to input connectLength!")

    # skip data
    if show_skip is not None:
        data_num = data_num // show_skip
        data_traj_temp = np.zeros((data_num, data_traj.shape[1], data_traj.shape[2]))
        data_time_temp = np.zeros(data_num)
        for i in range(data_num):
            data_traj_temp[i, :, :] = data_traj[i * show_skip, :, :]
            data_time_temp[i] = data_time[i * show_skip]
        data_traj = data_traj_temp
        data_time = data_time_temp
        if data_compare is not None:
            data_compare_temp = np.zeros((data_num, data_compare.shape[1], data_compare.shape[2]))
            for i in range(data_num):
                data_compare_temp[i, :, :] = data_compare[i * show_skip, :, :]
            data_compare = data_compare_temp
        if data_force is not None:
            data_force_temp = np.zeros((data_num, data_force.shape[1]))
            for i in range(data_num):
                data_force_temp[i, :] = data_force[i * show_skip, :]
            data_force = data_force_temp
        if data_virtual is not None:
            data_virtual_temp = np.zeros((data_num, data_virtual.shape[1], data_virtual.shape[2]))
            for i in range(data_num):
                data_virtual_temp[i, :, :] = data_virtual[i * show_skip, :, :]
            data_virtual = data_virtual_temp

    if data_virtual is not None:
        if multiChain:
            data_virtual2 = np.zeros_like(data_virtual)
            for i_time in range(data_num):
                data_virtual2[i_time, 0, 0] = data_virtual[i_time, 0, 0] + connectLength[0]
                data_virtual2[i_time, 0, 1] = data_virtual[i_time, 0, 1]
                data_virtual2[i_time, 0, 2] = data_virtual[i_time, 0, 2]

    # define 3D view limitation
    if data_compare is None:
        limit_refer = data_traj
    else:
        limit_refer = data_compare
    x_max = float('-inf')
    x_min = float('inf')
    y_max = float('-inf')
    y_min = float('inf')
    z_max = float('-inf')
    z_min = float('inf')
    for i_N in range(N):
        x_max = max(x_max, max(limit_refer[:, i_N, 0]))
        x_min = min(x_min, min(limit_refer[:, i_N, 0]))
        y_max = max(y_max, max(limit_refer[:, i_N, 1]))
        y_min = min(y_min, min(limit_refer[:, i_N, 1]))
        z_max = max(z_max, max(limit_refer[:, i_N, 2]))
        z_min = min(z_min, min(limit_refer[:, i_N, 2]))
    scale = x_max - x_min
    scale = max(scale, y_max - y_min)
    scale = max(scale, z_max - z_min)
    x_min_scaled = (x_max + x_min) / 2.0 - scale / 2.0
    x_max_scaled = (x_max + x_min) / 2.0 + scale / 2.0
    y_min_scaled = (y_max + y_min) / 2.0 - scale / 2.0
    y_max_scaled = (y_max + y_min) / 2.0 + scale / 2.0
    z_min_scaled = (z_max + z_min) / 2.0 - scale / 2.0
    z_max_scaled = (z_max + z_min) / 2.0 + scale / 2.0
    ax.set_xlim(x_min_scaled, x_max_scaled)
    ax.set_ylim(y_min_scaled, y_max_scaled)
    ax.set_zlim(z_min_scaled, z_max_scaled)

    # compute the connectLength of each edge: for data
    edge_length = np.zeros((data_num, data_traj.shape[1]))
    for i_time in range(data_num):
        R = data_traj[i_time, :, :]
        if data_virtual is None:  # first chain, let first edge connectLength = 0
            edge_length[i_time, :N_chain] = np.sqrt(
                np.square(
                    R[:N_chain] -
                    np.vstack(
                        [np.zeros_like(R[0]),
                         R[:N_chain - 1]]
                        )
                    ).sum(axis=1)
                )
            if multiChain and (not modelReduction):  # second chain
                edge_length[i_time, N_chain:] = np.sqrt(
                    np.square(
                        R[N_chain:] -
                        np.vstack(
                            [np.zeros_like(R[0]),
                             R[N_chain:-1]]
                            )
                        ).sum(axis=1)
                    )
        else:  # first edge connectLength = distance between first node and virtual node
            # first chain
            edge_length[i_time, :N_chain] = np.sqrt(
                np.square(
                    R[:N_chain] -
                    np.vstack(
                        [data_virtual[i_time],
                         R[:N_chain - 1]]
                        )
                    ).sum(axis=1)
                )
            if multiChain and (not modelReduction):  # second chain
                edge_length[i_time, N_chain:] = np.sqrt(
                    np.square(
                        R[N_chain:] -
                        np.vstack(
                            [data_virtual2[i_time],
                             R[N_chain:-1]]
                            )
                        ).sum(axis=1)
                    )
    # compute the connectLength of each edge: for comparison
    if data_compare is not None:
        edge_length_compare = np.zeros((data_num, data_compare.shape[1]))
        for i_time in range(data_num):
            R = data_compare[i_time, :, :]
            if data_virtual is None:
                edge_length_compare[i_time, :N_chain] = np.sqrt(
                    np.square(
                        R[:N_chain] -
                        np.vstack(
                            [np.zeros_like(R[0]),
                             R[:N_chain - 1]]
                            )
                        ).sum(axis=1)
                    )
                if multiChain and (not modelReduction):
                    edge_length_compare[i_time, N_chain:] = np.sqrt(
                        np.square(
                            R[N_chain:] -
                            np.vstack(
                                [np.zeros_like(R[0]),
                                 R[N_chain:-1]]
                                )
                            ).sum(axis=1)
                        )
            else:
                edge_length_compare[i_time, :N_chain] = np.sqrt(
                    np.square(
                        R[:N_chain] -
                        np.vstack(
                            [data_virtual[i_time],
                             R[:N_chain - 1]]
                            )
                        ).sum(axis=1)
                    )
                if multiChain and (not modelReduction):
                    edge_length_compare[i_time, N_chain:] = np.sqrt(
                        np.square(
                            R[N_chain:] -
                            np.vstack(
                                [data_virtual2[i_time],
                                 R[N_chain:-1]]
                                )
                            ).sum(axis=1)
                        )

    if show_history:
        lines = [ax.plot([], [], [], 'b-')[0] for _ in range(N)]
        points = [ax.plot([], [], [], 'o')[0] for _ in range(N)]

    if show_structure:
        # data object structure
        if multiChain and (not modelReduction):
            lines = [ax.plot([], [], [], 'b-')[0] for _ in range(N + N_chain)]  # connection between 2 chains
        else:
            lines = [ax.plot([], [], [], 'b-')[0] for _ in range(N)]
        points = [ax.plot([], [], [], 'ko')[0] for _ in range(N)]

        # comparison object structure
        if data_compare is not None:
            if multiChain:  # ground truth has multiple chains
                lines_compare = [ax.plot([], [], [], 'm:')[0] for _ in range(N + N_chain)]
            else:
                lines_compare = [ax.plot([], [], [], 'm:')[0] for _ in range(N)]
            points_compare = [ax.plot([], [], [], 'r+')[0] for _ in range(N)]

        # leading force vector
        if data_force is not None:
            quivers_length_max = np.max(np.abs(data_force))
            quivers_length_target = np.max(edge_length[:, 1])
            quivers_scaling = quivers_length_target * 0.8 / (quivers_length_max + const_numerical)

            def get_quiver_args(i_time):
                x = data_traj[i_time, 0, 0]
                y = data_traj[i_time, 0, 1]
                z = data_traj[i_time, 0, 2]
                u = data_force[i_time, 0] * quivers_scaling
                v = data_force[i_time, 1] * quivers_scaling
                w = data_force[i_time, 2] * quivers_scaling
                return x, y, z, u, v, w

            if multiChain and (not modelReduction):
                def get_quiver_args2(i_time):
                    x = data_traj[i_time, N_chain, 0]
                    y = data_traj[i_time, N_chain, 1]
                    z = data_traj[i_time, N_chain, 2]
                    u = data_force[i_time, 0] * quivers_scaling
                    v = data_force[i_time, 1] * quivers_scaling
                    w = data_force[i_time, 2] * quivers_scaling
                    return x, y, z, u, v, w

            quivers = ax.quiver(*get_quiver_args(0))
            if multiChain and (not modelReduction):
                quivers2 = ax.quiver(*get_quiver_args2(0))

        # virtual leading point
        if data_virtual is not None:
            if multiChain:
                points_virtual = [ax.plot([], [], [], 'ro')[0] for _ in range(2)]
            else:
                points_virtual = [ax.plot([], [], [], 'ro')[0] for _ in range(1)]

    def init():
        if show_history:
            for line, point in zip(lines, points):
                line.set_data([], [])
                line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
            return lines + points
        if show_structure:
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            for point in points:
                point.set_data([], [])
                point.set_3d_properties([])
            if data_compare is not None:
                for line in lines_compare:
                    line.set_data([], [])
                    line.set_3d_properties([])
                for point in points_compare:
                    point.set_data([], [])
                    point.set_3d_properties([])
            if data_virtual is not None:
                for point in points_virtual:
                    point.set_data([], [])
                    point.set_3d_properties([])
            return (lines + points
                    + (lines_compare + points_compare if data_compare is not None else [])
                    + (points_virtual if data_virtual is not None else []))

    def update(num):
        nonlocal quivers
        if multiChain and (not modelReduction):
            nonlocal quivers2

        title = f"Time: {data_time[num]:.2f} s"
        if show_dataDetail:
            title += f"\nData: l0={edge_length[num, 0]:.4f}"
            for i in range(1, N):
                title += f",l{i}={edge_length[num, i]:.2f}"
            if data_compare is not None:
                title += f"\nComp: l0={edge_length_compare[num, 0]:.4f}"
                for i in range(1, N):
                    title += f",l{i}={edge_length_compare[num, i]:.2f}"
            if data_force is not None:
                title += (f"\nForce Feedback: x={data_force[num, 0]:.2f},y={data_force[num, 1]:.2f},z="
                          f"{data_force[num, 2]:.2f}")
        ax.set_title(title)

        if show_history:
            for i, (line, point) in enumerate(zip(lines, points)):
                line.set_data(data_traj[:num, i, 0], data_traj[:num, i, 1])
                line.set_3d_properties(data_traj[:num, i, 2])
                point.set_data(data_traj[num - 1:num, i, 0], data_traj[num - 1:num, i, 1])
                point.set_3d_properties(data_traj[num - 1:num, i, 2])
            return lines + points

        if show_structure:
            # lines
            for i in range(N_chain):  # first chain (data object + comparison object)
                if i == 0:
                    if data_virtual is not None:  # point to virtual leading point
                        x_data = [data_virtual[num, 0, 0], data_traj[num, i, 0]]
                        y_data = [data_virtual[num, 0, 1], data_traj[num, i, 1]]
                        z_data = [data_virtual[num, 0, 2], data_traj[num, i, 2]]
                    else:  # point to (0,0,0)
                        x_data = [0, data_traj[num, i, 0]]
                        y_data = [0, data_traj[num, i, 1]]
                        z_data = [0, data_traj[num, i, 2]]
                else:
                    x_data = [data_traj[num, i - 1, 0], data_traj[num, i, 0]]
                    y_data = [data_traj[num, i - 1, 1], data_traj[num, i, 1]]
                    z_data = [data_traj[num, i - 1, 2], data_traj[num, i, 2]]
                lines[i].set_data(x_data, y_data)
                lines[i].set_3d_properties(z_data)
                if data_compare is not None:
                    if i == 0:
                        if data_virtual is not None:
                            x_data_compare = [data_virtual[num, 0, 0], data_compare[num, i, 0]]
                            y_data_compare = [data_virtual[num, 0, 1], data_compare[num, i, 1]]
                            z_data_compare = [data_virtual[num, 0, 2], data_compare[num, i, 2]]
                        else:
                            x_data_compare = [0, data_compare[num, i, 0]]
                            y_data_compare = [0, data_compare[num, i, 1]]
                            z_data_compare = [0, data_compare[num, i, 2]]
                    else:
                        x_data_compare = [data_compare[num, i - 1, 0], data_compare[num, i, 0]]
                        y_data_compare = [data_compare[num, i - 1, 1], data_compare[num, i, 1]]
                        z_data_compare = [data_compare[num, i - 1, 2], data_compare[num, i, 2]]
                    lines_compare[i].set_data(x_data_compare, y_data_compare)
                    lines_compare[i].set_3d_properties(z_data_compare)
            if multiChain:
                # second chain
                for i in range(N_chain, N):
                    if not modelReduction:  # second data object chain
                        if i == N_chain:
                            if data_virtual is not None:
                                x_data = [data_virtual2[num, 0, 0], data_traj[num, i, 0]]
                                y_data = [data_virtual2[num, 0, 1], data_traj[num, i, 1]]
                                z_data = [data_virtual2[num, 0, 2], data_traj[num, i, 2]]
                            else:
                                x_data = [0, data_traj[num, i, 0]]
                                y_data = [0, data_traj[num, i, 1]]
                                z_data = [0, data_traj[num, i, 2]]
                        else:
                            x_data = [data_traj[num, i - 1, 0], data_traj[num, i, 0]]
                            y_data = [data_traj[num, i - 1, 1], data_traj[num, i, 1]]
                            z_data = [data_traj[num, i - 1, 2], data_traj[num, i, 2]]
                        lines[i].set_data(x_data, y_data)
                        lines[i].set_3d_properties(z_data)
                    if data_compare is not None:  # second comparison object chain
                        if i == N_chain:
                            if data_virtual is not None:
                                x_data_compare = [data_virtual2[num, 0, 0], data_compare[num, i, 0]]
                                y_data_compare = [data_virtual2[num, 0, 1], data_compare[num, i, 1]]
                                z_data_compare = [data_virtual2[num, 0, 2], data_compare[num, i, 2]]
                            else:
                                x_data_compare = [0, data_compare[num, i, 0]]
                                y_data_compare = [0, data_compare[num, i, 1]]
                                z_data_compare = [0, data_compare[num, i, 2]]
                        else:
                            x_data_compare = [data_compare[num, i - 1, 0], data_compare[num, i, 0]]
                            y_data_compare = [data_compare[num, i - 1, 1], data_compare[num, i, 1]]
                            z_data_compare = [data_compare[num, i - 1, 2], data_compare[num, i, 2]]
                        lines_compare[i].set_data(x_data_compare, y_data_compare)
                        lines_compare[i].set_3d_properties(z_data_compare)
                # between chains
                for i in range(N_chain):
                    if not modelReduction:  # data object
                        x_data = [data_traj[num, i, 0], data_traj[num, i + N_chain, 0]]
                        y_data = [data_traj[num, i, 1], data_traj[num, i + N_chain, 1]]
                        z_data = [data_traj[num, i, 2], data_traj[num, i + N_chain, 2]]
                        lines[i + N].set_data(x_data, y_data)
                        lines[i + N].set_3d_properties(z_data)
                    if data_compare is not None:  # comparison object
                        x_data_compare = [data_compare[num, i, 0], data_compare[num, i + N_chain, 0]]
                        y_data_compare = [data_compare[num, i, 1], data_compare[num, i + N_chain, 1]]
                        z_data_compare = [data_compare[num, i, 2], data_compare[num, i + N_chain, 2]]
                        lines_compare[i + N].set_data(x_data_compare, y_data_compare)
                        lines_compare[i + N].set_3d_properties(z_data_compare)

            # object points
            for i in range(N_chain):  # first chain (data object + comparison object)
                points[i].set_data([data_traj[num, i, 0]], [data_traj[num, i, 1]])
                points[i].set_3d_properties([data_traj[num, i, 2]])
                if data_compare is not None:
                    points_compare[i].set_data([data_compare[num, i, 0]], [data_compare[num, i, 1]])
                    points_compare[i].set_3d_properties([data_compare[num, i, 2]])
            if multiChain:  # second chain
                for i in range(N_chain, N):
                    # data object
                    if not modelReduction:
                        points[i].set_data([data_traj[num, i, 0]], [data_traj[num, i, 1]])
                        points[i].set_3d_properties([data_traj[num, i, 2]])
                    # comparison object
                    if data_compare is not None:
                        points_compare[i].set_data([data_compare[num, i, 0]], [data_compare[num, i, 1]])
                        points_compare[i].set_3d_properties([data_compare[num, i, 2]])

            # virtual leading points
            if data_virtual is not None:
                if multiChain:
                    points_virtual[0].set_data([data_virtual[num, 0, 0]], [data_virtual[num, 0, 1]])
                    points_virtual[0].set_3d_properties([data_virtual[num, 0, 2]])
                    points_virtual[1].set_data([data_virtual2[num, 0, 0]], [data_virtual2[num, 0, 1]])
                    points_virtual[1].set_3d_properties([data_virtual2[num, 0, 2]])
                else:
                    points_virtual[0].set_data([data_virtual[num, 0, 0]], [data_virtual[num, 0, 1]])
                    points_virtual[0].set_3d_properties([data_virtual[num, 0, 2]])

            # leading force vector
            if data_force is not None:
                quivers.remove()
                quivers = ax.quiver(*get_quiver_args(num))
                if multiChain and (not modelReduction):
                    quivers2.remove()
                    quivers2 = ax.quiver(*get_quiver_args2(num))

            return (lines + points
                    + (lines_compare + points_compare if data_compare is not None else [])
                    + (points_virtual if data_virtual is not None else []))

    ani = FuncAnimation(fig, update, frames=data_num, init_func=init, blit=True)
    ani.save(filepath, writer=PillowWriter(fps=show_fps))
    plt.close(fig)  # not displayed


class KalmanFilter3D:
    """
    A 3D Kalman Filter implementation for tracking position, velocity, and acceleration in Cartesian space. The
    filter maintains a 9-dimensional state vector: [x, y, z, vx, vy, vz, ax, ay, az], representing position,
    velocity, and acceleration along each axis. It supports prediction and update phases using configurable process
    noise, measurement noise, and initial estimation covariance. The filter assumes that measurements provide only
    positional information, and it updates the internal state estimate and covariance matrix accordingly.
    """

    def __init__(self, process_var, measurement_var, estimated_var):
        """
        Initialize the Kalman filter with noise and covariance parameters.

        Parameters:
            process_var (float): Variance applied to the process noise for all state components.
            measurement_var (float): Variance of the measurement noise for positional inputs.
            estimated_var (float): Initial variance for the error covariance matrix P.
        """
        # Initialize the state vector [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros(9)  # Initial state estimate
        self.P = np.eye(9) * estimated_var  # Initial error covariance

        # State transition matrix (will be updated with dt)
        self.F = np.eye(9)

        # Measurement matrix
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Process noise covariance
        self.Q = np.zeros((9, 9))
        self.Q[0, 0] = process_var  # Position noise variance
        self.Q[1, 1] = process_var  # Position noise variance
        self.Q[2, 2] = process_var  # Position noise variance
        self.Q[3, 3] = process_var  # Velocity noise variance
        self.Q[4, 4] = process_var  # Velocity noise variance
        self.Q[5, 5] = process_var  # Velocity noise variance
        self.Q[6, 6] = process_var  # Acceleration noise variance
        self.Q[7, 7] = process_var  # Acceleration noise variance
        self.Q[8, 8] = process_var  # Acceleration noise variance

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_var

    def predict(self, dt):
        # Update state transition matrix with new dt
        self.F[0, 3] = dt
        self.F[0, 6] = 0.5 * dt ** 2
        self.F[1, 4] = dt
        self.F[1, 7] = 0.5 * dt ** 2
        self.F[2, 5] = dt
        self.F[2, 8] = 0.5 * dt ** 2

        # Prediction step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Measurement update step
        y = z - self.H @ self.x  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P  # Update error covariance

    def get_state(self):
        return self.x

    def get_state_position(self):
        return self.x[:3]

    def get_state_velocity(self):
        return self.x[3:6]
