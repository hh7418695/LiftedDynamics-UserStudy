import csv
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import fire
import jax.random as random
import matplotlib

from src.md import *
from src.utils import *

# Enable 64-bit (double precision) computations in JAX
jax.config.update("jax_enable_x64", True)

# Increase the maximum number of open figures warning in Matplotlib to 50
matplotlib.rcParams['figure.max_open_warning'] = 50


def wrap_main(f):
    def fn(*args, **kwargs):
        print(
            "\n******************************\n"
            f"Running {f.__name__}\n"
            "******************************\n"
            )
        print(f">> Args\n")
        for i in args:
            print(i)
        print(f">> KwArgs\n")
        for k, v in kwargs.items():
            print(f"{k}={v}")
        return f(*args, **kwargs)

    return fn


def Main(
        seed=42, dt=1.0e-4, stride=10, dim=3, socket_serverIP="127.0.0.1", socket_serverPort=12312,
        socket_bufferSize=512, participant_id=1
        ):
    """
    Wrapper around main to apply wrap_main decorator with default arguments.
    """
    return wrap_main(main)(
        seed=seed, dt=dt, stride=stride, dim=dim, socket_serverIP=socket_serverIP,
        socket_serverPort=socket_serverPort, socket_bufferSize=socket_bufferSize,
        participant_id=participant_id
        )


def main(seed, dt, stride, dim, socket_serverIP, socket_serverPort, socket_bufferSize, participant_id):
    """
    Main function to initialize and execute the primary workflow of the program.

    Parameters:
        seed (int): Random seed for reproducibility of results.
        dt (float): Time step for simulation.
        stride (int): Number of time steps ultimately used to generate the simulation data.
        dim (int): Dimensionality of the data.
        socket_serverIP (str): IP address of the server for socket communication.
        socket_serverPort (int): Port number of the server for socket communication.
        socket_bufferSize (int): Buffer size for receiving data over the socket.
        participant_id (int): Identifier for the current participant.

    Returns:
        None
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create UDP socket
    sock.bind((socket_serverIP, socket_serverPort))  # Bind the socket to the server address

    while True:
        print(
            "\n******************************\n"
            "Starting a new trial\n"
            "******************************\n"
            )

        # ******************** Trial settings ********************

        print(">> Input trial settings\n")

        print(
            "Format (separated by space):\n"
            "1) Object ID (default 1)\n"
            "2) If execute haptic rendering (default 1)\n"
            "3) If pre-check object animation before rendering (default 0)\n"
            "4) If pre-test object's resonance frequency before rendering (default 0)\n"
            "(Example input: 5 1 1 0)\n"
            )

        trialSettings_default = [1, 1, 0, 0]

        trialSettings_input = input("Enter here: ").strip().split()
        trialSettings_input = [int(x) for x in trialSettings_input]

        while len(trialSettings_input) < len(trialSettings_default):
            trialSettings_input.append(trialSettings_default[len(trialSettings_input)])

        object_id = trialSettings_input[0]
        enable_hapticRendering = bool(trialSettings_input[1])
        enable_preCheck_objectAnimation = bool(trialSettings_input[2])
        enable_preTest_resonanceFrequency = bool(trialSettings_input[3])

        if object_id > 10:
            print("\nInvalid object ID, please enter trail settings again\n")
            continue
        else:
            print(
                f"\nThis is well received! Current trial settings: "
                f"{object_id} {enable_hapticRendering} {enable_preCheck_objectAnimation} "
                f"{enable_preTest_resonanceFrequency}\n"
                )

        # ******************** Object parameters ********************

        print(">> Setting object parameters\n")

        N = 5  # number of nodes
        print(f"N = {N}")

        species = jnp.zeros(N, dtype=int)  # node types
        print(f"species = {species}")

        masses = np.array([40.0 for _ in range(N)])  # 40 g -> 200 g
        object_mass = np.sum(masses)  # remember that the haptic stylus itself has around 40g
        print(f"masses (g) = {masses} -> total weight (g) = {object_mass}")

        length = np.array([0.05 for _ in range(N - 1)])  # 5 cm * 4 -> 20 cm
        object_length = np.sum(length)
        print(f"lengths (m) = {length} -> total length (m) = {object_length}")

        enable_stretching = True
        stretching_val_list = np.linspace(50, 1000, num=5)
        if 1 <= object_id <= 5:
            object_stiffness_stretching_val = stretching_val_list[object_id - 1]
        else:  # object_id >= 6
            object_stiffness_stretching_val = stretching_val_list[2]
        object_stiffness_stretching_sim = np.array([object_stiffness_stretching_val for _ in range(N)]) * 1e3
        print(f"stretching stiffness (10^3 N/m) = {object_stiffness_stretching_sim / 1e3}")

        enable_bending = True
        bending_val_list = np.linspace(0, 0.1, num=5)
        if 6 <= object_id <= 10:
            object_stiffness_bending_val = bending_val_list[object_id - 6]
        else:  # object_id 1-5
            object_stiffness_bending_val = bending_val_list[2]
        object_stiffness_bending_sim = np.array([object_stiffness_bending_val for _ in range(N)]) * 1e3
        print(f"bending stiffness (10^3 N/m) = {object_stiffness_bending_sim / 1e3}")

        enable_damping = True
        object_damping_scale = 0.01 * 1e3
        object_damping_sim = np.array([0.1 for _ in range(N)]) * object_damping_scale  # previous 5.0
        print(f"general damping (10^3 Ns/m) = {object_damping_sim / 1e3}")

        enable_gravity = True

        enable_virtualCoupling = True
        VirtualCoupling_stiffness = 750 * 1e3
        print(f"virtual coupling stiffness (10^3 N/m) = {VirtualCoupling_stiffness / 1e3}")
        VirtualCoupling_damping = 25 * 1e3
        print(f"virtual coupling damping (10^3 N/m) = {VirtualCoupling_damping / 1e3}\n")

        # ******************** Computation methods ********************

        print(">> Setting computation methods\n")

        enable_KalmanFilter = True

        @jit
        def get_angle(vec1, vec2):
            vec1_norm = vec1 / (jnp.linalg.norm(vec1) + const_numerical)
            vec2_norm = vec2 / (jnp.linalg.norm(vec2) + const_numerical)
            angle_dot = jnp.dot(vec1_norm, vec2_norm)
            angle_dot = jnp.clip(angle_dot, -1.0, 1.0)
            angle = jnp.arccos(angle_dot)
            return angle

        @jit
        def getVector_relativeVelocity(vel_from, pos_from, vel_to, pos_to):
            pos_relative = pos_to - pos_from
            vel_relative = vel_to - vel_from
            distance = jnp.sqrt(jnp.square(pos_relative).sum())
            vel_angular = vel_relative / (distance + const_numerical)
            return vel_angular

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
        def getCoordinate_Python2TouchX(array):
            """
            Transform a 3D coordinate array from a Python-based coordinate system to a TouchX haptic device
            coordinate system.

            The transformation follows these rules:
                - x coordinate remains the same
                - y coordinate is replaced by the original z coordinate
                - z coordinate is replaced by the negative of the original y coordinate

            Parameters:
                array (jax.numpy.ndarray): Input array of coordinates, expected to be reshaped to a 3-element vector.

            Returns:
                jax.numpy.ndarray: Transformed coordinate array with the same shape as the input.
            """
            input_coordinate = array.reshape(dim)
            output_coordinate = jnp.zeros(dim)
            output_coordinate = output_coordinate.at[0].set(input_coordinate[0])  # x = x
            output_coordinate = output_coordinate.at[1].set(input_coordinate[2])  # y = z
            output_coordinate = output_coordinate.at[2].set(-input_coordinate[1])  # z = -y
            return output_coordinate.reshape(array.shape)

        @jit
        def getCoordinate_TouchX2Python(array):
            """
            Transform a 3D coordinate array from a TouchX haptic device coordinate system back to a Python-based
            coordinate system.

            The transformation follows these rules:
                - x coordinate remains the same
                - y coordinate is replaced by the negative of the original z coordinate
                - z coordinate is replaced by the original y coordinate

            Parameters:
                array (jax.numpy.ndarray): Input array of coordinates, expected to be reshaped to a 3-element vector.

            Returns:
                jax.numpy.ndarray: Transformed coordinate array with the same shape as the input.
            """
            input_coordinate = array.reshape(dim)
            output_coordinate = jnp.zeros(dim)
            output_coordinate = output_coordinate.at[0].set(input_coordinate[0])  # x = x
            output_coordinate = output_coordinate.at[1].set(-input_coordinate[2])  # y = -z
            output_coordinate = output_coordinate.at[2].set(input_coordinate[1])  # z = y
            return output_coordinate.reshape(array.shape)

        # ******************** Helper functions and parameters ********************

        print(">> Setting helper functions and parameters\n")

        sim_virtualCoupling_magnitude_xy = 0.1  # m
        sim_virtualCoupling_magnitude_z = 0.02  # m
        sim_virtualCoupling_period = 0.8  # s

        filepath_rootDirection = Path("../results")
        filepath_participantId = f"Participant-{participant_id}"
        filepath_objectId = f"Obj-{object_id}"
        filepath_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def get_fullFilepath(filepath_filename):
            """
            Generate and return the full file path for a given filename, ensuring that the directory structure exists
            by creating any missing directories.

            The full path is constructed as:
                <filepath_rootDirection>/<filepath_participantId>/<filepath_objectId>/<filepath_timestamp
                >/<filepath_filename>

            Parameters:
                filepath_filename (str): Name of the file (including extension) for which the full path is generated.

            Returns:
                str: Full path to the file as a string.
            """
            file_path = (filepath_rootDirection / filepath_participantId / filepath_objectId / filepath_timestamp /
                         filepath_filename)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            return str(file_path)

        def plot_exampleData(data, data_type, data_meaning, timestamp, index=0):
            """
            Plot a 3-axis example of time-series data and save the figure to a file.

            The function creates a 3-row subplot corresponding to the X, Y, and Z axes of the data, plotting the
            values from the 100th timestamp onward. The resulting figure is saved as a PNG file in the directory
            structure managed by `get_fullFilepath`.

            Parameters:
                data (numpy.ndarray or similar): Time-series data array with shape [time, samples, axes], where axes
                are expected to be X, Y, Z.
                data_type (str): Descriptive label for the type of data (e.g., sensor type).
                data_meaning (str): Human-readable description of the data's meaning.
                timestamp (array-like): Array of timestamps corresponding to the data.
                index (int, optional): Index of the sample to plot (default 0).

            Returns:
                None.
            """
            fig, axis = plt.subplots(3, 1)
            label_title_ax = ["X", "Y", "Z"]
            for i, ax in enumerate(axis):
                ax.plot(timestamp[100:], data[100:, index, i])
                ax.set_title(label_title_ax[i])
            plt.suptitle(f"{data_meaning} (Example {index}, {data_type})")
            plt.tight_layout()
            plt.savefig(get_fullFilepath(f"dataset_example{index}_{data_meaning}_{data_type}.png"))

        # ******************** Object dynamics ********************

        print(">> Setting object dynamics\n")

        def sim_shift(R, dR, V):
            return R + dR, V

        def sim_constraints_helper(R, l):
            out = jnp.sqrt(jnp.square(R[1:] - R[:-1]).sum(axis=1)) - l ** 2
            return out

        def sim_constraints(x, v, params):
            if enable_stretching:
                return jnp.zeros((1, N * dim))
            else:
                return jax.jacobian(lambda x: sim_constraints_helper(x.reshape(-1, dim), length), 0)(x)

        def sim_bendingEnergy(x):
            result = 0.0
            for i_N in range(2, N):
                vec1 = x[i_N - 1, :] - x[i_N - 2, :]
                vec2 = x[i_N, :] - x[i_N - 1, :]
                angle_temp = get_angle(vec1, vec2)
                result += getEnergy_bending(
                    angle_temp,
                    stiffness=object_stiffness_bending_sim[i_N]
                    )
            return result

        def sim_gravityEnergy_helper(R, g, mass):
            out = (mass * g * R[:, 2]).sum()
            return out

        sim_gravityEnergy = partial(sim_gravityEnergy_helper, g=const_gravity_acc, mass=masses)

        def sim_stretchingEnergy(x):
            result = 0.0
            for i_N in range(1, N):
                result += getEnergy_stretching(
                    x=x[i_N, :], x_lead=x[i_N - 1, :],
                    stiffness=object_stiffness_stretching_sim[i_N],
                    length=length[i_N - 1]
                    )
            return result

        def sim_potentialEnergy(x):
            result = 0.0
            if enable_gravity:
                result += sim_gravityEnergy(x)
            if enable_bending:
                result += sim_bendingEnergy(x)
            if enable_stretching:
                result += sim_stretchingEnergy(x)
            return result

        sim_kineticEnergy = partial(lnn._T, mass=masses)

        def sim_Lagrangian(x, v, params):
            return sim_kineticEnergy(v) - sim_potentialEnergy(x)

        def sim_externalForce(x, v, x_lead, v_lead, params):
            result = jnp.zeros((N, dim))

            if enable_virtualCoupling:
                result_virtualCoupling = jnp.zeros((N, dim))
                # print(f"sim_externalForce: x = {x}, v = {v}, x_lead = {x_lead}, v_lead = {v_lead}\n")
                if (x_lead is not None) and (v_lead is not None):
                    force_VirtualCoupling = getForce_virtualCoupling(x=x, v=v, x_lead=x_lead, v_lead=v_lead)
                    result_virtualCoupling = result_virtualCoupling.at[0, :].set(force_VirtualCoupling)
                else:
                    raise ValueError("Using virtual coupling but one/both of x_lead and v_lead is/are None!\n")
                result += result_virtualCoupling

            return result.reshape(-1, 1)

        def sim_damping(x, v, x_lead, v_lead, params):
            if enable_damping:
                vel_relative = jnp.zeros((N, dim))
                for i_N in range(1, N):
                    vel_relative = vel_relative.at[i_N, :].set(
                        getVector_relativeVelocity(
                            vel_from=v[i_N - 1, :], vel_to=v[i_N, :],
                            pos_from=x[i_N - 1, :], pos_to=x[i_N, :]
                            )
                        )
                damping_force_object = - vel_relative * object_damping_sim.reshape((N, 1))
                result = damping_force_object.reshape(-1, 1)
                return result
            else:
                return 0.0

        sim_acceleration = jit(
            lnn.accelerationFull(
                N, dim,
                lagrangian=sim_Lagrangian,
                non_conservative_forces=sim_damping,
                external_force=sim_externalForce,
                constraints=sim_constraints
                )
            )

        def sim_totalForce(R, V, R_lead, V_lead, params, mass):
            if mass is None:
                return sim_acceleration(R, V, R_lead, V_lead, params)
            else:
                return sim_acceleration(R, V, R_lead, V_lead, params) * mass.reshape(-1, 1)

        def sim_initialState(N, dim):
            # initial position
            R = jnp.zeros((N, dim))
            for i_N in range(N):
                # z direction
                result = 0.0
                for j in range(i_N):
                    result += length[j]
                R = R.at[i_N, 2].set(-result)
            # initial velocity
            V = jnp.zeros((N, dim))
            return R, V

        @partial(jax.jit, static_argnames=['runs'])
        def sim_nextState(R, V, R_lead, V_lead, runs):
            # print(f"sim_nextState: R_lead = {R_lead}, V_lead = {V_lead}")
            return predition(R, V, R_lead, V_lead, None, sim_totalForce, sim_shift, dt, masses, runs, stride)

        # ******************** Object animation check ********************

        def execute_objectAnimationCheck(test_num=2000):
            """
            Simulate and visualize the animation of objects in a 3D space using predefined initial conditions and
            sinusoidal leader motion.

            The function performs the following steps:
                1. Sets random seeds for reproducibility.
                2. Initializes object positions and velocities.
                3. Generates sinusoidal leader trajectories along X, Y, and Z axes with randomized periods and
                magnitudes.
                4. Iteratively updates object states using `sim_nextState` to simulate motion over `test_num` time
                steps.
                5. Prints progress and calculates average execution frequency.
                6. Plots and saves the resulting 3D trajectories as a GIF.

            Parameters:
                test_num (int, optional): Number of time steps to simulate (default 2000).

            Returns:
                None.
            """
            print(">> Generating object animation\n")

            np.random.seed(seed)
            key = random.PRNGKey(seed)

            test_timestamp = np.arange(test_num) * dt * stride
            Rs = np.empty((test_num, N, dim))
            Vs = np.empty((test_num, N, dim))
            Rs[0, :, :], Vs[0, :, :] = sim_initialState(N, dim)

            Rs_lead = np.zeros((test_num, 1, dim))
            Vs_lead = np.zeros((test_num, 1, dim))
            key, subkey = jax.random.split(key)
            period_train_x = jax.random.uniform(
                subkey,
                minval=sim_virtualCoupling_period * 0.5,
                maxval=sim_virtualCoupling_period * 1.5
                )
            omega_train_x = 2 * np.pi / period_train_x
            key, subkey = jax.random.split(key)
            magnitude_train_x = jax.random.uniform(
                subkey,
                minval=sim_virtualCoupling_magnitude_xy * 0.5,
                maxval=sim_virtualCoupling_magnitude_xy * 1.5
                )

            key, subkey = jax.random.split(key)
            period_train_y = jax.random.uniform(
                subkey,
                minval=sim_virtualCoupling_period * 0.5,
                maxval=sim_virtualCoupling_period * 1.5
                )
            omega_train_y = 2 * np.pi / period_train_y
            key, subkey = jax.random.split(key)
            magnitude_train_y = jax.random.uniform(
                subkey,
                minval=sim_virtualCoupling_magnitude_xy * 0.5,
                maxval=sim_virtualCoupling_magnitude_xy * 1.5
                )

            key, subkey = jax.random.split(key)
            period_train_z = jax.random.uniform(
                subkey,
                minval=sim_virtualCoupling_period * 0.5,
                maxval=sim_virtualCoupling_period * 1.5
                )
            omega_train_z = 2 * np.pi / period_train_z
            key, subkey = jax.random.split(key)
            magnitude_train_z = jax.random.uniform(
                subkey,
                minval=sim_virtualCoupling_magnitude_z * 0.5,
                maxval=sim_virtualCoupling_magnitude_z * 1.5
                )

            for i_time in range(test_num):
                index = i_time

                Rs_lead[index, 0, 0] = magnitude_train_x * np.sin(omega_train_x * test_timestamp[i_time])
                Vs_lead[index, 0, 0] = omega_train_x * magnitude_train_x * np.cos(
                    omega_train_x * test_timestamp[i_time]
                    )
                if test_timestamp[i_time] >= period_train_x:
                    Rs_lead[index, 0, 0] = 0.0
                    Vs_lead[index, 0, 0] = 0.0

                Rs_lead[index, 0, 1] = magnitude_train_y * np.sin(omega_train_y * test_timestamp[i_time])
                Vs_lead[index, 0, 1] = omega_train_y * magnitude_train_y * np.cos(
                    omega_train_y * test_timestamp[i_time]
                    )
                if test_timestamp[i_time] >= period_train_y:
                    Rs_lead[index, 0, 1] = 0.0
                    Vs_lead[index, 0, 1] = 0.0

                Rs_lead[index, 0, 2] = magnitude_train_z * np.sin(omega_train_z * test_timestamp[i_time])
                Vs_lead[index, 0, 2] = omega_train_z * magnitude_train_z * np.cos(
                    omega_train_z * test_timestamp[i_time]
                    )
                if test_timestamp[i_time] >= period_train_z:
                    Rs_lead[index, 0, 2] = 0.0
                    Vs_lead[index, 0, 2] = 0.0

            start_time = time.time()
            for i_time in range(test_num - 1):
                index = i_time
                # print(f"Rs = {Rs[index, :, :]}\n"
                #       f"Vs = {Vs[index, :, :]}\n"
                #       f"Rs_lead = {Rs_lead[index, :]}\n"
                #       f"Vs_lead = {Vs_lead[index, :]}\n")
                gen_data_train = sim_nextState(
                    Rs[index, :, :], Vs[index, :, :],
                    Rs_lead[index, :], Vs_lead[index, :],
                    1
                    )
                # print(f"index = {index}")
                # print(f"Rs[index, :, :] = {Rs[index, :, :]}")
                # print(f"Rs_lead[index, :] = {Rs_lead[index, :]}")
                # print(f"gen_data_train.position = {gen_data_train.position}")
                if index % (test_num / 5) == 0:
                    print(f"In progress: index = {index}")
                Rs[index + 1, :, :] = gen_data_train.position
                Vs[index + 1, :, :] = gen_data_train.velocity
            end_time = time.time()
            execution_time = end_time - start_time
            execution_time_aver = execution_time / test_num
            print(f"\nAverage execution frequency = {(1.0 / execution_time_aver):.2f} Hz\n")

            plot_trajectory(Rs, N, test_num, test_timestamp, get_fullFilepath("test_basic.gif"))
            print(f"Trajectory saved, test finished\n")

        # ******************** Resonance frequency test ********************

        def execute_resonanceFrequencyTest(enable_FFT=False, output_represent=False):
            """
            Perform a resonance frequency test on a multi-object system by sweeping through a range of input
            frequencies and evaluating the system's response.

            The function simulates sinusoidal leader motion along the X-axis and iteratively updates object positions
            and velocities using `sim_nextState`. The maximum relative displacement between the first and last
            objects is recorded for each frequency. Optionally, virtual coupling forces can be recorded for FFT
            analysis, and trajectories or force data can be saved as GIFs or PNGs.

            Parameters:
                enable_FFT (bool, optional): If True, calculates and records virtual coupling forces for frequency
                    analysis (default false).
                output_represent (bool, optional): If True, saves plots of trajectories and FFT results during the
                    sweep (default false).

            Returns:
                None.
            """
            print(">> Running resonance frequency test\n")

            # Explicit initialization for enable_FFT=True
            Ft = None
            Ft_magnitude = None

            result_MaxResponse = []
            result_freq = []

            freq_start = 0.5
            freq_stop = 5.0
            sweep_num = 450
            frequencies = np.linspace(freq_start, freq_stop, sweep_num)
            magnitude = sim_virtualCoupling_magnitude_xy

            progress = -1
            for freq in frequencies:
                progress += 1
                if progress % (int)(sweep_num / 10) == 0:
                    print(f"Progress {progress}: testing frequency {freq:.4f} Hz")

                curr_MaxResponse = 0

                if enable_FFT:
                    test_num = 10000
                else:
                    test_num = int(2000 / freq)  # 2 periods

                test_timestamp = np.arange(test_num) * dt * stride
                omega = 2 * np.pi * freq
                Rst_lead = np.zeros((test_num, 1, dim))
                Vst_lead = np.zeros((test_num, 1, dim))
                for i_time in range(test_num):
                    Rst_lead[i_time, 0, 0] = magnitude * np.sin(omega * test_timestamp[i_time])
                    Vst_lead[i_time, 0, 0] = omega * magnitude * np.cos(omega * test_timestamp[i_time])

                Rst = np.empty((test_num, N, dim))
                Vst = np.empty((test_num, N, dim))
                if enable_FFT:
                    Ft = np.zeros((test_num, N, dim))
                    Ft_magnitude = [0.0]
                Rst[0, :, :], Vst[0, :, :] = sim_initialState(N, dim)
                for i_time in range(test_num - 1):
                    gen_data_test = sim_nextState(
                        Rst[i_time, :, :],
                        Vst[i_time, :, :],
                        Rst_lead[i_time, :],
                        Vst_lead[i_time, :],
                        1
                        )
                    Rst[i_time + 1, :, :] = gen_data_test.position
                    Vst[i_time + 1, :, :] = gen_data_test.velocity
                    if enable_FFT:
                        Ft[i_time + 1, 0, :] = getForce_virtualCoupling(
                            Rst[i_time + 1, :, :],
                            Vst[i_time + 1, :, :],
                            Rst_lead[i_time, :],
                            Vst_lead[i_time, :]
                            )
                        Ft_magnitude.append(
                            (float(Ft[i_time + 1, 0, 0]) ** 2 + float(
                                Ft[i_time + 1, 0, 1]
                                ) ** 2 + float(Ft[i_time + 1, 0, 2]) ** 2) ** 0.5
                            )

                    # sqrt(x^2+y^2)
                    curr_response = (float(Rst[i_time + 1, -1, 0] - Rst[i_time + 1, 0, 0]) ** 2 +
                                     float(Rst[i_time + 1, -1, 1] - Rst[i_time + 1, 0, 1]) ** 2) ** 0.5
                    curr_MaxResponse = max(curr_MaxResponse, curr_response)

                if output_represent and progress % (int)(sweep_num / 4) == 0:
                    plot_trajectory(Rst, N, test_num, test_timestamp, get_fullFilepath(f"test_RF_{freq:.4f}.gif"))

                    if enable_FFT:
                        plot_exampleData(Ft, f"test_RF_{freq:.4f}", 'force', test_timestamp, 0)
                        FFT_dt = np.diff(test_timestamp).mean()
                        FFT_values = np.fft.fft(np.array(Ft_magnitude))
                        FFT_frequencies = np.fft.fftfreq(len(FFT_values), d=FFT_dt)
                        mask = (FFT_frequencies > 0) & (FFT_frequencies <= 10)
                        plt.figure(figsize=(8, 6))
                        plt.plot(
                            FFT_frequencies[mask], np.abs(FFT_values[mask]),
                            label='0–10 Hz'
                            )  # Positive frequencies in range
                        plt.title(f'FFT of the Virtual Coupling Force: {freq:.4f} Hz')
                        plt.xlabel('Frequency (Hz)')
                        plt.ylabel('Force Amplitude')
                        plt.grid()
                        plt.legend()
                        plt.savefig(get_fullFilepath(f"test_RF_FFT_{freq}.png"))

                result_MaxResponse.append(curr_MaxResponse)
                result_freq.append(freq)
                del Rst, Vst, Rst_lead, Vst_lead
                if enable_FFT:
                    del Ft

            result_freq = np.array(result_freq)
            print(f"\nresult_freq.shape = {result_freq.shape}\n")

            result_MaxResponse = np.array(result_MaxResponse)
            print(f"result_MaxResponse.shape = {result_MaxResponse.shape}\n")

            result_RF_response = max(result_MaxResponse)
            result_RF_freq = float(result_freq[list(result_MaxResponse).index(result_RF_response)])

            plt.figure(figsize=(10, 6))
            plt.plot(result_freq, result_MaxResponse, 'b.', alpha=0.6, label="Raw Data")
            plt.axvline(
                result_RF_freq, color='g', linestyle='--',
                label=f"{result_RF_response:.4f} @ {result_RF_freq:.4f} Hz"
                )
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Response")
            plt.title(f"Resonance Frequency Test: Object {object_id}")
            plt.legend()
            plt.grid(True)
            plt.savefig(get_fullFilepath(f"test_RF.png"))
            plt.show()

            print("Resonance frequency test finished\n")

        # ******************** Haptic rendering ********************

        def execute_hapticRendering(
                execute_MessagePrint=False, execute_SpeedAnalysis=False, execute_DataSave=True, execute_TrajPlot=False
                ):
            """
            Perform real-time haptic rendering of virtual objects with optional speed analysis, data saving,
            and trajectory plotting.

            The function receives position (and optionally velocity) data from a haptic device, computes virtual
            coupling forces, predicts object dynamics using `sim_nextState`, and sends the resulting forces back to
            the device. Optional Kalman filtering can be applied to estimate user velocity. Execution times for
            communication, force calculation, dynamic prediction, and overall rendering can be recorded if
            `execute_SpeedAnalysis=True`. Rendered object positions, user positions, and forces can be saved to a CSV
            file and plotted as trajectories.

            Notes:
                - `execute_TrajPlot=True` requires `execute_DataSave=True`.

            Parameters:
                execute_MessagePrint (bool, optional): Print incoming device positions and velocities (default false).
                execute_SpeedAnalysis (bool, optional): Record execution times for communication, force calculation,
                    dynamic prediction, and rendering (default false).
                execute_DataSave (bool, optional): Save rendered object positions, user positions, and forces to CSV
                    (default true).
                execute_TrajPlot (bool, optional): Plot and save object trajectories as a GIF (default false).

            Returns:
                None.
            """
            print(">> Running haptic rendering\n")

            print(
                "Please click the button on the haptic stylus and hold still until you can feel the weight\n"
                "Click the button again to stop the process\n"
                )

            # Explicit initialization for execute_SpeedAnalysis=True
            start_time_com = None
            start_time_cal_force = None
            start_time_cal_dynamic = None
            communication_time = None
            calculation_time_force = None
            calculation_time_dynamic = None
            calculation_time = None
            hist_com_count = None
            hist_execution_time_com = None
            hist_execution_time_CalForce = None
            hist_execution_time_CalDynamic = None
            hist_execution_time_render = None

            # Explicit initialization for execute_DataSave=True
            buffer_object_position = None
            buffer_user_position = None
            buffer_force = None
            buffer_time = None
            buffer_interval = None

            # Explicit initialization for enable_KalmanFilter=False
            data_velocity = None

            object_position, object_velocity = sim_initialState(N, dim)

            kf = KalmanFilter3D(0.001, 0.001, 0.01)  # Initialize the Kalman filter
            kf.x[:3] = [0.0, 0.0, 0.0]  # Initial state

            if execute_SpeedAnalysis:
                hist_execution_time_com = []
                hist_execution_time_CalForce = []
                hist_execution_time_CalDynamic = []
                hist_execution_time_render = []
                hist_com_count = []
            if execute_DataSave:
                buffer_size = 1000 * 20  # 50 s
                buffer_interval = 5  # 200 Hz
                buffer_object_position = np.zeros((buffer_size, N, dim), dtype=float)
                buffer_user_position = np.zeros((buffer_size, 1, dim), dtype=float)
                buffer_force = np.zeros((buffer_size, 1, dim), dtype=float)
                buffer_time = np.zeros(buffer_size, dtype=float)

            com_count = -1
            buffer_count = -1
            hist_time = []
            start_time_global = time.time()
            while True:
                if execute_SpeedAnalysis:
                    start_time_com = time.time()
                com_count += 1

                data_byte, address = sock.recvfrom(socket_bufferSize)
                data_received = bool(data_byte)
                # print(f"Received: {data_byte}")
                data_json = data_byte.decode('utf-8')
                data_dict = json.loads(data_json)  # dictionary
                data_position = data_dict.get('position', [])
                data_timestamp = data_dict.get('timestamp')
                if execute_MessagePrint:
                    print(f"Position = {data_position}")
                    print(f"Timestamp = {data_timestamp}\n")
                if not enable_KalmanFilter:
                    data_velocity = data_dict.get('velocity', [])
                    if execute_MessagePrint:
                        print(f"Velocity = {data_velocity}")

                # cipher for stopping
                if data_timestamp < 0:
                    break

                end_time_global = time.time()
                diff_time_global = end_time_global - start_time_global
                if execute_SpeedAnalysis:
                    end_time_com = time.time()
                    communication_time = end_time_com - start_time_com

                # compute force from position

                if execute_SpeedAnalysis:
                    start_time_cal_force = time.time()
                    hist_com_count.extend([com_count])

                hist_time.extend([diff_time_global])
                # hist_time.extend([data_timestamp * 1e-3])

                user_position = np.array([data_position]) * 1e-3  # from mm to m
                user_position = getCoordinate_TouchX2Python(user_position)

                if enable_KalmanFilter:
                    if com_count <= 1:
                        user_position = np.zeros((1, dim))
                        user_velocity = np.zeros((1, dim))
                    else:
                        hist_time_np = np.array(hist_time).reshape(-1)
                        delta_time = hist_time_np[com_count] - hist_time_np[com_count - 1]
                        # delta_time = 0.001
                        kf.predict(delta_time)

                        kf.update(user_position.reshape(dim))
                        state = kf.get_state()

                        # estimated_position = np.zeros((1, dim))
                        # for i_dim in range(dim):
                        #     estimated_position[0, i_dim] = state[i_dim]
                        #
                        # estimated_velocity = np.zeros((1, dim))
                        # for i_dim in range(dim):
                        #     estimated_velocity[0, i_dim] = state[3 + i_dim]

                        estimated_position = state[:3]
                        estimated_velocity = state[3:6]
                        user_position = estimated_position.reshape(1, dim)
                        user_velocity = estimated_velocity.reshape(1, dim)
                else:
                    user_velocity = np.array([data_velocity])
                    user_velocity = getCoordinate_TouchX2Python(user_velocity)

                # print(f"user_position = {user_position}, user_velocity = {user_velocity}")

                # print(f"user_position shape = {user_position.shape}")
                # print(f"object_position shape = {object_position.shape}")
                # print(f"Rst_lead shape = {Rst_lead[0, :, :].shape}")

                # object_acceleration = acceleration_fn_model(x=object_position,
                #                                             v=object_velocity,
                #                                             x_lead=user_position,
                #                                             v_lead=user_velocity,
                #                                             params=params)

                force = getForce_virtualCoupling(
                    x=object_position,
                    v=object_velocity,
                    x_lead=user_position,
                    v_lead=user_velocity
                    )
                force *= -1e-3

                # force test:
                # 2.0 yes
                # 2.5 ok but at corners will exist some sound
                # 3.0 ok but at limit (electrical sound), 4.0 no
                force = np.array(force)  # * 0.5

                force_max = np.max(np.abs(force))
                if force_max > 2.5:
                    force = force / force_max * 2.5  # can't do /= force_max * 1.7, will cause some unexpected value
                    # ~0.7

                # force_magnitude = np.linalg.norm(force)
                # if force_magnitude > 7.0:
                #     force = force / force_magnitude * 7.0
                # force_magnitude = np.linalg.norm(force)

                # for i_dim in range(dim):
                #     if force[i_dim] > 7.0 or force[i_dim] < -7.0:
                #         force_exceeded = True
                #         error_force.append(force.tolist())
                #         break
                #     else:
                #         force_exceeded = False
                # if force_exceeded:
                #     force = last_force
                # else:
                #     last_force = force

                # force = np.array(force)
                # np.where(force > 7.0, 7.0, force)
                # np.where(force < -7.0, -7.0, force)

                if execute_SpeedAnalysis:
                    end_time_cal_force = time.time()
                    calculation_time_force = end_time_cal_force - start_time_cal_force

                # predict next step dynamic

                if execute_SpeedAnalysis:
                    start_time_cal_dynamic = time.time()

                predict_traj = sim_nextState(
                    R=object_position,
                    V=object_velocity,
                    R_lead=user_position,
                    V_lead=user_velocity,
                    runs=1
                    )

                # object_position = np.zeros((N, dim))
                # object_velocity = np.zeros((N, dim))
                # for i_dim in range(dim):
                #     for i_N in range(N):
                #         object_position[i_N, i_dim] = predict_traj.position[0,i_N, i_dim]
                #         object_velocity[i_N, i_dim] = predict_traj.velocity[0,i_N, i_dim]
                object_position = predict_traj.position.reshape(
                    N,
                    dim
                    )  # [0,:,:] is causing time spikes, and only this line
                object_velocity = predict_traj.velocity.reshape(
                    N,
                    dim
                    )  # even used [0,:,:] as well, this line doesn't cause

                if execute_SpeedAnalysis:
                    end_time_cal_dynamic = time.time()
                    calculation_time_dynamic = end_time_cal_dynamic - start_time_cal_dynamic

                # if calculation_time_dynamic > 0.05:
                #     print(f"predict_traj = {predict_traj}")
                #     print(f"predict_traj.position.shape = {predict_traj.position.shape}")
                #     print(f"predict_traj.velocity.shape = {predict_traj.velocity.shape}")
                #     print("Time spike encountered!!!")
                #     break

                if execute_SpeedAnalysis:
                    calculation_time = calculation_time_force + calculation_time_dynamic

                # return force
                if execute_SpeedAnalysis:
                    start_time_com = time.time()
                output_dict = dict(force=getCoordinate_Python2TouchX(force).tolist())
                if data_received:
                    output_json = json.dumps(output_dict)
                    output_byte = output_json.encode('utf-8')
                    output_length = sock.sendto(output_byte, address)
                if execute_SpeedAnalysis:
                    end_time_com = time.time()
                    communication_time += end_time_com - start_time_com
                    render_time = communication_time + calculation_time
                    hist_execution_time_com.append(communication_time)
                    hist_execution_time_CalForce.append(calculation_time_force)
                    hist_execution_time_CalDynamic.append(calculation_time_dynamic)
                    hist_execution_time_render.append(render_time)
                if execute_DataSave and (com_count % buffer_interval == 0):
                    buffer_count += 1
                    buffer_object_position[buffer_count, :, :] = object_position
                    buffer_user_position[buffer_count, :, :] = user_position
                    buffer_force[buffer_count, :, :] = force
                    buffer_time[buffer_count] = hist_time[-1]

                # print(f"force = {force}")
                # print(f"average update rate = {(1.0 / (render_time + const_numerical)):.2f} Hz")
                # print(f"communication: {(communication_time * 1000.0):.2f} ms, "
                #       f"ratio {(communication_time / (render_time + const_numerical) * 100.0):.2f}%")
                # print(f"calculation: {(calculation_time * 1000.0):.2f} ms, "
                #       f"ratio {(calculation_time / (render_time + const_numerical) * 100.0):.2f}%")
                # print(f"cal - force: {(calculation_time_force * 1000.0):.2f} ms, "
                #       f"ratio {(calculation_time_force / (calculation_time + const_numerical) * 100.0):.2f}%")
                # print(f"cal - dynamic: {(calculation_time_dynamic * 1000.0):.2f} ms, "
                #       f"ratio {(calculation_time_dynamic / (calculation_time + const_numerical) * 100.0):.2f}%\n")

            hist_time = np.array(hist_time)
            print(f"Rendering process stopped, hist_time.shape = {hist_time.shape}")

            if execute_SpeedAnalysis:
                fig, axis = plt.subplots(4, 1)
                axis[0].plot(hist_com_count, hist_execution_time_com)
                axis[0].set_yscale('log')
                axis[0].set_title("Communication Time")
                axis[1].plot(hist_com_count, hist_execution_time_CalForce)
                axis[1].set_yscale('log')
                axis[1].set_title("Force Calculation Time")
                axis[2].plot(hist_com_count, hist_execution_time_CalDynamic)
                axis[2].set_yscale('log')
                axis[2].set_title("Dynamic Calculation Time")
                axis[3].plot(hist_com_count, hist_execution_time_render)
                axis[3].set_yscale('log')
                axis[3].set_title("Overall Rendering Time")
                plt.suptitle("Execution Time Observation")
                plt.tight_layout()
                plt.savefig(get_fullFilepath("render_execution_time.png"))
                print("Speed analysis finished")

            if execute_DataSave:
                print(f"buffer count = {buffer_count}")
                if execute_TrajPlot:
                    plot_trajectory(
                        buffer_object_position[:buffer_count, :, :],
                        N,
                        buffer_count,
                        buffer_time[:buffer_count],
                        get_fullFilepath("render.gif"),
                        data_force=buffer_force[:buffer_count, :, :],
                        data_virtual=buffer_user_position[:buffer_count, :, :],
                        show_skip=10
                        )
                with open(get_fullFilepath("render_hist.csv"), mode='w', newline='') as file:
                    writer = csv.writer(file)
                    title_whole = ["Timestamp (s)"]
                    title_dim = ["X", "Y", "Z"]
                    for i_dim in range(dim):
                        title_whole.append(f"Rendered Force {title_dim[i_dim]} (N)")
                    for i_dim in range(dim):
                        title_whole.append(f"User Position {title_dim[i_dim]} (m)")
                    for i_N in range(N):
                        for i_dim in range(dim):
                            title_whole.append(f"Node {i_N} Position {title_dim[i_dim]} (m)")
                    writer.writerow(title_whole)
                    for i_buffer in range(buffer_count):
                        curr_row = [buffer_time[i_buffer]]
                        for i_dim in range(dim):
                            curr_row.append(buffer_force[i_buffer, 0, i_dim])
                        for i_dim in range(dim):
                            curr_row.append(buffer_user_position[i_buffer, 0, i_dim])
                        for i_N in range(N):
                            for i_dim in range(dim):
                                curr_row.append(buffer_object_position[i_buffer, i_N, i_dim])
                        writer.writerow(curr_row)
                        del curr_row
                print("Render history data saved!")

            print("\n")
            del hist_time
            if execute_SpeedAnalysis:
                del hist_execution_time_com, hist_execution_time_CalForce, hist_execution_time_CalDynamic
                del hist_execution_time_render, hist_com_count
            if execute_DataSave:
                del buffer_user_position, buffer_object_position, buffer_force, buffer_time

        # ******************** Running trial ********************

        if enable_preCheck_objectAnimation:
            execute_objectAnimationCheck()
        if enable_preTest_resonanceFrequency:
            execute_resonanceFrequencyTest()
        if enable_hapticRendering:
            execute_hapticRendering()


def run_single_object(
        sock, object_id: int, participant_id: str,
        pi_input_fn=None,
        seed=42, dt=1.0e-4, stride=10, dim=3,
        socket_bufferSize=512,
        ):
    """
    Run haptic rendering for a single object using an externally managed socket.
    Called by experiment_runner.py instead of the interactive main() loop.

    Parameters:
        sock: bound UDP socket (shared across trials)
        object_id: 1-10
        participant_id: string, used for result file paths
        pi_input_fn: optional callable() -> str, checked each loop for 's'/'q'
        seed, dt, stride, dim, socket_bufferSize: same as main()

    Returns:
        'ok'   - rendering finished normally (stop signal received)
        'skip' - PI requested skip
        'quit' - PI requested quit
    """
    import csv
    import json
    import time
    from datetime import datetime
    from pathlib import Path
    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import jit

    jax.config.update("jax_enable_x64", True)

    from src.md import predition
    from src.utils import KalmanFilter3D, plot_trajectory
    from src import lnn

    const_numerical = 1e-10
    const_gravity_acc = -9.8

    rng = jax.random.PRNGKey(seed)

    # ── Object parameters (mirrors main()) ───────────────────────────────────
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

    # ── JIT helpers ───────────────────────────────────────────────────────────
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
        force_gravity = jnp.zeros(dim).at[2].set(object_mass * const_gravity_acc)
        return force + force_gravity

    @jit
    def getCoordinate_TouchX2Python(array):
        c = array.reshape(dim)
        return jnp.array([c[0], -c[2], c[1]]).reshape(array.shape)

    @jit
    def getCoordinate_Python2TouchX(array):
        c = array.reshape(dim)
        return jnp.array([c[0], c[2], -c[1]]).reshape(array.shape)

    # ── Physics ───────────────────────────────────────────────────────────────
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

    # initial state
    R_init = jnp.zeros((N, dim))
    for i in range(N):
        R_init = R_init.at[i, 2].set(-sum(length[:i]))
    V_init = jnp.zeros((N, dim))

    @partial(jax.jit, static_argnames=['runs'])
    def sim_nextState(R, V, R_lead, V_lead, runs):
        return predition(R, V, R_lead, V_lead, None, sim_totalForce,
                         lambda R, dR, V: (R + dR, V), dt, masses, runs, stride)

    # ── Result path ───────────────────────────────────────────────────────────
    result_dir = (Path("../results") / f"Participant-{participant_id}" /
                  f"Obj-{object_id}" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── Rendering loop ────────────────────────────────────────────────────────
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

        # Check PI emergency input (non-blocking)
        if pi_input_fn is not None:
            cmd = pi_input_fn()
            if cmd == "q":
                return "quit"
            if cmd == "s":
                return "skip"

        data_byte, address = sock.recvfrom(socket_bufferSize)
        data_dict = json.loads(data_byte.decode("utf-8"))
        data_position  = data_dict.get("position", [])
        data_timestamp = data_dict.get("timestamp")

        # Stop signal from C++ (or mock)
        if data_timestamp < 0:
            break

        diff_time = time.time() - start_time_global
        hist_time.append(diff_time)

        user_position = getCoordinate_TouchX2Python(np.array([data_position]) * 1e-3)

        if com_count <= 1:
            kf.x[:3] = np.array(user_position).flatten()[:3]
            user_velocity = np.zeros((1, dim))
        else:
            kf.predict(dt)
            kf.update(np.array(user_position).flatten()[:3])
            user_velocity = np.array(kf.x[3:6]).reshape(1, dim)

        force = np.array(getForce_virtualCoupling(
            x=object_position, v=object_velocity,
            x_lead=user_position, v_lead=user_velocity,
        ))
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

    # ── Save CSV ──────────────────────────────────────────────────────────────
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
    print(f"  Render history saved → {csv_path}")

    return "ok"


if __name__ == "__main__":
    fire.Fire(Main)
