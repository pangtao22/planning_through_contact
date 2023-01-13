import os
import pickle
from pathlib import Path

import numpy as np
from pydrake.all import (
    AngleAxis,
    Quaternion,
    PiecewisePolynomial,
    PiecewiseQuaternionSlerp,
)

import matplotlib.pyplot as plt


# %%
def get_quaternion(q_unnormalized: np.ndarray) -> Quaternion:
    return Quaternion(q_unnormalized / np.linalg.norm(q_unnormalized))


def get_X_WB_ref_trj(
    q_u_knots_ref: np.ndarray, t_knots: np.ndarray
) -> [PiecewiseQuaternionSlerp, PiecewisePolynomial]:
    Q_WB_list = [get_quaternion(q_u[:4]) for q_u in q_u_knots_ref]

    return (
        PiecewiseQuaternionSlerp(t_knots, Q_WB_list),
        PiecewisePolynomial.FirstOrderHold(t_knots, q_u_knots_ref[:, 4:].T),
    )


def recover_q_u_knots_ref_from_q_u_ref(
    q_u_ref: np.ndarray, x_log_times: np.ndarray, t_knots: np.ndarray
):
    """
    Terry saved q_u_ref evaluated at x_log_time, instead of only the knots
     corresponding to t_knots.
    This functions recover the knots from the refs.
    """
    q_u_ref_trj = PiecewisePolynomial.ZeroOrderHold(x_log_times, q_u_ref.T)
    q_u_knots_ref = np.zeros((len(t_knots), q_u_ref.shape[1]))

    for i, t in enumerate(t_knots):
        q_u_knots_ref[i] = q_u_ref_trj.value(t).squeeze()

    return q_u_knots_ref


def calc_errors_for_segment_3d(
    q_u_knots_ref: np.ndarray,
    t_knots: np.ndarray,
    x_log_time: np.ndarray,
    q_u_mbp: np.ndarray,
):
    Q_WB_trj_ref, p_WB_trj_ref = get_X_WB_ref_trj(q_u_knots_ref, t_knots)
    Q_WB_refs = np.array([Q_WB_trj_ref.value(t).squeeze() for t in x_log_time])
    p_WB_refs = np.array([p_WB_trj_ref.value(t).squeeze() for t in x_log_time])

    orientation_error_for_segment = []
    for Q_WB_ref, Q_WB in zip(Q_WB_refs, q_u_mbp[:, :4]):
        Q_WB_ref = Quaternion(Q_WB_ref)
        Q_WB = get_quaternion(Q_WB)
        orientation_error_for_segment.append(
            AngleAxis(Q_WB_ref.inverse().multiply(Q_WB)).angle()
        )

    position_error_for_segment = np.linalg.norm(
        p_WB_refs - q_u_mbp[:, 4:], axis=1
    )

    return position_error_for_segment, orientation_error_for_segment


def calc_errors_for_segment_2d(
    q_u_knots_ref: np.ndarray,
    t_knots: np.ndarray,
    x_log_time: np.ndarray,
    q_u_mbp: np.ndarray,
):
    q_u_trj_ref = PiecewisePolynomial.FirstOrderHold(t_knots, q_u_knots_ref.T)
    q_u_ref = np.array([q_u_trj_ref.value(t).squeeze() for t in x_log_time])

    orientation_error_for_segment = np.abs(q_u_ref[2] - q_u_mbp[2])

    position_error_for_segment = np.linalg.norm(
        q_u_ref[:, :2] - q_u_mbp[:, :2], axis=1
    )

    return position_error_for_segment, orientation_error_for_segment


def calc_X_WB_ref_trj_length_3d(q_u_knots_ref: np.ndarray):
    Q_WB_list = [get_quaternion(q_u[:4]) for q_u in q_u_knots_ref]
    T = len(q_u_knots_ref) - 1
    total_angular_displacement = 0
    total_position_displacement = 0
    for t in range(T):
        Q_WB_t = Q_WB_list[t]
        Q_WB_t1 = Q_WB_list[t + 1]
        d_angle = AngleAxis(Q_WB_t.inverse().multiply(Q_WB_t1)).angle()
        d_position = np.linalg.norm(
            q_u_knots_ref[t, 4:] - q_u_knots_ref[t + 1, 4:]
        )
        total_angular_displacement += d_angle
        total_position_displacement += d_position

    return total_angular_displacement, total_position_displacement


def calc_X_WB_ref_trj_length_2d(q_u_knots_ref: np.ndarray):
    angles = q_u_knots_ref[:, 2]
    positions = q_u_knots_ref[:, :2]

    total_angular_displacement = np.abs(angles[1:] - angles[:-1]).sum()
    total_position_displacement = np.linalg.norm(
        positions[1:] - positions[:-1], axis=1
    ).sum()

    return total_angular_displacement, total_position_displacement


def calc_errors_and_path_lengths(file_paths):
    position_errors = []
    position_path_lengths = []
    angular_errors = []
    angular_path_lengths = []

    for file_path in file_paths:
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        # unpack the dictionary.
        q_u_knots_ref = data_dict["q_u_ref"]
        t_knots = data_dict["t_knots"]
        q_u_mbp = data_dict["q_u_mbp"]
        x_log_time = data_dict["x_log_time"]

        if len(q_u_knots_ref) != len(t_knots) and len(q_u_knots_ref) == len(
            x_log_time
        ):
            # Terry's logs.
            q_u_knots_ref = recover_q_u_knots_ref_from_q_u_ref(
                q_u_knots_ref, x_log_time, t_knots
            )

        dim_q_u = q_u_knots_ref.shape[1]
        calc_errors_for_segment = None
        calc_X_WB_ref_trj_length = None
        if dim_q_u == 7:
            # SE(3)
            calc_errors_for_segment = calc_errors_for_segment_3d
            calc_X_WB_ref_trj_length = calc_X_WB_ref_trj_length_3d
        elif dim_q_u == 3:
            # SE(2)
            calc_errors_for_segment = calc_errors_for_segment_2d
            calc_X_WB_ref_trj_length = calc_X_WB_ref_trj_length_2d
        elif dim_q_u == 2:
            # door
            calc_errors_for_segment = calc_errors_for_segment_door
            calc_X_WB_ref_trj_length = calc_X_WB_ref_trj_length_door

        (
            position_error_for_segment,
            orientation_error_for_segment,
        ) = calc_errors_for_segment(
            q_u_knots_ref,
            t_knots,
            x_log_time,
            q_u_mbp,
        )

        angular_errors.append(np.mean(orientation_error_for_segment))
        position_errors.append(np.mean(position_error_for_segment))
        angular_length, position_length = calc_X_WB_ref_trj_length(
            q_u_knots_ref
        )
        angular_path_lengths.append(angular_length)
        position_path_lengths.append(position_length)

    return {
        "position_errors": position_errors,
        "position_path_lengths": position_path_lengths,
        "angular_errors": angular_errors,
        "angular_path_lengths": angular_path_lengths,
    }


# %%
results_dict = {}
system_names = [
    "allegro_hand",
    "allegro_hand_pen",
    "allegro_hand_plate",
    "planar_pushing",
    "planar_hand",
    "iiwa_bimanual",
]

for system_name in system_names:
    data_dir_path = Path(Path.home(), "ptc_data", system_name, "sim2real")
    file_paths = [
        Path(data_dir_path, file_name)
        for file_name in os.listdir(data_dir_path)
    ]
    file_paths.sort()
    results_dict[system_name] = calc_errors_and_path_lengths(file_paths)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

for system_name in system_names:
    position_path_lengths = results_dict[system_name]["position_path_lengths"]
    position_errors = results_dict[system_name]["position_errors"]
    angular_path_lengths = results_dict[system_name]["angular_path_lengths"]
    angular_errors = results_dict[system_name]["angular_errors"]

    axes[0].scatter(position_path_lengths, position_errors, label=system_name)
    axes[1].scatter(angular_path_lengths, angular_errors, label=system_name)

# reference line
x = np.linspace(1e-3, 5, 10)
y = 0.1 * x

axes[0].set_xlabel("path length [m]")
axes[0].set_ylabel("Average position error [m]")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].axis("equal")
axes[0].plot(x, y, color="b", linestyle="--")

axes[1].set_xlabel("path length [rad]")
axes[1].set_ylabel("Average angular error [rad]")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].legend()
axes[1].axis("equal")
axes[1].plot(x, y, color="b", linestyle="--")


plt.show()
