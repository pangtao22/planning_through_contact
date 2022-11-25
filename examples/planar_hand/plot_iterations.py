from distutils.dir_util import create_tree
import os.path
from struct import pack
import time
from matplotlib import patches
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import networkx as nx

import cProfile
from dash_vis.dash_common import create_pca_plots

from pydrake.all import PiecewisePolynomial

from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from irs_mpc.irs_mpc_quasistatic import IrsMpcQuasistatic
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsRrt, IrsNode


from tqdm import tqdm


def get_cost_array(irs_rrt, global_metric):
    n_nodes = len(irs_rrt.graph.nodes)
    costs = np.zeros(n_nodes)

    irs_rrt.rrt_params.global_metric = global_metric
    for n in range(1, n_nodes):
        costs[n] = np.min(
            irs_rrt.calc_distance_batch_global(
                irs_rrt.goal, n, is_q_u_only=True
            )
        )

    return costs[1:]


def get_packing_ratio_array(irs_rrt, sampling_function, n_samples, threshold):
    n_nodes = len(irs_rrt.graph.nodes)
    costs = np.zeros(n_nodes)

    for n in tqdm(range(1, n_nodes)):
        samples = sampling_function(n_samples)
        pairwise_distance = irs_rrt.calc_pairwise_distance_batch_local(
            samples, n, is_q_u_only=True
        )
        dist = np.min(pairwise_distance, axis=1)
        costs[n] = np.sum(dist < threshold) / n_samples

    return costs[1:]


def compute_statistics(filename):
    with open(filename, "rb") as f:
        tree = pickle.load(f)

    """Modify the below lines for specific implementations."""

    irs_rrt = IrsRrt.make_from_pickled_tree(tree)
    global_metric = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.2])
    n_samples = 100
    threshold = 3

    def sampling_function(n_samples):
        samples = np.random.rand(n_samples, 7)
        samples[:, 4] = 0.2 * samples[:, 4] - 0.1
        samples[:, 5] = 0.2 * samples[:, 5] + 0.3
        samples[:, 6] = (np.pi + 0.01) * samples[:, 6] - 0.01
        return samples

    cost_array = get_cost_array(irs_rrt, global_metric)
    packing_ratio_array = get_packing_ratio_array(
        irs_rrt, sampling_function, n_samples, threshold
    )

    return cost_array, packing_ratio_array


def plot_filename_array(filename_array, color, label):
    cost_array_lst = []
    packing_ratio_lst = []

    for filename in filename_array:
        cost_array, packing_ratio_array = compute_statistics(filename)
        cost_array_lst.append(cost_array)
        packing_ratio_lst.append(packing_ratio_array)

    plt.subplot(1, 2, 1)
    cost_array_lst = np.array(cost_array_lst)
    mean_cost = np.mean(cost_array_lst, axis=0)
    std_cost = np.std(cost_array_lst, axis=0)

    plt.plot(range(len(mean_cost)), mean_cost, "-", color=color, label=label)
    plt.fill_between(
        range(len(mean_cost)),
        mean_cost - std_cost,
        mean_cost + std_cost,
        color=color,
        alpha=0.1,
    )
    plt.xlabel("Iterations")
    plt.ylabel("Closest Distance to Goal")

    plt.subplot(1, 2, 2)
    packing_ratio_lst = np.array(packing_ratio_lst)
    mean_cost = np.mean(packing_ratio_lst, axis=0)
    std_cost = np.std(packing_ratio_lst, axis=0)

    plt.plot(range(len(mean_cost)), mean_cost, "-", color=color, label=label)
    plt.fill_between(
        range(len(mean_cost)),
        mean_cost - std_cost,
        mean_cost + std_cost,
        color=color,
        alpha=0.1,
    )
    plt.xlabel("Iterations")
    plt.ylabel("Packing Ratio")


fig = plt.figure(figsize=(16, 4))
plt.rcParams["font.size"] = "16"
filename_array = [
    "data/planar_hand/projection/ours/tree_2000_planar_hand_rg_1.pkl",
    "data/planar_hand/projection/ours/tree_2000_planar_hand_rg_2.pkl",
    "data/planar_hand/projection/ours/tree_2000_planar_hand_rg_3.pkl",
    "data/planar_hand/projection/ours/tree_2000_planar_hand_rg_4.pkl",
    "data/planar_hand/projection/ours/tree_2000_planar_hand_rg_5.pkl",
]
plot_filename_array(filename_array, "springgreen", "iRS-RRT")

filename_array = [
    "data/planar_hand/projection/global/tree_2000_planar_hand_rg_global_1.pkl",
    "data/planar_hand/projection/global/tree_2000_planar_hand_rg_global_2.pkl",
    "data/planar_hand/projection/global/tree_2000_planar_hand_rg_global_3.pkl",
    "data/planar_hand/projection/global/tree_2000_planar_hand_rg_global_4.pkl",
    "data/planar_hand/projection/global/tree_2000_planar_hand_rg_global_5.pkl",
]
plot_filename_array(filename_array, "red", "Global Metric")

filename_array = [
    "data/planar_hand/projection/nocontact/tree_2000_planar_hand_rg_nocontact_1.pkl",
    "data/planar_hand/projection/nocontact/tree_2000_planar_hand_rg_nocontact_2.pkl",
    "data/planar_hand/projection/nocontact/tree_2000_planar_hand_rg_nocontact_3.pkl",
    "data/planar_hand/projection/nocontact/tree_2000_planar_hand_rg_nocontact_4.pkl",
    "data/planar_hand/projection/nocontact/tree_2000_planar_hand_rg_nocontact_5.pkl",
]
plot_filename_array(filename_array, "royalblue", "No Contact")

plt.subplot(1, 2, 1)
plt.legend()
plt.subplot(1, 2, 2)
plt.legend()

fig.set_figheight(6)
fig.set_figwidth(12)
plt.show()
