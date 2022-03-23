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
from irs_mpc.quasistatic_dynamics_parallel import (
    QuasistaticDynamicsParallel)
from irs_mpc.irs_mpc_quasistatic import (
    IrsMpcQuasistatic)
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters

from irs_rrt.irs_rrt import IrsRrt, IrsNode
from irs_rrt.rrt_params import IrsRrtParams
from irs_rrt.irs_rrt_random_grasp import IrsRrtRandomGrasp

from allegro_hand_setup import *

from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix

from tqdm import tqdm


def get_cost_array(irs_rrt, global_metric):
    n_nodes = len(irs_rrt.graph.nodes)
    costs = np.zeros(n_nodes)

    irs_rrt.params.global_metric = global_metric
    for n in range(1,n_nodes):
        costs[n] = np.min(
                irs_rrt.calc_distance_batch_global(
                    irs_rrt.goal, n, is_q_u_only=True))

    return costs[1:]

def get_packing_ratio_array(irs_rrt, sampling_function, n_samples, threshold):
    n_nodes = len(irs_rrt.graph.nodes)
    costs = np.zeros(n_nodes)

    for n in tqdm(range(1,n_nodes)):
        samples = sampling_function(n_samples)
        pairwise_distance = irs_rrt.calc_pairwise_distance_batch_local(
            samples, n, is_q_u_only=True)
        dist = np.min(pairwise_distance, axis=1)
        costs[n] = np.sum(dist < threshold) / n_samples

    return costs[1:]

def compute_statistics(filename):
    with open(filename, 'rb') as f:
        tree = pickle.load(f)

    """Modify the below lines for specific implementations."""

    irs_rrt = IrsRrt.make_from_pickled_tree(tree)
    global_metric = np.ones(26)
    n_samples = 100
    threshold = 5

    def sampling_function(n_samples):
        samples = np.random.rand(n_samples,26)

        rpy_random = np.random.rand(n_samples, 3)
        rpy_random[:,0] = 0.5 * rpy_random[:,0]
        rpy_random[:,1] = 0.5 * rpy_random[:,1]
        rpy_random[:,2] = 1.2 * rpy_random[:,2]

        q_array = np.zeros((n_samples,4))

        for k in range(n_samples):
            q_array[k,:] = RollPitchYaw(rpy_random[k,:]).ToQuaternion().wxyz()

        samples[:,19:23] = q_array
        samples[:,23] = 0.2 * samples[:,4]
        samples[:,24] = 0.2 * samples[:,5]
        samples[:,25] = 0.2 * samples[:,6]
        return samples

    cost_array = get_cost_array(irs_rrt, global_metric)
    packing_ratio_array = get_packing_ratio_array(irs_rrt,
        sampling_function, n_samples, threshold)

    return cost_array, packing_ratio_array

def plot_filename_array(filename_array, color, label):
    cost_array_lst = []
    packing_ratio_lst = []

    for filename in filename_array:
        cost_array, packing_ratio_array = compute_statistics(filename)
        cost_array_lst.append(cost_array)
        packing_ratio_lst.append(packing_ratio_array)

    plt.subplot(1,2,1)
    cost_array_lst = np.array(cost_array_lst)
    mean_cost = np.mean(cost_array_lst, axis=0)
    std_cost  = np.std(cost_array_lst, axis=0)

    plt.plot(range(len(mean_cost)), mean_cost, '-', color=color,
        label=label)
    plt.fill_between(range(len(mean_cost)),
        mean_cost - std_cost, mean_cost + std_cost, color=color, alpha=0.1)
    plt.xlabel('Iterations')
    plt.ylabel('Closest Distance to Goal')

    plt.subplot(1,2,2)
    packing_ratio_lst = np.array(packing_ratio_lst)
    mean_cost = np.mean(packing_ratio_lst, axis=0)
    std_cost = np.std(packing_ratio_lst, axis=0)

    plt.plot(range(len(mean_cost)), mean_cost, '-', color=color,
        label=label)
    plt.fill_between(range(len(mean_cost)),
        mean_cost - std_cost, mean_cost + std_cost, color=color, alpha=0.1)
    plt.xlabel('Iterations')
    plt.ylabel('Packing Ratio')

fig = plt.figure(figsize=(16,4))

plt.rcParams['font.size'] = '16'
filename_array = [
    "data/trajopt/ours/tree_50_allegro_hand_random_grasp_1.pkl",
    "data/trajopt/ours/tree_50_allegro_hand_random_grasp_2.pkl",
    "data/trajopt/ours/tree_50_allegro_hand_random_grasp_3.pkl",
    "data/trajopt/ours/tree_50_allegro_hand_random_grasp_4.pkl",
    "data/trajopt/ours/tree_50_allegro_hand_random_grasp_5.pkl",
]
plot_filename_array(filename_array, 'springgreen', 'iRS-RRT')

filename_array = [
    "data/trajopt/global/tree_50_allegro_hand_random_grasp_1.pkl",
    "data/trajopt/global/tree_50_allegro_hand_random_grasp_2.pkl",
    "data/trajopt/global/tree_50_allegro_hand_random_grasp_3.pkl",
    "data/trajopt/global/tree_50_allegro_hand_random_grasp_4.pkl",
    "data/trajopt/global/tree_50_allegro_hand_random_grasp_5.pkl",
]
plot_filename_array(filename_array, 'red', 'Global Metric')


filename_array = [
    "data/trajopt/nocontact/tree_50_allegro_hand_random_grasp_1.pkl",
    "data/trajopt/nocontact/tree_50_allegro_hand_random_grasp_2.pkl",
    "data/trajopt/nocontact/tree_50_allegro_hand_random_grasp_3.pkl",
    "data/trajopt/nocontact/tree_50_allegro_hand_random_grasp_4.pkl",
    "data/trajopt/nocontact/tree_50_allegro_hand_random_grasp_5.pkl",
]
plot_filename_array(filename_array, 'royalblue', 'No Contact')

plt.subplot(1,2,1)
plt.legend()
plt.subplot(1,2,2)
plt.legend()

fig.set_figheight(6)
fig.set_figwidth(12)
# plt.show()
plt.savefig("trajopt_pen.png")