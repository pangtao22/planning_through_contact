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

from planar_hand_setup import *
from contact_sampler_2d import PlanarHandContactSampler2D

from tqdm import tqdm


def get_cost_array(irs_rrt, global_metric):
    n_nodes = len(irs_rrt.graph.nodes)
    costs = np.zeros(n_nodes)

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
    global_metric = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.2])
    n_samples = 100
    threshold = 3

    def sampling_function(n_samples):
        samples = np.random.rand(n_samples,7)
        samples[:,4] = 0.2 * samples[:,4] - 0.1
        samples[:,5] = 0.2 * samples[:,5] + 0.3
        samples[:,6] = (np.pi + 0.01) * samples[:,6] - 0.01
        return samples

    cost_array = get_cost_array(irs_rrt, global_metric)
    packing_ratio_array = get_packing_ratio_array(irs_rrt,
        sampling_function, n_samples, threshold=3)

    return cost_array, packing_ratio_array

def plot_filename_array(filename_array):
    cost_array_lst = []
    packing_ratio_lst = []

    for filename in filename_array:
        cost_array, packing_ratio_array = compute_statistics(filename)
        cost_array_lst.append(cost_array)
        packing_ratio_lst.append(packing_ratio_array)

    plt.figure()
    plt.subplot(1,2,1)
    cost_array_lst = np.log(np.array(cost_array_lst))
    mean_cost = np.mean(cost_array_lst, axis=0)
    std_cost  = np.std(cost_array_lst, axis=0)

    plt.plot(range(len(mean_cost)), mean_cost, 'r-')
    plt.fill_between(range(len(mean_cost)),
        mean_cost - std_cost, mean_cost + std_cost,
        color=[1,0,0,0.1])
    plt.xlabel('Iterations')
    plt.ylabel('L2 Distance to Root Node')

    plt.subplot(1,2,2)
    packing_ratio_lst = np.array(packing_ratio_lst)
    mean_cost = np.mean(packing_ratio_lst, axis=0)
    std_cost = np.std(packing_ratio_lst, axis=0)

    plt.plot(range(len(mean_cost)), mean_cost, '-', color='springgreen')
    plt.fill_between(range(len(mean_cost)),
        mean_cost - std_cost, mean_cost + std_cost, color='springgreen',
        alpha=0.1)
    plt.xlabel('Iterations')
    plt.ylabel('Packing Ratio')

    plt.show()

filename_array = [
    "tree_2000_planar_hand_rg_1.pkl",
    "tree_2000_planar_hand_rg_2.pkl",
    #"tree_2000_planar_hand_regrasp_2d_3.pkl",
    #"tree_2000_planar_hand_regrasp_2d_4.pkl",
    #"tree_2000_planar_hand_regrasp_2d_5.pkl"
]

plot_filename_array(filename_array)
