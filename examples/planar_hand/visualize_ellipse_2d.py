from distutils.dir_util import create_tree
import os.path
import time
from matplotlib import patches
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

import cProfile
from dash_vis.dash_common import create_pca_plots

from pydrake.all import PiecewisePolynomial

from qsim.simulator import QuasistaticSimulator, GradientMode
from qsim_cpp import QuasistaticSimulatorCpp

from irs_rrt.irs_rrt import IrsRrt, IrsNode
from irs_rrt.rrt_params import IrsRrtParams
from irs_rrt.irs_rrt_projection import IrsRrtProjection

from planar_hand_setup import *

parser = argparse.ArgumentParser()
parser.add_argument("tree_file_path")
args = parser.parse_args()


# %% Construct computational tools.
with open(args.tree_file_path, "rb") as f:
    tree = pickle.load(f)

irs_rrt = IrsRrt.make_from_pickled_tree(tree)
q_dynamics = irs_rrt.q_dynamics
q_sim_py = q_dynamics.q_sim_py
n_nodes = len(tree.nodes)

n_q_u = q_dynamics.dim_x - q_dynamics.dim_u
q_nodes = np.zeros((n_nodes, q_dynamics.dim_x))

# node coordinates.
for i in range(n_nodes):
    node = tree.nodes[i]["node"]
    q_nodes[i] = node.q

print(q_dynamics.get_q_u_indices_into_x())

q_u_nodes = q_nodes[:, q_dynamics.get_q_u_indices_into_x()]

# Edges. Assuming that the GRAPH IS A TREE.
z_edges = []
theta_edges = []
for i_node in tree.nodes:
    if i_node == 0:
        continue
    i_parents = list(tree.predecessors(i_node))
    i_parent = i_parents[0]
    z_edges += [q_u_nodes[i_node, 0], q_u_nodes[i_parent, 0], None]
    theta_edges += [q_u_nodes[i_node, 1], q_u_nodes[i_parent, 1], None]

r = 0.1


def compute_min_metric(num_nodes: int, q_query):

    dist = irs_rrt.calc_distance_batch_local(
        np.array([0.0, 0.0, 0.0, 0.0, q_query[0], q_query[1]]),
        num_nodes,
        is_q_u_only=True,
    )
    return np.min(dist), np.argmin(dist)


def create_tree_plot_up_to_node(num_nodes: int):
    nodes_plot = plt.plot(
        q_u_nodes[:num_nodes, 0], q_u_nodes[:num_nodes, 1], "ro"
    )

    edges_plot = plt.plot(
        z_edges[: (num_nodes - 1) * 3], theta_edges[: (num_nodes - 1) * 3], "k"
    )

    for i in range(num_nodes):
        node = irs_rrt.get_node_from_id(i)
        U, Sigma, Vh = np.linalg.svd(node.covinv_u)

        major = r / np.sqrt(Sigma[0])
        minor = r / np.sqrt(Sigma[1])

        major_vec = Vh[0, :]
        minor_vec = Vh[1, :]

        theta = np.arctan2(major_vec[1], major_vec[0])
        ellipse = patches.Ellipse(
            node.q[-2:], major, minor, theta, facecolor=(0, 1, 0, 0.1)
        )
        plt.gca().add_patch(ellipse)

    x = np.linspace(0.3, 0.5, 100)
    y = np.linspace(-0.01, np.pi, 100)
    xv, yv = np.meshgrid(x, y)

    zv = np.zeros((100, 100))
    zv_ind = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            zv[i, j], zv_ind[i, j] = compute_min_metric(
                num_nodes, np.array([xv[i, j], yv[i, j]])
            )
    cp = plt.gca().contour(xv, yv, zv, 100)

    plt.gca().clabel(cp, inline=1, fontsize=10)

    # Assign colors.
    unique_coloring = np.unique(zv_ind)
    colormap = cm.get_cmap("gist_rainbow")

    colored_coords = np.zeros((10000, 6))
    k = 0
    for i in range(100):
        for j in range(100):
            color = colormap(
                np.argwhere(unique_coloring == zv_ind[i, j])
                / len(unique_coloring)
            )[0][0]

            colored_coords[k, :] = np.array(
                [x[j], y[i], color[0], color[1], color[2], 0.5]
            )

            k = k + 1

    print(colored_coords[:, 2:6].shape)

    # plt.scatter(colored_coords[:,0],
    #    colored_coords[:,1], color=colored_coords[:,2:6].transpose(0,1))


plt.figure()
create_tree_plot_up_to_node(2000)
plt.show()
