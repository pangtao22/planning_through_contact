import pickle

import numpy as np

import plotly.graph_objects as go
from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import InternalVisualizationType

import plotly.io as pio

pio.renderers.default = "browser"  # see plotly charts in pycharm.

#%%
tree_file_path = "tree_2000_good.pkl"
with open(tree_file_path, "rb") as f:
    tree = pickle.load(f)

rrt_obj = IrsRrt.make_from_pickled_tree(
    tree, internal_vis=InternalVisualizationType.Cpp
)
q_sim, q_sim_py = rrt_obj.q_sim, rrt_obj.q_sim_py
plant = q_sim.get_plant()

#%%
n_nodes = len(tree.nodes)
n_q = plant.num_positions()
n_q_u = q_sim.num_unactuated_dofs()

q_nodes = rrt_obj.q_matrix[:n_nodes]

# ellipsoid volumes
ellipsoid_volumes = []
for i in range(n_nodes):
    node = tree.nodes[i]["node"]
    ellipsoid_volumes.append(np.sqrt(node.cov_u[0, 0]))

v_95 = np.percentile(ellipsoid_volumes, 95)
v_normalized = np.minimum(ellipsoid_volumes / v_95, 1)

#%%
layout = go.Layout(
    autosize=True,
    height=900,
    legend=dict(orientation="h"),
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        camera_projection_type="perspective",
        xaxis_title_text="x_a",
        yaxis_title_text="y_a",
        zaxis_title_text="x_u",
        aspectmode="data",
        aspectratio=dict(x=1.0, y=1.0, z=1.0),
    ),
    showlegend=True,
)

x_a_nodes = q_nodes[:, 1]
y_a_nodes = q_nodes[:, 2]
x_u_nodes = q_nodes[:, 0]

q_plot = go.Scatter3d(
    x=x_a_nodes,
    y=y_a_nodes,
    z=x_u_nodes,
    mode="markers",
    marker=dict(size=3, color=v_normalized, colorscale="jet", showscale=True),
)

# draw goal plane
x_a_min = np.min(x_a_nodes)
x_a_max = np.max(x_a_nodes)
y_a_min = np.min(y_a_nodes)
y_a_max = np.max(y_a_nodes)
plane_corners = np.array(
    [
        [x_a_min, y_a_min],
        [x_a_min, y_a_max],
        [x_a_max, y_a_min],
        [x_a_max, y_a_max],
    ]
)

n_points = 5
x_a_line = np.linspace(x_a_min, x_a_max, 10)
y_a_line = np.linspace(y_a_min, y_a_max, 10)

bright_blue = [[0, "#7DF9FF"], [1, "#7DF9FF"]]
bright_pink = [[0, "#FF007F"], [1, "#FF007F"]]
light_yellow = [[0, "#FFDB58"], [1, "#FFDB58"]]

z_goal = np.ones((n_points, n_points)) * rrt_obj.goal[0]
goal_plane_plot = go.Surface(
    z=z_goal,
    x=plane_corners[:, 0],
    y=plane_corners[:, 1],
    opacity=0.2,
    showscale=False,
    colorscale=bright_blue,
    surfacecolor=np.ones((n_points, n_points)) * 0.8,
)

z_start = np.zeros((n_points, n_points))
start_plane_plot = go.Surface(
    z=z_start,
    x=plane_corners[:, 0],
    y=plane_corners[:, 1],
    opacity=0.2,
    showscale=False,
    colorscale=bright_pink,
    surfacecolor=np.ones((n_points, n_points)) * 0.2,
)


fig = go.Figure(data=[start_plane_plot, goal_plane_plot, q_plot], layout=layout)
fig.show()
