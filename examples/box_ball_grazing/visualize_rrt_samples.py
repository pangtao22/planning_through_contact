import pickle

import numpy as np

import plotly.graph_objects as go
from irs_rrt.irs_rrt import IrsRrt
from irs_mpc2.quasistatic_visualizer import InternalVisualizationType


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
)

q_plot = go.Scatter3d(
    x=q_nodes[:, 1],
    y=q_nodes[:, 2],
    z=q_nodes[:, 0],
    mode="markers",
    marker=dict(size=3, color=v_normalized, colorscale="jet", showscale=True),
)

fig = go.Figure(data=[q_plot], layout=layout)
fig.show()
