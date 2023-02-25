#!/usr/bin/env python3

import argparse
import json
import pickle
import time

import dash
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from matplotlib import cm
from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix
from qsim.simulator import InternalVisualizationType

from dash_vis.dash_common import (
    set_orthographic_camera_yz,
    hover_template_y_z_theta,
    layout,
    make_large_point_3d,
    make_ellipsoid_plotly,
    calc_X_WG,
    trace_path_to_root_from_node,
)
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from irs_rrt.irs_rrt import IrsRrt

parser = argparse.ArgumentParser()
parser.add_argument("tree_file_path", type=str)
parser.add_argument("--two_d", default=False, action="store_true")
parser.add_argument("--draw_ellipsoids", default=False, action="store_true")
parser.add_argument("--port", default=8050, type=int)
args = parser.parse_args()

# %% Construct computational tools.
with open(args.tree_file_path, "rb") as f:
    tree = pickle.load(f)

irs_rrt_obj = IrsRrt.make_from_pickled_tree(
    tree, internal_vis=InternalVisualizationType.Python
)
q_sim, q_sim_py = irs_rrt_obj.q_sim, irs_rrt_obj.q_sim_py
plant = q_sim.get_plant()
q_vis = QuasistaticVisualizer(q_sim=q_sim, q_sim_py=q_sim_py)
meshcat_vis = q_sim_py.viz.vis  # meshcat.Visualizer (from meshcat-python)

if args.two_d:
    set_orthographic_camera_yz(meshcat_vis)
z_height = 0.25
q_goal = tree.graph["irs_rrt_params"].goal
q_u_goal = q_goal[q_sim.get_q_u_indices_into_q()]

q_vis.draw_configuration(tree.nodes[0]["node"].q)
q_vis.draw_object_triad(
    length=0.4, radius=0.005, opacity=1, path="sphere/sphere"
)
kGoalVisPrefix = q_vis.draw_goal_triad(
    length=0.4,
    radius=0.01,
    opacity=0.7,
    X_WG=RigidTransform(
        RollPitchYaw(q_u_goal[2], 0, 0), np.hstack([[z_height], q_u_goal[:2]])
    ),
)

# %%
"""
This visualizer works only for 2D systems with 3 DOFs, which are
    [y, z, theta].
"""
n_nodes = len(tree.nodes)
n_q_u = q_sim.num_unactuated_dofs()
n_q = plant.num_positions()
if not (n_q_u == 3 or n_q_u == 2):
    raise RuntimeError("Visualizing planar systems only.")

q_nodes = np.zeros((n_nodes, n_q))

# node coordinates.
for i in range(n_nodes):
    node = tree.nodes[i]["node"]
    q_nodes[i] = node.q

idx_q_u_into_x = q_sim.get_q_u_indices_into_q()
q_u_nodes = q_nodes[:, idx_q_u_into_x]
if n_q_u == 2:
    q_u_nodes = np.hstack([q_u_nodes, np.zeros((len(q_u_nodes), 1))])

# Edges. Assuming that the GRAPH IS A TREE.
y_edges = []
z_edges = []
theta_edges = []
for i_node in tree.nodes:
    if i_node == 0:
        continue
    i_parents = list(tree.predecessors(i_node))
    i_parent = i_parents[0]
    y_edges += [q_u_nodes[i_node, 0], q_u_nodes[i_parent, 0], None]
    z_edges += [q_u_nodes[i_node, 1], q_u_nodes[i_parent, 1], None]
    theta_edges += [q_u_nodes[i_node, 2], q_u_nodes[i_parent, 2], None]


def scalar_to_rgb255(v: float):
    """
    v is a scalar between 0 and 1.
    """
    r, g, b = cm.jet(v)[:3]
    return int(r * 255), int(g * 255), int(b * 255)


# Draw ellipsoids.
ellipsoid_mesh_points = []
ellipsoid_volumes = []
for i in range(n_nodes):
    node = tree.nodes[i]["node"]
    cov_inv_u = node.covinv_u
    p_center = node.q[idx_q_u_into_x]
    e_points, volume = make_ellipsoid_plotly(cov_inv_u, p_center, 0.08, 8)
    ellipsoid_mesh_points.append(e_points)
    ellipsoid_volumes.append(volume)

# compute color
ellipsoid_volumes = np.array(ellipsoid_volumes)
v_95 = np.percentile(ellipsoid_volumes, 95)
v_normalized = np.minimum(ellipsoid_volumes / v_95, 1)
e_plot_list = []
for i, (x, v) in enumerate(zip(ellipsoid_mesh_points, v_normalized)):
    r, g, b = scalar_to_rgb255(v)
    e_plot_list.append(
        go.Mesh3d(
            dict(
                x=x[0],
                y=x[1],
                z=x[2],
                alphahull=0,
                name=f"ellip{i}",
                color=f"rgb({r}, {g}, {b})",
                opacity=0.5,
            )
        )
    )


def create_tree_plot_up_to_node(num_nodes: int):
    nodes_plot = go.Scatter3d(
        x=q_u_nodes[:num_nodes, 0],
        y=q_u_nodes[:num_nodes, 1],
        z=q_u_nodes[:num_nodes, 2],
        name="nodes",
        mode="markers",
        hovertemplate=hover_template_y_z_theta,
        marker=dict(
            size=3, color=v_normalized, colorscale="jet", showscale=True
        ),
    )

    edges_plot = go.Scatter3d(
        x=y_edges[: (num_nodes - 1) * 3],
        y=z_edges[: (num_nodes - 1) * 3],
        z=theta_edges[: (num_nodes - 1) * 3],
        name="edges",
        mode="lines",
        line=dict(color="blue", width=2),
        opacity=0.5,
    )

    path_plot = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        name="path",
        mode="lines",
        line=dict(color="crimson", width=5),
    )

    root_plot = make_large_point_3d(q_u_nodes[0], name="root")
    goal_plot = make_large_point_3d(
        q_u_goal, name=kGoalVisPrefix, color="green"
    )

    return [nodes_plot, edges_plot, root_plot, goal_plot, path_plot]


"""
It is important to put the ellipsoid list in the front, so that the 
curveNumber of the first ellipsoid is 0, which can be used to index into tree 
nodes.
"""
if args.draw_ellipsoids:
    fig = go.Figure(
        data=e_plot_list + create_tree_plot_up_to_node(n_nodes), layout=layout
    )
else:
    fig = go.Figure(data=create_tree_plot_up_to_node(n_nodes), layout=layout)

# global variable for histogram figure.
fig_hist_local = {}
fig_hist_local_u = {}
fig_hist_global = {}

# %% dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="tree-fig", figure=fig),
                    width={"size": 6, "offset": 0, "order": 0},
                ),
                dbc.Col(
                    html.Iframe(
                        src=meshcat_vis.url(),
                        height=800,
                        width=1000,
                    ),
                    width={"size": 6, "offset": 0, "order": 0},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Tree Growth"),
                        dcc.Slider(
                            id="tree-progress",
                            min=0,
                            max=n_nodes - 1,
                            value=0,
                            step=1,
                            marks={
                                0: {"label": "0"},
                                n_nodes: {"label": f"{n_nodes}"},
                            },
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                    ],
                    width={"size": 6, "offset": 0, "order": 0},
                ),
                dbc.Col(
                    [
                        html.H5("Aspect mode"),
                        dcc.Dropdown(
                            id="aspect-mode",
                            options=[
                                {"label": "auto", "value": "auto"},
                                {"label": "data", "value": "data"},
                                {"label": "manual", "value": "manual"},
                                {"label": "cube", "value": "cube"},
                            ],
                            value="data",
                            placeholder="Select aspect mode",
                        ),
                    ],
                    width={"size": 1, "offset": 0, "order": 0},
                ),
                dbc.Col(
                    [
                        html.H5("Distance metric"),
                        dcc.Dropdown(
                            id="metric-to-plot",
                            options=[
                                {"label": i, "value": i}
                                for i in [
                                    "local",
                                    "local_u",
                                    "global",
                                    "global_u",
                                ]
                            ],
                            value="local_u",
                            placeholder="Select metric to plot",
                        ),
                    ],
                    width={"size": 1, "offset": 0, "order": 0},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id="cost-histogram-local", figure=fig_hist_local
                        )
                    ],
                    width={"size": 4, "offset": 0, "order": 0},
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id="cost-histogram-local-u", figure=fig_hist_local_u
                        )
                    ],
                    width={"size": 4, "offset": 0, "order": 0},
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id="cost-histogram-global", figure=fig_hist_global
                        )
                    ],
                    width={"size": 4, "offset": 0, "order": 0},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(
                            """
                **Hover Data**

                Mouse over values in the graph.
            """
                        ),
                        html.Pre(id="hover-data", style=styles["pre"]),
                    ],
                    width={"size": 3, "offset": 0, "order": 0},
                )
            ]
        ),
    ],
    fluid=True,
)


def get_tree_node_idx(point, curve):
    if curve["name"] == "nodes":
        return point["pointNumber"]

    if curve["name"].startswith("ellip"):
        return point["curveNumber"]

    return None


def plot_best_nodes(q_g_u: np.ndarray, distances: np.ndarray, best_n: int):
    if len(distances) <= best_n:
        indices = np.argsort(distances)
    else:
        # indices of the best (small cost) (best_n) nodes
        indices = np.argsort(distances)[:best_n]
    max_of_best_distances = np.max(distances[indices])
    best_n_plots = []
    for i, idx in enumerate(indices):
        width = 7 if i == 0 else 2
        q_u = tree.nodes[idx]["node"].q[irs_rrt_obj.q_u_indices_into_x]
        r, g, b = scalar_to_rgb255(distances[idx] / max_of_best_distances)
        best_n_plots.append(
            go.Scatter3d(
                x=[q_u[0], q_g_u[0]],
                y=[q_u[1], q_g_u[1]],
                z=[q_u[2], q_g_u[2]] if len(q_u) == 3 else [0, 0],
                name=f"best{i}",
                mode="lines",
                line=dict(
                    color=f"rgb({r}, {g}, {b})", width=width, dash="dash"
                ),
            )
        )

    print("best costs:", distances[indices])
    return best_n_plots


@app.callback(Output("hover-data", "children"), Input("tree-fig", "hoverData"))
def display_config_in_meshcat(hover_data):
    if hover_data is None:
        return json.dumps(hover_data, indent=2)

    point = hover_data["points"][0]
    curve = fig.data[point["curveNumber"]]
    i_node = get_tree_node_idx(point, curve)

    if i_node is None:
        return json.dumps(hover_data, indent=2)
    q_vis.draw_configuration(tree.nodes[i_node]["node"].q)
    return json.dumps(hover_data, indent=2)


@app.callback(
    [
        Output("tree-fig", "figure"),
        Output("cost-histogram-local", "figure"),
        Output("cost-histogram-local-u", "figure"),
        Output("cost-histogram-global", "figure"),
    ],
    [
        Input("tree-fig", "clickData"),
        Input("tree-progress", "value"),
        Input("aspect-mode", "value"),
        Input("metric-to-plot", "value"),
    ],
    [State("tree-fig", "relayoutData"), State("aspect-mode", "value")],
)
def tree_fig_callback(
    click_data,
    slider_value,
    aspect_mode,
    metric_to_plot,
    relayout_data,
    aspect_mode_as_state,
):
    ctx = dash.callback_context
    histograms = [fig_hist_local, fig_hist_local_u, fig_hist_global]

    if not ctx.triggered:
        return fig, *histograms
    else:
        input_name = ctx.triggered[0]["prop_id"].split(".")[0]

    num_nodes = slider_value + 1
    if input_name == "tree-fig":
        figs_list = click_callback(click_data, relayout_data)
        figs_list[0].update_scenes({"aspectmode": aspect_mode_as_state})
        return figs_list
    if input_name == "tree-progress" or input_name == "metric-to-plot":
        figs_list = slider_callback(num_nodes, metric_to_plot, relayout_data)
        figs_list[0].update_scenes({"aspectmode": aspect_mode_as_state})
        return figs_list
    if input_name == "aspect-mode":
        fig.update_scenes({"aspectmode": aspect_mode})
        return fig, *histograms


def click_callback(click_data, relayout_data):
    if click_data is None:
        return fig

    point = click_data["points"][0]
    curve = fig.data[point["curveNumber"]]
    i_node = get_tree_node_idx(point, curve)
    if i_node is None:
        return fig

    q_u_path, x_trj = trace_path_to_root_from_node(
        i_node=i_node,
        q_u_nodes=q_u_nodes,
        q_nodes=q_nodes,
        tree=tree,
    )
    fig.update_traces(
        x=q_u_path[:, 0],
        y=q_u_path[:, 1],
        z=q_u_path[:, 2],
        selector=dict(name="path"),
    )
    try:
        fig.update_layout(scene_camera=relayout_data["scene.camera"])
    except KeyError:
        pass

    # show path in meshcat
    q_vis.publish_trajectory(x_trj, h=irs_rrt_obj.rrt_params.h)

    return fig, fig_hist_local, fig_hist_local_u, fig_hist_global


def slider_callback(num_nodes, metric_to_plot, relayout_data):
    # Reachability ellipsoids.
    if args.draw_ellipsoids:
        traces_list = e_plot_list[:num_nodes]
    else:
        traces_list = []
    # Tree nodes and edges
    traces_list += create_tree_plot_up_to_node(num_nodes)
    global fig, fig_hist_local, fig_hist_local_u, fig_hist_global

    i_node = num_nodes - 1
    if i_node == 0:
        return fig, fig_hist_local, fig_hist_local_u, fig_hist_global

    # Subgoal
    # When the newest leaf in the tree is node_parent,
    # the tree is extended towards node_current.subgoal, and node_current is
    # added to the tree, replacing node_parent as the newest leaf.
    node_current = tree.nodes[i_node]["node"]
    i_parent = list(tree.predecessors(i_node))[0]
    node_parent = tree.nodes[i_parent]["node"]

    q_p = node_parent.q
    q_p_u = q_p[irs_rrt_obj.q_u_indices_into_x]
    q_u = node_current.q[irs_rrt_obj.q_u_indices_into_x]
    q_g = node_current.subgoal
    q_g_u = q_g[irs_rrt_obj.q_u_indices_into_x]
    traces_list.append(
        make_large_point_3d(p=q_g_u, name="subgoal", color="red")
    )
    traces_list.append(
        make_large_point_3d(p=q_p_u, name="parent", color="darkslateblue")
    )
    traces_list.append(
        make_large_point_3d(
            p=q_u, name="new leaf", color="red", symbol="diamond"
        )
    )

    # show subgoal in meshcat
    X_WG = calc_X_WG(
        y=q_g_u[0], z=q_g_u[1], theta=q_g_u[2] if len(q_g_u) == 3 else 0
    )
    meshcat_vis["goal"].set_transform(X_WG.GetAsMatrix4())

    # Subgoal to parent in red dashed line.
    distance_metric = tree.graph["irs_rrt_params"].distance_metric
    edge_to_parent = go.Scatter3d(
        x=[q_p_u[0], q_g_u[0]],
        y=[q_p_u[1], q_g_u[1]],
        z=[q_p_u[2], q_g_u[2]] if len(q_g_u) == 3 else [0, 0],
        name=f"attempt {distance_metric}",
        mode="lines",
        line=dict(color="red", width=5),
        opacity=0.3,
    )
    traces_list.append(edge_to_parent)

    # Subgoal cost histogram
    d_local = irs_rrt_obj.calc_distance_batch(
        q_query=q_g, n_nodes=num_nodes - 1, distance_metric="local"
    )
    d_local_u = irs_rrt_obj.calc_distance_batch(
        q_query=q_g, n_nodes=num_nodes - 1, distance_metric="local_u"
    )
    d_global = irs_rrt_obj.calc_distance_batch(
        q_query=q_g, n_nodes=num_nodes - 1, distance_metric="global"
    )
    d_global_u = irs_rrt_obj.calc_distance_batch(
        q_query=q_g, n_nodes=num_nodes - 1, distance_metric="global_u"
    )
    assert len(d_local) == num_nodes - 1
    assert len(d_global) == num_nodes - 1

    d_dict = {
        "local": d_local,
        "local_u": d_local_u,
        "global": d_global,
        "global_u": d_global_u,
    }

    # Best 10 nodes in tree to the subgoal.
    # TODO: try not to recompute every distance metric when updating the tree
    #  plot for distance metrics.
    traces_list += plot_best_nodes(
        q_g_u=q_g_u, distances=d_dict[metric_to_plot], best_n=10
    )

    # distance histograms.
    df_local = pd.DataFrame(
        {
            "log10_distance": np.log10(d_local).tolist(),
            "metric": ["local"] * (num_nodes - 1),
        }
    )
    df_local_u = pd.DataFrame(
        {
            "log10_distance": np.log10(d_local_u).tolist(),
            "metric": ["local_u"] * (num_nodes - 1),
        }
    )
    distances_global = (
        np.log10(d_global).tolist() + np.log10(d_global_u).tolist()
    )
    metric_global = ["global"] * (num_nodes - 1)
    metric_global += ["global_u"] * (num_nodes - 1)
    df_global = pd.DataFrame(
        {"log10_distance": distances_global, "metric": metric_global}
    )
    fig_hist_local = px.histogram(
        df_local,
        x="log10_distance",
        color="metric",
        nbins=40,
        color_discrete_sequence=["chartreuse"],
    )
    fig_hist_local_u = px.histogram(
        df_local_u,
        x="log10_distance",
        color="metric",
        color_discrete_sequence=["deeppink"],
        nbins=40,
    )
    fig_hist_global = px.histogram(
        df_global, x="log10_distance", color="metric", nbins=40
    )

    # update fig.
    fig = go.Figure(data=traces_list, layout=layout)

    try:
        fig.update_layout(scene_camera=relayout_data["scene.camera"])
    except KeyError:
        pass

    return fig, fig_hist_local, fig_hist_local_u, fig_hist_global


if __name__ == "__main__":
    plt.hist(ellipsoid_volumes, bins=50)
    plt.title("ellipsoid_volumes")
    plt.show()
    app.run_server(debug=False, port=args.port)
