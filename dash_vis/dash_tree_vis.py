#!/usr/bin/env python3

import argparse
import json
import pickle

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_vis.dash_common import (hover_template_y_z_theta,
                                  layout, make_large_point_3d,
                                  make_ellipsoid_plotly,
                                  set_orthographic_camera_yz,
                                  add_goal_meshcat,
                                  calc_X_WG)
from dash.exceptions import PreventUpdate

import matplotlib.pyplot as plt
from matplotlib import cm

from irs_rrt.irs_rrt import IrsRrt

parser = argparse.ArgumentParser()
parser.add_argument("tree_file_path")
args = parser.parse_args()

# %% Construct computational tools.
with open(args.tree_file_path, 'rb') as f:
    tree = pickle.load(f)

irs_rrt = IrsRrt.make_from_pickled_tree(tree)
q_dynamics = irs_rrt.q_dynamics
q_sim_py = q_dynamics.q_sim_py
vis = q_dynamics.q_sim_py.viz.vis
set_orthographic_camera_yz(q_dynamics.q_sim_py.viz.vis)
add_goal_meshcat(vis)

# %%
"""
This visualizer works only for 2D systems with 3 DOFs, which are
    [y, z, theta].
"""
n_nodes = len(tree.nodes)
n_q_u = q_dynamics.dim_x - q_dynamics.dim_u
assert n_q_u == 3
q_nodes = np.zeros((n_nodes, q_dynamics.dim_x))

# node coordinates.
for i in range(n_nodes):

    node = tree.nodes[i]["node"]
    q_nodes[i] = node.q

q_u_nodes = q_nodes[:, q_dynamics.get_q_u_indices_into_x()]

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


def create_tree_plot_up_to_node(num_nodes: int):
    nodes_plot = go.Scatter3d(x=q_u_nodes[:num_nodes, 0],
                              y=q_u_nodes[:num_nodes, 1],
                              z=q_u_nodes[:num_nodes, 2],
                              name='nodes',
                              mode='markers',
                              hovertemplate=hover_template_y_z_theta,
                              marker=dict(size=3, color='azure'))

    edges_plot = go.Scatter3d(x=y_edges[:(num_nodes - 1) * 3],
                              y=z_edges[:(num_nodes - 1) * 3],
                              z=theta_edges[:(num_nodes - 1) * 3],
                              name='edges',
                              mode='lines',
                              line=dict(color='blue', width=2),
                              opacity=0.5)

    path_plot = go.Scatter3d(x=[],
                             y=[],
                             z=[],
                             name='path',
                             mode='lines',
                             line=dict(color='crimson', width=5))

    root_plot = make_large_point_3d(q_u_nodes[0], name='root')

    return [nodes_plot, edges_plot, root_plot, path_plot]


# Draw ellipsoids.
ellipsoid_mesh_points = []
ellipsoid_volumes = []
for i in range(n_nodes):
    node = tree.nodes[i]["node"]
    cov_inv_u = node.covinv_u
    p_center = node.q[-3:]
    e_points, volume = make_ellipsoid_plotly(cov_inv_u, p_center, 0.1, 10)
    ellipsoid_mesh_points.append(e_points)
    ellipsoid_volumes.append(volume)


def scalar_to_rgb255(v: float):
    """
    v is a scalar between 0 and 1.
    """
    r, g, b = cm.jet(v)[:3]
    return int(r * 255), int(g * 255), int(b * 255)


# compute color
ellipsoid_volumes = np.array(ellipsoid_volumes)
v_99 = np.percentile(ellipsoid_volumes, 99)
v_normalized = np.minimum(ellipsoid_volumes / v_99, 1)
e_plot_list = []
for i, (x, v) in enumerate(zip(ellipsoid_mesh_points, v_normalized)):
    r, g, b = scalar_to_rgb255(v)
    e_plot_list.append(
        go.Mesh3d(dict(x=x[0], y=x[1], z=x[2], alphahull=0, name=f"ellip{i}",
                       color=f"rgb({r}, {g}, {b})", opacity=0.5)))

'''
It is important to put the ellipsoid list in the front, so that the 
curveNumber of the first ellipsoid is 0, which can be used to index into tree 
nodes.
'''
fig = go.Figure(data=e_plot_list + create_tree_plot_up_to_node(n_nodes),
                layout=layout)

# global variable for histrogram figure.
fig_hist_local = {}
fig_hist_local_u = {}
fig_hist_global = {}

# %% dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='tree-fig',
                figure=fig),
            width={'size': 6, 'offset': 0, 'order': 0},
        ),
        dbc.Col(
            html.Iframe(src='http://127.0.0.1:7000/static/',
                        height=800, width=1000),
            width={'size': 6, 'offset': 0, 'order': 0},
        )
    ]),
    dbc.Row([
        dbc.Col([
            html.H3('Tree Growth'),
            dcc.Slider(id='tree-progress', min=0, max=n_nodes - 1,
                       value=0, step=1,
                       marks={0: {'label': '0'},
                              n_nodes: {'label': f'{n_nodes}'}},
                       tooltip={"placement": "bottom", "always_visible": True}
                       )],
            width={'size': 6, 'offset': 0, 'order': 0}),
        dbc.Col([
            html.H5('Aspect mode'),
            dcc.Dropdown(id='aspect-mode',
                         options=[{'label': 'auto', 'value': 'auto'},
                                  {'label': 'data', 'value': 'data'},
                                  {'label': 'manual', 'value': 'manual'},
                                  {'label': 'cube', 'value': 'cube'}],
                         value='data',
                         placeholder="Select aspect mode",
                         )],
            width={'size': 1, 'offset': 0, 'order': 0}),
        dbc.Col([
            html.H5('Distance metric'),
            dcc.Dropdown(id='metric-to-plot',
                         options=[{'label': i, 'value': i}
                                  for i in ['local', 'local_u',
                                            'global', 'global_u']],
                         value='local_u',
                         placeholder="Select metric to plot",
                         )],
            width={'size': 1, 'offset': 0, 'order': 0}),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='cost-histogram-local', figure=fig_hist_local)],
            width={'size': 4, 'offset': 0, 'order': 0}),
        dbc.Col([
            dcc.Graph(id='cost-histogram-local-u', figure=fig_hist_local_u)],
            width={'size': 4, 'offset': 0, 'order': 0}),
        dbc.Col([
            dcc.Graph(id='cost-histogram-global', figure=fig_hist_global)],
            width={'size': 4, 'offset': 0, 'order': 0})
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Markdown("""
                **Hover Data**

                Mouse over values in the graph.
            """),
            html.Pre(id='hover-data', style=styles['pre'])],
            width={'size': 3, 'offset': 0, 'order': 0})
    ]),
], fluid=True)


def get_tree_node_idx(point, curve):
    if curve['name'] == 'nodes':
        return point['pointNumber']

    if curve['name'].startswith('ellip'):
        return point['curveNumber']

    return None


@app.callback(
    Output('hover-data', 'children'),
    Input('tree-fig', 'hoverData'))
def display_config_in_meshcat(hover_data):
    if hover_data is None:
        return json.dumps(hover_data, indent=2)

    point = hover_data['points'][0]
    curve = fig.data[point['curveNumber']]
    i_node = get_tree_node_idx(point, curve)

    if i_node is None:
        return json.dumps(hover_data, indent=2)

    q_sim_py.update_mbp_positions_from_vector(tree.nodes[i_node]["node"].q)
    q_sim_py.draw_current_configuration()

    return json.dumps(hover_data, indent=2)


@app.callback(
    [Output('tree-fig', 'figure'), Output('cost-histogram-local', 'figure'),
     Output('cost-histogram-local-u', 'figure'),
     Output('cost-histogram-global', 'figure')],
    [Input('tree-fig', 'clickData'), Input('tree-progress', 'value'),
     Input('aspect-mode', 'value'), Input('metric-to-plot', 'value')],
    [State('tree-fig', 'relayoutData'), State('aspect-mode', 'value')])
def tree_fig_callback(click_data, slider_value, aspect_mode,
                      metric_to_plot, relayout_data, aspect_mode_as_state):
    ctx = dash.callback_context
    histograms = [fig_hist_local, fig_hist_local_u, fig_hist_global]

    if not ctx.triggered:
        return fig, *histograms
    else:
        input_name = ctx.triggered[0]['prop_id'].split('.')[0]

    num_nodes = slider_value + 1
    if input_name == 'tree-fig':
        figs_list = click_callback(click_data, relayout_data)
        figs_list[0].update_scenes({'aspectmode': aspect_mode_as_state})
        return figs_list
    if input_name == 'tree-progress' or input_name == "metric-to-plot":
        figs_list = slider_callback(num_nodes, metric_to_plot, relayout_data)
        figs_list[0].update_scenes({'aspectmode': aspect_mode_as_state})
        return figs_list
    if input_name == 'aspect-mode':
        fig.update_scenes({'aspectmode': aspect_mode})
        return fig, *histograms


def click_callback(click_data, relayout_data):
    if click_data is None:
        return fig

    point = click_data['points'][0]
    curve = fig.data[point['curveNumber']]
    i_node = get_tree_node_idx(point, curve)
    if i_node is None:
        return fig

    # trace back to root to get path.
    y_path = []
    z_path = []
    theta_path = []
    idx_path = []

    while True:
        y_path.append(q_u_nodes[i_node, 0])
        z_path.append(q_u_nodes[i_node, 1])
        theta_path.append(q_u_nodes[i_node, 2])
        idx_path.append(i_node)

        i_parents = list(tree.predecessors(i_node))
        assert len(i_parents) <= 1
        if len(i_parents) == 0:
            break

        i_node = i_parents[0]

    fig.update_traces(x=y_path, y=z_path, z=theta_path,
                      selector=dict(name='path'))
    try:
        fig.update_layout(scene_camera=relayout_data['scene.camera'])
    except KeyError:
        pass

    # show path in meshcat
    idx_path.reverse()
    q_dynamics.publish_trajectory(q_nodes[idx_path], h=2 / len(idx_path))

    return fig, fig_hist_local, fig_hist_local_u, fig_hist_global


def slider_callback(num_nodes, metric_to_plot, relayout_data):
    # Reachability ellipsoids.
    traces_list = e_plot_list[:num_nodes]
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
    node_current = tree.nodes[i_node]['node']
    i_parent = list(tree.predecessors(i_node))[0]
    node_parent = tree.nodes[i_parent]['node']

    q_p = node_parent.q
    q_p_u = q_p[irs_rrt.q_u_indices_into_x]
    q_u = node_current.q[irs_rrt.q_u_indices_into_x]
    q_g = node_current.subgoal
    q_g_u = q_g[irs_rrt.q_u_indices_into_x]
    traces_list.append(make_large_point_3d(
        p=q_g_u, name='subgoal', color='red'))
    traces_list.append(make_large_point_3d(
        p=q_p_u, name='parent', color='darkslateblue'))
    traces_list.append(make_large_point_3d(
        p=q_u, name='new leaf', color='red', symbol='diamond'))

    # show subgoal in meshcat
    X_WG = calc_X_WG(y=q_g_u[0], z=q_g_u[1], theta=q_g_u[2])
    vis['goal'].set_transform(X_WG)

    # Subgoal to parent in red dashed line.
    distance_metric = tree.graph['irs_rrt_params'].distance_metric
    edge_to_parent = go.Scatter3d(
        x=[q_p_u[0], q_g_u[0]],
        y=[q_p_u[1], q_g_u[1]],
        z=[q_p_u[2], q_g_u[2]],
        name=f'attempt {distance_metric}',
        mode='lines',
        line=dict(color='red', width=5),
        opacity=0.3)
    traces_list.append(edge_to_parent)

    # Subgoal cost histogram
    d_local = irs_rrt.calc_metric_batch(
        q_query=q_g, n_nodes=num_nodes - 1, distance_metric='local')
    d_local_u = irs_rrt.calc_metric_batch(
        q_query=q_g, n_nodes=num_nodes - 1, distance_metric='local_u')
    d_global = irs_rrt.calc_metric_batch(
        q_query=q_g, n_nodes=num_nodes - 1, distance_metric='global')
    d_global_u = irs_rrt.calc_metric_batch(
        q_query=q_g, n_nodes=num_nodes - 1, distance_metric='global_u')
    assert len(d_local) == num_nodes - 1
    assert len(d_global) == num_nodes - 1

    d_dict = {"local": d_local, "local_u": d_local_u,
              "global": d_global, "global_u": d_global_u}

    # Best 10 nodes in tree to the subgoal.
    # TODO: try not to recompute every distance metric when updating the tree
    #  plot for distance metrics.
    traces_list += plot_best_nodes(q_g_u=q_g_u,
                                   distances=d_dict[metric_to_plot],
                                   best_n=10)

    # distance histograms.
    df_local = pd.DataFrame(
        {'log10_distance': np.log10(d_local).tolist(),
         'metric': ['local'] * (num_nodes - 1)})
    df_local_u = pd.DataFrame(
        {'log10_distance': np.log10(d_local_u).tolist(),
         'metric': ['local_u'] * (num_nodes - 1)})
    distances_global = (np.log10(d_global).tolist()
                        + np.log10(d_global_u).tolist())
    metric_global = ['global'] * (num_nodes - 1)
    metric_global += ['global_u'] * (num_nodes - 1)
    df_global = pd.DataFrame(
        {'log10_distance': distances_global, 'metric': metric_global})
    fig_hist_local = px.histogram(df_local, x='log10_distance', color='metric',
                                  nbins=40,
                                  color_discrete_sequence=['chartreuse'])
    fig_hist_local_u = px.histogram(df_local_u, x='log10_distance',
                                    color='metric',
                                    color_discrete_sequence=['deeppink'],
                                    nbins=40)
    fig_hist_global = px.histogram(df_global, x='log10_distance',
                                   color='metric',
                                   nbins=40)

    # update fig.
    fig = go.Figure(data=traces_list, layout=layout)

    try:
        fig.update_layout(scene_camera=relayout_data['scene.camera'])
    except KeyError:
        pass

    return fig, fig_hist_local, fig_hist_local_u, fig_hist_global


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
        q_u = tree.nodes[idx]['node'].q[irs_rrt.q_u_indices_into_x]
        r, g, b = scalar_to_rgb255(distances[idx] / max_of_best_distances)
        best_n_plots.append(
            go.Scatter3d(x=[q_u[0], q_g_u[0]],
                         y=[q_u[1], q_g_u[1]],
                         z=[q_u[2], q_g_u[2]],
                         name=f'best{i}',
                         mode='lines',
                         line=dict(color=f"rgb({r}, {g}, {b})", width=width,
                                   dash='dash')))

    print('best costs:', distances[indices])
    return best_n_plots


if __name__ == '__main__':
    app.run_server(debug=True)
