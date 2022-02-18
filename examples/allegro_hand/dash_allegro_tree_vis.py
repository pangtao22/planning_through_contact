#!/usr/bin/env python3

import argparse
import json
import pickle

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_vis.dash_common import (hover_template_y_z_theta,
                                  layout, make_large_point_3d,
                                  make_ellipsoid_plotly,
                                  set_orthographic_camera_yz)
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.reachable_set import ReachableSet
from matplotlib import cm

from pydrake.all import AngleAxis, Quaternion

parser = argparse.ArgumentParser()
parser.add_argument("tree_file_path")
args = parser.parse_args()


# %% Construct computational tools.
with open(args.tree_file_path, 'rb') as f:
    tree = pickle.load(f)
irs_rrt_param = tree.graph['irs_rrt_params']
q_model_path = irs_rrt_param.q_model_path
h = irs_rrt_param.h
q_dynamics = QuasistaticDynamics(h=h, q_model_path=q_model_path,
                                 internal_viz=True)
reachable_set = ReachableSet(q_dynamics, irs_rrt_param)
q_sim_py = q_dynamics.q_sim_py
# set_orthographic_camera_yz(q_dynamics.q_sim_py.viz.vis)

# %%
"""
This visualizer works for 3D allegro hand with [roll, pitch, yaw].
"""
n_nodes = len(tree.nodes)
q_nodes = np.zeros((n_nodes, q_dynamics.dim_x))

# node coordinates.
for i in range(n_nodes):
    node = tree.nodes[i]["node"]
    q_nodes[i] = node.q

q_u_nodes = q_nodes[:, q_dynamics.get_q_u_indices_into_x()]
q_u_nodes_rot = np.zeros((n_nodes, 3))
for i in range(n_nodes):
    aa = AngleAxis(Quaternion(q_u_nodes[i][:4]))
    q_u_nodes_rot[i] = aa.axis() * aa.angle()

# edges.
x_edges = []
y_edges = []
z_edges = []
for i_node in tree.nodes:
    if i_node == 0:
        continue
    i_parents = list(tree.predecessors(i_node))
    i_parent = i_parents[0]
    x_edges += [q_u_nodes_rot[i_node, 0], q_u_nodes_rot[i_parent, 0], None]
    y_edges += [q_u_nodes_rot[i_node, 1], q_u_nodes_rot[i_parent, 1], None]
    z_edges += [q_u_nodes_rot[i_node, 2], q_u_nodes_rot[i_parent, 2], None]


def create_tree_plot_up_to_node(num_nodes: int):
    nodes_plot = go.Scatter3d(x=q_u_nodes_rot[:num_nodes, 0],
                              y=q_u_nodes_rot[:num_nodes, 1],
                              z=q_u_nodes_rot[:num_nodes, 2],
                              name='nodes',
                              mode='markers',
                              hovertemplate=hover_template_y_z_theta,
                              marker=dict(size=3))

    edges_plot = go.Scatter3d(x=x_edges[:num_nodes * 3],
                              y=y_edges[:num_nodes * 3],
                              z=z_edges[:num_nodes * 3],
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

    root_plot = make_large_point_3d(q_u_nodes_rot[0], name='root')

    return [nodes_plot, edges_plot, root_plot, path_plot]

# # Draw ellipsoids.
# ellipsoid_mesh_points = []
# ellipsoid_volumes = []
# for i in range(n_nodes):
#     node = tree.nodes[i]["node"]
#     cov_u, _ = reachable_set.calc_unactuated_metric_parameters(
#         node.Bhat, node.chat)
#     cov_inv = np.linalg.inv(cov_u)
#     p_center = node.q[-3:]
#     e_points, volume = make_ellipsoid_plotly(cov_inv, p_center, 0.05, 8)
#     ellipsoid_mesh_points.append(e_points)
#     ellipsoid_volumes.append(volume)

# # compute color
# ellipsoid_volumes = np.array(ellipsoid_volumes)
# v_99 = np.percentile(ellipsoid_volumes, 99)
# v_normalized = np.minimum(ellipsoid_volumes / v_99, 1)
# e_plot_list = []
# for i, (x, v) in enumerate(zip(ellipsoid_mesh_points, v_normalized)):
#     r, g, b = cm.jet(v)[:3]
#     r = int(r * 255)
#     g = int(g * 255)
#     b = int(b * 255)
#     e_plot_list.append(
#         go.Mesh3d(dict(x=x[0], y=x[1], z=x[2], alphahull=0, name=f"ellip{i}",
#                        color=f"rgb({r}, {g}, {b})", opacity=0.5)))

'''
It is important to put the ellipsoid list in the front, so that the 
curveNumber of the first ellipsoid is 0, which can be used to index into tree 
nodes.
'''
fig = go.Figure(data=create_tree_plot_up_to_node(n_nodes),
                layout=layout)
fig.update_layout(scene=dict(xaxis_title_text='x_rot',
                             yaxis_title_text='y_rot',
                             zaxis_title_text='z_rot',))

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
            dcc.Slider(id='tree-progress', min=0, max=n_nodes, value=0, step=1,
                       marks={0: {'label': '0'},
                              n_nodes: {'label': f'{n_nodes}'}},
                       tooltip={"placement": "bottom", "always_visible": True}
                       ), ],
            width={'size': 6, 'offset': 0, 'order': 0}),
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
    hover_data_json = json.dumps(hover_data, indent=2)
    if hover_data is None:
        return hover_data_json

    point = hover_data['points'][0]
    curve = fig.data[point['curveNumber']]

    i_node = get_tree_node_idx(point, curve)
    if i_node is None:
        return hover_data_json

    q_sim_py.update_mbp_positions_from_vector(tree.nodes[i_node]["node"].q)
    q_sim_py.draw_current_configuration()

    return hover_data_json


@app.callback(
    Output('tree-fig', 'figure'),
    [Input('tree-fig', 'clickData'), Input('tree-progress', 'value')],
    [State('tree-fig', 'relayoutData'), State('tree-progress', 'value')])
def tree_fig_callback(click_data, slider_value, relayout_data,
                      slider_value_as_state):
    ctx = dash.callback_context

    if not ctx.triggered:
        return fig
    else:
        input_name = ctx.triggered[0]['prop_id'].split('.')[0]

    num_nodes = slider_value_as_state + 1
    if input_name == 'tree-fig':
        return click_callback(click_data, relayout_data)

    if input_name == 'tree-progress':
        return slider_callback(num_nodes, relayout_data)


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
        y_path.append(q_u_nodes_rot[i_node, 0])
        z_path.append(q_u_nodes_rot[i_node, 1])
        theta_path.append(q_u_nodes_rot[i_node, 2])
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

    return fig


def slider_callback(num_nodes, relayout_data):
    print(num_nodes)
    traces_list = []
    traces_list += create_tree_plot_up_to_node(num_nodes)
    global fig
    fig = go.Figure(data=traces_list, layout=layout)

    try:
        fig.update_layout(scene_camera=relayout_data['scene.camera'])
    except KeyError:
        pass

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
