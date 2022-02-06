import os
import json
import pickle

import meshcat
import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from planar_hand_setup import (q_model_path, h,
                               robot_l_name, robot_r_name, object_name)

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from rrt.utils import set_orthographic_camera_yz

from dash_common import (add_goal_meshcat, hover_template_y_z_theta,
                         hover_template_trj, layout, calc_principal_points,
                         create_pca_plots, calc_X_WG, create_q_u0_plot)


#%%
with open('tree_1000.pkl', 'rb') as f:
    tree = pickle.load(f)

q_dynamics = QuasistaticDynamics(h=h, q_model_path=q_model_path,
                                 internal_viz=True)

#%%
n_nodes = len(tree.nodes)
n_q_u = 3
q_nodes = np.zeros((n_nodes, 7))

# node coordinates.
for i in range(n_nodes):
    node = tree.nodes[i]
    q_nodes[i] = node['q']

q_u_nodes = q_nodes[:, -3:]


# edges.
y_edges = []
z_edges = []
theta_edges = []
for i_u, i_v in tree.edges:
    y_edges += [q_u_nodes[i_u, 0], q_u_nodes[i_v, 0], None]
    z_edges += [q_u_nodes[i_u, 1], q_u_nodes[i_v, 1], None]
    theta_edges += [q_u_nodes[i_u, 2], q_u_nodes[i_v, 2], None]


nodes_plot = go.Scatter3d(x=q_u_nodes[:, 0],
                          y=q_u_nodes[:, 1],
                          z=q_u_nodes[:, 2],
                          name='nodes',
                          mode='markers',
                          hovertemplate=hover_template_y_z_theta,
                          marker=dict(size=3))

edges_plot = go.Scatter3d(x=y_edges,
                          y=z_edges,
                          z=theta_edges,
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

root_plot = create_q_u0_plot(q_u_nodes[0], name='root')


fig = go.Figure(data=[nodes_plot, edges_plot, root_plot, path_plot],
                layout=layout)

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
    ])
], fluid=True)


@app.callback(
    Output('tree-fig', 'figure'), Input('tree-fig', 'clickData'))
def click_callback(click_data):
    print(click_data)

    if click_data is None:
        return fig

    point = click_data['points'][0]
    if fig.data[point['curveNumber']]['name'] != 'nodes':
        return fig

    # trace back to root to get path.
    y_path = []
    z_path = []
    theta_path = []
    idx_path = []
    i_node = point['pointNumber']

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

    # show path in meshcat
    idx_path.reverse()
    q_dynamics.publish_trajectory(q_nodes[idx_path], h=2 / len(idx_path))

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
