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

from dash_common import (add_goal_meshcat, hover_template_y_z_theta,
                         hover_template_trj, layout, calc_principal_points,
                         create_pca_plots, calc_X_WG, create_q_u0_plot,
                         make_ellipsoid_plotly)
from matplotlib import cm


#%%
with open('./data/tree_1000.pkl', 'rb') as f:
    tree = pickle.load(f)

q_dynamics = QuasistaticDynamics(h=h, q_model_path=q_model_path,
                                 internal_viz=True)
q_sim_py = q_dynamics.q_sim_py

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

tree_plot_list = [nodes_plot, edges_plot, root_plot, path_plot]

# Draw ellipsoids.
ellipsoid_mesh_points = []
ellipsoid_volumes = []
for i in range(n_nodes):
    node = tree.nodes[i]
    B_u = node['Bhat'][-3:, :]
    cov_inv = np.linalg.inv(B_u @ B_u.T + 1e-6 * np.eye(3))
    p_center = node['q'][-3:]
    e_points, volume = make_ellipsoid_plotly(cov_inv, p_center, 0.05, 8)
    ellipsoid_mesh_points.append(e_points)
    ellipsoid_volumes.append(volume)

# compute color
ellipsoid_volumes = np.array(ellipsoid_volumes)
v_99 = np.percentile(ellipsoid_volumes, 99)
v_normalized = np.minimum(ellipsoid_volumes / v_99, 1)
e_plot_list = []
for i, (x, v) in enumerate(zip(ellipsoid_mesh_points, v_normalized)):
    r, g, b = cm.jet(v)[:3]
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    e_plot_list.append(
        go.Mesh3d(dict(x=x[0], y=x[1], z=x[2], alphahull=0, name=f"ellip{i}",
                       color=f"rgb({r}, {g}, {b})", opacity=0.5)))

'''
It is important to put the ellipsoid list in the front, so that the 
curveNumber of the first ellipsoid is 0, which can be used to index into tree 
nodes.
'''
fig = go.Figure(data=e_plot_list + tree_plot_list,
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
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Markdown("""
                **Hover Data**

                Mouse over values in the graph.
            """),
            html.Pre(id='hover-data', style=styles['pre'])],
            width={'size': 3, 'offset': 0, 'order': 0}
        )
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

    q_sim_py.update_mbp_positions_from_vector(tree.nodes[i_node]['q'])
    q_sim_py.draw_current_configuration()

    return hover_data_json


@app.callback(
    Output('tree-fig', 'figure'), Input('tree-fig', 'clickData'),
    State('tree-fig', 'relayoutData'))
def click_callback(click_data, relayout_data):
    print(click_data)

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

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
