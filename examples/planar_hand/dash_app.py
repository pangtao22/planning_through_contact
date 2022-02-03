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

from dash_app_common import (add_goal_meshcat, hover_template_reachability,
                             hover_template_trj, layout, calc_principal_points,
                             create_pca_plots, calc_X_WG, create_q_u0_plot)

# %% quasistatic dynamics
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=True)
q_sim_py = q_dynamics.q_sim_py
plant = q_dynamics.plant

model_a_l = plant.GetModelInstanceByName(robot_l_name)
model_a_r = plant.GetModelInstanceByName(robot_r_name)
model_u = plant.GetModelInstanceByName(object_name)

# %% meshcat
vis = q_dynamics.q_sim_py.viz.vis
set_orthographic_camera_yz(vis)
add_goal_meshcat(vis)

# %% load data from disk and format data.
'''
data format
name: reachability_trj_opt_xx.pkl
{# key: item
    'qu_0': (3,) array, initial pose of the sphere.
    'reachable_set_radius': float, radius of the box from which 1 step reachable
        set commands are sampled.
    'trj_data': List[Dict], where Dict is
        {
            'cost': {'Qu': float, 'Qu_f': float, 'Qa': float, 'Qa_f': float, 
                     'R': float, 'all': float},
            'x_trj': (T+1, n_q) array.
            'u_trj': (T, n_a) array.
            'dqu_goal': (3,) array. dqu_goal + qu_0 gives the goal which this 
                x_trj and u_trj tries to reach.
        }
    'reachable_set_data': # samples used to generate 1-step or multi-step 
        reachable sets.
    {
        'du': (n_samples, n_a) array,
        'qa_l': {'1_step': (n_samples, 2) array, 
                 'multi_step': (n_samples, 2) array.},
        'qa_r': {'1_step': (n_samples, 2) array, 
                 'multi_step': (n_samples, 2) array.},
        'qu': {'1_step': (n_samples, 3) array, 
               'multi_step': (n_samples, 3) array.}
    }
}
'''

with open('./data/reachability_trj_opt_00.pkl', 'rb') as f:
    reachability_trj_opt = pickle.load(f)

du = reachability_trj_opt['reachable_set_data']['du']
qa_l = reachability_trj_opt['reachable_set_data']['qa_l']
qa_r = reachability_trj_opt['reachable_set_data']['qa_r']
qu = reachability_trj_opt['reachable_set_data']['qu']

# the first row in all trajectories have the same initial object pose.
q_u0 = reachability_trj_opt['qu_0']
trj_data = reachability_trj_opt['trj_data']
dqu_goal = np.array([result['dqu_goal'] for result in trj_data])

#%% PCA of 1-step reachable set.
principal_points = calc_principal_points(qu_samples=qu['1_step'], r=0.5)

# %%
plot_1_step = go.Scatter3d(x=qu['1_step'][:, 0],
                           y=qu['1_step'][:, 1],
                           z=qu['1_step'][:, 2],
                           name='1_step',
                           mode='markers',
                           hovertemplate=hover_template_reachability,
                           marker=dict(size=2))
# plot_multi = go.Scatter3d(x=qu['multi_step'][:, 0],
#                           y=qu['multi_step'][:, 1],
#                           z=qu['multi_step'][:, 2],
#                           name='multi_step',
#                           mode='markers',
#                           hovertemplate=hover_template_reachability,
#                           marker=dict(size=2))

plot_trj = go.Scatter3d(
    x=q_u0[0] + dqu_goal[:, 0],
    y=q_u0[1] + dqu_goal[:, 1],
    z=q_u0[2] + dqu_goal[:, 2],
    name='goals',
    mode='markers',
    hovertemplate=hover_template_trj,
    marker=dict(size=5,
                color=[result['cost']['Qu_f'] for result in trj_data],
                cmin=0,
                cmax=60,
                colorscale='jet',
                showscale=True,
                opacity=0.8))

plot_qu0 = create_q_u0_plot(q_u0)

# PCA lines
principal_axes_plots = create_pca_plots(principal_points)
fig = go.Figure(data=[plot_1_step, plot_trj, plot_qu0] + principal_axes_plots,
                layout=layout)


# %%
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
                id='reachable-sets',
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
        ),
        dbc.Col([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style=styles['pre'])],
            width={'size': 3, 'offset': 0, 'order': 0}
        ),
        dbc.Col([
            dcc.Markdown("""
                **Selection Data**

                Choose the lasso or rectangle tool in the graph's menu
                bar and then select points in the graph.

                Note that if `layout.clickmode = 'event+select'`, selection data also
                accumulates (or un-accumulates) selected data if you hold down the shift
                button while clicking.
            """),
            html.Pre(id='selected-data', style=styles['pre'])],
            width={'size': 3, 'offset': 0, 'order': 0}
        ),
        dbc.Col([
            dcc.Markdown("""
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """),
            html.Pre(id='relayout-data', style=styles['pre'])],
            width={'size': 3, 'offset': 0, 'order': 0}
        )
    ])
], fluid=True)


@app.callback(
    Output('hover-data', 'children'),
    Input('reachable-sets', 'hoverData'),
    State('reachable-sets', 'figure'))
def display_hover_data(hoverData, figure):
    hover_data_json = json.dumps(hoverData, indent=2)
    if hoverData is None:
        return hover_data_json
    point = hoverData['points'][0]
    idx_fig = point['curveNumber']
    name = figure['data'][idx_fig]['name']
    idx = point["pointNumber"]

    if name == 'goals':
        pass
    elif name.startswith('pca'):
        return hover_data_json
    else:
        q_dict = {
            model_u: qu[name][idx],
            model_a_l: qa_l[name][idx],
            model_a_r: qa_r[name][idx]}

        q_sim_py.update_mbp_positions(q_dict)
        q_sim_py.draw_current_configuration()

    return hover_data_json


@app.callback(
    Output('click-data', 'children'),
    Input('reachable-sets', 'clickData'),
    State('reachable-sets', 'figure'))
def display_click_data(click_data, figure):
    click_data_json = json.dumps(click_data, indent=2)
    if click_data is None:
        return click_data_json
    point = click_data['points'][0]
    idx_fig = point['curveNumber']
    name = figure['data'][idx_fig]['name']
    idx = point["pointNumber"]

    if name == 'goals':
        X_WG = calc_X_WG(y=point['x'], z=point['y'], theta=point['z'])
        vis['goal'].set_transform(X_WG)
        q_dynamics.publish_trajectory(trj_data[idx]['x_trj'])

    return click_data_json


@app.callback(
    Output('selected-data', 'children'),
    Input('reachable-sets', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('relayout-data', 'children'),
    Input('reachable-sets', 'relayoutData'))
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=False)
