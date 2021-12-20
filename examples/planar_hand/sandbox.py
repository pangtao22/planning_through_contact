import os
import json
import pickle

import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from dash.dependencies import Input, Output
import plotly.graph_objects as go

from qsim.simulator import (QuasistaticSimulator, QuasistaticSimParameters)
from planar_hand_setup import (model_directive_path,
                               robot_stiffness_dict, object_sdf_dict,
                               robot_l_name, robot_r_name, object_name)

from rrt.utils import set_orthographic_camera_yz

# %% load data from disk and format data.
file_names_prefix = ['du_', 'qa_l_', 'qa_r_', 'qu_']
suffix = '_r0.2'

data = []
for name in file_names_prefix:
    path = os.path.join('data', f'{name}{suffix}.pkl')
    with open(path, 'rb') as f:
        data.append(pickle.load(f))

du, qa_l, qa_r, qu = data


#%% prepare panda data frame
df = pd.DataFrame(np.vstack([qu['1_step'], qu['multi_step']]),
                  columns=['y', 'z', 'theta'])
df['type'] = ['1_step'] * len(qu['1_step']) + ['multi_step'] * len(qu['multi_step'])


#%%
fig2 = px.scatter_3d(df, x='y', y='z', z='theta', color='type')

#%% combine reachability and trj_opt data into one dictionary.
path = os.path.join('data', 'traj_opt_fingers_and_sphere_on_a_circle_config_2.pkl')
with open(path, 'rb') as f:
    trj_data = pickle.load(f)

file_names_prefix = ['du_', 'qa_l_', 'qa_r_', 'qu_']
suffix = '_r0.2'

reachable_set_data = []
for name in file_names_prefix:
    path = os.path.join('data', f'{name}{suffix}.pkl')
    with open(path, 'rb') as f:
        reachable_set_data.append(pickle.load(f))

du, qa_l, qa_r, qu = reachable_set_data


reachability_trj_opt = dict(
    qu_0=np.array([-0.2, 0.3, 0]),
    reachable_set_radius=0.2,
    trj_data=trj_data,
    reachable_set_data=dict(du=du, qa_l=qa_l, qa_r=qa_r, qu=qu))

#%%
with open(f"reachability_trj_opt_01.pkl", 'wb') as f:
    pickle.dump(reachability_trj_opt, f)

