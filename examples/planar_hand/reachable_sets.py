import pickle

import numpy as np
from tqdm import tqdm
import meshcat
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from qsim.simulator import (QuasistaticSimulator, QuasistaticSimParameters)
from qsim.system import (cpp_params_from_py_params)
from quasistatic_simulator.examples.setup_simulation_diagram import (
    create_dict_keyed_by_model_instance_index)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics

from planar_hand_setup import (object_sdf_path, model_directive_path, Kp,
                               robot_stiffness_dict, object_sdf_dict,
                               gravity, contact_detection_tolerance,
                               gradient_lstsq_tolerance,
                               robot_l_name, robot_r_name, object_name)
from rrt.planner import ConfigurationSpace

viz = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
pio.renderers.default = "browser"  # see plotly charts in pycharm.

#%%
h = 0.1

sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

# robot
nq_a = 4

# Python sim.
q_sim_py = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params,
    internal_vis=True)

# C++ sim.
sim_params_cpp = cpp_params_from_py_params(sim_params)
sim_params_cpp.gradient_lstsq_tolerance = gradient_lstsq_tolerance
q_sim_cpp = QuasistaticSimulatorCpp(
    model_directive_path=model_directive_path,
    robot_stiffness_str=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params_cpp)

q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)
n_a = q_dynamics.dim_u
n_u = q_dynamics.dim_x - n_a

model_a_l = q_sim_py.plant.GetModelInstanceByName(robot_l_name)
model_a_r = q_sim_py.plant.GetModelInstanceByName(robot_r_name)
model_u = q_sim_py.plant.GetModelInstanceByName(object_name)

# cspace object for sampling configurations.
cspace = ConfigurationSpace(model_u=model_u, model_a_l=model_a_l, model_a_r=model_a_r,
                            q_sim=q_sim_py)

#%% Get initial config from sampling.
q_u0 = np.array([-0.2, 0.3, 0])
q_dict = cspace.sample_contact(q_u=q_u0)
q_sim_py.update_mbp_positions(q_dict)
q_sim_py.draw_current_configuration()

#%% generate samples
n_samples = 5000
radius = 0.2
qu_samples = {"1_step": np.zeros((n_samples, n_u)),
              "multi_step": np.zeros((n_samples, n_u))}

qa_l_samples = {"1_step": np.zeros((n_samples, 2)),
                "multi_step": np.zeros((n_samples, 2))}

qa_r_samples = {"1_step": np.zeros((n_samples, 2)),
                "multi_step": np.zeros((n_samples, 2))}

du = np.random.rand(n_samples, n_a) * radius * 2 - radius

x0 = q_dynamics.get_x_from_q_dict(q_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q_dict)


def save_x(x: np.ndarray, sim_type: str):
    q_dict = q_dynamics.get_q_dict_from_x(x)
    qu_samples[sim_type][i] = q_dict[model_u]
    qa_l_samples[sim_type][i] = q_dict[model_a_l]
    qa_r_samples[sim_type][i] = q_dict[model_a_r]


for i in tqdm(range(n_samples)):
    u = u0 + du[i]
    x_1 = q_dynamics.dynamics(x0, u, requires_grad=False)
    x_multi = q_dynamics.dynamics_more_steps(x0, u, n_steps=10)
    save_x(x_1, "1_step")
    save_x(x_multi, "multi_step")


#%%
layout = go.Layout(scene=dict(aspectmode='data'), height=1000)
data_1_step = go.Scatter3d(x=qu_samples['1_step'][:, 0],
                           y=qu_samples['1_step'][:, 1],
                           z=qu_samples['1_step'][:, 2],
                           mode='markers',
                           marker=dict(color=0x00ff00,
                                       size=1.5,
                                       sizemode='diameter'))
data_multi = go.Scatter3d(x=qu_samples['multi_step'][:, 0],
                          y=qu_samples['multi_step'][:, 1],
                          z=qu_samples['multi_step'][:, 2],
                          mode='markers',
                          marker=dict(color=0x00ff00,
                                      size=1.5,
                                      sizemode='diameter'))

fig = go.Figure(data=[data_1_step, data_multi],
                layout=layout)
fig.update_scenes(camera_projection_type='orthographic',
                  xaxis_title_text='y',
                  yaxis_title_text='z',
                  zaxis_title_text='theta')
fig.show()


#%% visualize
__, axes = plt.subplots(1, 2)
plt.title('qa_samples')
axes[0].scatter(qa_l_samples[:, 0], qa_l_samples[:, 1])
axes[1].scatter(qa_r_samples[:, 0], qa_r_samples[:, 1])
for ax in axes:
    ax.axis('equal')
    ax.grid(True)
plt.show()

#%% save data to disk.
data_file_suffix = '_r0.2'
with open(f"du_{data_file_suffix}.pkl", 'wb') as f:
    pickle.dump(du, f)

with open(f"qa_l_{data_file_suffix}.pkl", 'wb') as f:
    pickle.dump(qa_l_samples, f)

with open(f"qa_r_{data_file_suffix}.pkl", 'wb') as f:
    pickle.dump(qa_r_samples, f)

with open(f"qu_{data_file_suffix}.pkl", 'wb') as f:
    pickle.dump(qu_samples, f)
