import copy
import pickle

import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px

from irs_mpc.irs_mpc_quasistatic import IrsMpcQuasistatic
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters
from planar_hand_setup import *

from contact_sampler import sample_on_sphere, PlanarHandContactSampler

import plotly.io as pio
pio.renderers.default = "browser"  # see plotly charts in pycharm.


#%% Quasistatic Dynamics
q_dynamics = QuasistaticDynamics(h=h, q_model_path=q_model_path,
                                 internal_viz=False)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
q_sim_py = q_dynamics.q_sim_py

plant = q_dynamics.plant
model_a_l = plant.GetModelInstanceByName(robot_l_name)
model_a_r = plant.GetModelInstanceByName(robot_r_name)
model_u = plant.GetModelInstanceByName(object_name)
contact_sampler = PlanarHandContactSampler(q_dynamics, 0.5)


#%% random 1-step forward sim for reachable set computation.
n_samples = 10
radius = 0.2
du = np.random.rand(n_samples, 4) * radius * 2 - radius
qu_samples = np.zeros((n_samples, 3))
qa_l_samples = np.zeros((n_samples, 2))
qa_r_samples = np.zeros((n_samples, 2))
contact_results = []


def save_x(x: np.ndarray, i: int):
    q_dict = q_dynamics.get_q_dict_from_x(x)
    qu_samples[i] = q_dict[model_u]
    qa_l_samples[i] = q_dict[model_a_l]
    qa_r_samples[i] = q_dict[model_a_r]


q_u0 = np.array([0, 0.35, 0])
q0_dict = contact_sampler.calc_enveloping_grasp(q_u=q_u0)
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q0_dict)


for i in range(n_samples):
    x_next = q_dynamics.dynamics_py(x=x0, u=u0 + du[i])
    save_x(x_next, i)
    contact_results.append(
        copy.deepcopy(q_dynamics.q_sim_py.serialize_my_contact_results(
            f_threshold=0.1)))


for i, my_contact_results in enumerate(contact_results):
    print(i, len(my_contact_results))


#%%
n_c = q_dynamics.q_sim_py.contact_results.num_point_pair_contacts()
for i in range(n_c):
    print(q_dynamics.q_sim_py.contact_results.point_pair_contact_info(
        i).contact_force())

