import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import plotly.express as px

from irs_mpc.irs_mpc_quasistatic import IrsMpcQuasistatic
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.irs_mpc_params import IrsMpcQuasistaticParameters
from planar_hand_setup import *

from rrt.planner import ConfigurationSpace
from rrt.utils import sample_on_sphere

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
cspace = ConfigurationSpace(model_u=model_u, model_a_l=model_a_l,
                            model_a_r=model_a_r, q_sim=q_sim_py)

#%% Irs-Mpc
params = IrsMpcQuasistaticParameters()
params.Q_dict = {
    model_u: np.array([10, 10, 10]),
    model_a_l: np.array([1e-3, 1e-3]),
    model_a_r: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    model_a_l: 5 * np.array([1, 1]),
    model_a_r: 5 * np.array([1, 1])}

T = int(round(2 / h))  # num of time steps to simulate forward.
params.T = T

params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])

params.calc_std_u = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3

params.decouple_AB = decouple_AB
params.num_samples = num_samples
params.bundle_mode = bundle_mode
params.parallel_mode = parallel_mode

irs_mpc = IrsMpcQuasistatic(q_dynamics=q_dynamics, params=params)
q_dynamics_p = irs_mpc.q_dynamics_parallel

#%% traj opt
q_u0 = np.array([0, 0.35, 0])
# q_u0 = np.array([-0.2, 0.3, 0])
q0_dict = cspace.sample_contact(q_u=q_u0)
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q0_dict)

delta_q_u_samples = sample_on_sphere(radius=0.5, n_samples=500)

trj_data = []
for delta_q_u in tqdm(delta_q_u_samples):
    x_goal = np.array(x0)
    x_goal[q_sim_py.velocity_indices[model_u]] += delta_q_u
    irs_mpc.initialize_problem(
        x0=x0,
        x_trj_d=np.tile(x_goal, (T + 1, 1)),
        u_trj_0=np.tile(u0, (T, 1)))

    irs_mpc.iterate(max_iterations=10)
    result = irs_mpc.package_solution()
    result["dqu_goal"] = delta_q_u
    trj_data.append(result)


#%% random 1-step forward sim for reachable set computation.
n_samples = 2000
radius = 0.2
du = np.random.rand(n_samples, 4) * radius * 2 - radius
qu_samples = np.zeros((n_samples, 3))
qa_l_samples = np.zeros((n_samples, 2))
qa_r_samples = np.zeros((n_samples, 2))


def save_x(x: np.ndarray, i: int):
    q_dict = q_dynamics.get_q_dict_from_x(x)
    qu_samples[i] = q_dict[model_u]
    qa_l_samples[i] = q_dict[model_a_l]
    qa_r_samples[i] = q_dict[model_a_r]


x_batch = q_dynamics_p.dynamics_batch(
    x_batch=np.repeat(x0[None, :], n_samples, axis=0),
    u_batch=u0 + du)

for i in range(n_samples):
    save_x(x_batch[i], i)

#%% save traj opt data and reachable set data to disk, used by dash app.
reachability_trj_opt = dict(
    qu_0=q_u0,
    reachable_set_radius=radius,
    trj_data=trj_data,
    reachable_set_data=dict(du=du,
                            qa_l={'1_step': qa_l_samples},
                            qa_r={'1_step': qa_r_samples},
                            qu={'1_step': qu_samples}))

with open("./data/reachability_trj_opt_02.pkl", 'wb') as f:
    pickle.dump(reachability_trj_opt, f)


#%%
df = pd.DataFrame({
    'd_qu_y': delta_q_u_samples[:, 0],
    'd_qu_z': delta_q_u_samples[:, 1],
    'd_qu_theta': delta_q_u_samples[:, 2],
    'cost': [result['cost']['Qu_f'] for result in trj_data]})

fig = px.scatter_3d(df, x='d_qu_y', y='d_qu_z', z='d_qu_theta', color='cost')
fig.update_scenes(camera_projection_type='orthographic',
                  aspectmode='data')
fig.show()
