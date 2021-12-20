from tqdm import tqdm
import pandas as pd
import plotly.express as px

from qsim.simulator import QuasistaticSimulator, QuasistaticSimParameters
from qsim.system import cpp_params_from_py_params
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)
from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from irs_lqr.irs_lqr_quasistatic import IrsLqrQuasistatic

from planar_hand_setup import *

from rrt.planner import ConfigurationSpace
from rrt.utils import sample_on_sphere

import plotly.io as pio
pio.renderers.default = "browser"  # see plotly charts in pycharm.

#%% Quasistatic Simulator
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)

q_sim_py = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params,
    internal_vis=True)

# construct C++ backend.
sim_params_cpp = cpp_params_from_py_params(sim_params)
sim_params_cpp.gradient_lstsq_tolerance = gradient_lstsq_tolerance
q_sim_cpp = QuasistaticSimulatorCpp(
    model_directive_path=model_directive_path,
    robot_stiffness_str=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params_cpp)

plant = q_sim_cpp.get_plant()
q_sim_py.get_robot_name_to_model_instance_dict()
model_a_l = plant.GetModelInstanceByName(robot_l_name)
model_a_r = plant.GetModelInstanceByName(robot_r_name)
model_u = plant.GetModelInstanceByName(object_name)

#%% Quasistatic Dynamics
q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
cspace = ConfigurationSpace(model_u=model_u, model_a_l=model_a_l,
                            model_a_r=model_a_r, q_sim=q_sim_py)

#%% Irs-Lqr
params = IrsLqrQuasistaticParameters()
params.Q_dict = {
    model_u: np.array([10, 10, 10]),
    model_a_l: np.array([1e-3, 1e-3]),
    model_a_r: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    model_a_l: 5 * np.array([1, 1]),
    model_a_r: 5 * np.array([1, 1])}

params.sampling = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3

params.decouple_AB = decouple_AB
params.use_workers = use_workers
params.gradient_mode = gradient_mode
params.task_stride = task_stride
params.num_samples = num_samples
params.u_bounds_abs = np.array([-np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])
params.publish_every_iteration = False

T = int(round(2 / h))  # num of time steps to simulate forward.
params.T = T
duration = T * h

irs_lqr_q = IrsLqrQuasistatic(q_dynamics=q_dynamics, params=params)

#%%
# q_u0 = np.array([0, 0.35, 0])
q_u0 = np.array([-0.2, 0.3, 0])
q0_dict = cspace.sample_contact(q_u=q_u0)
x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q0_dict)

delta_q_u_samples = sample_on_sphere(radius=0.5, n_samples=500)

results = []
for delta_q_u in tqdm(delta_q_u_samples):
    x_goal = np.array(x0)
    x_goal[q_sim_py.velocity_indices[model_u]] += delta_q_u
    irs_lqr_q.initialize_problem(
        x0=x0,
        x_trj_d=np.tile(x_goal, (T + 1, 1)),
        u_trj_0=np.tile(u0, (T, 1)))

    irs_lqr_q.iterate(max_iterations=10)
    result = irs_lqr_q.package_solution()
    result["dqu_goal"] = delta_q_u
    results.append(result)


#%%
df = pd.DataFrame({
    'd_qu_y': delta_q_u_samples[:, 0],
    'd_qu_z': delta_q_u_samples[:, 1],
    'd_qu_theta': delta_q_u_samples[:, 2],
    'cost': [result['cost']['Qu_f'] for result in results]})

fig = px.scatter_3d(df, x='d_qu_y', y='d_qu_z', z='d_qu_theta', color='cost')
fig.update_scenes(camera_projection_type='orthographic',
                  aspectmode='data')
fig.show()




