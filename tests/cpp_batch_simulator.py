import os

import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from examples.allegro_hand.allegro_hand_setup import *

#%%
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=False)

q_dynamics_p = QuasistaticDynamicsParallel(
    q_dynamics=q_dynamics, use_zmq_workers=False)

#%%
q_sim_py = q_dynamics.q_sim_py
q_sim_cpp = q_dynamics.q_sim
plant = q_sim_py.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)

# initial conditions.
q_a0 = np.array([0.03501504, 0.75276565, 0.74146232, 0.83261002, 0.63256269,
                 1.02378254, 0.64089555, 0.82444782, -0.1438725, 0.74696812,
                 0.61908827, 0.70064279, -0.06922541, 0.78533142, 0.82942863,
                 0.90415436])
q_u0 = np.array([1, 0, 0, 0, -0.081, 0.001, 0.071])
q0_dict = {idx_a: q_a0, idx_u: q_u0}

x0 = q_dynamics.get_x_from_q_dict(q0_dict)
u0 = q_dynamics.get_u_from_q_cmd_dict(q0_dict)

n_batch = 100
x_batch = np.zeros((n_batch, q_dynamics.dim_x))
u_batch = np.zeros((n_batch, q_dynamics.dim_u))

x_batch[:] = x0
u_batch[:] = u0
u_batch += np.random.normal(0, 0.1, u_batch.shape)


x_next_batch_py = q_dynamics_p.dynamics_batch_serial(x_batch, u_batch)
x_next_batch_cpp = q_dynamics_p.dynamics_batch(x_batch, u_batch)

