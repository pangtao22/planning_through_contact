from contact_sampler import *

from irs_mpc.irs_mpc_quasistatic import QuasistaticDynamics
from planar_hand_setup import q_model_path, h
from contact_sampler import PlanarHandContactSampler

#%%
q_dynamics = QuasistaticDynamics(h=h,
                                 q_model_path=q_model_path,
                                 internal_viz=True)
q_sim_py = q_dynamics.q_sim_py
contact_sampler = PlanarHandContactSampler(q_dynamics, 0.5)



#%%
q_u = np.array([-0.2, 0.4, 0])
q_dict = contact_sampler.calc_enveloping_grasp(q_u)
q_sim_py.update_mbp_positions(q_dict)
q_sim_py.draw_current_configuration(False)

#%%
q_dict_list = contact_sampler.sample_pinch_grasp(q_u, n_samples=50)

#%%
for q_dict in q_dict_list:
    q_sim_py.update_mbp_positions(q_dict)
    q_sim_py.draw_current_configuration(False)
    input()
