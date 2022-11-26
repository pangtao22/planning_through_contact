from contact_sampler import *

from planar_hand_setup import q_model_path, h
from contact_sampler import PlanarHandContactSampler
from qsim.parser import QuasistaticParser
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# %%
q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(h=h)

q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py

contact_sampler = PlanarHandContactSampler(
    q_sim=q_sim,
    q_sim_py=q_sim_py,
    pinch_prob=0.5,
)


# %%
q_u = np.array([-0.2, 0.4, 0])
q_dict = contact_sampler.calc_enveloping_grasp(q_u)
q_vis.draw_configuration(q_sim.get_q_vec_from_dict(q_dict))

# %%
q_dict_list = contact_sampler.sample_pinch_grasp(q_u, n_samples=50)

# %%
for q_dict in q_dict_list:
    q_vis.draw_configuration(q_sim.get_q_vec_from_dict(q_dict))
    input()
