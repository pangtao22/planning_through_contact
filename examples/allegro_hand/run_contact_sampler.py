import numpy as np

from allegro_hand_setup import q_model_path, h
from irs_rrt.contact_sampler_allegro import AllegroHandContactSampler
from qsim.parser import QuasistaticParser
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# %%
q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(h=h)

q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py

contact_sampler = AllegroHandContactSampler(
    q_sim=q_sim,
    q_sim_py=q_sim_py,
)

# %%
q_u0 = np.array([1, 0, 0, 0, -0.081, 0.001, 0.071])
q = contact_sampler.sample_contact(q_u0)
q_vis.draw_configuration(q)
