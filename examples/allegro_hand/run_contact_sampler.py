import numpy as np

from allegro_hand_setup import q_model_path, h, robot_name, object_name
from irs_rrt.contact_sampler_allegro import AllegroHandContactSampler
from qsim.parser import QuasistaticParser
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

# %%
q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(h=h)

q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
q_sim, q_sim_py = q_vis.q_sim, q_vis.q_sim_py
plant = q_sim.get_plant()

contact_sampler = AllegroHandContactSampler(
    q_sim=q_sim,
    q_sim_py=q_sim_py,
)

# %%
q_a0 = np.zeros(16)  # The values do not matter.
q_u0 = np.array([1, 0, 0, 0, -0.081, 0.001, 0.071])
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)
q0 = q_sim.get_q_vec_from_dict({idx_a: q_a0, idx_u: q_u0})
q = contact_sampler.sample_contact(q0)
q_vis.draw_configuration(q)
