from qsim.simulator import QuasistaticSimulator, QuasistaticSimParameters

from planar_hand_setup import *

from rrt.planner import ConfigurationSpace

from rrt.utils import set_orthographic_camera_yz

#%% sim setup
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h
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
set_orthographic_camera_yz(q_sim_py.viz.vis)

plant = q_sim_py.get_plant()
model_a_l = plant.GetModelInstanceByName(robot_l_name)
model_a_r = plant.GetModelInstanceByName(robot_r_name)
model_u = plant.GetModelInstanceByName(object_name)

cspace = ConfigurationSpace(
    model_u=model_u, model_a_l=model_a_l, model_a_r=model_a_r, q_sim=q_sim_py)

#%%
# some configuration.
q_u0 = np.array([-0.2, 0.3, 0])
q_dict = cspace.sample_contact(q_u=q_u0)

q_sim_py.update_mbp_positions(q_dict)
q_sim_py.draw_current_configuration()
