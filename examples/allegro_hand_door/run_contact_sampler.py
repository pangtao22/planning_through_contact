import numpy as np

from irs_mpc.quasistatic_dynamics import QuasistaticDynamics

from contact_sampler_allegro_door import AllegroHandPlateContactSampler

from allegro_hand_setup import *


q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path, internal_viz=True
)

contact_sampler = AllegroHandPlateContactSampler(q_dynamics=q_dynamics)

#%%
from pydrake.all import JointIndex

plant = contact_sampler.plant
for i in range(plant.num_joints()):
    print(plant.get_joint(JointIndex(i)))

#%%
contact_sampler.sample_contact(np.array([0, 0]))
