# Model-based Contact-rich Manipulation Planning
[![Deepnote](https://deepnote.com/buttons/launch-in-deepnote.svg)](https://deepnote.com/workspace/pang-a928705f-1da7-4aa9-a64c-e5cdfb9b162e/project/planning-through-contact-c39756a2-acd7-4fb0-8443-38625de2db2e/notebook/allegro_hand_irs_mpc-f9677ed681c742b0b6577c7f1f86d4cc)

![](/media/planar_hand.gif) ![](/media/allegro_hand_ball.gif) ![](/media/allegro_hand_door.gif)

This repo provides implementation of model-based planners for contact-rich manipulation. The attached Deepnote project includes two notebooks, illustrating two planning algorithms on the Allegro hand dexterous manipulation example:
- A trajectory optimizer using iterative MPC (iMPC),
- A sampling-based planner capable of handling contact dynamics constraints. 

Details of the planning algorithms can be found in 
- [Global Planning for Contact-Rich Manipulation via
Local Smoothing of Quasi-dynamic Contact Models](https://arxiv.org/abs/2206.10787), currently under review.

Our quasidynamic simulator can be found in 
- https://github.com/pangtao22/quasistatic_simulator
