#!/usr/bin/env python3
from irs_lqr.zmq_dynamics_worker import launch_workers

from planar_hand_setup import q_model_path, h

if __name__ == "__main__":
    launch_workers(q_model_path, h)
