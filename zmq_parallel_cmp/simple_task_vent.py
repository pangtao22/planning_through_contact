# Task ventilator
# Binds PUSH socket to tcp://localhost:5557
# Sends batch of tasks to workers via that socket
#
# Original Author: Lev Givon <lev(at)columbia(dot)edu>

import sys
import random
import time

from array_io import *

context = zmq.Context()

# Socket to send messages on
sender = context.socket(zmq.PUSH)
sender.bind("tcp://*:5557")

# Socket to receive messages on
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5558")

print("Press Enter when the workers are ready: ")
input()
print("Sending tasks to workers...")

# Initialize random number generator
random.seed()

n_tasks = 100
# Send 100 tasks
for task_nbr in range(n_tasks):
    A = np.ones(random.randint(1, 10000))
    send_array(sender, A, t=[0, 1], n_samples=100, std=[0.1])
    print(task_nbr, A.sum())


# Start our clock now
# Each request sleeps for 100ms. The total elapsed time should be close to
#  (10s / num_workers).
tstart = time.time()

# Process 100 confirmations
for task_nbr in range(n_tasks):
    B, t, n_samples, std = recv_array(receiver)
    print(task_nbr, B.sum(), t, n_samples, std)
    sys.stdout.flush()

# Calculate and report duration of batch
tend = time.time()
print(f"Total elapsed time: {(tend-tstart)*1000} msec")
