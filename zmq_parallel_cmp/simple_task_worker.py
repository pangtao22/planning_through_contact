# Task worker
# Connects PULL socket to tcp://localhost:5557
# Collects workloads from ventilator via that socket
# Connects PUSH socket to tcp://localhost:5558
# Sends results to sink via that socket
#
# Original Author: Lev Givon <lev(at)columbia(dot)edu>

import multiprocessing
import sys
import time

from array_io import *


def f_worker_simple():
    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:5557")

    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:5558")

    pid = multiprocessing.current_process().pid
    print("worker", pid, "ready.")

    # Process tasks forever
    i_tasks = 0
    while True:
        A, t, n_samples, std = recv_array(receiver)

        # Pretending to do some work.
        time.sleep(n_samples / 1000)
        # Send results to sink
        send_array(sender, A * 2, t=t, n_samples=n_samples, std=std)

        i_tasks += 1
        if i_tasks % 10 == 0:
            print(pid, "has processed", i_tasks, "tasks.")


if __name__ == "__main__":
    p_list = []
    try:
        for _ in range(5):
            p = multiprocessing.Process(target=f_worker_simple)
            p_list.append(p)
            p.start()
        time.sleep(100000)
    except KeyboardInterrupt:
        for p in p_list:
            p.terminate()
            p.join()


