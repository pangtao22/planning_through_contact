import multiprocessing
import time
import spdlog

from .quasistatic_dynamics import QuasistaticDynamics
from .irs_lqr_params import IrsLqrGradientMode
from zmq_parallel_cmp.array_io import *

kTaskVentSocket = 5557
kTaskSinkSocket = 5558


def f_worker(q_model_path: str, h: float):
    """
    :param q_model_path: path to a yml file describing the quasistatic system.
    :param h: simulation time step.
    :return:
    """
    pid = multiprocessing.current_process().pid
    logger = spdlog.ConsoleLogger(str(pid))

    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect(f"tcp://localhost:{kTaskVentSocket}")

    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    sender.connect(f"tcp://localhost:{kTaskSinkSocket}")
    logger.info(f"worker {pid} is ready.")

    q_dynamics = QuasistaticDynamics(
        h=h, quasistatic_model_path=q_model_path,
        internal_viz=False)

    # Process tasks forever
    i_tasks = 0
    while True:
        x_u_nominal, t_list, n_samples, std = recv_array(receiver)
        assert len(x_u_nominal.shape) == 2
        x_nominals = x_u_nominal[:, :q_dynamics.dim_x]
        u_nominals = x_u_nominal[:, q_dynamics.dim_x:]

        ABhat = q_dynamics.calc_AB_batch(
            x_nominals=x_nominals,
            u_nominals=u_nominals,
            n_samples=n_samples,
            std_u=std,
            mode=IrsLqrGradientMode.kFirst)

        # Send results to sink
        send_array(sender, A=ABhat, t=t_list, n_samples=-1, std=[-1])

        i_tasks += 1
        if i_tasks % 50 == 0:
            logger.info(f"{pid} has processed {i_tasks} tasks.")


def launch_workers(q_model_path: str, h: float):
    p_list = []
    try:
        for _ in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=f_worker, args=(q_model_path, h))
            p_list.append(p)
            p.start()
        time.sleep(100000)
    except KeyboardInterrupt:
        for p in p_list:
            p.terminate()
            p.join()
