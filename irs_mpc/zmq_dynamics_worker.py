import multiprocessing
import time
import spdlog

from .quasistatic_dynamics import QuasistaticDynamics
from .irs_mpc_params import BundleMode
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
        h=h, q_model_path=q_model_path,
        internal_viz=False)

    # Process tasks forever
    i_tasks = 0
    while True:
        x_and_u, data = recv_x_and_u(receiver)
        t = data['t']
        n_samples = data['n_samples']
        std = data['std']
        bundle_mode = BundleMode(data['bundle_mode'])

        assert len(x_and_u.shape) == 1

        ABhat = q_dynamics.calc_bundled_AB(
            x_nominals=x_and_u[None, :q_dynamics.dim_x],
            u_nominals=x_and_u[None, q_dynamics.dim_x:],
            n_samples=n_samples,
            std_u=std,
            bundle_mode=bundle_mode)

        # Send results to sink
        send_bundled_AB(sender, AB=ABhat, t=t)

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
