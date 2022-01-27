from typing import List
import numpy as np
import zmq
from irs_mpc.irs_mpc_params import BundleMode


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(dtype = str(A.dtype), shape = A.shape)
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(memoryview(msg), dtype=md['dtype'])
    return A.reshape(md['shape'])


def send_x_and_u(socket: zmq.Socket, x_u: np.ndarray, t: int,
                 n_samples: int, std: List[float],
                 bundle_mode: BundleMode):
    data = dict(t=t, n_samples=n_samples, std=std,
                bundle_mode=bundle_mode.value)
    socket.send_json(data, zmq.SNDMORE)
    return send_array(socket, x_u)


def recv_x_and_u(socket: zmq.Socket):
    data = socket.recv_json()
    x_and_u = recv_array(socket)
    return x_and_u, data


def send_bundled_AB(socket: zmq.Socket, AB: np.ndarray, t: int):
    socket.send_json({'t': t}, zmq.SNDMORE)
    return send_array(socket, AB)


def recv_bundled_AB(socket: zmq.Socket):
    t = socket.recv_json()['t']
    AB = recv_array(socket)
    return AB, t


