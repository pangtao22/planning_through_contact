from typing import List
import numpy as np
import zmq


def send_array(socket, A: np.ndarray, t: List[int],
               n_samples: int, std: List[float],
               flags=0, copy=True, track=False):
    """send a numpy array with metadata
    A can be:
        xu: (n, n_x + n_u): 2D array containing the nominal points about which
         gradients are evaluated.
    t: (n,) List containing the indices into the trajectory.
    """
    md = dict(dtype=str(A.dtype), shape=A.shape, t=t, n_samples=n_samples,
              std=std)
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    xu = np.frombuffer(memoryview(msg), dtype=md['dtype'])
    return xu.reshape(md['shape']), md['t'], md['n_samples'], md['std']
