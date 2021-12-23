import numpy as np
import meshcat
import tqdm
import pickle


def set_orthographic_camera_yz(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show YZ plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-1, right=0.5, bottom=-0.5, top=1, near=-1000, far=1000)
    vis['/Cameras/default/rotated'].set_object(camera)
    vis['/Cameras/default/rotated/<object>'].set_property(
        "position", [0, 0, 0])
    vis['/Cameras/default'].set_transform(
        meshcat.transformations.translation_matrix([1, 0, 0]))


def sample_on_sphere(radius: float, n_samples: int):
    """
    Uniform sampling on a sphere with radius r.
    http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    u = np.random.rand(n_samples, 2)  # uniform samples.
    u_theta = u[:, 0]
    u_phi = u[:, 1]
    theta = u_theta * 2 * np.pi
    phi = np.arccos(1 - 2 * u_phi)

    xyz_samples = np.zeros((n_samples, 3))
    xyz_samples[:, 0] = radius * np.sin(phi) * np.cos(theta)
    xyz_samples[:, 1] = radius * np.sin(phi) * np.sin(theta)
    xyz_samples[:, 2] = radius * np.cos(phi)

    return xyz_samples


def reachable_sets(x0, u0, q_dynamics, model_u, model_a_l, model_a_r,
                   n_samples=2000, radius=0.2):
    du = np.random.rand(n_samples, 4) * radius * 2 - radius
    qu_samples = np.zeros((n_samples, 3))
    qa_l_samples = np.zeros((n_samples, 2))
    qa_r_samples = np.zeros((n_samples, 2))

    def save_x(x: np.ndarray):
        q_dict = q_dynamics.get_q_dict_from_x(x)
        qu_samples[i] = q_dict[model_u]
        qa_l_samples[i] = q_dict[model_a_l]
        qa_r_samples[i] = q_dict[model_a_r]

    for i in tqdm.tqdm(range(n_samples)):
        u = u0 + du[i]
        x_1 = q_dynamics.dynamics(x0, u, requires_grad=False)
        save_x(x_1)

    return qu_samples


def pca_gaussian(qu_samples, scale_rad=np.pi, r=0.5):
    '''
    Ellipsoid x.T inv(cov) x = 1
    '''
    # Scale radian to metric
    qu_samples[:, 2] /= scale_rad
    qu_mean = qu_samples.mean(axis=0)
    _, sigma, Vh = np.linalg.svd(qu_samples - qu_mean)
    scale = r/sigma[0]
    sigma *= scale
    cov = Vh.T@np.diag(sigma**2)@Vh
    return qu_mean, sigma, cov, Vh


def solve_irs_lqr(irs_lqr_q, q_dynamics, q_start, q_goal, T, num_iters,
                  x_trj_d=None):
    """
    x_trj_d: initial guess of the object trajectory
    """
    xd = q_dynamics.get_x_from_q_dict(q_goal)
    u0 = q_dynamics.get_u_from_q_cmd_dict(q_start)

    if x_trj_d is None:
        x_trj_d = np.tile(xd, (T + 1, 1))

    irs_lqr_q.initialize_problem(
        x0=q_dynamics.get_x_from_q_dict(q_start),
        x_trj_d=x_trj_d,
        u_trj_0=np.tile(u0, (T, 1)))

    irs_lqr_q.iterate(num_iters)


def save_rrt(rrt):
    model_u, model_a_l, model_a_r = rrt.cspace.model_u, rrt.cspace.model_a_l, rrt.cspace.model_a_r
    int_model_u = int(model_u)
    int_model_a_l = int(model_a_l)
    int_model_a_r = int(model_a_r)

    def recur(node):
        q = {}
        q[int_model_u] = node.q[model_u]
        q[int_model_a_l] = node.q[model_a_l]
        q[int_model_a_r] = node.q[model_a_r]
        node.q = q
        node.q_goal = None
        for child in node.children:
            recur(child)

    recur(rrt.root)

    rrt.cspace = None

    with open("rrt.pkl", "wb") as file:
        pickle.dump(rrt, file, pickle.HIGHEST_PROTOCOL)


def load_rrt(rrt_file):
    with open(rrt_file, "rb") as file:
        rrt = pickle.load(file)
    return rrt
