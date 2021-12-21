import numpy as np
import meshcat


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
