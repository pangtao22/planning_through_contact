import numpy as np
import meshcat
import plotly.graph_objects as go


#%% meshcat
X_WG0 = meshcat.transformations.rotation_matrix(np.pi/2, [0, 0, 1])


def set_orthographic_camera_yz(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show YZ plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-1, right=0.5, bottom=-0.5, top=1, near=-1000, far=1000)
    vis['/Cameras/default/rotated'].set_object(camera)
    vis['/Cameras/default/rotated/<object>'].set_property(
        "position", [0, 0, 0])
    vis['/Cameras/default'].set_transform(
        meshcat.transformations.translation_matrix([1, 0, 0]))


def add_goal_meshcat(vis: meshcat.Visualizer):
    # goal
    vis["goal/cylinder"].set_object(
        meshcat.geometry.Cylinder(height=0.001, radius=0.25),
        meshcat.geometry.MeshLambertMaterial(color=0xdeb948, reflectivity=0.8))
    vis['goal/box'].set_object(
        meshcat.geometry.Box([0.02, 0.005, 0.25]),
        meshcat.geometry.MeshLambertMaterial(color=0x00ff00, reflectivity=0.8))
    vis['goal/box'].set_transform(
        meshcat.transformations.translation_matrix([0, 0, 0.125]))

    # rotate cylinder so that it faces the x-axis.
    vis['goal'].set_transform(X_WG0)


def calc_X_WG(y: float, z: float , theta: float):
    """
    Goal pose.
    """
    p_WG = np.array([y, 0, z])
    X_G0G = (meshcat.transformations.translation_matrix(p_WG) @
             meshcat.transformations.rotation_matrix(-theta, [0, 1, 0]))
    return X_WG0 @ X_G0G


#%% hover templates and layout
hover_template_y_z_theta = (
        '<i>y</i>: %{x:.4f}<br>' +
        '<i>z</i>: %{y:.4f}<br>' +
        '<i>theta</i>: %{z:.4f}')

hover_template_trj = (hover_template_y_z_theta +
                      '<br><i>cost</i>: %{marker.color:.4f}')

layout = go.Layout(autosize=True, height=900,
                   legend=dict(orientation="h"),
                   margin=dict(l=0, r=0, b=0, t=0),
                   scene=dict(camera_projection_type='perspective',
                              xaxis_title_text='y',
                              yaxis_title_text='z',
                              zaxis_title_text='theta',
                              aspectmode='data',
                              aspectratio=dict(x=1.0, y=1.0, z=1.0)))


#%% PCA of reachable set samples
def calc_principal_points(qu_samples: np.ndarray, r: float):
    qu_mean = qu_samples.mean(axis=0)
    U, sigma, Vh = np.linalg.svd(qu_samples - qu_mean)
    principal_points = np.zeros((3, 2, 3))
    for i in range(3):
        principal_points[i, 0] = qu_mean - Vh[i] * sigma[i] / sigma[0] * r
        principal_points[i, 1] = qu_mean + Vh[i] * sigma[i] / sigma[0] * r
    return principal_points


def create_pca_plots(principal_points: np.ndarray):
    colors = ['red', 'green', 'blue']
    pca_names = 'xyz'
    principal_axes_plots = []
    for i in range(3):
        principal_axes_plots.append(
            go.Scatter3d(
                x=principal_points[i, :, 0],
                y=principal_points[i, :, 1],
                z=principal_points[i, :, 2],
                name=f'pca_{pca_names[i]}',
                mode='lines',
                line=dict(color=colors[i], width=4)
            )
        )
    return principal_axes_plots


#%% plotly figure components
def create_q_u0_plot(q_u0: np.ndarray, name='q_u0'):
    return go.Scatter3d(x=[q_u0[0]],
                        y=[q_u0[1]],
                        z=[q_u0[2]],
                        name=name,
                        mode='markers',
                        hovertemplate=hover_template_y_z_theta,
                        marker=dict(size=12, symbol='cross', opacity=1.0))


def make_ellipsoid_plotly(A_inv: np.ndarray, p_center: np.ndarray, r: float,
                          n: int = 20):
    """
    Make a plotly 3d mesh object for an ellipsoid described by
     (x - p_center).T @ A_inv @ (x - p_center) = r**2.

    A = R.T @ Sigma @ R, as A_inv is symmetric.
    Let z = R * (x - p_center), i.e. x = R.T @ z + p_center.

    The original ellipsoid becomes z.T @ Sigma @ z = r**2, which is axis-aligned
     and centered around the origin.

     Returns: (points of the ellipsoid mesh, ellipsoid volume)
    """
    # Points for sphere.
    phi = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(-np.pi / 2, np.pi / 2, n)
    phi, theta = np.meshgrid(phi, theta)

    z = np.zeros([3, n**2])
    z[0] = (np.cos(theta) * np.sin(phi)).ravel()
    z[1] = (np.cos(theta) * np.cos(phi)).ravel()
    z[2] = np.sin(theta).ravel()

    # Find shape of ellipsoid.
    U, Sigma, Vh = np.linalg.svd(A_inv)
    z[0] *= r / np.sqrt(Sigma[0])
    z[1] *= r / np.sqrt(Sigma[1])
    z[2] *= r / np.sqrt(Sigma[2])
    R = U.T
    x = R.T @ z + p_center[:, None]

    return x, 1 / np.prod(np.sqrt(Sigma))
