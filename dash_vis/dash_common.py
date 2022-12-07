import networkx
import numpy as np
import plotly.graph_objects as go

from pydrake.all import RigidTransform, RollPitchYaw
import meshcat


# %% meshcat
def set_orthographic_camera_yz(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show YZ plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-1, right=0.5, bottom=-0.5, top=1, near=-1000, far=1000
    )
    vis["/Cameras/default/rotated"].set_object(camera)
    vis["/Cameras/default/rotated/<object>"].set_property("position", [0, 0, 0])
    vis["/Cameras/default"].set_transform(
        meshcat.transformations.translation_matrix([1, 0, 0])
    )


def add_goal_meshcat(vis: meshcat.Visualizer):
    # goal
    vis["goal/cylinder"].set_object(
        meshcat.geometry.Cylinder(height=0.001, radius=0.25),
        meshcat.geometry.MeshLambertMaterial(color=0xDEB948, reflectivity=0.8),
    )
    vis["goal/box"].set_object(
        meshcat.geometry.Box([0.02, 0.005, 0.25]),
        meshcat.geometry.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8),
    )
    vis["goal/box"].set_transform(
        meshcat.transformations.translation_matrix([0, 0, 0.125])
    )

    # rotate cylinder so that it faces the x-axis.
    vis["goal"].set_transform(X_WG0)


def calc_X_WG(y: float, z: float, theta: float):
    """
    Goal pose.
    """
    X_WG = RigidTransform(RollPitchYaw(theta, 0, 0), np.array([0, y, z]))
    return X_WG


# %% hover templates and layout
hover_template_y_z_theta = (
    "<i>y</i>: %{x:.4f}<br>"
    + "<i>z</i>: %{y:.4f}<br>"
    + "<i>theta</i>: %{z:.4f}"
)

hover_template_trj = (
    hover_template_y_z_theta + "<br><i>cost</i>: %{marker.color:.4f}"
)

layout = go.Layout(
    autosize=True,
    height=900,
    legend=dict(orientation="h"),
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        camera_projection_type="perspective",
        xaxis_title_text="y",
        yaxis_title_text="z",
        zaxis_title_text="theta",
        aspectmode="data",
        aspectratio=dict(x=1.0, y=1.0, z=1.0),
    ),
)


# %% PCA of reachable set samples
def calc_principal_points(qu_samples: np.ndarray, r: float):
    qu_mean = qu_samples.mean(axis=0)
    U, sigma, Vh = np.linalg.svd(qu_samples - qu_mean)
    principal_points = np.zeros((3, 2, 3))
    for i in range(3):
        principal_points[i, 0] = qu_mean - Vh[i] * sigma[i] / sigma[0] * r
        principal_points[i, 1] = qu_mean + Vh[i] * sigma[i] / sigma[0] * r
    return principal_points


def create_pca_plots(principal_points: np.ndarray):
    colors = ["red", "green", "blue"]
    pca_names = "xyz"
    principal_axes_plots = []
    for i in range(3):
        principal_axes_plots.append(
            go.Scatter3d(
                x=principal_points[i, :, 0],
                y=principal_points[i, :, 1],
                z=principal_points[i, :, 2],
                name=f"pca_{pca_names[i]}",
                mode="lines",
                line=dict(color=colors[i], width=4),
            )
        )
    return principal_axes_plots


# %% plotly figure components
def make_large_point_3d(
    p: np.ndarray, name="q_u0", symbol="cross", color="magenta"
):
    return go.Scatter3d(
        x=[p[0]],
        y=[p[1]],
        z=[p[2] if len(p) == 3 else 0],
        name=name,
        mode="markers",
        hovertemplate=hover_template_y_z_theta,
        marker=dict(size=12, symbol=symbol, opacity=1.0, color=color),
    )


def make_ellipsoid_plotly(
    A_inv: np.ndarray, p_center: np.ndarray, r: float, n: int = 20
):
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
    if len(p_center) == 3:
        z[2] *= r / np.sqrt(Sigma[2])
    else:
        z[2] *= 1e-3
    R = U.T
    if len(p_center) == 3:
        x = R.T @ z + p_center[:, None]
    else:
        # p_center == 2
        R3 = np.zeros((3, 3))
        R3[:2, :2] = R
        R3[2, 2] = 1
        x = R3.T @ z + np.array([p_center[0], p_center[1], 0])[:, None]

    return x, 1 / np.prod(np.sqrt(Sigma))


def edges_have_trj(tree: networkx.DiGraph):
    for edge in tree.edges(0):
        break
    return not (tree.edges[edge]["edge"].trj is None)


def trace_nodes_to_root_from(i_node: int, tree: networkx.DiGraph):
    node_idx_path = []
    # trace back to root to get path.
    while True:
        node_idx_path.append(i_node)

        i_parents = list(tree.predecessors(i_node))
        assert len(i_parents) <= 1
        if len(i_parents) == 0:
            break

        i_node = i_parents[0]

    node_idx_path.reverse()
    return node_idx_path


def trace_path_to_root_from_node(
    i_node: int,
    q_u_nodes: np.ndarray,
    q_nodes: np.ndarray,
    tree: networkx.DiGraph,
):
    node_idx_path = trace_nodes_to_root_from(i_node, tree)
    q_u_path = q_u_nodes[node_idx_path]

    # Trajectory.
    if edges_have_trj(tree):
        n_edges = len(node_idx_path) - 1
        x_trj_list = []
        x_trj_sizes_list = []
        for i in range(n_edges):
            node_i = node_idx_path[i]
            node_j = node_idx_path[i + 1]
            x_trj_i = tree.edges[node_i, node_j]["edge"].trj["x_trj"]
            x_trj_list.append(x_trj_i)
            x_trj_sizes_list.append(len(x_trj_i))

        dim_x = q_nodes.shape[1]
        x_trj = np.zeros((np.sum(x_trj_sizes_list), dim_x))
        i_start = 0
        for x_trj_i, size_i in zip(x_trj_list, x_trj_sizes_list):
            x_trj[i_start : i_start + size_i] = x_trj_i
            i_start += size_i
    else:
        x_trj = q_nodes[node_idx_path]

    return q_u_path, x_trj
