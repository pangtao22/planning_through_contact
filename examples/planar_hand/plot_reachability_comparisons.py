from typing import Tuple, List

import meshcat
import numpy as np
import plotly.graph_objects as go
from dash_vis.dash_common import (
    add_goal_meshcat,
    hover_template_y_z_theta,
    layout,
    calc_principal_points,
    calc_X_WG,
    make_large_point_3d,
)

from irs_mpc.quasistatic_dynamics_parallel import QuasistaticDynamicsParallel
from irs_mpc.quasistatic_dynamics import QuasistaticDynamics
from irs_rrt.reachable_set import ReachableSet
from irs_rrt.rrt_params import IrsRrtParams

from planar_hand_setup import (
    h,
    q_model_path,
    decouple_AB,
    bundle_mode,
    num_samples,
    robot_l_name,
    robot_r_name,
    object_name,
)

from contact_sampler import PlanarHandContactSampler, sample_on_sphere
from dash_vis.dash_common import make_ellipsoid_plotly

from planar_hand_plotly_vis import (
    pink,
    dark_red,
    bright_blue,
    blue,
    rgb_tuple_2_rgba_str,
    plot_robots_with_major_and_minor_goals,
    plot_planar_hand_from_vector,
)

import plotly.io as pio

pio.renderers.default = "browser"  # see plotly charts in pycharm.

line_width = 12

# %% quasistatic dynamics
q_dynamics = QuasistaticDynamics(
    h=h, q_model_path=q_model_path, internal_viz=True
)
plant = q_dynamics.plant
q_sim_py = q_dynamics.q_sim_py
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
model_a_l = plant.GetModelInstanceByName(robot_l_name)
model_a_r = plant.GetModelInstanceByName(robot_r_name)
model_u = plant.GetModelInstanceByName(object_name)

contact_sampler = PlanarHandContactSampler(
    q_dynamics=q_dynamics, pinch_prob=0.5
)
q_dynamics_p = QuasistaticDynamicsParallel(q_dynamics)
IrsRrtParams(q_model_path, contact_sampler.joint_limits)
reachable_set = ReachableSet(
    q_dynamics=q_dynamics,
    rrt_params=IrsRrtParams(q_model_path, contact_sampler.joint_limits),
    q_dynamics_p=q_dynamics_p,
)

vis = q_dynamics.q_sim_py.viz.vis
add_goal_meshcat(vis)

# %%
qu0 = np.array([-0.22, 0.52, 0])
q_dict0 = {
    model_u: qu0,
    model_a_l: np.array([-0.7539515572457925, -0.0007669903939428206]),
    model_a_r: np.array([1.5700293364009537, 0.10507768397016642]),
}

qu1 = np.array([0, 0.3, 0])
q_dict1 = {
    model_u: np.array([0, 0.3, 0]),
    model_a_l: np.array([-0.45582961764157437, -1.4673076392961526]),
    model_a_r: np.array([0.5033647632644459, 1.3918307002902661]),
}

x0 = q_dynamics.get_x_from_q_dict(q_dict0)
u0 = q_dynamics.get_u_from_q_cmd_dict(q_dict0)
x1 = q_dynamics.get_x_from_q_dict(q_dict1)
u1 = q_dynamics.get_u_from_q_cmd_dict(q_dict1)


def create_ellipsoid_plot(x_bar, u_bar, color: str):
    Bhat, chat = reachable_set.calc_bundled_Bc_randomized(q=x_bar, ubar=u_bar)
    cov_u, c_u_hat = reachable_set.calc_unactuated_metric_parameters(Bhat, chat)
    cov_u_inv = np.linalg.inv(cov_u)
    r = 1
    e_points, volume = make_ellipsoid_plotly(
        cov_u_inv, np.zeros_like(c_u_hat), r, 20
    )
    return (
        go.Mesh3d(
            dict(
                x=e_points[0],
                y=e_points[1],
                z=e_points[2],
                alphahull=0,
                opacity=0.3,
                name="ellipsoid",
                color=color,
            )
        ),
        c_u_hat,
        cov_u,
    )


def create_pca_plots(
    cov_u: np.ndarray, name: str, color: str, line_width=line_width
):
    U, sigma, Vh = np.linalg.svd(cov_u)

    # Principal points in world frame.
    principal_points_W = np.zeros((3, 2, 3))
    for i in range(3):
        # E: ellipsoid frame.
        principal_points_E = np.zeros((2, 3))
        principal_points_E[0, i] = np.sqrt(sigma[i])
        principal_points_E[1, i] = -np.sqrt(sigma[i])
        principal_points_W[i] = (U @ principal_points_E.T).T

    # principal points W x, y, z.
    ppW_x = []
    ppW_y = []
    ppW_z = []
    for i in range(3):
        ppW_x += [
            principal_points_W[i, 0, 0],
            principal_points_W[i, 1, 0],
            None,
        ]
        ppW_y += [
            principal_points_W[i, 0, 1],
            principal_points_W[i, 1, 1],
            None,
        ]
        ppW_z += [
            principal_points_W[i, 0, 2],
            principal_points_W[i, 1, 2],
            None,
        ]

    principal_axes_plot = go.Scatter3d(
        x=ppW_x,
        y=ppW_y,
        z=ppW_z,
        name=name,
        mode="lines",
        line=dict(color=color, width=line_width),
    )

    p_W_major = principal_points_W[0, 0]
    p_W_minor = principal_points_W[-1, 0]
    p_W_minor /= np.linalg.norm(p_W_minor)
    p_W_minor *= np.linalg.norm(p_W_major)

    return principal_axes_plot, p_W_major, p_W_minor


# %%
ellipsoid_plot0, c_u_hat_0, cov_u0 = create_ellipsoid_plot(x0, u0)
ellipsoid_plot1, c_u_hat_1, cov_u1 = create_ellipsoid_plot(x1, u1)

# %%
ellipsoid_colors = ["blue", "red"]
colors0 = [blue, bright_blue]
colors1 = [dark_red, pink]

pca0, p_W_major0, p_W_minor0 = create_pca_plots(
    cov_u0, name="pca0", color=ellipsoid_colors[0]
)
pca1, p_W_major1, p_W_minor1 = create_pca_plots(
    cov_u1, name="pca1", color=ellipsoid_colors[1]
)


def create_major_minor_plots(
    idx: int, p_W_major: np.ndarray, p_W_minor: np.ndarray, colors: List[Tuple]
):
    p_W_major_plot = go.Scatter3d(
        x=[p_W_major[0]],
        y=[p_W_major[1]],
        z=[p_W_major[2]],
        name=f"p_W_major{idx}",
        mode="markers",
        hovertemplate=hover_template_y_z_theta,
        marker=dict(
            size=20,
            symbol="circle",
            opacity=1.0,
            color=rgb_tuple_2_rgba_str(colors[0], 1.0),
        ),
    )

    p_W_minor_plot = go.Scatter3d(
        x=[p_W_minor[0]],
        y=[p_W_minor[1]],
        z=[p_W_minor[2]],
        name=f"p_W_minor{idx}",
        mode="markers",
        hovertemplate=hover_template_y_z_theta,
        marker=dict(
            size=20,
            symbol="square",
            opacity=1.0,
            color=rgb_tuple_2_rgba_str(colors[1], 1.0),
        ),
    )

    dashed_line = go.Scatter3d(
        x=[0.0, p_W_minor[0]],
        y=[0.0, p_W_minor[1]],
        z=[0.0, p_W_minor[2]],
        mode="lines",
        line=dict(width=line_width, color=ellipsoid_colors[idx], dash="dash"),
    )
    return p_W_major_plot, p_W_minor_plot, dashed_line


# %%
plots = [ellipsoid_plot0, ellipsoid_plot1, pca0, pca1]
plots += create_major_minor_plots(0, p_W_major0, p_W_minor0, colors=colors0)
plots += create_major_minor_plots(1, p_W_major1, p_W_minor1, colors=colors1)
fig = go.Figure(data=plots, layout=layout)

ellipsoid_plot_layout = dict(
    height=1200,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="v"),
    yaxis=dict(scaleanchor="x", scaleratio=1),
    scene=dict(
        camera_projection_type="orthographic",
        aspectmode="auto",
        xaxis=dict(
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="rgb(125,125,125)",
            showticklabels=False,
            title="X",
            titlefont=dict(size=30),
        ),
        yaxis=dict(
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="rgb(125,125,125)",
            showticklabels=False,
            title="Y",
            titlefont=dict(size=30),
        ),
        zaxis=dict(
            backgroundcolor="rgb(255, 255, 255)",
            gridcolor="rgb(125,125,125)",
            showticklabels=False,
            title="Theta",
            titlefont=dict(size=30),
        ),
    ),
)

fig.update_layout(ellipsoid_plot_layout)
fig.show()

# %%%
# these are read from the ellipsoid plots.
qu0_goal_major = c_u_hat_0 + p_W_major0
qu0_goal_minor = c_u_hat_0 + p_W_minor0

qu1_goal_major = c_u_hat_1 + p_W_major1
qu1_goal_minor = c_u_hat_1 + p_W_minor1

# %%
plot_robots_with_major_and_minor_goals(
    x=x0,
    qu_goal_major=qu0_goal_major,
    qu_goal_minor=qu0_goal_minor,
    colors=colors0,
)

# %%
plot_robots_with_major_and_minor_goals(
    x=x1,
    qu_goal_major=qu1_goal_major,
    qu_goal_minor=qu1_goal_minor,
    colors=colors1,
)


# %%
def draw_nominal_config_and_extreme_point(q_dict_nominal, qu_extreme):
    X_WG = calc_X_WG(y=qu_extreme[0], z=qu_extreme[1], theta=qu_extreme[2])
    vis["goal"].set_transform(X_WG)
    q_dynamics.q_sim_py.update_mbp_positions(q_dict_nominal)
    q_dynamics.q_sim_py.draw_current_configuration()


draw_nominal_config_and_extreme_point(q_dict0, qu0_goal_major)
# draw_nominal_config_and_extreme_point(q_dict1, qu1_goal_major)

# %% Visualize reachable set volume -- robot configurations.
q0 = contact_sampler.sample_contact(qu1)

q1 = np.array(q0)
q1[3:5] += 0.1
q1[5:7] -= 0.1

q2 = np.array(q0)
q2[3:5] += 0.2
q2[5:7] -= 0.2

red_rgb = (255, 0, 0)
green_rgb = (0, 255, 127)
blue_rgb = (0, 0, 255)
robot_arm_line_opacity = 0.7

fig = go.Figure()
plot_planar_hand_from_vector(
    x=q0,
    fig=fig,
    color_ball_fill="rgba(255,255,255,0.1)",
    color_ball_line="rgba(0,0,0,1)",
    color_arms_fill=f"rgba({red_rgb[0]}, {red_rgb[1]}, {red_rgb[2]}, 0.2)",
    color_arms_line=f"rgba({red_rgb[0]}, {red_rgb[1]}, {red_rgb[2]}, "
    f"{robot_arm_line_opacity})",
)

plot_planar_hand_from_vector(
    x=q1,
    fig=fig,
    color_ball_fill="rgba(255,255,255,0.1)",
    color_ball_line="rgba(0,0,0,1)",
    color_arms_fill=f"rgba({green_rgb[0]}, {green_rgb[1]}, {green_rgb[2]}, "
    f"0.2)",
    color_arms_line=f"rgba({green_rgb[0]}, {green_rgb[1]}, {green_rgb[2]}, "
    f"{robot_arm_line_opacity})",
)

plot_planar_hand_from_vector(
    x=q2,
    fig=fig,
    color_ball_fill="rgba(255,255,255,0.1)",
    color_ball_line="rgba(0,0,0,1)",
    color_arms_fill=f"rgba({blue_rgb[0]}, {blue_rgb[1]}, {blue_rgb[2]}, 0.2)",
    color_arms_line=f"rgba({blue_rgb[0]}, {blue_rgb[1]}, {blue_rgb[2]}, "
    f"{robot_arm_line_opacity})",
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    width=1000,
    height=1000,
    autosize=False,
    yaxis=dict(scaleanchor="x", scaleratio=1),
    scene=dict(aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=1.0)),
)
fig.update_yaxes(showticklabels=False)
fig.update_xaxes(showticklabels=False)

fig.show()

# %% Visualize reachable set volume -- ellipsoids.
ellips_colors = ["red", "springgreen", "blue"]

ellips_plots = []
pca_plots = []

for i, q in enumerate([q0, q1, q2]):
    c = ellips_colors[i]

    e, c_u_hat, cov_u = create_ellipsoid_plot(
        q, q[q_dynamics.get_q_a_indices_into_x()], color=c
    )
    pca, _, _ = create_pca_plots(cov_u, name=c, color=c, line_width=5)

    ellips_plots.append(e)
    pca_plots.append(pca)


fig = go.Figure(data=ellips_plots + pca_plots, layout=ellipsoid_plot_layout)
fig.update_layout(
    yaxis=dict(scaleanchor="x", scaleratio=1),
    # zaxis=dict(scaleanchor='x', scaleratio=1),
    scene=dict(aspectmode="data", aspectratio=dict(x=1.0, y=1.0, z=1.0)),
)
fig.show()
