from typing import Union

import numpy as np
import plotly.graph_objects as go


line_width = 5
pink = (255, 105, 180)
dark_red = (212, 20, 20)
blue = (0, 0, 255)
bright_blue = (0, 150, 255)


def rgb_tuple_2_rgba_str(rgb, a):
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {a})"


def draw_ball(
    y: float,
    z: float,
    theta: float,
    fill_color: str,
    line_color: str,
    line_dash: Union[str, None],
    fig,
):
    fig.add_shape(
        type="circle",
        x0=y - 0.25,
        x1=y + 0.25,
        y0=z - 0.25,
        y1=z + 0.25,
        fillcolor=fill_color,
        line_color=line_color,
        line_dash=line_dash,
        line_width=line_width,
    )

    fig.add_shape(
        type="line",
        x0=y,
        y0=z,
        x1=y + 0.25 * np.sin(theta),
        y1=z + 0.25 * np.cos(theta),
        line_width=line_width,
        line=dict(color=line_color, dash=line_dash, width=line_width),
    )


def plot_planar_hand_from_vector(
    x, fig, color_ball_fill, color_ball_line, color_arms_fill, color_arms_line
):
    def make_R(theta):
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return R

    polygon1 = np.array(
        [
            0.3 * np.array([1, 1, 0, 0, 1]),
            0.1 * np.array([0.5, -0.5, -0.5, 0.5, 0.5]),
            np.array([1, 1, 1, 1, 1]),
        ]
    )

    polygon2 = np.array(
        [
            0.2 * np.array([1, 1, 0, 0, 1]),
            0.1 * np.array([0.5, -0.5, -0.5, 0.5, 0.5]),
            np.array([1, 1, 1, 1, 1]),
        ]
    )

    y_ball = x[0]
    z_ball = x[1]
    theta_ball = x[2]
    q1_l = x[3] + np.pi
    q2_l = x[4]

    q1_r = x[5]
    q2_r = x[6]

    circle_l0 = np.array([-0.1, 0.0])
    circle_l1 = circle_l0 + make_R(q1_l)[:2, :2].dot(np.array([0.3, 0.0]))
    circle_l2 = circle_l1 + make_R(q1_l + q2_l)[:2, :2].dot(
        np.array([0.2, 0.0])
    )

    circle_r0 = np.array([0.1, 0.0])
    circle_r1 = circle_r0 + make_R(q1_r)[:2, :2].dot(np.array([0.3, 0.0]))
    circle_r2 = circle_r1 + make_R(q1_r + q2_r)[:2, :2].dot(
        np.array([0.2, 0.0])
    )

    fig.add_shape(
        type="circle",
        x0=circle_l0[0] - 0.05,
        x1=circle_l0[0] + 0.05,
        y0=circle_l0[1] - 0.05,
        y1=circle_l0[1] + 0.05,
        fillcolor=color_arms_fill,
        line_color=color_arms_line,
        line_width=line_width,
    )
    fig.add_shape(
        type="circle",
        x0=circle_l1[0] - 0.05,
        x1=circle_l1[0] + 0.05,
        y0=circle_l1[1] - 0.05,
        y1=circle_l1[1] + 0.05,
        fillcolor=color_arms_fill,
        line_color=color_arms_line,
        line_width=line_width,
    )
    fig.add_shape(
        type="circle",
        x0=circle_l2[0] - 0.05,
        x1=circle_l2[0] + 0.05,
        y0=circle_l2[1] - 0.05,
        y1=circle_l2[1] + 0.05,
        fillcolor=color_arms_fill,
        line_color=color_arms_line,
        line_width=line_width,
    )

    fig.add_shape(
        type="circle",
        x0=circle_r0[0] - 0.05,
        x1=circle_r0[0] + 0.05,
        y0=circle_r0[1] - 0.05,
        y1=circle_r0[1] + 0.05,
        fillcolor=color_arms_fill,
        line_color=color_arms_line,
        line_width=line_width,
    )

    fig.add_shape(
        type="circle",
        x0=circle_r1[0] - 0.05,
        x1=circle_r1[0] + 0.05,
        y0=circle_r1[1] - 0.05,
        y1=circle_r1[1] + 0.05,
        fillcolor=color_arms_fill,
        line_color=color_arms_line,
        line_width=line_width,
    )

    fig.add_shape(
        type="circle",
        x0=circle_r2[0] - 0.05,
        x1=circle_r2[0] + 0.05,
        y0=circle_r2[1] - 0.05,
        y1=circle_r2[1] + 0.05,
        fillcolor=color_arms_fill,
        line_color=color_arms_line,
        line_width=line_width,
    )

    draw_ball(
        y=y_ball,
        z=z_ball,
        theta=theta_ball,
        fill_color=color_ball_fill,
        line_color=color_ball_line,
        line_dash=None,
        fig=fig,
    )

    body_1 = make_R(q1_l).dot(polygon1)[0:2] + np.array([circle_l0]).transpose()
    fig.add_trace(
        go.Scatter(
            x=body_1[0, :],
            y=body_1[1, :],
            fill="toself",
            fillcolor=color_arms_fill,
            line_color=color_arms_line,
            line_width=line_width,
        )
    )

    body_2 = (
        make_R(q1_l + q2_l).dot(polygon2)[0:2]
        + np.array([circle_l1]).transpose()
    )
    fig.add_trace(
        go.Scatter(
            x=body_2[0, :],
            y=body_2[1, :],
            fill="toself",
            fillcolor=color_arms_fill,
            line_color=color_arms_line,
            line_width=line_width,
        )
    )

    body_3 = make_R(q1_r).dot(polygon1)[0:2] + np.array([circle_r0]).transpose()
    fig.add_trace(
        go.Scatter(
            x=body_3[0, :],
            y=body_3[1, :],
            fill="toself",
            fillcolor=color_arms_fill,
            line_color=color_arms_line,
            line_width=line_width,
        )
    )

    body_4 = (
        make_R(q1_r + q2_r).dot(polygon2)[0:2]
        + np.array([circle_r1]).transpose()
    )
    fig.add_trace(
        go.Scatter(
            x=body_4[0, :],
            y=body_4[1, :],
            fill="toself",
            fillcolor=color_arms_fill,
            line_color=color_arms_line,
            line_width=line_width,
        )
    )


def plot_robots_with_major_and_minor_goals(
    x, qu_goal_major, qu_goal_minor, colors
):
    fig = go.Figure()
    plot_planar_hand_from_vector(
        x,
        fig,
        color_ball_fill="rgba(255,255,255,0.1)",
        color_ball_line="rgba(0,0,0,1)",
        color_arms_fill="rgba(105, 105, 105, 0.5)",  # 20, 180, 128
        color_arms_line="rgba(105, 105, 105, 1.0)",
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

    # draw goal
    draw_ball(
        y=qu_goal_major[0],
        z=qu_goal_major[1],
        theta=qu_goal_major[2],
        fill_color=rgb_tuple_2_rgba_str(colors[0], 0.1),
        line_color=rgb_tuple_2_rgba_str(colors[0], 1.0),
        line_dash="dash",
        fig=fig,
    )

    draw_ball(
        y=qu_goal_minor[0],
        z=qu_goal_minor[1],
        theta=qu_goal_minor[2],
        fill_color=rgb_tuple_2_rgba_str(colors[1], 0.1),
        line_color=rgb_tuple_2_rgba_str(colors[1], 1.0),
        line_dash="dash",
        fig=fig,
    )

    fig.show()
