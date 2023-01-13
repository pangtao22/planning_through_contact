import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import plotly.io as pio

pio.renderers.default = "browser"  # see plotly charts in pycharm.

#%%
q0 = 0.0
v0 = 0.0
m = 1.0
k = 100.0
d = 5  # damping ratio = 1.


#%% explicit Euler
def calc_implicit(q, v, h, u):
    k_tqv = -k * (q - u) - d * v
    v_next = (m * v + h * k_tqv + h * d * v) / (m + h * h * k + h * d)
    q_next = q + h * v_next

    return np.array([q_next, v_next])


h = 5e-1
T = 3.0
x_trj_implicit = np.zeros([int(T / h) + 1, 2])
x_trj_implicit[0] = [q0, v0]
for i in range(len(x_trj_implicit) - 1):
    q, v = x_trj_implicit[i]
    x_trj_implicit[i + 1] = calc_implicit(q, v, h, 1.0)


t = np.arange(len(x_trj_implicit)) * h


plt.axhline(1.0, linestyle="--")
plt.step(t, x_trj_implicit[:, 0], where="pre", label="implicit")
plt.xlabel("t [s]")
plt.ylabel("u")
plt.show()


#%%
def integrate_to_T(u, T: float, h: float):
    x_trj_implicit = np.zeros([int(T / h) + 1, 2])

    for i in range(len(x_trj_implicit) - 1):
        q, v = x_trj_implicit[i]
        x_trj_implicit[i + 1] = calc_implicit(q, v, h, u)

    return x_trj_implicit


x_finals = np.zeros((51, 51, 2))
u_values = np.linspace(0, 5.0, 51)
t_values = np.linspace(0, 2.0, 51)

for i, u in enumerate(u_values):
    for j, T in enumerate(t_values):
        x_finals[i, j] = integrate_to_T(u, T, 1e-3)[-1]


#%%
layout = go.Layout(
    autosize=False,
    width=600,
    height=400,
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        camera_projection_type="perspective",
        xaxis_title_text="u",
        yaxis_title_text="t [s]",
        zaxis_title_text="q_a",
        aspectmode="cube",
        aspectratio=dict(x=1.0, y=1.0, z=1.0),
    ),
)

hover_template_u_t_q = (
    "<i>u</i>: %{x:.2f}<br>" + "<i>t</i>: %{y:.2f}<br>" + "<i>q_a</i>: %{z:.2f}"
)

transients_plot = go.Surface(
    z=x_finals[:, :, 0].T,
    x=u_values,
    y=t_values,
    opacity=1.0,
    name="transients",
    hovertemplate=hover_template_u_t_q,
)

u_max = np.max(u_values)
t_max = np.max(t_values)

steady_state_plot = go.Scatter3d(
    x=[0, u_max],
    y=[t_max, t_max],
    z=[0, u_max],
    name="steady_state",
    mode="lines",
    line=dict(color="cyan", width=10),
    hovertemplate=hover_template_u_t_q,
)

fig = go.Figure(data=[transients_plot, steady_state_plot], layout=layout)
fig.show()
