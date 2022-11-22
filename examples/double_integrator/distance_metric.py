import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import GurobiSolver
from pydrake.solvers import mathematicalprogram as mp

solver = GurobiSolver()
A_continuous = np.array([[0, 1.0], [0, 0]])
B_continuous = np.array([[0], [1]])

h = 0.1
A = np.eye(2) + A_continuous * h
B = B_continuous * h


# %% State of double integrator: [theta, theta_dot]
def calc_optimal_trajectory(x0: np.ndarray, x_goal: np.ndarray):
    T = 50
    h = 0.01
    A = np.eye(2) + A_continuous * h
    B = B_continuous * h
    Q = np.eye(2)
    R = 0.1

    prog = mp.MathematicalProgram()
    x_trj = prog.NewContinuousVariables(T + 1, 2, "x")
    u_trj = prog.NewContinuousVariables(T, "u")

    # x0
    prog.AddLinearEqualityConstraint(np.eye(2), x0, x_trj[0])

    # x_T
    prog.AddLinearEqualityConstraint(np.eye(2), x_goal, x_trj[T])

    for t in range(T):
        # Costs.
        prog.AddQuadraticErrorCost(Q, x_goal, x_trj[t])
        prog.AddQuadraticCost(R * u_trj[t] ** 2)

        # x[t + 1] = A * x[t] + B * u[t]
        prog.AddLinearEqualityConstraint(
            np.hstack([np.eye(2), -A, -B]),
            np.zeros(2),
            np.hstack([x_trj[t + 1], x_trj[t], u_trj[t]]),
        )

    result = solver.Solve(prog)

    return (
        result.GetSolution(x_trj),
        result.GetSolution(u_trj),
        result.get_optimal_cost(),
    )


x0 = np.array([1, 1.0])
x_next_nominal = A @ x0
x_goal_right = x_next_nominal + np.array([0.5, 0.0])  # hard
x_goal_up = x_next_nominal + np.array([0, 0.5])  # easy

x_trj_right, u_trj_right, cost_right = calc_optimal_trajectory(x0, x_goal_right)

x_trj_up, u_trj_up, cost_up = calc_optimal_trajectory(x0, x_goal_up)

print(cost_right, cost_up)

# %%
r = 3
n = 101
theta_list = np.linspace(x_next_nominal[0] - r, x_next_nominal[0] + r, n)
theta_dot_list = np.linspace(x_next_nominal[1] - r, x_next_nominal[1] + r, n)
Theta, Theta_dot = np.meshgrid(theta_list, theta_dot_list)

Sigma = B @ B.T + np.eye(2) * 1e-4

Z = (Theta - x_next_nominal[0]) ** 2 / Sigma[0, 0] + (
    Theta_dot - x_next_nominal[1]
) ** 2 / Sigma[1, 1]

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(Theta, Theta_dot, Z, levels=20)
fig.colorbar(cp)  # Add a colorbar to a plot
ax.set_title("Filled Contours Plot")
ax.set_xlabel("theta")
ax.set_ylabel("theta_dot")

ax.plot(x_trj_right[:, 0], x_trj_right[:, 1], color="r")
ax.plot(x_trj_up[:, 0], x_trj_up[:, 1], color="b")
ax.axis("equal")

plt.show()
