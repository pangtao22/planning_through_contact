import numpy as np
from pydrake.all import MathematicalProgram, GurobiSolver


def solve_mpc(
    At,
    Bt,
    ct,
    Q,
    Qd,
    R,
    x0,
    x_trj_d,
    solver: GurobiSolver,
    indices_u_into_x=None,
    x_bound_abs=None,
    u_bound_abs=None,
    x_bound_rel=None,
    u_bound_rel=None,
    xinit=None,
    uinit=None,
):
    """
    Solve time-varying LQR problem as an instance of a quadratic program (QP).
    Uses Drake's OSQP solver by default. Can use other solvers that Drake
    supports, but OSQP will often result in fastest time (while being slightly inexact)
    args:
     - At   (np.array, dim: T x n x n) : time-varying dynamics matrix
     - Bt   (np.array, dim: T x n x m) : time-varying actuation matrix.
     - ct   (np.array, dim: T x n x 1) : bias term for affine dynamics.
     - Q    (np.array, dim: n x n): Quadratic cost on state error x(t) - xd(t)
     - Qd    (np.array, dim: n x n): Quadratic cost on final state error x(T) - xd(T)
     - R    (np.array, dim: m x m): Quadratic cost on actuation.
     - x0   (np.array, dim: n): Initial state of the problem.
     - x_trj_d  (np.array, dim: (T + 1) x n): Desired trajectory of the system.
     - solver (Drake's solver class): solver. Initialized outside the loop for
             better performance.
     - indices_u_into_x: (np.array): in a quasistatic system, x is the configuration
         of the whole system, whereas u is the commanded configuration of the
         actuated DOFs. This is used to detect whether or not the system is using
         position-controlled dynamics, or force/velocity controlled dynamics.
     - xbound_abs (np.array, dim: 2 x n): (lb, ub) Bound on state variables (abs).
         constrains x globally.
     - ubound_abs (np.array, dim: 2 x u): (lb, ub) Bound on input variables (abs).
         constrains u globally.
     - xbound_rel (np.array, dim: 2 x n): (lb, ub) Bound on state variables (rel).
         constrains the of x relative to the last x (Bound on x[t] - x[t-1]).
     - ubound_rel (np.array, dim: 2 x u): (lb, ub) Bound on input variables (rel).
         constrains the of u relative to the last u (Bound on u[t] - u[t-1]).
     - solver (Drake's solver class): solver. Initialized outside the loop for
             better performance.
     - xinit (np.array, dim: (T + 1) x n): initial guess for state.
     - uinit (np.array, dim: T x m): initial guess for input.
    NOTE(terry-suh): This implementation needs to be "blazing fast.". It is
    performed O(iterations * timesteps^2).
    """
    if np.isnan(At).any() or np.isnan(Bt).any():
        raise RuntimeError("At or Bt is nan.")

    prog = MathematicalProgram()

    T = At.shape[0]
    n_x = Q.shape[0]
    n_u = R.shape[0]

    # 1. Declare new variables corresponding to optimal state and input.
    xt = prog.NewContinuousVariables(T + 1, n_x, "state")
    ut = prog.NewContinuousVariables(T, n_u, "input")

    if xinit is not None:
        prog.SetInitialGuess(xt, xinit)
    if uinit is not None:
        prog.SetInitialGuess(ut, uinit)

    # 2. Initial constraint.
    prog.AddLinearEqualityConstraint(np.eye(n_x), x0, xt[0])

    # 3. Loop over to add dynamics constraints and costs.
    for t in range(T):
        # Add affine dynamics constraint.
        prog.AddLinearEqualityConstraint(
            np.hstack((At[t], Bt[t], -np.eye(n_x))),
            -ct[t],
            np.hstack((xt[t], ut[t], xt[t + 1])),
        )

        # Compute differences.
        if indices_u_into_x is not None:
            if t == 0:
                du = ut[t] - xt[t, indices_u_into_x]
            else:
                du = ut[t] - ut[t - 1]

            prog.AddQuadraticCost(du.dot(R).dot(du))

        # Add constraints.
        if x_bound_abs is not None:
            prog.AddBoundingBoxConstraint(
                x_bound_abs[0, t], x_bound_abs[1, t], xt[t, :]
            )
        if u_bound_abs is not None:
            prog.AddBoundingBoxConstraint(
                u_bound_abs[0, t], u_bound_abs[1, t], ut[t, :]
            )
        if x_bound_rel is not None:
            raise NotImplementedError
        if u_bound_rel is not None:
            raise NotImplementedError

        # Add cost.
        prog.AddQuadraticErrorCost(Q, x_trj_d[t], xt[t])

    # Add final constraint.
    prog.AddQuadraticErrorCost(Qd, x_trj_d[T], xt[T])

    if x_bound_abs is not None:
        prog.AddBoundingBoxConstraint(
            x_bound_abs[0, T], x_bound_abs[1, T], xt[T]
        )

    # 4. Solve the program.
    result = solver.Solve(prog)

    if not result.is_success():
        raise RuntimeError("TV_LQR failed. Optimization problem is not solved.")

    return result.GetSolution(xt), result.GetSolution(ut)
