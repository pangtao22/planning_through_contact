class DynamicalSystem:
    def __init__(self): 
        """
        Base virtual dynamical systems class. 
        Any dynamics as an input to the system must inherit from this class.
        TODO(terry-suh): Consider using ABC?
        """
        self.h = 0
        self.dim_x = 0
        self.dim_u = 0

    def dynamics(self, x, u):
        """
        Numerical expression for dynamics in state-space form.
        args:
        - x_t (np.array, dim: n): state
        - u_t (np.array, dim: m): action
        returns 
        - x_{t+1} (np.array, dim: n), next state.
        """

        raise NotImplementedError("This class is virtual.")

    def dynamics_batch(self, x, u):
        """
        Special batch implementation of dynamics that allows 
        parallel evaluation. If the dynamics cannot be easily
        batched, replace this method with a for loop over the 
        dynamics function.
        args:
        - x_t (np.array, dim: B x n): batched state
        - u_t (np.array, dim: B x m): batched action
        returns 
        - x_{t+1} (np.array, dim: B x n): next batched state.
        """

        raise NotImplementedError("This class is virtual.")        

    def jacobian_xu(self, x, u):
        """
        Numerical jacobian of dynamics w.r.t. x and u. 
        Should be a fat matrix with the first n columns corresponding
        to dfdx, and the last m columns corresponding to dfdu.
        args:
        - x_t (np.array, dim: n): state
        - u_t (np.array, dim: m): action
        returns:
        - J_xu (np.array, dim: n x (n + m)): df/dxu
        """

        raise NotImplementedError("This class is virtual.")

    def jacobian_xu_batch(self, x, u):
        """
        Batch jacobian of dynamics w.r.t. x and u that allows for faster 
        parallelized computations. If Jacobian computation cannot be
        easily batched, replace this method with a for loop over the 
        jacobian_xu function.
        args:
        - x_t (np.array, dim: B x n): state
        - u_t (np.array, dim: B x m): action
        returns:
        - J_xu (np.array, dim: B x n x (n + m)): batched Jacobians.
        """

        raise NotImplementedError("This class is virtual.")
