# Planar Hand Ball Manipulation Example

# Setups 

1. Move right
```
qa_l_knots[0] = [-np.pi / 4, -np.pi / 4]
qa_r_knots[0] = [np.pi / 4, np.pi / 4]
q_u0 = np.array([0, 0.35, 0])
params.Q_dict = {
    idx_u: np.array([10, 10, 1e-3]),
    idx_a_l: np.array([1e-3, 1e-3]),
    idx_a_r: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    idx_a_l: 5 * np.array([1, 1]),
    idx_a_r: 5 * np.array([1, 1])}

xd_dict = {idx_u: q_u0 + np.array([0.3, -0.1, 0]),
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}    

params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 0.5 * h, np.ones(dim_u) * 0.5 * h])

def sampling(u_initial, iter):
    return u_initial / (iter ** 0.8)

params.sampling = sampling
params.std_u_initial = np.ones(dim_u) * 0.3
```

2. Rotate in place.

```
qa_l_knots[0] = [-np.pi / 2 + 0.5, -np.pi / 2 + 0.5]
qa_r_knots[0] = [np.pi / 2 -0.5, np.pi / 2 - 0.5]
q_u0 = np.array([0, 0.6, 0])
params.Q_dict = {
    idx_u: np.array([10, 1, 10]),
    idx_a_l: np.array([1e-3, 1e-3]),
    idx_a_r: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 10 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    idx_a_l: 1e2 * np.array([1, 1]),
    idx_a_r: 1e2 * np.array([1, 1])}

xd_dict = {idx_u: q_u0 + np.array([0.0, -0.2, -np.pi/4]),
           idx_a_l: qa_l_knots[0],
           idx_a_r: qa_r_knots[0]}    

params.u_bounds_abs = np.array([
    -np.ones(dim_u) * 0.1 * h, np.ones(dim_u) * 0.1 * h])

def sampling(u_initial, iter):
    return u_initial / (iter ** 0.8)

params.sampling = sampling
params.std_u_initial = np.ones(dim_u) * 0.3
```