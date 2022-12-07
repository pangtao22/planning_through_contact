import matplotlib.pyplot as plt
import numpy as np
import pickle

from irs_rrt.irs_rrt import IrsRrt, IrsNode


from tqdm import tqdm


def get_cost_array(irs_rrt, global_metric):
    n_nodes = len(irs_rrt.graph.nodes)
    costs = np.zeros(n_nodes)

    irs_rrt.rrt_params.global_metric = global_metric
    for n in range(1, n_nodes):
        costs[n] = np.min(
            irs_rrt.calc_distance_batch_global(
                irs_rrt.goal, n, is_q_u_only=True
            )
        )

    return costs[1:]


def get_packing_ratio_array(irs_rrt, sampling_function, n_samples, threshold):
    n_nodes = len(irs_rrt.graph.nodes)
    costs = np.zeros(n_nodes)

    for n in tqdm(range(1, n_nodes)):
        samples = sampling_function(n_samples)
        pairwise_distance = irs_rrt.calc_pairwise_distance_batch_local(
            samples, n, is_q_u_only=True
        )
        dist = np.min(pairwise_distance, axis=1)
        costs[n] = np.sum(dist < threshold) / n_samples

    return costs[1:]


def compute_statistics(filename):
    with open(filename, "rb") as f:
        tree = pickle.load(f)

    """Modify the below lines for specific implementations."""

    irs_rrt = IrsRrt.make_from_pickled_tree(tree)
    global_metric = np.ones(21) * 0.1
    global_metric[-2:] = 1
    n_samples = 100
    threshold = 3

    def sampling_function(n_samples):
        samples = np.random.rand(n_samples, 21)
        door_angle_goal = -np.pi / 12 * 5
        samples[:, -2] = door_angle_goal * samples[:, 0]
        samples[:, -1] = samples[:, 1] * np.pi / 4 + np.pi / 4
        return samples

    cost_array = get_cost_array(irs_rrt, global_metric)
    packing_ratio_array = get_packing_ratio_array(
        irs_rrt, sampling_function, n_samples, threshold
    )

    return cost_array, packing_ratio_array


def plot_filename_array(filename_array, color, label):
    cost_array_lst = []
    packing_ratio_lst = []

    for filename in filename_array:
        cost_array, packing_ratio_array = compute_statistics(filename)
        cost_array_lst.append(cost_array)
        packing_ratio_lst.append(packing_ratio_array)

    plt.subplot(1, 2, 1)
    cost_array_lst = np.array(cost_array_lst)
    mean_cost = np.mean(cost_array_lst, axis=0)
    std_cost = np.std(cost_array_lst, axis=0)

    plt.plot(range(len(mean_cost)), mean_cost, "-", color=color, label=label)
    plt.fill_between(
        range(len(mean_cost)),
        mean_cost - std_cost,
        mean_cost + std_cost,
        color=color,
        alpha=0.1,
    )
    plt.xlabel("Iterations")
    plt.ylabel("Closest Distance to Goal")

    plt.subplot(1, 2, 2)
    packing_ratio_lst = np.array(packing_ratio_lst)
    mean_cost = np.mean(packing_ratio_lst, axis=0)
    std_cost = np.std(packing_ratio_lst, axis=0)

    plt.plot(range(len(mean_cost)), mean_cost, "-", color=color, label=label)
    plt.fill_between(
        range(len(mean_cost)),
        mean_cost - std_cost,
        mean_cost + std_cost,
        color=color,
        alpha=0.1,
    )
    plt.xlabel("Iterations")
    plt.ylabel("Packing Ratio")


fig = plt.figure(figsize=(16, 4))
plt.rcParams["font.size"] = "16"
prefix = "tree"  # tree_traj_opt
n_nodes = 500  # 100
filename_array = [f"{prefix}_local_u_{n_nodes}_{i}.pkl" for i in range(5)]
plot_filename_array(filename_array, "springgreen", "iRS-RRT")

filename_array = [f"{prefix}_global_u_{n_nodes}_{i}.pkl" for i in range(5)]
plot_filename_array(filename_array, "red", "Global Metric")

filename_array = [
    f"{prefix}_local_u_{n_nodes}_no_contact_{i}.pkl" for i in range(5)
]
plot_filename_array(filename_array, "royalblue", "No Contact")

#%%
plt.subplot(1, 2, 1)
plt.legend()
plt.subplot(1, 2, 2)
plt.legend()

fig.set_figheight(6)
fig.set_figwidth(12)
plt.show()
