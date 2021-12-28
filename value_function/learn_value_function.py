from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from planning_through_contact.examples.planar_hand.planar_hand_setup import *
from rrt.utils import load_rrt

import torch
import numpy as np


def setup_relu(relu_layer_width: tuple,
               params=None,
               negative_slope: float = 0.01,
               bias: bool = True,
               dtype=torch.float64):
    """
    Setup a relu network.
    @param negative_slope The negative slope of the leaky relu units.
    @param bias whether the linear layer has bias or not.
    """
    assert (isinstance(relu_layer_width, tuple))
    if params is not None:
        assert (isinstance(params, torch.Tensor))

    def set_param(linear, param_count):
        linear.weight.data = params[param_count:param_count +
                                    linear.in_features *
                                    linear.out_features].clone().reshape(
                                        (linear.out_features,
                                         linear.in_features))
        param_count += linear.in_features * linear.out_features
        if bias:
            linear.bias.data = params[param_count:param_count +
                                      linear.out_features].clone()
            param_count += linear.out_features
        return param_count

    linear_layers = [None] * (len(relu_layer_width) - 1)
    param_count = 0
    for i in range(len(linear_layers)):
        next_layer_width = relu_layer_width[i + 1]
        linear_layers[i] = torch.nn.Linear(relu_layer_width[i],
                                           next_layer_width,
                                           bias=bias).type(dtype)
        if params is None:
            pass
        else:
            param_count = set_param(linear_layers[i], param_count)
    layers = [None] * (len(linear_layers) * 2 - 1)
    for i in range(len(linear_layers) - 1):
        layers[2 * i] = linear_layers[i]
        layers[2 * i + 1] = torch.nn.LeakyReLU(negative_slope)
    layers[-1] = linear_layers[-1]
    relu = torch.nn.Sequential(*layers)
    return relu


def train_approximator(dataset,
                       model,
                       batch_size,
                       num_epochs,
                       lr,
                       verbose=True):

    train_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_set_size, test_set_size])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)

    variables = model.parameters()
    optimizer = torch.optim.Adam(variables, lr=lr)
    loss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.
        for i, data in enumerate(train_loader, 0):
            input_samples, target = data
            optimizer.zero_grad()

            output_samples = model(input_samples)
            batch_loss = loss(output_samples, target)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
        test_input_samples, test_target = test_set[:]
        test_output_samples = model(test_input_samples)
        test_loss = loss(test_output_samples, test_target)

        if verbose:
            print(f"epoch {epoch} training loss " +
                  f"{running_loss/len(train_loader)}," +
                  f" test loss {test_loss}")

q_dynamics = QuasistaticDynamics(h=h,
                                 quasistatic_model_path=quasistatic_model_path,
                                 internal_viz=True)
q_sim_py = q_dynamics.q_sim_py

plant = q_sim_py.get_plant()
q_sim_py.get_robot_name_to_model_instance_dict()
model_a_l = plant.GetModelInstanceByName(robot_l_name)
model_a_r = plant.GetModelInstanceByName(robot_r_name)
model_u = plant.GetModelInstanceByName(object_name)
rrt = load_rrt("rrt.pkl")

x = []
y = []

for node in list(rrt.nodes):
    xdata = node.q[int(model_u)]
    for model in [model_a_l, model_a_r]:
        xdata = np.concatenate((xdata, node.q[int(model)]))
    x.append(xdata)
    y.append(node.value)

dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(x)), torch.tensor(np.array(y)))

model = setup_relu((7, 12, 12, 1),
                    params=None,
                    negative_slope=0.1,
                    bias=True,
                    dtype=torch.float64)

train_approximator(dataset, model, batch_size=50, num_epochs=200, lr=0.001)