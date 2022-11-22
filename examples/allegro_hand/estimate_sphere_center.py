import numpy as np
import torch


#%%
def estimate_center_and_r(p_surface: torch.Tensor, r_initial_guess: float):
    p = p_surface.mean(axis=0).requires_grad_(True)
    r = torch.tensor(r_initial_guess, requires_grad=True)

    optim = torch.optim.Adam([p, r], lr=0.1)
    for i in range(500):
        optim.zero_grad()
        loss = (
            ((((p - p_surface) ** 2).sum(axis=1)).sqrt() - r) ** 2
        ).sum() + 1e-4 * r**2

        loss.backward()
        optim.step()

    print(f"final loss: {loss} after {i + 1} iterations")
    return p.detach().numpy(), r.item()


def estimate_center(p_surface: torch.Tensor, r: float):
    p_mean = p_surface.mean(axis=0)
    p_surface_centered = p_surface - p_mean
    p = p_mean.clone().requires_grad_(True)
    r = torch.Tensor([r])
    optim = torch.optim.Adam([p, r], lr=0.1)
    for i in range(500):
        optim.zero_grad()
        loss = (
            ((((p - p_surface_centered) ** 2).sum(axis=1)).sqrt() - r) ** 2
        ).sum() + 1e-4 * r**2

        loss.backward()
        optim.step()

    print(f"final loss: {loss} after {i + 1} iterations")
    return (p + p_mean).detach().numpy()


if __name__ == "__main__":
    n = 4
    theta = np.random.rand(n) * np.pi * 2
    phi = np.random.rand(n) * np.pi

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    p_surface = torch.from_numpy(np.vstack([x, y, z]).T)
    p_center, r = estimate_center_and_r(p_surface, r_initial_guess=0.9)
    print(p_center, r)
