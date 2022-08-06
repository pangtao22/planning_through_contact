import numpy as np
import torch

#%%
n = 4
theta = np.random.rand(n) * np.pi * 2
phi = np.random.rand(n) * np.pi

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
p_surface = torch.from_numpy(np.vstack([x, y, z]).T)

p = p_surface.mean(axis=0).requires_grad_(True)
r = torch.tensor(0.9, requires_grad=True)

optim = torch.optim.Adam([p, r], lr=0.1)
for i in range(500):
    optim.zero_grad()
    loss = (((((p - p_surface) ** 2).sum(axis=1)).sqrt() - r) ** 2).sum()
    loss.backward()
    optim.step()
    print(i, loss)
