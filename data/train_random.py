import torch
import torch.optim as optim

from model import PRSNet
from loss import symmetry_distance_loss, regularization_loss, prsnet_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PRSNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 100
w_r = 0.1

for epoch in range(epochs):
    # Use random data for demonstration
    voxel = torch.randn(1, 1, 32, 32, 32).to(device)
    sampled_surface_points = torch.rand(500, 3).to(device) * 31
    closest_grid = torch.rand(32, 32, 32, 3).to(device)

    reflection_params, rotation_params = model(voxel)

    reflection_params = reflection_params[0]
    rotation_params = rotation_params[0]

    L_sd = symmetry_distance_loss(
        sampled_surface_points,
        reflection_params,
        rotation_params,
        closest_grid
    )

    L_r = regularization_loss(
        reflection_params,
        rotation_params
    )

    L = prsnet_loss(L_sd, L_r, w_r)

    optimizer.zero_grad()

    L.backward()

    optimizer.step()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"L_sd: {L_sd.item():.4f} "
        f"L_r: {L_r.item():.4f} "
        f"Total: {L.item():.4f}"
    )