import torch

# ============================================================
# Generate 32^3 voxel shapes
# ============================================================

resolution = 32

# ------------------------------------------------------------
# cube
# ------------------------------------------------------------

cube = torch.zeros(resolution, resolution, resolution)

cube[8:24, 8:24, 8:24] = 1.0

torch.save(cube, "cube.pt")

# ------------------------------------------------------------
# sphere
# ------------------------------------------------------------

sphere = torch.zeros(resolution, resolution, resolution)

center = resolution // 2

radius = 8

for x in range(resolution):
    for y in range(resolution):
        for z in range(resolution):

            dx = x - center
            dy = y - center
            dz = z - center

            if dx * dx + dy * dy + dz * dz <= radius * radius:
                sphere[x, y, z] = 1.0

torch.save(sphere, "sphere.pt")

# ------------------------------------------------------------
# cylinder
# ------------------------------------------------------------

cylinder = torch.zeros(resolution, resolution, resolution)

radius = 6

for x in range(resolution):
    for y in range(resolution):

        dx = x - center
        dy = y - center

        if dx * dx + dy * dy <= radius * radius:

            cylinder[x, y, :] = 1.0

torch.save(cylinder, "cylinder.pt")

print("Generated:")
print("cube.pt")
print("sphere.pt")
print("cylinder.pt")