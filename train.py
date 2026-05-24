import argparse
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from model import PRSNet
from dataset import PRSNetDataset
from loss import symmetry_distance_loss, regularization_loss, prsnet_loss

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=str, default="ModelNet40_processed")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--w-r", type=float, default=25)
parser.add_argument("--save-path", type=str, default="prsnet_model.pth")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PRSNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

dataset = PRSNetDataset(data_root=args.data_root, split=args.split)

dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True
)

epochs = args.epochs
w_r = args.w_r

for epoch in range(epochs):

    model.train()

    epoch_L_sd = 0.0
    epoch_L_r = 0.0
    epoch_total = 0.0

    for voxels, surface_points, closest_grids in dataloader:

        voxels = voxels.to(device)
        surface_points = surface_points.to(device)
        closest_grids = closest_grids.to(device)

        batch_L_sd = 0.0
        batch_L_r = 0.0

        ref_params_batch, rot_params_batch = model(voxels)

        for i in range(voxels.size(0)):

            ref_p = ref_params_batch[i]
            rot_p = rot_params_batch[i]

            sp = surface_points[i]
            cg = closest_grids[i]

            L_sd = symmetry_distance_loss(sp, ref_p, rot_p, cg)

            L_r = regularization_loss(ref_p, rot_p)

            batch_L_sd += L_sd
            batch_L_r += L_r

        batch_L_sd = batch_L_sd / voxels.size(0)
        batch_L_r = batch_L_r / voxels.size(0)

        L = prsnet_loss(batch_L_sd, batch_L_r, w_r)

        optimizer.zero_grad()

        L.backward()

        optimizer.step()

        epoch_L_sd += batch_L_sd.item()
        epoch_L_r += batch_L_r.item()
        epoch_total += L.item()

    num_batches = len(dataloader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"L_sd: {epoch_L_sd / num_batches:.4f} "
        f"L_r: {epoch_L_r / num_batches:.4f} "
        f"Total: {epoch_total / num_batches:.4f}"
    )

torch.save(model.state_dict(), args.save_path)

print(f"Model saved to {args.save_path}")
