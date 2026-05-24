import torch
import torch.nn as nn
import torch.nn.functional as F

class PRSNet(nn.Module):
    """
    PRS-Net predictor:
    Input: voxel grid x with shape (B, 1, 32, 32, 32)
    Output: 3 reflection planes and 3 rotation axes, each with shape (B, 3, 4)
    """
    def __init__(self):
        super().__init__()
        # 3D CNN encoder with 5 convolutional layers and 4x4x4 max-pooling.
        self.conv1 = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(32, 64, kernel_size=2)
        self.pool = nn.MaxPool3d(2)
        # Shared MLP maps the global voxel feature to a compact latent vector.
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)

        # Three plane heads predict P_i = (n_i, d_i).
        reflection_heads = []
        for _ in range(3):
            reflection_heads.append(nn.Linear(16, 4))
        self.reflection_heads = nn.ModuleList(reflection_heads)
        # Three rotation heads predict quaternions p_j.
        rotation_heads = []
        for _ in range(3):
            rotation_heads.append(nn.Linear(16, 4))
        self.rotation_heads = nn.ModuleList(rotation_heads)

        self._initialize_symmetry_heads()
    
    def _initialize_symmetry_heads(self):
        """
        Initialize the heads with axis-aligned candidates.
        """
        reflection_biases = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        rotation_biases = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # Initialize with small random noise
        # Reflection
        for head, bias in zip(self.reflection_heads, reflection_biases):
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            # Not added to compute gradients
            with torch.no_grad():
                head.bias.copy_(bias)
        # Rotation
        for head, bias in zip(self.rotation_heads, rotation_biases):
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            # Not added to compute gradients
            with torch.no_grad():
                head.bias.copy_(bias)
    
    def forward(self, x):
        # 32^3 -> 16^3 -> 8^3 -> 4^3 -> 2^3 -> 1^3
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        # Flatten (B, 64, 1, 1, 1) to (B, 64).
        x = x.view(x.shape[0], 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        reflection_outputs = []
        for head in self.reflection_heads:
            r = head(x)
            normal = r[:, :3]
            offset = r[:, 3:]
            # Force ||n|| = 1
            normal = normal / (torch.norm(normal, dim=1, keepdim=True) + 1e-8)
            # Put into plane parameters P_i = (n_i, d_i)
            r = torch.cat([normal, offset], dim=1)
            reflection_outputs.append(r)

        reflection_params = torch.stack(
            reflection_outputs,
            dim=1
        )

        rotation_outputs = []
        for head in self.rotation_heads:
            p = head(x)
            # Unit quaternions represent valid 3D rotations.
            p = p / (torch.norm(p, dim=1, keepdim=True) + 1e-8)
            rotation_outputs.append(p)

        rotation_params = torch.stack(
            rotation_outputs,
            dim=1
        )

        return reflection_params, rotation_params
