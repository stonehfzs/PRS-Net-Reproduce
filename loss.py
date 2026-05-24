# loss.py: Define the loss functions for PRS-Net training
import torch
# Constants
EPS = 1e-8
VOXEL_SIZE = 32.0
NUM_REFLECTIONS = 3
NUM_ROTATIONS = 3

def reflection_symmetry_point(q_k, n_i, d_i):
    """
    q'_k = q_k - 2 * ((q_k · n_i + d_i) / ||n_i||^2) * n_i
    """
    signed_plane_value = torch.matmul(q_k, n_i) + d_i
    normal_squared_norm = torch.sum(n_i * n_i) + EPS
    reflection_scale = 2.0 * signed_plane_value / normal_squared_norm
    q_k_prime = q_k - reflection_scale.unsqueeze(-1) * n_i
    return q_k_prime

def quaternion_multiply(a, b):
    """
    Define quaternion multiplication for batches of quaternions:
    a = aw + ax*i + ay*j + az*k
    b = bw + bx*i + by*j + bz*k
    """
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    product = torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dim=1)
    return product

def rotation_symmetry_point(q_k, p_j):
    """
    q_hat'_k = p_j * q_hat_k * p_j^{-1}
    q'_k = Im(q_hat'_k) + c
    """
    # Centralize and convert to quaternion format
    num_points = q_k.shape[0]
    center = 15.5
    q_k_centered = q_k - center
    q_k_hat = torch.cat([
        torch.zeros((num_points, 1), device=q_k.device),
        q_k_centered
    ], dim=1)
    # Compute inverse and rotation
    p_j_inv = p_j.clone()
    p_j_inv[1:] = -p_j_inv[1:]
    p_j_batch = p_j.unsqueeze(0).expand(num_points, -1)
    p_j_inv_batch = p_j_inv.unsqueeze(0).expand(num_points, -1)
    q_k_hat_prime = quaternion_multiply(
        quaternion_multiply(p_j_batch, q_k_hat),
        p_j_inv_batch
    )
    return q_k_hat_prime[:, 1:] + center

def closest_distance_batch(q_k_primes, closest_grid, voxel_size=VOXEL_SIZE):
    """
    D_k = ||q'_k - o(q'_k)||^2 / L^2
    """
    # Convert to grid points
    grid_size = closest_grid.shape[0]
    idx_x = torch.clamp(q_k_primes[:, 0].long(), 0, grid_size - 1)
    idx_y = torch.clamp(q_k_primes[:, 1].long(), 0, grid_size - 1)
    idx_z = torch.clamp(q_k_primes[:, 2].long(), 0, grid_size - 1)
    # Get closest surface points from the grid
    closest_surface_points = closest_grid[idx_x, idx_y, idx_z]
    displacement = q_k_primes - closest_surface_points
    squared_distance_voxel = torch.sum(displacement**2, dim=1)
    # Normalize by voxel size squared error
    normalized_squared_distance = squared_distance_voxel / (voxel_size ** 2)
    return normalized_squared_distance

def symmetry_distance_loss(sampled_surface_points, reflection_params, rotation_params, closest_grid):
    """
    L_sd = sum_i sum_k D_reflection(i, k) + sum_j sum_k D_rotation(j, k)
    """
    L_sd = 0.0
    # Reflection
    for i in range(NUM_REFLECTIONS):
        n_i = reflection_params[i][:3]
        d_i = reflection_params[i][3]
        q_k_primes = reflection_symmetry_point(sampled_surface_points, n_i, d_i)
        L_sd += torch.sum(closest_distance_batch(q_k_primes, closest_grid))
    # Rotation
    for j in range(NUM_ROTATIONS):
        p_j = rotation_params[j]
        q_k_primes = rotation_symmetry_point(sampled_surface_points, p_j)
        L_sd += torch.sum(closest_distance_batch(q_k_primes, closest_grid))
    return L_sd

def regularization_loss(reflection_params, rotation_params):
    """
    M1 = [n_1; n_2; n_3]
    M2 = [v_1; v_2; v_3]
    A  = M1 M1^T - I
    B  = M2 M2^T - I
    L_r = ||A||_F^2 + ||B||_F^2
    """
    # Reflection
    M1 = reflection_params[:, :3]
    I = torch.eye(NUM_REFLECTIONS, device=M1.device)
    A = torch.matmul(M1, M1.T) - I
    # Rotation
    rotation_axes = rotation_params[:, 1:]
    M2 = rotation_axes / (torch.norm(rotation_axes, dim=1, keepdim=True) + EPS)
    B = torch.matmul(M2, M2.T) - I

    # Frobenius norm squared
    L_r = torch.norm(A, p='fro')**2 + torch.norm(B, p='fro')**2
    return L_r

def prsnet_loss(L_sd, L_r, w_r):
    """
        Loss = L_sd + w_r * L_r
    """
    return L_sd + w_r * L_r
