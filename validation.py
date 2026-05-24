import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-8
VOXEL_SIZE = 32.0
NUM_REFLECTIONS = 3
NUM_ROTATIONS = 3
OVERLAP_ANGLE_THRESHOLD = np.pi / 6


def unit_vector(v):
    """
    Normalize a vector
    """
    return v / (np.linalg.norm(v) + EPS)

def angle_between_directions(u, v):
    """
    Angle between unoriented directions.
    """
    cosine = abs(np.dot(unit_vector(u), unit_vector(v)))
    angle = np.arccos(np.clip(cosine, 0, 1))
    return angle

def calculate_sde(S_voxels, transform_type, params, tree, L=VOXEL_SIZE):
    """
    E(S, tau) = (1 / |S|) * sum (dist(tau(p), S)^2)
    """
    p = S_voxels / L
    center = 15.5 / L

    if transform_type == 'reflection':
        # Normalize normal vector
        n = params[:3]
        n_norm = np.linalg.norm(n)
        n_hat = n / n_norm + EPS
        d_hat = params[3] / (L * n_norm)

        # Find symmetric point: p' = p - 2 * (p · n_hat + d_hat) * n_hat
        signed_plane_value = np.dot(p, n_hat) + d_hat
        p_prime = p - 2.0 * np.outer(signed_plane_value, n_hat)

    elif transform_type == 'rotation':
        # Normalize axis and convert angle to radians
        theta = params[0]
        axis = params[1:]
        axis_hat = unit_vector(axis)
        
        # Find symmetric point: p' = p cos(theta) + (a x p) sin(theta) + a (a · p) (1 - cos(theta))
        p_centered = p - center
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        axis_dot_point = np.dot(p_centered, axis_hat)
        axis_cross_point = np.cross(axis_hat, p_centered)

        # Back to original coordinates
        p_prime_centered = (
            p_centered * cos_theta
            + axis_cross_point * sin_theta
            + np.outer(axis_dot_point, axis_hat) * (1.0 - cos_theta)
        )
        p_prime = p_prime_centered + center
    
    else:
        return 1.0
    # Query closest surface point
    distances_voxel, _ = tree.query(p_prime * L)
    normalized_distances = distances_voxel / L
    
    # Normalized squared distance
    return np.mean(normalized_distances**2)


def validate_results(voxels, reflection_params, rotation_params, threshold=4e-4, return_errors=False):
    """
    Validation following:
        1. SDE error <= threshold
        2. Angle between plane normals > 30 degrees
        3. Rotation axis must pass every 1-degree rotation test
        4. Angle between rotation axes > 30 degrees
    """
    S_voxels = np.argwhere(voxels).astype(float)
    if len(S_voxels) == 0:
        if return_errors:
            return [], [], {"reflection": [], "rotation": []}
        return [], []
    
    errors = {"reflection": [], "rotation": []}
    tree = cKDTree(S_voxels)

    # 1. Reflection candidate filtering by SDE.
    reflection_candidates = []
    for i in range(NUM_REFLECTIONS):
        plane_params = reflection_params[i]
        error = calculate_sde(S_voxels, 'reflection', plane_params, tree)
        errors["reflection"].append(error)

        if error < threshold:
            reflection_candidates.append({
                'params': plane_params,
                'error': error,
            })

    # 2. Keep lower-error planes first, then remove near-duplicate normals.
    reflection_candidates.sort(key=lambda x: x['error'])

    final_planes = []
    for candidate in reflection_candidates:
        candidate_normal = candidate['params'][:3]
        overlaps_existing_plane = False

        for accepted in final_planes:
            accepted_normal = accepted['params'][:3]
            angle = angle_between_directions(candidate_normal, accepted_normal)
            if angle < OVERLAP_ANGLE_THRESHOLD:
                overlaps_existing_plane = True
                break

        if not overlaps_existing_plane:
            final_planes.append(candidate)

    # 3. Rotation candidate filtering.
    final_axes = []
    for j in range(NUM_ROTATIONS):
        quaternion = rotation_params[j]
        axis = quaternion[1:]

        is_revolution_axis = True
        accumulated_error = 0

        # A revolution axis should stay symmetric for all tested angles.
        for deg in range(1, 360):
            theta = np.deg2rad(deg)
            rotation_params_at_theta = np.array([theta, axis[0], axis[1], axis[2]])
            error = calculate_sde(S_voxels, 'rotation', rotation_params_at_theta, tree)

            if error > threshold:
                is_revolution_axis = False
                break

            accumulated_error += error

        average_error = accumulated_error / 359.0 if is_revolution_axis else error
        errors["rotation"].append(average_error)

        if not is_revolution_axis:
            continue

        # 4. Remove near-duplicate axis directions.
        overlaps_existing_axis = False
        for accepted_axis in final_axes:
            accepted_direction = accepted_axis['params'][1:]
            angle = angle_between_directions(axis, accepted_direction)
            if angle < OVERLAP_ANGLE_THRESHOLD:
                overlaps_existing_axis = True
                break

        if not overlaps_existing_axis:
            final_axes.append({
                'params': quaternion,
                'error': average_error,
            })

    if return_errors:
        return final_planes, final_axes, errors
    return final_planes, final_axes
