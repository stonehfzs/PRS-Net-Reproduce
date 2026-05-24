import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm

def random_rotation_matrix(rng):
    """
    Sample random 3D rotation.
    """
    # Uniformly sample a unit quaternion on S^3
    u1, u2, u3 = rng.random(3)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    # Convert quaternion to a 3D rotation matrix
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw), 0.0],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw), 0.0],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def voxelize_and_dist(off_path, voxel_size=32, rotation_seed=None):
    """
    Preprocess one mesh into PRS-Net training tensors.
    Input:
        off_path: *.off file path
        voxel_size: voxel grid resolution L=32
        rotation_seed: random seed for train augmentation
    Output:
        voxels:
            shape = (L, L, L)
            meaning = surface occupancy grid used as network input.
        surface_points: surface points used in L_sd, shape = (1000, 3)
        closest_grid:
            shape = (L, L, L, 3)
            meaning = closest_grid[x, y, z] stores the approximate closest
                      surface point o(x, y, z).
    """
    try:
        mesh = trimesh.load(off_path)
        # Augmentation training: O_aug = R * O
        if rotation_seed is not None:
            rng = np.random.default_rng(rotation_seed)
            mesh.apply_transform(random_rotation_matrix(rng))

        # Normalize mesh coordinates into the voxel coordinate system:
        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        grid_center = (voxel_size - 1) / 2.0
        # Avoid whole space occupied
        scale = (voxel_size - 5) / max(extents)

        # Convert to voxel coordinates and center the mesh
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.apply_scale(scale)
        mesh.apply_translation([grid_center, grid_center, grid_center])

        # Surface points used for L_sd and visualization
        surface_points, _ = trimesh.sample.sample_surface(mesh, 1000)

        # Sample 10000 points and turn to voxel grids
        points_for_voxels, _ = trimesh.sample.sample_surface(mesh, 10000)
        voxel_indices = np.floor(points_for_voxels).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, voxel_size - 1)
        # Occupancy grid
        voxels = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
        voxels[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0

        # Precompute closest points and using KDTree to search
        dense_surface, _ = trimesh.sample.sample_surface(mesh, 2000)
        tree = KDTree(dense_surface)
        # Enumerate all L^3 voxel grid points and query their closest surface points
        grid_coords = np.indices(
            (voxel_size, voxel_size, voxel_size)
        ).reshape(3, -1).T.astype(np.float32)

        # closest_grid[x, y, z] stores the approximate closest surface point
        _, nearest_indices = tree.query(grid_coords)
        closest_points = dense_surface[nearest_indices]
        closest_grid = closest_points.reshape(voxel_size, voxel_size, voxel_size, 3)

        return voxels, surface_points.astype(np.float32), closest_grid.astype(np.float32)
    
    except Exception as exc:
        print(f"Failed to process {off_path}: {exc}")
        return None, None, None


def worker(task):
    """
    Save *.off file to *.pt after voxelization and precomputation
    """
    off_file, save_path, voxel_size, rotation_seed = task

    # Skip already generated files
    if save_path.exists():
        return
    
    # Process and save
    voxels, surface_points, closest_grid = voxelize_and_dist(
        str(off_file),
        voxel_size=voxel_size,
        rotation_seed=rotation_seed,
    )
    if voxels is None:
        return
    
    # Convert to PyTorch tensors and save
    data = {
        "voxels": torch.from_numpy(voxels),
        "surface_points": torch.from_numpy(surface_points),
        "closest_grid": torch.from_numpy(closest_grid),
        "source_off": str(off_file),
        "rotation_seed": rotation_seed,
    }
    torch.save(data, str(save_path))


def build_standard_tasks(root_path, output_path, voxel_size):
    """
    Build tasks for standard preprocessing without augmentation
    """
    tasks = []
    for off_file in sorted(root_path.glob("**/*.off")):
        relative_path = off_file.relative_to(root_path)
        save_path = output_path / relative_path.with_suffix(".pt")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        tasks.append((off_file, save_path, voxel_size, None))
    return tasks


def build_augmented_tasks(root_path, output_path, voxel_size, target_per_category, seed):
    """
    Build tasks for augmentation preprocessing with random rotations
    """
    tasks = []
    # Random number generator
    rng = np.random.default_rng(seed)
    category_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])

    # Generate augmented samples for training
    for category_dir in category_dirs:
        category = category_dir.name
        train_files = sorted((category_dir / "train").glob("*.off"))
        test_files = sorted((category_dir / "test").glob("*.off"))

        if train_files:
            train_output_dir = output_path / category / "train"
            train_output_dir.mkdir(exist_ok=True, parents=True)
            # Sample source meshes and random seeds for augmentation
            sampled_indices = rng.integers(0, len(train_files), size=target_per_category)
            rotation_seeds = rng.integers(0, np.iinfo(np.int32).max, size=target_per_category)

            for aug_idx, (file_idx, rotation_seed) in enumerate(zip(sampled_indices, rotation_seeds)):
                off_file = train_files[int(file_idx)]
                save_path = train_output_dir / f"{category}_aug_{aug_idx:05d}.pt"
                tasks.append((off_file, save_path, voxel_size, int(rotation_seed)))

        if test_files:
            for off_file in test_files:
                relative_path = off_file.relative_to(root_path)
                save_path = output_path / relative_path.with_suffix(".pt")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                tasks.append((off_file, save_path, voxel_size, None))

    return tasks


def process_dataset(
    root_dir,
    output_dir,
    num_workers=16,
    voxel_size=32,
    augment_train=False,
    target_per_category=4000,
    seed=1234,
):
    """
    Preprocess the ModelNet40 dataset for PRS-Net training
    """
    root_path = Path(root_dir)

    if output_dir is None:
        dataset_name = root_path.name
        output_dir = (
            f"{dataset_name}_augmented_processed"
            if augment_train
            else f"{dataset_name}_processed"
        )
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if augment_train:
        tasks = build_augmented_tasks(
            root_path=root_path,
            output_path=output_path,
            voxel_size=voxel_size,
            target_per_category=target_per_category,
            seed=seed,
        )
    else:
        tasks = build_standard_tasks(root_path, output_path, voxel_size)
    
    # Output meta information about the preprocessing run
    print(f"Prepared {len(tasks)} preprocessing tasks.")
    print(f"Source: {root_path}")
    print(f"Destination: {output_path}")
    print(f"Workers: {num_workers}, voxel size: {voxel_size}")

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)))

# Preprocess entrance
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="ModelNet40")
    parser.add_argument("--dest", type=str, default=None)
    parser.add_argument("--workers", type=int, default=max(cpu_count() - 1, 1))
    parser.add_argument("--voxel-size", type=int, default=32)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--target-per-category", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    process_dataset(
        root_dir=args.src,
        output_dir=args.dest,
        num_workers=args.workers,
        voxel_size=args.voxel_size,
        augment_train=args.augment_train,
        target_per_category=args.target_per_category,
        seed=args.seed,
    )
