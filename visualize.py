import torch
import numpy as np
import os
import plotly.graph_objects as go
from model import PRSNet
from validation import validate_results

def _normalized(v, eps=1e-8):
    """
    Normalize a vector for reporting and drawing.
    """
    return v / (np.linalg.norm(v) + eps)

def _quaternion_to_axis_angle(q):
    """
    Convert unit quaternion into axes and angle
    """
    q = q / (np.linalg.norm(q) + 1e-8)
    axis = _normalized(q[1:])
    theta = 2.0 * np.arctan2(np.linalg.norm(q[1:]), q[0])
    return axis, theta

def _add_reflection_plane(fig, params, name, color, opacity):
    """
    Draw the reflection plane
    """
    n = params[:3].astype(float)
    d = float(params[3])
    norm = np.linalg.norm(n) + 1e-8
    n_hat = n / norm
    d_hat = d / norm

    # Generate 2D grid
    u = np.linspace(-15, 15, 10)
    v = np.linspace(-15, 15, 10)
    UU, VV = np.meshgrid(u, v)
    # v1 and v2 are two orthogonal vectors to define the plane
    if abs(n_hat[0]) > 0.1:
        v1 = np.array([-n_hat[1], n_hat[0], 0.0])
    else:
        v1 = np.array([0.0, -n_hat[2], n_hat[1]])
    v1 = _normalized(v1)
    v2 = np.cross(n_hat, v1)

    # Centralize the grid
    center = np.array([15.5, 15.5, 15.5])
    origin = center - (np.dot(center, n_hat) + d_hat) * n_hat
    X = origin[0] + UU * v1[0] + VV * v2[0]
    Y = origin[1] + UU * v1[1] + VV * v2[1]
    Z = origin[2] + UU * v1[2] + VV * v2[2]

    # Add the plane to the figure
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name
    ))

def _add_rotation_axis(fig, params, name, color, width):
    """
    Draw the rotation axis
    """
    axis, _ = _quaternion_to_axis_angle(params)
    p1 = 15.5 * np.ones(3) - 20 * axis
    p2 = 15.5 * np.ones(3) + 20 * axis

    # Add the axis line to the figure
    fig.add_trace(go.Scatter3d(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
        mode='lines',
        line=dict(color=color, width=width),
        name=name
    ))

def visualize_symmetries(data_path, model_path, output_html="visual_result.html"):
    """
    Run PRS-Net on one preprocessed sample and write to *.html files
    """
    log_path = os.path.splitext(output_html)[0] + ".log"
    output_dir = os.path.dirname(output_html)
    log_dir = os.path.dirname(log_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_file = open(log_path, "w", encoding="utf-8")

    def log(message):
        print(message)
        log_file.write(str(message) + "\n")
        log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = PRSNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    data = torch.load(data_path, weights_only=False)
    voxels = data["voxels"].float()
    surface_points = data.get("surface_points")
    voxel_input = voxels.unsqueeze(0).unsqueeze(0).to(device)
    
    # Get raw parameters from network
    with torch.no_grad():
        reflection_params, rotation_params = model(voxel_input)
        reflection_params = reflection_params[0].cpu().numpy()
        rotation_params = rotation_params[0].cpu().numpy()
    
    # Validation removes candidates that fail the checks
    log("Performing validation...")
    valid_planes, valid_axes, errors = validate_results(
        voxels.numpy(),
        reflection_params,
        rotation_params,
        return_errors=True,
        surface_points=surface_points.numpy() if surface_points is not None else None
    )
    
    log("-" * 30)
    log(f"Results for {data_path}:")
    log(f"Validated Planes: {len(valid_planes)} / 3")
    log(f"Validated Axes: {len(valid_axes)} / 3")
    log("Candidate Reflection Errors:")
    for i, err in enumerate(errors["reflection"]):
        log(f"  Plane {i}: {err:.6e}")
    log("Candidate Reflection Parameters:")
    for i, p in enumerate(reflection_params):
        n = _normalized(p[:3])
        log(
            f"  Plane {i}: "
            f"n=({n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f}), "
            f"d={p[3]:.6f}"
        )
    log("Candidate Rotation Errors:")
    for j, err in enumerate(errors["rotation"]):
        log(f"  Axis {j}: {err:.6e}")
    log("Candidate Rotation Diagnostics:")
    for j, diagnostic in enumerate(errors["rotation_diagnostics"]):
        failed_angle = diagnostic["first_failed_angle"]
        failed_angle_text = "None" if failed_angle is None else f"{failed_angle} deg"
        log(
            f"  Axis {j}: "
            f"passed={diagnostic['pass_count']}/{diagnostic['total_count']}, "
            f"mean={diagnostic['mean_error']:.6e}, "
            f"max={diagnostic['max_error']:.6e}, "
            f"first_failed_angle={failed_angle_text}"
        )
    log("Candidate Rotation Parameters:")
    for j, q in enumerate(rotation_params):
        axis, theta = _quaternion_to_axis_angle(q)
        log(
            f"  Axis {j}: "
            f"q=({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}), "
            f"axis=({axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}), "
            f"theta={np.degrees(theta):.3f} deg"
        )
    
    # Visualization starts with the raw occupied voxel points.
    indices = np.argwhere(voxels.numpy())
    fig = go.Figure()
    
    # Add Voxel Points
    fig.add_trace(go.Scatter3d(
        x=indices[:, 0], y=indices[:, 1], z=indices[:, 2],
        mode='markers',
        marker=dict(size=2, color='gray', opacity=0.3),
        name='Object Points'
    ))
    # Add Symmetry Center
    fig.add_trace(go.Scatter3d(
        x=[15.5], y=[15.5], z=[15.5],
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='Center (15.5)'
    ))

    # Draw only validated outputs
    # Add validated reflection planes only.
    for i, p_info in enumerate(valid_planes):
        _add_reflection_plane(
            fig,
            p_info['params'],
            f'Plane {i} valid (Err: {p_info["error"]:.2e})',
            'red',
            0.5
        )
    
    # Add validated rotation axes only.
    for j, a_info in enumerate(valid_axes):
        _add_rotation_axis(
            fig,
            a_info['params'],
            f'Axis {j} valid (Err: {a_info["error"]:.2e})',
            'green',
            10
        )
    
    # Set the layout for better visualization
    fig.update_layout(
        title=f"PRS-Net Validated Results: {os.path.basename(data_path)}",
        scene=dict(
            xaxis=dict(range=[0, 32]),
            yaxis=dict(range=[0, 32]),
            zaxis=dict(range=[0, 32]),
            aspectmode='cube'
        ),
        margin=dict(r=0, l=0, b=0, t=40)
    )

    fig.write_html(output_html, include_plotlyjs=True)
    log(f"Validated interactive results saved to: {output_html}")
    log(f"Validation log saved to: {log_path}")
    log("-" * 30)

    log_file.close()

# Visualization entrances
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .pt file")
    parser.add_argument("--model", type=str, default="prsnet_model.pth")
    parser.add_argument("--output", type=str, default="visual_result.html", help="Output HTML path")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        visualize_symmetries(args.data, args.model, args.output)
    else:
        print(f"Model {args.model} not found.")
