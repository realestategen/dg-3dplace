#!/usr/bin/env python3
"""
Gaussian Splatting Viewer for 3DGS Room Scene

This script uses the gsplat library to render a true Gaussian splatted view of a 3DGS checkpoint.
"""
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import gsplat
from gsplat import rasterization

def load_gaussians(ckpt_path: str):
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"] if "pipeline" in ckpt else ckpt
    means = state["_model.means"].numpy()
    features_dc = state["_model.features_dc"].numpy()
    if features_dc.ndim == 3:
        features_dc = features_dc.squeeze(1)
    C0 = 0.28209479177387814
    colors = C0 * features_dc + 0.5
    colors = np.clip(colors, 0, 1)
    opacities = state["_model.opacities"].numpy()
    opacities = 1 / (1 + np.exp(-opacities))
    opacities = opacities.squeeze()
    scales = state["_model.scales"].numpy()
    quats = state["_model.quats"].numpy()
    return means, colors, opacities, scales, quats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="room.ckpt")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--output", type=str, default="gsplat_render.png")
    parser.add_argument("--opacity-thresh", type=float, default=0.1)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=None, help="Camera position x y z")
    parser.add_argument("--lookat", type=float, nargs=3, default=None, help="Look-at point x y z")
    args = parser.parse_args()

    means, colors, opacities, scales, quats = load_gaussians(args.checkpoint)
    mask = opacities > args.opacity_thresh
    means = means[mask]
    colors = colors[mask]
    scales = scales[mask]
    quats = quats[mask]
    opacities = opacities[mask]

    # Camera setup
    if args.camera_pos is not None and args.lookat is not None:
        cam_pos = np.array(args.camera_pos)
        lookat = np.array(args.lookat)
    else:
        # Default: orbit around center
        center = means.mean(axis=0)
        cam_pos = center + np.array([0, -2, 0.5])
        lookat = center
    up = np.array([0, 0, 1])
    fov_rad = np.radians(args.fov)
    width, height = args.width, args.height

    # Compute camera rotation (OpenGL convention)
    forward = (lookat - cam_pos)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_vec = np.cross(right, forward)
    up_vec = up_vec / np.linalg.norm(up_vec)
    rot_matrix = np.column_stack([right, up_vec, -forward])
    from scipy.spatial.transform import Rotation as R
    rot_obj = R.from_matrix(rot_matrix)
    quat_xyzw = rot_obj.as_quat()
    wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # Build camera intrinsics
    fy = (height / 2) / np.tan(fov_rad / 2)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Camera-to-world
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot_matrix
    c2w[:3, 3] = cam_pos
    # World-to-camera (OpenCV)
    w2c = np.linalg.inv(c2w)
    w2c[1, :] *= -1
    w2c[2, :] *= -1

    # Render with gsplat
    print("Rendering with gsplat...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    quats_t = torch.tensor(quats, dtype=torch.float32, device=device)
    colors_t = torch.tensor(colors, dtype=torch.float32, device=device)
    opacities_t = torch.tensor(opacities, dtype=torch.float32, device=device)
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)
    K_t = torch.tensor(K, dtype=torch.float32, device=device)
    renders, alphas, meta = rasterization(
        means=means_t,
        quats=quats_t / quats_t.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales_t),
        opacities=opacities_t,
        colors=colors_t,
        viewmats=viewmat.unsqueeze(0),
        Ks=K_t.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=None,
        backgrounds=torch.ones(1, 3, device=device),
    )
    rgb = renders[0].cpu().numpy()
    plt.imsave(args.output, np.clip(rgb, 0, 1))
    print(f"Saved gsplat render to {args.output}")
    plt.figure(figsize=(width/100, height/100))
    plt.imshow(np.clip(rgb, 0, 1))
    plt.axis('off')
    plt.title('Gaussian Splatting Render')
    plt.show()

if __name__ == "__main__":
    main()
