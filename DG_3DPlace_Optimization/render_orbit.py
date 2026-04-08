#!/usr/bin/env python3
"""
Gaussian Splatting Orbit Viewer for 3DGS Room Scene

This script uses the gsplat library to render a high-performance 
360-degree orbit around a 3DGS checkpoint and saves it to data/view/.
"""
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Render a 360 orbit of a 3DGS model.")
    parser.add_argument("checkpoint", nargs="?", default="room.ckpt", help="Path to the .ckpt file")
    parser.add_argument("--width", type=int, default=1280, help="Resolution width")
    parser.add_argument("--height", type=int, default=720, help="Resolution height")
    parser.add_argument("--fov", type=float, default=60.0, help="Field of view in degrees")
    parser.add_argument("--opacity-thresh", type=float, default=0.1, help="Filter out low opacity splats")
    parser.add_argument("--lookat", type=float, nargs=3, default=None, help="Look-at point x y z")
    
    # Arguments for the Orbit
    parser.add_argument("--num-frames", type=int, default=36, help="Number of images to generate for the orbit")
    parser.add_argument("--radius", type=float, default=3.0, help="Distance of the camera from the center")
    parser.add_argument("--height-offset", type=float, default=1.0, help="Height of the camera above the center")
    
    args = parser.parse_args()

    # 1. Load Data
    means, colors, opacities, scales, quats = load_gaussians(args.checkpoint)
    
    # Filter based on opacity threshold
    mask = opacities > args.opacity_thresh
    means = means[mask]
    colors = colors[mask]
    scales = scales[mask]
    quats = quats[mask]
    opacities = opacities[mask]

    # 2. Pre-load to GPU once for maximum performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Transferring {len(means)} splats to {device}...")
    
    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    quats_t = torch.tensor(quats, dtype=torch.float32, device=device)
    colors_t = torch.tensor(colors, dtype=torch.float32, device=device)
    opacities_t = torch.tensor(opacities, dtype=torch.float32, device=device)

    # 3. Setup Orbit Parameters and Hardcoded Directory
    center = means.mean(axis=0) if args.lookat is None else np.array(args.lookat)
    
    out_dir = "data/view_output/"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting render loop for {args.num_frames} frames...")
    print(f"Outputs will be saved strictly to: {out_dir}")

    # 4. Render Loop
    for i in range(args.num_frames):
        # Calculate angle for current frame
        angle = (i / args.num_frames) * 2 * np.pi
        
        # Calculate new camera position on the circle
        cam_x = center[0] + args.radius * np.cos(angle)
        cam_y = center[1] + args.radius * np.sin(angle)
        cam_z = center[2] + args.height_offset
        cam_pos = np.array([cam_x, cam_y, cam_z])
        
        # Compute camera rotation matrices
        forward = (center - cam_pos)
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_vec = np.cross(right, forward)
        up_vec = up_vec / np.linalg.norm(up_vec)
        rot_matrix = np.column_stack([right, up_vec, -forward])
        
        # Camera-to-world
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot_matrix
        c2w[:3, 3] = cam_pos
        
        # World-to-camera (OpenCV convention)
        w2c = np.linalg.inv(c2w)
        w2c[1, :] *= -1
        w2c[2, :] *= -1

        # Build Camera Intrinsics
        fov_rad = np.radians(args.fov)
        fy = (args.height / 2) / np.tan(fov_rad / 2)
        fx = fy
        cx = args.width / 2.0
        cy = args.height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        # Move matrices to GPU
        viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)
        K_t = torch.tensor(K, dtype=torch.float32, device=device)
        
        # Execute Rasterization
        renders, alphas, meta = rasterization(
            means=means_t,
            quats=quats_t / quats_t.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_t),
            opacities=opacities_t,
            colors=colors_t,
            viewmats=viewmat.unsqueeze(0),
            Ks=K_t.unsqueeze(0),
            width=args.width,
            height=args.height,
            sh_degree=None,
            backgrounds=torch.ones(1, 3, device=device),
        )
        
        # Save output image
        rgb = renders[0].cpu().numpy()
        out_filename = os.path.join(out_dir, f"frame_{i:03d}.png")
        plt.imsave(out_filename, np.clip(rgb, 0, 1))
        print(f"Saved {out_filename}")
        
    print("Orbit rendering complete!")

if __name__ == "__main__":
    main()