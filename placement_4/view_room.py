#!/usr/bin/env python3
"""
View 3DGS Room Scene with Viser

Simple viewer to visualize room.ckpt using viser point cloud.
"""

import torch
import numpy as np
import viser
import argparse
import time


def load_gaussians(ckpt_path: str):
    """Load Gaussian positions and colors from checkpoint."""
    print(f"Loading: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Get pipeline state
    if "pipeline" in ckpt:
        state = ckpt["pipeline"]
    else:
        state = ckpt
    
    # Extract positions
    means = state["_model.means"].numpy()
    print(f"  Gaussians: {len(means):,}")
    
    # Extract colors from SH DC coefficients
    features_dc = state["_model.features_dc"].numpy()
    if features_dc.ndim == 3:
        features_dc = features_dc.squeeze(1)
    
    # SH to RGB: color = C0 * sh + 0.5
    C0 = 0.28209479177387814
    colors = C0 * features_dc + 0.5
    colors = np.clip(colors, 0, 1)
    
    # Extract opacities for filtering
    opacities = state["_model.opacities"].numpy()
    opacities = 1 / (1 + np.exp(-opacities))  # sigmoid
    opacities = opacities.squeeze()
    
    print(f"  Position range: {means.min(axis=0)} to {means.max(axis=0)}")
    
    return means, colors, opacities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="room.ckpt")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--opacity-thresh", type=float, default=0.1)
    args = parser.parse_args()
    
    # Load data
    means, colors, opacities = load_gaussians(args.checkpoint)
    
    # Filter by opacity
    mask = opacities > args.opacity_thresh
    means = means[mask]
    colors = colors[mask]
    print(f"  After opacity filter: {len(means):,} points")
    
    # Subsample if needed
    if len(means) > args.max_points:
        idx = np.random.choice(len(means), args.max_points, replace=False)
        means = means[idx]
        colors = colors[idx]
        print(f"  Subsampled to: {len(means):,} points")
    
    # Convert colors to uint8
    colors_u8 = (colors * 255).astype(np.uint8)
    
    # Start viser server
    print(f"\nStarting viser on port {args.port}...")
    server = viser.ViserServer(port=args.port)
    
    # Add point cloud
    print("Adding point cloud...")
    server.scene.add_point_cloud(
        name="/room",
        points=means.astype(np.float32),
        colors=colors_u8,
        point_size=0.01,
    )
    
    # Add axes at origin
    server.scene.add_frame(
        name="/origin",
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        axes_length=1.0,
        axes_radius=0.02,
    )
    
    # Scene info
    center = means.mean(axis=0)
    print(f"\n✓ Viewer ready at http://localhost:{args.port}")
    print(f"  Center: {center}")
    print(f"  Points: {len(means):,}")
    print("\nPress Ctrl+C to stop")
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        server.stop()


if __name__ == "__main__":
    main()
