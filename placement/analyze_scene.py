#!/usr/bin/env python3
"""
Analyze the 3D scene to find good placement coordinates for the vase.
"""

import torch
import numpy as np
import json
from pathlib import Path

print("=" * 80)
print("                  ANALYZE 3D SCENE FOR PLACEMENT")
print("=" * 80)

# Load checkpoint
CHECKPOINT_PATH = Path("/home/cse_g2/RealEstateGen/DG-3DPlace/room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/step-000006999.ckpt")
CAMERA_META_PATH = Path("camera_meta.json")

print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
state_dict = checkpoint['pipeline']

# Get scene Gaussians
if '_model.means' in state_dict:
    means = state_dict['_model.means'].numpy()
else:
    means = state_dict['means'].numpy()

print(f"\n[SCENE ANALYSIS]")
print("-" * 80)
print(f"Total Gaussians: {len(means):,}")
print(f"\nScene bounds:")
print(f"  X: [{means[:, 0].min():.3f}, {means[:, 0].max():.3f}]")
print(f"  Y: [{means[:, 1].min():.3f}, {means[:, 1].max():.3f}]")
print(f"  Z: [{means[:, 2].min():.3f}, {means[:, 2].max():.3f}]")
print(f"\nScene center: [{means.mean(axis=0)}]")
print(f"Scene size: {means.max(axis=0) - means.min(axis=0)}")

# Analyze density in different regions
center = means.mean(axis=0)
distances = np.linalg.norm(means - center, axis=1)
print(f"\nDistance from center:")
print(f"  Min: {distances.min():.3f}")
print(f"  Max: {distances.max():.3f}")
print(f"  Mean: {distances.mean():.3f}")
print(f"  Median: {np.median(distances):.3f}")

# Suggest placement coordinates
# Place vase at ground level (low Z), in front area (negative Y), centered X
suggested_x = means[:, 0].mean()  # Center X
suggested_y = np.percentile(means[:, 1], 30)  # Forward area
suggested_z = np.percentile(means[:, 2], 10)  # Near ground

print(f"\n[SUGGESTED VASE PLACEMENT]")
print("-" * 80)
print(f"Position: [{suggested_x:.3f}, {suggested_y:.3f}, {suggested_z:.3f}]")
print(f"  (Center X, Forward 30%, Ground level)")

# Load camera data if available
if CAMERA_META_PATH.exists():
    with open(CAMERA_META_PATH) as f:
        camera_data = json.load(f)
    
    print(f"\n[CAMERA INFORMATION]")
    print("-" * 80)
    print(f"Camera position: {camera_data['camera_position']}")
    print(f"Camera direction: {camera_data['camera_direction']}")
    
    # Calculate distance from camera to suggested position
    cam_pos = np.array(camera_data['camera_position'])
    suggested_pos = np.array([suggested_x, suggested_y, suggested_z])
    distance_to_cam = np.linalg.norm(suggested_pos - cam_pos)
    print(f"Distance from camera to suggested position: {distance_to_cam:.3f} meters")

print("\n" + "=" * 80)
print("Recommended settings for inject.py:")
print("=" * 80)
print(f"VASE_POSITION = np.array([{suggested_x:.3f}, {suggested_y:.3f}, {suggested_z:.3f}])")
print(f"VASE_SCALE = 0.15  # Adjust based on desired size")
print(f"NUM_POINTS = 50000  # More points for better visibility")
print("=" * 80)
