#!/usr/bin/env python3
"""
Place vase directly in front of camera for guaranteed visibility.
"""

import numpy as np
import torch
import trimesh
import json
from pathlib import Path

print("=" * 80)
print("     INJECT VASE DIRECTLY IN CAMERA VIEW (TEST)")
print("=" * 80)

# Paths
CHECKPOINT_PATH = Path("/home/cse_g2/RealEstateGen/DG-3DPlace/room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/step-000006999.ckpt")
MESH_PATH = Path("vase.obj")
CAMERA_META_PATH = Path("camera_meta.json")
OUTPUT_CHECKPOINT = Path("scene_with_vase.ckpt")

# Place vase at scene center (most visible location)
print("\n[STEP 1] Calculating Vase Position")
print("-" * 80)

# Load scene first to find center
checkpoint_temp = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
state_dict_temp = checkpoint_temp['pipeline']
if '_model.means' in state_dict_temp:
    scene_means_temp = state_dict_temp['_model.means'].numpy()
else:
    scene_means_temp = state_dict_temp['means'].numpy()

# Place at scene center (most visible)
scene_center = scene_means_temp.mean(axis=0)
VASE_POSITION = scene_center.copy()

print(f"Scene center: {scene_center}")
print(f"✓ Placing vase at scene center for maximum visibility")
VASE_SCALE = 0.2  # Large size
NUM_POINTS = 100000  # Lots of points
GAUSSIAN_SCALE = 0.01  # Large Gaussians
VASE_COLOR = [1.0, 0.0, 0.0]  # BRIGHT RED - impossible to miss
VASE_OPACITY = 1.0  # Full opacity

print(f"\n✓ Vase will be placed at: {VASE_POSITION}")
print(f"  (At scene center for maximum visibility)")
print(f"  Color: BRIGHT RED for visibility test")
print(f"  Points: {NUM_POINTS:,}")

# ==================== LOAD MESH ====================
print(f"\n[STEP 2] Loading Vase Mesh")
print("-" * 80)

mesh = trimesh.load_mesh(str(MESH_PATH))
print(f"✓ Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

mesh.vertices -= mesh.vertices.mean(axis=0)
mesh.vertices *= VASE_SCALE
print(f"✓ Scaled by {VASE_SCALE}x")

# Sample points
points, _ = trimesh.sample.sample_surface(mesh, NUM_POINTS)
points += VASE_POSITION
print(f"✓ Sampled {len(points):,} points")

# ==================== CREATE GAUSSIANS ====================
print(f"\n[STEP 3] Creating Gaussians")
print("-" * 80)

num_vase = len(points)
means = torch.from_numpy(points).float()

# Large scales for visibility
scales = torch.full((num_vase, 3), np.log(GAUSSIAN_SCALE), dtype=torch.float32)

# Identity rotations
quats = torch.zeros((num_vase, 4), dtype=torch.float32)
quats[:, 0] = 1.0

# BRIGHT RED color
features_dc = torch.zeros((num_vase, 3), dtype=torch.float32)
color_tensor = torch.tensor(VASE_COLOR, dtype=torch.float32)
features_dc[:] = (color_tensor - 0.5) / 0.28209479177387814

# SH rest
features_rest = torch.zeros((num_vase, 15, 3), dtype=torch.float32)

# Full opacity
opacity_logit = 10.0  # Very high = near 100% opacity
opacities = torch.full((num_vase, 1), opacity_logit, dtype=torch.float32)

print(f"✓ Created {num_vase:,} Gaussians")
print(f"  - Scale: {GAUSSIAN_SCALE}")
print(f"  - Color: BRIGHT RED {VASE_COLOR}")
print(f"  - Opacity: ~100%")

# ==================== LOAD SCENE ====================
print(f"\n[STEP 4] Loading Scene")
print("-" * 80)

checkpoint = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
state_dict = checkpoint['pipeline']

if '_model.means' in state_dict:
    key_prefix = '_model.'
else:
    key_prefix = ''

scene_means = state_dict[f'{key_prefix}means']
scene_scales = state_dict[f'{key_prefix}scales']
scene_quats = state_dict[f'{key_prefix}quats']
scene_features_dc = state_dict[f'{key_prefix}features_dc']
scene_features_rest = state_dict.get(f'{key_prefix}features_rest', 
                                      torch.zeros((len(scene_means), 15, 3), dtype=torch.float32))
scene_opacities = state_dict[f'{key_prefix}opacities']

print(f"✓ Scene: {len(scene_means):,} Gaussians")

# ==================== MERGE ====================
print(f"\n[STEP 5] Merging")
print("-" * 80)

merged_means = torch.cat([scene_means, means], dim=0)
merged_scales = torch.cat([scene_scales, scales], dim=0)
merged_quats = torch.cat([scene_quats, quats], dim=0)
merged_features_dc = torch.cat([scene_features_dc, features_dc], dim=0)
merged_features_rest = torch.cat([scene_features_rest, features_rest], dim=0)
merged_opacities = torch.cat([scene_opacities, opacities], dim=0)

print(f"✓ Total: {len(merged_means):,} Gaussians")
print(f"  - Scene: {len(scene_means):,}")
print(f"  - Vase: {num_vase:,}")

# Update state dict
state_dict[f'{key_prefix}means'] = merged_means
state_dict[f'{key_prefix}scales'] = merged_scales
state_dict[f'{key_prefix}quats'] = merged_quats
state_dict[f'{key_prefix}features_dc'] = merged_features_dc
state_dict[f'{key_prefix}features_rest'] = merged_features_rest
state_dict[f'{key_prefix}opacities'] = merged_opacities

# ==================== SAVE ====================
print(f"\n[STEP 6] Saving")
print("-" * 80)

checkpoint['pipeline'] = state_dict
torch.save(checkpoint, str(OUTPUT_CHECKPOINT))
print(f"✓ Saved: {OUTPUT_CHECKPOINT}")

# Copy to nerfstudio
import subprocess
subprocess.run([
    'sudo', 'cp', 
    str(OUTPUT_CHECKPOINT),
    '/home/cse_g2/RealEstateGen/DG-3DPlace/room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/step-999999999.ckpt'
], check=True)
print(f"✓ Copied to nerfstudio models directory")

print("\n" + "=" * 80)
print("✅ DONE - BRIGHT RED VASE IN CAMERA VIEW")
print("=" * 80)
print(f"\n🎯 Vase position: {VASE_POSITION}")
print(f"📍 At scene center (most visible location)")
print(f"🔴 BRIGHT RED color - should be impossible to miss!")
print(f"📊 {len(merged_means):,} total Gaussians ({num_vase:,} for vase)")

print("\n🎬 Restart viewer:")
print("cd /home/cse_g2/RealEstateGen/DG-3DPlace && \\")
print("sudo docker run --rm -it --gpus all \\")
print("  -v $(pwd)/room:/workspace -p 7007:7007 \\")
print("  nerfstudio/nerfstudio:latest \\")
print("  ns-viewer --load-config /workspace/output/my_scene/data/splatfacto/2026-02-02_124835/config.yml")
print("\nOpen: http://localhost:7007")
print("=" * 80)
