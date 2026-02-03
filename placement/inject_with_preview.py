#!/usr/bin/env python3
"""
Inject vase with Open3D preview to verify placement before saving.
"""

import numpy as np
import torch
import trimesh
import open3d as o3d
from pathlib import Path

print("=" * 80)
print("          INJECT VASE WITH OPEN3D PREVIEW")
print("=" * 80)

# ==================== CONFIGURATION ====================
CHECKPOINT_PATH = Path("/home/cse_g2/RealEstateGen/DG-3DPlace/room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/step-000006999.ckpt")
MESH_PATH = Path("vase.obj")
OUTPUT_CHECKPOINT = Path("scene_with_vase.ckpt")

# Vase placement - will be calculated based on scene analysis
VASE_POSITION = None  # To be filled after analysis
VASE_SCALE = 0.15
NUM_POINTS = 50000
GAUSSIAN_SCALE = 0.008
VASE_COLOR = [0.8, 0.5, 0.3]  # Terracotta/clay color
VASE_OPACITY = 0.98

# ==================== LOAD AND ANALYZE SCENE ====================
print(f"\n[STEP 1] Loading and Analyzing Scene")
print("-" * 80)

checkpoint = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
state_dict = checkpoint['pipeline']

if '_model.means' in state_dict:
    key_prefix = '_model.'
else:
    key_prefix = ''

scene_means = state_dict[f'{key_prefix}means']
scene_means_np = scene_means.numpy()

print(f"✓ Loaded scene: {len(scene_means):,} Gaussians")
print(f"  Scene bounds:")
print(f"    X: [{scene_means_np[:, 0].min():.3f}, {scene_means_np[:, 0].max():.3f}]")
print(f"    Y: [{scene_means_np[:, 1].min():.3f}, {scene_means_np[:, 1].max():.3f}]")
print(f"    Z: [{scene_means_np[:, 2].min():.3f}, {scene_means_np[:, 2].max():.3f}]")

# Calculate good vase position - center of floor area
vase_x = scene_means_np[:, 0].mean()  # Center X
vase_y = np.percentile(scene_means_np[:, 1], 25)  # Forward area
vase_z = np.percentile(scene_means_np[:, 2], 5)  # Ground level

VASE_POSITION = np.array([vase_x, vase_y, vase_z])
print(f"\n✓ Calculated vase position: {VASE_POSITION}")

# ==================== LOAD VASE MESH ====================
print(f"\n[STEP 2] Loading Vase Mesh")
print("-" * 80)

mesh = trimesh.load_mesh(str(MESH_PATH))
print(f"✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Center and scale
mesh.vertices -= mesh.vertices.mean(axis=0)
mesh.vertices *= VASE_SCALE
print(f"✓ Scaled by {VASE_SCALE}x")

# Sample points
points, face_indices = trimesh.sample.sample_surface(mesh, NUM_POINTS)
points += VASE_POSITION
print(f"✓ Sampled {len(points)} points at position {VASE_POSITION}")

# ==================== OPEN3D PREVIEW ====================
print(f"\n[STEP 3] Creating Open3D Preview")
print("-" * 80)

# Create point clouds for visualization
# Scene point cloud (sample 10% for speed)
scene_sample_idx = np.random.choice(len(scene_means_np), size=min(50000, len(scene_means_np)), replace=False)
scene_pcd = o3d.geometry.PointCloud()
scene_pcd.points = o3d.utility.Vector3dVector(scene_means_np[scene_sample_idx])
scene_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray

# Vase point cloud
vase_pcd = o3d.geometry.PointCloud()
vase_pcd.points = o3d.utility.Vector3dVector(points)
vase_pcd.paint_uniform_color(VASE_COLOR)  # Terracotta

# Add coordinate frame for reference
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=VASE_POSITION)

print(f"✓ Created point clouds:")
print(f"  - Scene: {len(scene_pcd.points):,} points (gray)")
print(f"  - Vase: {len(vase_pcd.points):,} points (terracotta)")
print(f"\n[PREVIEW] Opening Open3D viewer...")
print("  - Gray points = original scene")
print("  - Terracotta points = vase")
print("  - RGB axes = vase position")
print("\nClose the window to continue with injection...")

o3d.visualization.draw_geometries(
    [scene_pcd, vase_pcd, coord_frame],
    window_name="Scene Preview - Close to Continue",
    width=1280,
    height=720,
    point_show_normal=False
)

# ==================== CREATE GAUSSIANS ====================
print(f"\n[STEP 4] Creating Gaussian Parameters")
print("-" * 80)

num_vase = len(points)
means = torch.from_numpy(points).float()

# Scales (log space) - slightly larger for better visibility
scales = torch.full((num_vase, 3), np.log(GAUSSIAN_SCALE), dtype=torch.float32)

# Rotations (identity quaternions)
quats = torch.zeros((num_vase, 4), dtype=torch.float32)
quats[:, 0] = 1.0

# Colors (spherical harmonics DC term)
features_dc = torch.zeros((num_vase, 3), dtype=torch.float32)
color_tensor = torch.tensor(VASE_COLOR, dtype=torch.float32)
features_dc[:] = (color_tensor - 0.5) / 0.28209479177387814

# Higher-order SH coefficients
scene_features_rest = state_dict.get(f'{key_prefix}features_rest', 
                                      torch.zeros((len(scene_means), 15, 3), dtype=torch.float32))
features_rest = torch.zeros((num_vase, 15, 3), dtype=torch.float32)

# Opacities (inverse sigmoid) - high opacity
opacity_logit = np.log(VASE_OPACITY / (1 - VASE_OPACITY))
opacities = torch.full((num_vase, 1), opacity_logit, dtype=torch.float32)

print(f"✓ Created {num_vase:,} Gaussians with:")
print(f"  - Scale: {GAUSSIAN_SCALE} (log={np.log(GAUSSIAN_SCALE):.4f})")
print(f"  - Color: RGB{VASE_COLOR}")
print(f"  - Opacity: {VASE_OPACITY}")

# ==================== MERGE ====================
print(f"\n[STEP 5] Merging with Scene")
print("-" * 80)

scene_scales = state_dict[f'{key_prefix}scales']
scene_quats = state_dict[f'{key_prefix}quats']
scene_features_dc = state_dict[f'{key_prefix}features_dc']
scene_opacities = state_dict[f'{key_prefix}opacities']

merged_means = torch.cat([scene_means, means], dim=0)
merged_scales = torch.cat([scene_scales, scales], dim=0)
merged_quats = torch.cat([scene_quats, quats], dim=0)
merged_features_dc = torch.cat([scene_features_dc, features_dc], dim=0)
merged_features_rest = torch.cat([scene_features_rest, features_rest], dim=0)
merged_opacities = torch.cat([scene_opacities, opacities], dim=0)

print(f"✓ Merged: {len(merged_means):,} total Gaussians")
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
print(f"\n[STEP 6] Saving Checkpoint")
print("-" * 80)

checkpoint['pipeline'] = state_dict
torch.save(checkpoint, str(OUTPUT_CHECKPOINT))
print(f"✓ Saved: {OUTPUT_CHECKPOINT} ({OUTPUT_CHECKPOINT.stat().st_size / 1024**2:.1f} MB)")

# Copy to nerfstudio directory
import shutil
dest = Path("/home/cse_g2/RealEstateGen/DG-3DPlace/room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/step-999999999.ckpt")
shutil.copy(OUTPUT_CHECKPOINT, dest)
print(f"✓ Copied to: {dest}")

print("\n" + "=" * 80)
print("✅ VASE INJECTION COMPLETE")
print("=" * 80)
print(f"\n📍 Vase position: {VASE_POSITION}")
print(f"📊 Total Gaussians: {len(merged_means):,}")
print(f"   - Scene: {len(scene_means):,}")
print(f"   - Vase: {num_vase:,}")
print("\n🎬 View with:")
print("cd /home/cse_g2/RealEstateGen/DG-3DPlace && \\")
print("sudo docker run --rm -it --gpus all \\")
print("  -v $(pwd)/room:/workspace -p 7007:7007 \\")
print("  nerfstudio/nerfstudio:latest \\")
print("  ns-viewer --load-config /workspace/output/my_scene/data/splatfacto/2026-02-02_124835/config.yml")
print("\nThen open: http://localhost:7007")
print("=" * 80)
