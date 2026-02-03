#!/usr/bin/env python3
"""
Simple script to inject vase.obj into the 3D Gaussian Splatting scene.
Places vase at a fixed location and saves modified checkpoint for viewing.
"""

import numpy as np
import torch
import trimesh
from pathlib import Path

print("=" * 80)
print("                  INJECT VASE INTO 3D SCENE")
print("=" * 80)

# ==================== CONFIGURATION ====================
# Input paths
CHECKPOINT_PATH = Path("/home/cse_g2/RealEstateGen/DG-3DPlace/room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/step-000006999.ckpt")
MESH_PATH = Path("vase.obj")

# Output path - save in placement directory (easier permissions)
OUTPUT_CHECKPOINT = Path("scene_with_vase.ckpt")

# Vase placement - offset from scene center, on the floor
VASE_POSITION = None  # Will be calculated from scene
OFFSET_FROM_CENTER = np.array([0.5, 0.0, 0.0])  # Slightly to the side

# Vase appearance - EXACT REPLICA
NUM_POINTS = 50000        # High density for detail
VASE_SCALE = 0.008        # Much smaller scale
GAUSSIAN_SCALE = 0.002    # Small tight Gaussians for detail
USE_MESH_COLORS = True    # Extract actual colors from .obj
VASE_OPACITY = 0.95       # Opacity

print(f"\n[CONFIG]")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Mesh: {MESH_PATH}")
print(f"  Output: {OUTPUT_CHECKPOINT}")
print(f"  Vase Position: {VASE_POSITION}")
print(f"  Vase Scale: {VASE_SCALE}")
print(f"  Num Points: {NUM_POINTS}")

# ==================== ANALYZE SCENE ====================
print(f"\n[STEP 1] Analyzing Scene for Placement")
print("-" * 80)

# Load scene first to calculate proper placement
checkpoint_temp = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
state_dict_temp = checkpoint_temp['pipeline']
if '_model.means' in state_dict_temp:
    scene_means_np = state_dict_temp['_model.means'].numpy()
else:
    scene_means_np = state_dict_temp['means'].numpy()

# Calculate floor position (low Z percentile) and centered XY
floor_z = np.percentile(scene_means_np[:, 2], 5)
center_x = scene_means_np[:, 0].mean()
center_y = scene_means_np[:, 1].mean()

VASE_POSITION = np.array([center_x, center_y, floor_z]) + OFFSET_FROM_CENTER
print(f"✓ Scene bounds:")
print(f"    X: [{scene_means_np[:, 0].min():.2f}, {scene_means_np[:, 0].max():.2f}]")
print(f"    Y: [{scene_means_np[:, 1].min():.2f}, {scene_means_np[:, 1].max():.2f}]")
print(f"    Z: [{scene_means_np[:, 2].min():.2f}, {scene_means_np[:, 2].max():.2f}]")
print(f"✓ Vase will be placed at: {VASE_POSITION}")
print(f"  (Floor level, slightly offset from center)")

# ==================== LOAD MESH ====================
print(f"\n[STEP 2] Loading Vase Mesh with Colors")
print("-" * 80)

mesh = trimesh.load_mesh(str(MESH_PATH))
print(f"✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Check if mesh has colors
has_vertex_colors = mesh.visual.kind == 'vertex'
has_texture = mesh.visual.kind == 'texture'
print(f"  Vertex colors: {has_vertex_colors}")
print(f"  Texture: {has_texture}")

# Center and scale mesh
mesh.vertices -= mesh.vertices.mean(axis=0)
mesh.vertices *= VASE_SCALE
print(f"✓ Centered and scaled by {VASE_SCALE}x")
print(f"  Mesh bounds: {mesh.bounds}")

# ==================== SAMPLE POINTS ====================
print(f"\n[STEP 3] Sampling Points from Mesh Surface")
print("-" * 80)

points, face_indices = trimesh.sample.sample_surface(mesh, NUM_POINTS)
print(f"✓ Sampled {len(points)} points from mesh surface")

# Extract colors at sampled points
if USE_MESH_COLORS and (has_vertex_colors or has_texture):
    print(f"✓ Extracting colors from mesh...")
    
    # Get barycentric coordinates for interpolation
    if has_vertex_colors:
        # Interpolate vertex colors
        colors = []
        for face_idx in face_indices:
            face = mesh.faces[face_idx]
            # Get vertex colors for this face
            v_colors = mesh.visual.vertex_colors[face][:, :3]  # RGB only
            # Average color of the face (simple approach)
            colors.append(v_colors.mean(axis=0))
        colors = np.array(colors) / 255.0  # Normalize to [0, 1]
    else:
        # Use texture colors if available
        colors = np.array([mesh.visual.material.main_color[:3] for _ in range(len(points))]) / 255.0
    
    print(f"✓ Extracted colors for {len(colors)} points")
    print(f"  Color range: [{colors.min():.3f}, {colors.max():.3f}]")
else:
    # Fallback: use a neutral terracotta/clay color
    print(f"⚠ No colors in mesh, using default terracotta color")
    colors = np.tile([0.8, 0.6, 0.4], (len(points), 1))

# Translate points to target position
points += VASE_POSITION
print(f"✓ Translated to position: {VASE_POSITION}")
print(f"  Final point cloud bounds:")
print(f"    Min: {points.min(axis=0)}")
print(f"    Max: {points.max(axis=0)}")

# ==================== CREATE GAUSSIANS ====================
print(f"\n[STEP 4] Initializing Gaussian Parameters")
print("-" * 80)

num_vase = len(points)

# Convert points to torch tensor
means = torch.from_numpy(points).float()

# Scales (log space) - small for detail
scales = torch.full((num_vase, 3), np.log(GAUSSIAN_SCALE), dtype=torch.float32)

# Rotations (identity quaternions: w=1, x=y=z=0)
quats = torch.zeros((num_vase, 4), dtype=torch.float32)
quats[:, 0] = 1.0

# Colors (spherical harmonics DC term) - USE ACTUAL MESH COLORS
# SH DC formula: (color - 0.5) / 0.28209479177387814
features_dc = torch.zeros((num_vase, 3), dtype=torch.float32)
colors_tensor = torch.from_numpy(colors).float()
features_dc[:] = (colors_tensor - 0.5) / 0.28209479177387814

# Higher-order SH coefficients (zeros for simplicity)
features_rest = torch.zeros((num_vase, 15, 3), dtype=torch.float32)

# Opacities (inverse sigmoid space)
opacity_value = VASE_OPACITY
opacity_logit = np.log(opacity_value / (1 - opacity_value))
opacities = torch.full((num_vase, 1), opacity_logit, dtype=torch.float32)

print(f"✓ Created {num_vase} Gaussians:")
print(f"  - Means: {means.shape}")
print(f"  - Scales: {scales.shape} (log value: {np.log(GAUSSIAN_SCALE):.4f})")
print(f"  - Quats: {quats.shape}")
print(f"  - Features DC: {features_dc.shape}")
print(f"  - Features Rest: {features_rest.shape}")
print(f"  - Opacities: {opacities.shape} (logit value: {opacity_logit:.4f})")
print(f"  - Colors: From mesh ({colors.shape[0]} unique colors)")
print(f"  - Opacity: {VASE_OPACITY}")

# ==================== LOAD CHECKPOINT ====================
print(f"\n[STEP 5] Loading Scene Checkpoint")
print("-" * 80)

checkpoint = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
state_dict = checkpoint['pipeline']

# Detect key format and extract scene Gaussians
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

print(f"✓ Loaded scene checkpoint")
print(f"  Scene Gaussians: {len(scene_means):,}")
print(f"  - means: {scene_means.shape}")
print(f"  - scales: {scene_scales.shape}")
print(f"  - quats: {scene_quats.shape}")
print(f"  - features_dc: {scene_features_dc.shape}")
print(f"  - features_rest: {scene_features_rest.shape}")
print(f"  - opacities: {scene_opacities.shape}")

# ==================== MERGE ====================
print(f"\n[STEP 6] Merging Vase with Scene")
print("-" * 80)

# Concatenate all parameters
merged_means = torch.cat([scene_means, means], dim=0)
merged_scales = torch.cat([scene_scales, scales], dim=0)
merged_quats = torch.cat([scene_quats, quats], dim=0)
merged_features_dc = torch.cat([scene_features_dc, features_dc], dim=0)
merged_features_rest = torch.cat([scene_features_rest, features_rest], dim=0)
merged_opacities = torch.cat([scene_opacities, opacities], dim=0)

print(f"✓ Merged Gaussians: {len(merged_means):,}")
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
print(f"\n[STEP 6] Saving Modified Checkpoint")
print("-" * 80)

checkpoint['pipeline'] = state_dict
torch.save(checkpoint, str(OUTPUT_CHECKPOINT))

file_size_mb = OUTPUT_CHECKPOINT.stat().st_size / (1024 * 1024)
print(f"✓ Saved: {OUTPUT_CHECKPOINT}")
print(f"  Size: {file_size_mb:.1f} MB")

# ==================== INSTRUCTIONS ====================
print("\n" + "=" * 80)
print("✅ VASE INJECTION COMPLETE!")
print("=" * 80)
print(f"\n📦 Output checkpoint: {OUTPUT_CHECKPOINT.name}")
print(f"🎯 Vase placed at: {VASE_POSITION}")
print(f"📊 Total Gaussians: {len(merged_means):,}")
print(f"   - Scene: {len(scene_means):,}")
print(f"   - Vase: {num_vase:,}")

print("\n" + "=" * 80)
print("🎬 HOW TO VIEW THE RESULT:")
print("=" * 80)
print("\n1. First, rename the checkpoint to a step number format:\n")
print("sudo mv room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/scene_with_vase.ckpt \\")
print("       room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/step-999999999.ckpt")
print("\n2. Then run the viewer from the DG-3DPlace directory:\n")
print("sudo docker run --rm -it \\")
print("  --gpus all \\")
print("  -v $(pwd)/room:/workspace \\")
print("  -p 7007:7007 \\")
print("  nerfstudio/nerfstudio:latest \\")
print("  ns-viewer --load-config /workspace/output/my_scene/data/splatfacto/2026-02-02_124835/config.yml")
print("\n3. Open browser: http://localhost:7007")
print("\nNote: The viewer will automatically load the latest checkpoint (step-999999999.ckpt with vase)")
print("=" * 80)
