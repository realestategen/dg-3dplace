#!/usr/bin/env python3
"""
Inject vase.obj into 3D Gaussian Splatting scene using detected properties.
Uses YOLO detection results to properly scale, rotate, and place the vase.
"""

import numpy as np
import torch
import trimesh
from pathlib import Path
import json

def rotation_matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

print("=" * 80)
print("                  INJECT VASE INTO 3D SCENE")
print("=" * 80)

# ==================== CONFIGURATION ====================
# Input paths
CHECKPOINT_PATH = Path("/home/cse_g2/RealEstateGen/DG-3DPlace/room/output/my_scene/data/splatfacto/2026-02-02_124835/nerfstudio_models/step-000006999.ckpt")
MESH_PATH = Path("vase.obj")
PLACEMENT_3D = Path("3d_placement.json")  # Camera-based 3D mapping

# Output path - save in placement directory (easier permissions)
OUTPUT_CHECKPOINT = Path("scene_with_vase.ckpt")

# Vase appearance - VISIBLE but not huge
NUM_POINTS = 200000       # Much higher density for exact replica (4x increase)
GAUSSIAN_SCALE = 0.01     # Moderate Gaussian size
USE_MESH_COLORS = False   # Use red color for visibility
VASE_OPACITY = 1.0        # Full opacity

# These will be loaded from 3D placement
VASE_POSITION = None      # From camera unprojection
VASE_SCALE = None         # From 3D dimensions
VASE_ROTATION_INFO = None  # Rotation in 3D space

# ==================== LOAD 3D PLACEMENT DATA ====================
print(f"\n[STEP 0] Loading 3D Placement from Camera Unprojection")
print("-" * 80)

with open(PLACEMENT_3D, 'r') as f:
    placement = json.load(f)

# Extract 3D position from camera unprojection
VASE_POSITION = np.array([
    placement['position_3d']['x'],
    placement['position_3d']['y'],
    placement['position_3d']['z']
])

# Extract 3D dimensions
detected_height_3d = placement['scale_3d']['height']
detected_width_3d = placement['scale_3d']['width']

# Extract rotation information
VASE_ROTATION_INFO = placement['rotation']
yolo_rotation_deg = VASE_ROTATION_INFO['yolo_degrees']
cam_forward = np.array(VASE_ROTATION_INFO['camera_forward'])
cam_right = np.array(VASE_ROTATION_INFO['camera_right'])
cam_up = np.array(VASE_ROTATION_INFO['camera_up'])

print(f"✓ Loaded 3D placement (camera-based unprojection):")
print(f"  Position: [{VASE_POSITION[0]:.3f}, {VASE_POSITION[1]:.3f}, {VASE_POSITION[2]:.3f}]")
print(f"  3D Height: {detected_height_3d:.3f} m")
print(f"  3D Width: {detected_width_3d:.3f} m")
print(f"  Rotation: {yolo_rotation_deg:.2f}° (in camera image plane)")
print(f"  Camera forward: {cam_forward}")

print(f"\n[CONFIG]")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Mesh: {MESH_PATH}")
print(f"  Placement: {PLACEMENT_3D} (camera-based)")
print(f"  Output: {OUTPUT_CHECKPOINT}")
print(f"  Num Points: {NUM_POINTS}")

# ==================== VERIFY SCENE BOUNDS ====================
print(f"\n[STEP 1] Verifying Scene Bounds")
print("-" * 80)

# Load scene to verify bounds
checkpoint_temp = torch.load(str(CHECKPOINT_PATH), map_location='cpu', weights_only=False)
state_dict_temp = checkpoint_temp['pipeline']
if '_model.means' in state_dict_temp:
    scene_means_np = state_dict_temp['_model.means'].numpy()
else:
    scene_means_np = state_dict_temp['means'].numpy()

# Show scene bounds for reference
x_min, x_max = scene_means_np[:, 0].min(), scene_means_np[:, 0].max()
y_min, y_max = scene_means_np[:, 1].min(), scene_means_np[:, 1].max()
z_min, z_max = scene_means_np[:, 2].min(), scene_means_np[:, 2].max()

print(f"✓ Scene bounds:")
print(f"    X: [{x_min:.2f}, {x_max:.2f}]")
print(f"    Y: [{y_min:.2f}, {y_max:.2f}]")
print(f"    Z: [{z_min:.2f}, {z_max:.2f}]")
print(f"✓ Vase will be placed at: {VASE_POSITION}")
print(f"  (From camera-based 3D unprojection)")

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

# Center mesh
mesh.vertices -= mesh.vertices.mean(axis=0)

# Calculate scale - use 3D height from camera unprojection
mesh_height = mesh.bounds[1][2] - mesh.bounds[0][2]  # Z dimension
VASE_SCALE = detected_height_3d / mesh_height
print(f"✓ Mesh height in obj units: {mesh_height:.2f}")
print(f"✓ Target height from 3D unprojection: {detected_height_3d:.3f} m")
print(f"✓ Calculated scale factor: {VASE_SCALE:.6f}")

# Apply scale
mesh.vertices *= VASE_SCALE
print(f"✓ Scaled mesh to match detected size")
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
    # Use bright red for visibility
    print(f"✓ Using bright red color for visibility")
    colors = np.tile([1.0, 0.0, 0.0], (len(points), 1))  # Pure red

# Apply rotation based on camera orientation
print(f"\\n[STEP 3.5] Applying Rotation in Camera Space")
print("-" * 80)

# The YOLO rotation is in the image plane, which corresponds to rotation around the camera's forward axis
# We need to construct a rotation matrix that rotates around the camera forward vector

rotation_rad = np.radians(yolo_rotation_deg)

# Create rotation matrix around camera forward axis
# Using Rodrigues' rotation formula
k = cam_forward / np.linalg.norm(cam_forward)  # Unit vector
K = np.array([
    [0, -k[2], k[1]],
    [k[2], 0, -k[0]],
    [-k[1], k[0], 0]
])

rotation_matrix = np.eye(3) + np.sin(rotation_rad) * K + (1 - np.cos(rotation_rad)) * (K @ K)

points = points @ rotation_matrix.T
print(f"✓ Rotated points by {yolo_rotation_deg:.2f}° around camera forward axis")
print(f"  Camera forward: {cam_forward}")

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

# Rotations - convert rotation matrix to quaternions for each Gaussian
# We apply the same rotation to all Gaussians
rotation_quat = rotation_matrix_to_quaternion(rotation_matrix)
quats = torch.zeros((num_vase, 4), dtype=torch.float32)
quats[:, :] = torch.from_numpy(rotation_quat).float()

# Colors (spherical harmonics DC term) - USE ACTUAL MESH COLORS
# SH DC formula: (color - 0.5) / 0.28209479177387814
features_dc = torch.zeros((num_vase, 3), dtype=torch.float32)
colors_tensor = torch.from_numpy(colors).float()
features_dc[:] = (colors_tensor - 0.5) / 0.28209479177387814

# Higher-order SH coefficients (zeros for simplicity)
features_rest = torch.zeros((num_vase, 15, 3), dtype=torch.float32)

# Opacities (inverse sigmoid space)
opacity_value = VASE_OPACITY
if opacity_value >= 0.9999:
    opacity_logit = 10.0  # Very high value for near-1.0 opacity
else:
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
