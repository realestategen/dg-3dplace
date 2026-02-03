# 3D Gaussian Splatting Object Injection - Complete Guide

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
   - [What is 3D Gaussian Splatting?](#what-is-3d-gaussian-splatting)
   - [Understanding Checkpoints (.ckpt)](#understanding-checkpoints-ckpt)
   - [How Rendering Works](#how-rendering-works)
3. [Camera Extraction & Rendering](#camera-extraction--rendering)
4. [Object to Gaussian Conversion](#object-to-gaussian-conversion)
5. [The Merge Process](#the-merge-process)
6. [Placement & Iteration](#placement--iteration)
7. [Code Walkthrough](#code-walkthrough)
8. [Practical Examples](#practical-examples)

---

## Introduction

This guide explains how to inject 3D objects (like a vase) into trained 3D Gaussian Splatting scenes. Unlike traditional 3D rendering where you add meshes, in Gaussian Splatting we represent everything as collections of 3D Gaussian distributions ("splats"). This document covers the complete pipeline from understanding the underlying data structures to iteratively placing objects with perfect visual fidelity.

**What we achieve:**
- Extract a trained 3D scene from a checkpoint
- Convert a 3D mesh (.obj) to Gaussian representation
- Merge the object into the scene without destroying structure
- Render and view the combined result in real-time
- Iterate on placement until perfect

---

## Theoretical Foundation

### What is 3D Gaussian Splatting?

3D Gaussian Splatting (3DGS) is a novel scene representation technique that represents 3D scenes as collections of 3D Gaussian distributions rather than meshes or voxels.

#### **Core Concept:**

Instead of triangles (traditional 3D), we use **millions of oriented 3D "blobs"** (Gaussians):

```
Traditional Mesh:          3D Gaussian Splatting:
    /\                         ●  ●
   /  \                       ● ○ ● ●
  /____\                     ● ○ ○ ○ ●
 Triangles                   Gaussian Blobs
```

#### **Each Gaussian is defined by:**

1. **Position (μ)** - 3D coordinates `[x, y, z]`
2. **Covariance (Σ)** - Represented as:
   - **Scale (s)**: Size in 3 directions `[sx, sy, sz]`
   - **Rotation (r)**: Quaternion `[w, x, y, z]`
   - Together: `Σ = R S Sᵀ Rᵀ` (rotation × scale × scaleᵀ × rotationᵀ)
3. **Color (c)** - Spherical Harmonics coefficients for view-dependent appearance
4. **Opacity (α)** - Transparency from 0 (invisible) to 1 (opaque)

#### **Mathematical Definition:**

A 3D Gaussian at position **μ** with covariance **Σ** is:

```
G(x) = exp(-½(x - μ)ᵀ Σ⁻¹ (x - μ))
```

Where:
- `x` = any point in 3D space
- `μ` = center position of Gaussian
- `Σ` = covariance matrix (defines shape/orientation)

#### **Why Gaussians?**

1. **Differentiable**: Can optimize with gradient descent
2. **Fast to render**: Efficient splatting algorithms exist
3. **Continuous**: Smooth interpolation between discrete points
4. **Compact**: Few parameters per Gaussian
5. **Real-time**: 60+ FPS rendering on modern GPUs

---

### Understanding Checkpoints (.ckpt)

A **checkpoint file** is a snapshot of the trained neural network's state - essentially a save file for your 3D scene.

#### **Structure of a .ckpt file:**

```python
checkpoint = {
    'pipeline': {
        # Model parameters (the actual 3D Gaussians)
        '_model.means': Tensor[N, 3],         # Positions
        '_model.scales': Tensor[N, 3],        # Sizes (log-space)
        '_model.quats': Tensor[N, 4],         # Rotations
        '_model.features_dc': Tensor[N, 3],   # Base color (SH order 0)
        '_model.features_rest': Tensor[N, 15, 3],  # Color details (SH order 1-3)
        '_model.opacities': Tensor[N, 1],     # Transparency (logit-space)
        
        # Additional model state
        'datamanager.train_camera_optimizer.pose_adjustment': ...,
        ...
    },
    'optimizers': {
        # Training state (we don't need this for inference)
        ...
    },
    'step': 6999,  # Training iteration when saved
    'scalars': {...}  # Training metrics
}
```

Where `N` = number of Gaussians (e.g., 343,694 for our room scene).

#### **Why Special Encodings?**

**1. Scales are in log-space:**
```python
# Stored value
log_scale = -5.5

# Actual scale
actual_scale = exp(-5.5) = 0.004 units

# Why? 
# - Scales must be positive (log ensures this)
# - Numerical stability during optimization
# - Easier gradient flow
```

**2. Opacities are in logit-space:**
```python
# Stored value (logit)
logit = 2.944

# Actual opacity (sigmoid)
opacity = 1 / (1 + exp(-2.944)) = 0.95

# Why?
# - Opacities must be in [0, 1]
# - Sigmoid function naturally bounds output
# - Better gradient behavior at extremes
```

**3. Colors use Spherical Harmonics:**
```python
# DC coefficient (order 0) - base color
features_dc = (RGB - 0.5) / 0.28209479177387814

# Why this constant?
# 0.28209479177387814 = sqrt(1 / (4π))
# This is the SH basis function for order 0

# Higher orders (features_rest) encode view-dependent effects
# Like how surfaces look different from different angles
```

#### **Memory Layout:**

For a scene with 343,694 Gaussians:

```
Parameter           Shape              Size (float32)
─────────────────────────────────────────────────────
means               [343694, 3]        4.1 MB
scales              [343694, 3]        4.1 MB
quats               [343694, 4]        5.5 MB
features_dc         [343694, 3]        4.1 MB
features_rest       [343694, 15, 3]    61.9 MB
opacities           [343694, 1]        1.4 MB
─────────────────────────────────────────────────────
TOTAL                                  81.1 MB (parameters only)
```

Full checkpoint: ~242 MB (includes optimizer state, metadata, etc.)

---

### How Rendering Works

#### **The Rendering Pipeline:**

```
1. Camera Setup
   ↓
2. Ray Generation (one per pixel)
   ↓
3. For each ray:
   ├─ Find Gaussians intersecting ray
   ├─ Sort by depth (back to front)
   └─ Alpha composite
   ↓
4. Output pixel color
```

#### **Detailed Breakdown:**

**Step 1: Project Gaussians to 2D**

For a Gaussian at 3D position **μ** with covariance **Σ**:

```python
# Transform to camera space
μ_cam = W2C @ μ  # World-to-camera transformation

# Project to image plane
u, v = K @ μ_cam  # K = camera intrinsics matrix
# u, v = pixel coordinates

# Project covariance (Jacobian of projection)
Σ_2D = J @ W2C @ Σ @ W2Cᵀ @ Jᵀ
```

Where:
- `W2C` = World-to-camera transformation matrix (from camera extrinsics)
- `K` = Camera intrinsics (focal length, principal point)
- `J` = Jacobian of perspective projection

**Step 2: Alpha Compositing (The Magic)**

For each pixel, we composite all Gaussians along the ray:

```python
def render_pixel(ray):
    # Find all Gaussians that might affect this pixel
    gaussians = find_nearby_gaussians(ray)
    
    # Sort by depth (furthest first)
    gaussians.sort(key=lambda g: depth(g))
    
    # Alpha compositing
    color = [0, 0, 0]
    alpha = 0
    
    for gaussian in gaussians:
        # Evaluate Gaussian at pixel location
        weight = gaussian.opacity * exp(-distance² / (2σ²))
        
        # Blend color (front-to-back compositing)
        color += weight * gaussian.color * (1 - alpha)
        alpha += weight * (1 - alpha)
        
        # Early termination
        if alpha > 0.99:
            break  # Fully opaque, no need to continue
    
    return color
```

This is similar to compositing in Photoshop - each Gaussian is a semi-transparent layer.

#### **Why This Is Fast:**

1. **Parallel**: Each pixel is independent → GPU parallelization
2. **Early termination**: Stop when pixel is opaque
3. **Tile-based**: Process pixels in groups for better cache locality
4. **Efficient sorting**: Use GPU-friendly radix sort

**Typical Performance:**
- RTX 4070 Ti: 60-120 FPS at 1080p
- 343,694 Gaussians processed in ~8-16ms per frame

---

## Camera Extraction & Rendering

### Understanding `render.py`

The `render.py` script extracts a single viewpoint from the trained scene. Let's break it down:

#### **Step 1: Load Configuration**

```python
def load_config_and_patch_paths(config_path, new_data_path):
    # Load YAML config
    config = yaml.load(config_path.open(), Loader=yaml.Loader)
    
    # The config contains:
    # - Pipeline settings (model type, hyperparameters)
    # - Data paths (where training images are)
    # - Camera information (intrinsics, extrinsics)
    
    # Patch data path (training data location may have changed)
    config.data = new_data_path
    config.pipeline.datamanager.data = new_data_path
```

**Why patch paths?**
Training might have happened in a different directory. We need to tell the system where the training images are now.

#### **Step 2: Initialize Pipeline**

```python
# Setup pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = config.pipeline.setup(device=device, test_mode="inference")

# Load checkpoint
ckpt_path = sorted(ckpt_dir.glob("*.ckpt"))[-1]  # Get latest
loaded_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
```

**What happens here:**
1. Create the model architecture (SplatfactoModel)
2. Load trained parameters from checkpoint
3. Move to GPU for fast rendering

#### **Step 3: Get Camera**

```python
# Get camera 0 from training set
camera = pipeline.datamanager.train_dataset.cameras[0]

# Camera object contains:
# - Intrinsics: focal length, principal point
# - Extrinsics: position, orientation in world
# - Image dimensions
```

**Camera Intrinsics:**
```python
fx, fy = camera.fx, camera.fy  # Focal lengths (in pixels)
cx, cy = camera.cx, camera.cy  # Principal point (image center offset)

# Intrinsics matrix K:
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

**Camera Extrinsics (c2w - camera-to-world):**
```python
c2w = camera.camera_to_worlds  # [3, 4] matrix

# Structure:
# [R | t] where R is 3×3 rotation, t is 3×1 translation
# 
# Camera position in world = t
# Camera direction = -R[:, 2] (negative z-axis)
```

#### **Step 4: Render**

```python
with torch.no_grad():
    # Generate rays for this camera
    camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)
    
    # Render (this calls the alpha compositing pipeline)
    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    
    # Extract RGB
    rendered_rgb = outputs["rgb"].cpu().numpy()  # [H, W, 3]
```

**What's a ray bundle?**
```python
RayBundle = {
    'origins': Tensor[H*W, 3],     # Ray start points (camera position)
    'directions': Tensor[H*W, 3],  # Ray directions
    'pixel_area': Tensor[H*W, 1],  # Pixel footprint
    'camera_indices': Tensor[H*W], # Which camera (for multi-cam)
    'nears': Tensor[H*W, 1],       # Near clipping plane
    'fars': Tensor[H*W, 1],        # Far clipping plane
}
```

Each ray corresponds to one pixel. For 960×540 image = 518,400 rays.

#### **Step 5: Save Outputs**

```python
# Save RGB image
img = Image.fromarray((rendered_rgb * 255).astype(np.uint8))
img.save("background.png")

# Save depth map
depth_map = outputs["depth"].cpu().numpy()
np.save("depth.npy", depth_map)

# Save camera metadata
camera_data = {
    'intrinsics': [...],
    'extrinsics': [...],
    'width': 960,
    'height': 540
}
json.dump(camera_data, open("camera_meta.json", "w"))
```

---

## Object to Gaussian Conversion

### The Challenge

**Question:** How do we convert a triangular mesh (.obj) to Gaussians without losing detail?

**Answer:** Surface sampling + per-point Gaussian initialization.

### The Process

#### **Step 1: Load the Mesh**

```python
import trimesh

mesh = trimesh.load_mesh("vase.obj")

# Mesh structure:
# - vertices: [5816, 3]  - 3D points
# - faces: [9456, 3]     - Triangle connectivity (indices into vertices)
# - visual:              - Colors/textures

# Example:
# vertices = [[x1, y1, z1],
#             [x2, y2, z2],
#             ...]
# faces = [[0, 1, 2],    - Triangle using vertices 0, 1, 2
#          [1, 3, 2],    - Triangle using vertices 1, 3, 2
#          ...]
```

#### **Step 2: Center and Scale**

```python
# Center the mesh at origin
center = mesh.vertices.mean(axis=0)
mesh.vertices -= center

# Scale to desired size
mesh.vertices *= VASE_SCALE  # e.g., 0.008
```

**Why center?** So we can easily position it anywhere in the scene later.

#### **Step 3: Surface Sampling**

This is the key step - we need to convert a mesh surface to a point cloud.

```python
NUM_POINTS = 50000
points, face_indices = trimesh.sample.sample_surface(mesh, NUM_POINTS)

# Returns:
# points: [50000, 3] - Random points ON the mesh surface
# face_indices: [50000] - Which face each point came from
```

**How does surface sampling work?**

```python
def sample_surface(mesh, num_points):
    # 1. Calculate area of each face
    areas = []
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        # Triangle area = 0.5 * ||(v1-v0) × (v2-v0)||
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas.append(area)
    
    # 2. Sample faces proportional to area
    #    (larger faces get more points)
    probabilities = areas / sum(areas)
    face_indices = np.random.choice(
        len(mesh.faces), 
        size=num_points, 
        p=probabilities
    )
    
    # 3. For each sampled face, pick random point on triangle
    points = []
    for face_idx in face_indices:
        v0, v1, v2 = mesh.vertices[mesh.faces[face_idx]]
        
        # Random barycentric coordinates
        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        
        # Point on triangle
        point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
        points.append(point)
    
    return np.array(points), face_indices
```

**Result:** Dense point cloud uniformly distributed on the mesh surface.

#### **Step 4: Extract Colors**

```python
# If mesh has per-vertex colors
if mesh.visual.kind == 'vertex':
    colors = []
    for face_idx in face_indices:
        face = mesh.faces[face_idx]
        # Get vertex colors for this face
        v_colors = mesh.visual.vertex_colors[face][:, :3]  # RGB only
        # Average (or use barycentric interpolation)
        colors.append(v_colors.mean(axis=0))
    colors = np.array(colors) / 255.0  # Normalize to [0, 1]

# If mesh has texture
elif mesh.visual.kind == 'texture':
    # Use texture coordinates to sample texture image
    # (more complex - involves UV mapping)
    colors = sample_texture(mesh, points, face_indices)
```

#### **Step 5: Create Gaussians**

Now we convert each point to a Gaussian:

```python
# Positions
means = torch.from_numpy(points).float()  # [50000, 3]

# Scales (all Gaussians same size)
GAUSSIAN_SCALE = 0.002
scales = torch.full((50000, 3), np.log(GAUSSIAN_SCALE))  # [50000, 3]
# Note: log-space, so actual size = exp(scales)

# Rotations (identity - no rotation)
quats = torch.zeros((50000, 4))
quats[:, 0] = 1.0  # w=1, x=y=z=0 → no rotation

# Colors (convert RGB to spherical harmonics)
features_dc = torch.zeros((50000, 3))
colors_tensor = torch.from_numpy(colors).float()
features_dc = (colors_tensor - 0.5) / 0.28209479177387814

# Higher-order SH (for view-dependent effects - start with zeros)
features_rest = torch.zeros((50000, 15, 3))

# Opacity (high - we want the vase to be opaque)
VASE_OPACITY = 0.95
opacity_logit = np.log(VASE_OPACITY / (1 - VASE_OPACITY))
opacities = torch.full((50000, 1), opacity_logit)
```

**Why this preserves structure:**

1. **Dense sampling**: 50,000 points capture fine details
2. **Tight Gaussians**: Small scale (0.002) means each Gaussian is tiny
3. **Overlapping coverage**: Multiple Gaussians per surface region
4. **Correct colors**: Inherited from original mesh

The result is a "Gaussian mesh" - the Gaussians are so dense and small that they form a continuous surface, just like the original mesh.

---

## The Merge Process

### Conceptual Overview

**Key insight:** Merging is just array concatenation!

In Gaussian Splatting, a "scene" is just a collection of Gaussians. Adding an object = adding more Gaussians.

```
Original Scene:    New Object:       Merged Scene:
[Gaussian 1]       [Gaussian A]      [Gaussian 1]
[Gaussian 2]       [Gaussian B]      [Gaussian 2]
[Gaussian 3]   +   [Gaussian C]  =   [Gaussian 3]
   ...                ...             [Gaussian A]
[Gaussian N]       [Gaussian M]      [Gaussian B]
                                     [Gaussian C]
                                        ...
                                     [Gaussian N]
                                     [Gaussian M]
```

### Step-by-Step Code Walkthrough

#### **1. Load Scene Checkpoint**

```python
checkpoint = torch.load("step-000006999.ckpt", map_location='cpu', weights_only=False)
state_dict = checkpoint['pipeline']

# Extract scene Gaussians
if '_model.means' in state_dict:
    key_prefix = '_model.'
else:
    key_prefix = ''

scene_means = state_dict[f'{key_prefix}means']           # [343694, 3]
scene_scales = state_dict[f'{key_prefix}scales']         # [343694, 3]
scene_quats = state_dict[f'{key_prefix}quats']           # [343694, 4]
scene_features_dc = state_dict[f'{key_prefix}features_dc']  # [343694, 3]
scene_features_rest = state_dict.get(f'{key_prefix}features_rest',
                                      torch.zeros((343694, 15, 3)))
scene_opacities = state_dict[f'{key_prefix}opacities']   # [343694, 1]
```

**Scene has 343,694 Gaussians** representing the room.

#### **2. Prepare Vase Gaussians**

From the previous section, we have:

```python
# Vase Gaussians (from mesh conversion)
vase_means = means              # [50000, 3]
vase_scales = scales            # [50000, 3]
vase_quats = quats              # [50000, 4]
vase_features_dc = features_dc  # [50000, 3]
vase_features_rest = features_rest  # [50000, 15, 3]
vase_opacities = opacities      # [50000, 1]
```

**Vase has 50,000 Gaussians** representing the object.

#### **3. Translate Vase to Target Position**

```python
# Calculate target position (e.g., floor center)
VASE_POSITION = np.array([0.70, 0.30, -0.77])

# Translate vase points
vase_means_positioned = vase_means + torch.tensor(VASE_POSITION).float()
```

**This moves all vase Gaussians from origin to the target location.**

#### **4. Concatenate (The Merge)**

```python
merged_means = torch.cat([scene_means, vase_means_positioned], dim=0)
merged_scales = torch.cat([scene_scales, vase_scales], dim=0)
merged_quats = torch.cat([scene_quats, vase_quats], dim=0)
merged_features_dc = torch.cat([scene_features_dc, vase_features_dc], dim=0)
merged_features_rest = torch.cat([scene_features_rest, vase_features_rest], dim=0)
merged_opacities = torch.cat([scene_opacities, vase_opacities], dim=0)

# Result shapes:
# merged_means: [393694, 3]  (343694 + 50000)
# merged_scales: [393694, 3]
# etc.
```

**That's it!** We now have a single set of Gaussians representing the combined scene.

#### **5. Update Checkpoint**

```python
# Replace old Gaussians with merged ones
state_dict[f'{key_prefix}means'] = merged_means
state_dict[f'{key_prefix}scales'] = merged_scales
state_dict[f'{key_prefix}quats'] = merged_quats
state_dict[f'{key_prefix}features_dc'] = merged_features_dc
state_dict[f'{key_prefix}features_rest'] = merged_features_rest
state_dict[f'{key_prefix}opacities'] = merged_opacities

# Save updated checkpoint
checkpoint['pipeline'] = state_dict
torch.save(checkpoint, "scene_with_vase.ckpt")
```

**We've created a new checkpoint with the merged scene!**

### Why This Works

**No spatial data structures needed:** Unlike meshes (which need spatial indexing, collision detection, etc.), Gaussians are independent. Order doesn't matter.

**No discontinuities:** Each Gaussian smoothly blends with neighbors via alpha compositing. No seams or artifacts.

**Preserves both structures:**
- Room Gaussians: Still in their original positions
- Vase Gaussians: Now positioned at target location
- No modification to existing Gaussians
- No interaction between old and new

**Rendering handles everything:** The renderer doesn't care which Gaussians came from where. It just composites all of them.

---

## Placement & Iteration

### Finding the Right Position

#### **Strategy 1: Analyze Scene Geometry**

```python
# Load scene to understand its structure
scene_means_np = scene_means.numpy()

# Find bounds
x_min, x_max = scene_means_np[:, 0].min(), scene_means_np[:, 0].max()
y_min, y_max = scene_means_np[:, 1].min(), scene_means_np[:, 1].max()
z_min, z_max = scene_means_np[:, 2].min(), scene_means_np[:, 2].max()

print(f"Scene spans: {x_max - x_min:.2f} × {y_max - y_min:.2f} × {z_max - z_min:.2f}")

# Find floor (lowest Z values)
floor_z = np.percentile(scene_means_np[:, 2], 5)  # 5th percentile

# Find center
center_x = scene_means_np[:, 0].mean()
center_y = scene_means_np[:, 1].mean()

# Place vase at floor center
VASE_POSITION = np.array([center_x, center_y, floor_z])
```

#### **Strategy 2: Use Depth Map**

If you have a depth map from a specific viewpoint:

```python
# Load depth and camera data
depth = np.load("depth.npy")  # [H, W]
camera_meta = json.load(open("camera_meta.json"))

# Find empty floor space (high depth values in bottom region)
bottom_region = depth[int(H*0.6):, :]  # Bottom 40% of image
floor_pixels = np.where(bottom_region > percentile(bottom_region, 80))

# Pick center of floor region
pixel_y, pixel_x = floor_pixels[0].mean(), floor_pixels[1].mean()

# Unproject to 3D
depth_value = depth[int(pixel_y), int(pixel_x)]
ray_direction = pixel_to_ray(pixel_x, pixel_y, camera_meta)
position_3d = camera_meta['position'] + depth_value * ray_direction

VASE_POSITION = position_3d
```

#### **Strategy 3: Interactive Placement**

```python
# Create visualization with Open3D
import open3d as o3d

# Create point cloud from scene
scene_pcd = o3d.geometry.PointCloud()
scene_pcd.points = o3d.utility.Vector3dVector(scene_means_np)
scene_pcd.paint_uniform_color([0.7, 0.7, 0.7])

# Create vase point cloud
vase_pcd = o3d.geometry.PointCloud()
vase_pcd.points = o3d.utility.Vector3dVector(vase_points)
vase_pcd.paint_uniform_color([0.8, 0.5, 0.3])

# Visualize and manually adjust
o3d.visualization.draw_geometries([scene_pcd, vase_pcd])

# User can rotate/pan to see if placement looks good
# Then extract final position
VASE_POSITION = vase_pcd.get_center()
```

### Iterative Refinement

Create a script to test multiple placements:

```python
# test_placements.py

placements = [
    {"name": "Center", "pos": [0.0, 0.0, floor_z]},
    {"name": "Left", "pos": [-1.0, 0.0, floor_z]},
    {"name": "Right", "pos": [1.0, 0.0, floor_z]},
    {"name": "Back", "pos": [0.0, -2.0, floor_z]},
]

for placement in placements:
    print(f"\nTesting: {placement['name']}")
    
    # Inject vase at this position
    inject_vase(position=placement['pos'], output=f"test_{placement['name']}.ckpt")
    
    # Render from same viewpoint
    render_scene(f"test_{placement['name']}.ckpt", output=f"render_{placement['name']}.png")
    
print("\nCompare render_*.png images to choose best placement")
```

### Automated Placement Validation

```python
def validate_placement(vase_position, scene_means, threshold=0.1):
    """
    Check if vase would intersect with scene geometry
    """
    # For each vase Gaussian, check distance to nearest scene Gaussian
    from scipy.spatial import cKDTree
    
    scene_tree = cKDTree(scene_means)
    
    for vase_point in vase_points + vase_position:
        distance, _ = scene_tree.query(vase_point)
        if distance < threshold:
            return False, f"Collision at {vase_point}"
    
    return True, "No collisions detected"

# Use it
valid, message = validate_placement(VASE_POSITION, scene_means_np)
if not valid:
    print(f"Warning: {message}")
    print("Try a different position")
```

---

## Code Walkthrough

### inject.py - Complete Breakdown

```python
#!/usr/bin/env python3
"""
Inject 3D object into Gaussian Splatting scene
"""

import numpy as np
import torch
import trimesh
from pathlib import Path

# ==================== CONFIGURATION ====================
CHECKPOINT_PATH = Path("step-000006999.ckpt")
MESH_PATH = Path("vase.obj")
OUTPUT_CHECKPOINT = Path("scene_with_vase.ckpt")

# Vase appearance
NUM_POINTS = 50000        # Number of Gaussians
VASE_SCALE = 0.008        # Mesh scale multiplier
GAUSSIAN_SCALE = 0.002    # Individual Gaussian size
USE_MESH_COLORS = True    # Extract colors from .obj
VASE_OPACITY = 0.95       # Opacity
OFFSET_FROM_CENTER = np.array([0.5, 0.0, 0.0])  # Placement offset

# ==================== STEP 1: ANALYZE SCENE ====================
print("[STEP 1] Analyzing Scene")

# Load scene checkpoint
checkpoint_temp = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
state_dict_temp = checkpoint_temp['pipeline']

# Get scene Gaussians
if '_model.means' in state_dict_temp:
    scene_means_np = state_dict_temp['_model.means'].numpy()
else:
    scene_means_np = state_dict_temp['means'].numpy()

# Calculate placement
floor_z = np.percentile(scene_means_np[:, 2], 5)  # Find floor
center_x = scene_means_np[:, 0].mean()             # Find center X
center_y = scene_means_np[:, 1].mean()             # Find center Y

VASE_POSITION = np.array([center_x, center_y, floor_z]) + OFFSET_FROM_CENTER
print(f"✓ Vase will be placed at: {VASE_POSITION}")

# ==================== STEP 2: LOAD MESH ====================
print("[STEP 2] Loading Mesh")

mesh = trimesh.load_mesh(str(MESH_PATH))
print(f"✓ Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Check for colors
has_vertex_colors = mesh.visual.kind == 'vertex'
has_texture = mesh.visual.kind == 'texture'

# Center and scale
mesh.vertices -= mesh.vertices.mean(axis=0)
mesh.vertices *= VASE_SCALE
print(f"✓ Scaled by {VASE_SCALE}x")

# ==================== STEP 3: SAMPLE SURFACE ====================
print("[STEP 3] Sampling Surface")

points, face_indices = trimesh.sample.sample_surface(mesh, NUM_POINTS)
print(f"✓ Sampled {len(points)} points")

# Extract colors
if USE_MESH_COLORS and (has_vertex_colors or has_texture):
    if has_vertex_colors:
        colors = []
        for face_idx in face_indices:
            face = mesh.faces[face_idx]
            v_colors = mesh.visual.vertex_colors[face][:, :3]
            colors.append(v_colors.mean(axis=0))
        colors = np.array(colors) / 255.0
    else:
        # Use texture material color
        colors = np.array([mesh.visual.material.main_color[:3] 
                          for _ in range(len(points))]) / 255.0
else:
    # Default color
    colors = np.tile([0.8, 0.6, 0.4], (len(points), 1))

print(f"✓ Extracted colors")

# Translate to target position
points += VASE_POSITION

# ==================== STEP 4: CREATE GAUSSIANS ====================
print("[STEP 4] Creating Gaussians")

num_vase = len(points)
means = torch.from_numpy(points).float()
scales = torch.full((num_vase, 3), np.log(GAUSSIAN_SCALE), dtype=torch.float32)
quats = torch.zeros((num_vase, 4), dtype=torch.float32)
quats[:, 0] = 1.0

# Colors (SH DC term)
features_dc = torch.zeros((num_vase, 3), dtype=torch.float32)
colors_tensor = torch.from_numpy(colors).float()
features_dc[:] = (colors_tensor - 0.5) / 0.28209479177387814

# SH rest (higher orders)
features_rest = torch.zeros((num_vase, 15, 3), dtype=torch.float32)

# Opacities
opacity_logit = np.log(VASE_OPACITY / (1 - VASE_OPACITY))
opacities = torch.full((num_vase, 1), opacity_logit, dtype=torch.float32)

print(f"✓ Created {num_vase} Gaussians")

# ==================== STEP 5: LOAD SCENE ====================
print("[STEP 5] Loading Scene")

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
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
                                      torch.zeros((len(scene_means), 15, 3)))
scene_opacities = state_dict[f'{key_prefix}opacities']

print(f"✓ Scene has {len(scene_means)} Gaussians")

# ==================== STEP 6: MERGE ====================
print("[STEP 6] Merging")

merged_means = torch.cat([scene_means, means], dim=0)
merged_scales = torch.cat([scene_scales, scales], dim=0)
merged_quats = torch.cat([scene_quats, quats], dim=0)
merged_features_dc = torch.cat([scene_features_dc, features_dc], dim=0)
merged_features_rest = torch.cat([scene_features_rest, features_rest], dim=0)
merged_opacities = torch.cat([scene_opacities, opacities], dim=0)

print(f"✓ Merged: {len(merged_means)} total Gaussians")

# Update state dict
state_dict[f'{key_prefix}means'] = merged_means
state_dict[f'{key_prefix}scales'] = merged_scales
state_dict[f'{key_prefix}quats'] = merged_quats
state_dict[f'{key_prefix}features_dc'] = merged_features_dc
state_dict[f'{key_prefix}features_rest'] = merged_features_rest
state_dict[f'{key_prefix}opacities'] = merged_opacities

# ==================== STEP 7: SAVE ====================
print("[STEP 7] Saving")

checkpoint['pipeline'] = state_dict
torch.save(checkpoint, OUTPUT_CHECKPOINT)

print(f"✓ Saved: {OUTPUT_CHECKPOINT}")
print("\n✅ Done!")
```

### Key Parameters to Adjust

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `NUM_POINTS` | Vase detail/density | 10,000 - 100,000 |
| `VASE_SCALE` | Overall vase size | 0.001 - 0.1 |
| `GAUSSIAN_SCALE` | Individual splat size | 0.001 - 0.01 |
| `VASE_OPACITY` | Transparency | 0.8 - 1.0 |
| `VASE_POSITION` | Location in scene | Depends on scene |

---

## Practical Examples

### Example 1: Place Vase on Table

```python
# Find table surface (assume it's at a known height)
table_z = -0.5  # meters

# Find table center
table_points = scene_means_np[
    (scene_means_np[:, 2] > table_z - 0.1) &
    (scene_means_np[:, 2] < table_z + 0.1)
]
table_center_x = table_points[:, 0].mean()
table_center_y = table_points[:, 1].mean()

VASE_POSITION = np.array([table_center_x, table_center_y, table_z + 0.1])
```

### Example 2: Create Array of Vases

```python
# Create grid of vases
for i in range(3):
    for j in range(3):
        position = np.array([i * 0.5, j * 0.5, floor_z])
        
        # Create vase at this position
        vase_gaussians = create_vase_gaussians(position)
        
        # Add to scene
        scene_gaussians = concatenate_gaussians(scene_gaussians, vase_gaussians)
```

### Example 3: Different Sizes and Colors

```python
vases = [
    {"scale": 0.008, "color": [1.0, 0.0, 0.0], "pos": [0, 0, floor_z]},    # Red, small
    {"scale": 0.012, "color": [0.0, 1.0, 0.0], "pos": [1, 0, floor_z]},    # Green, medium
    {"scale": 0.016, "color": [0.0, 0.0, 1.0], "pos": [2, 0, floor_z]},    # Blue, large
]

for vase_config in vases:
    # Load and scale mesh
    mesh = load_mesh("vase.obj")
    mesh.vertices *= vase_config["scale"]
    
    # Sample and colorize
    points, _ = sample_surface(mesh, 50000)
    colors = np.tile(vase_config["color"], (len(points), 1))
    
    # Create Gaussians
    vase_gaussians = create_gaussians(points + vase_config["pos"], colors)
    
    # Merge
    scene_gaussians = merge(scene_gaussians, vase_gaussians)
```

---

## Viewing & Validation

### Starting the Viewer

```bash
# Copy checkpoint to nerfstudio models directory
sudo cp scene_with_vase.ckpt \
    room/output/my_scene/data/splatfacto/*/nerfstudio_models/step-999999999.ckpt

# Start viewer
cd /path/to/DG-3DPlace
sudo docker run --rm -it --gpus all \
    -v $(pwd)/room:/workspace \
    -p 7007:7007 \
    nerfstudio/nerfstudio:latest \
    ns-viewer --load-config /workspace/output/my_scene/data/splatfacto/*/config.yml
```

### What the Viewer Does

1. **Loads checkpoint** → reads all Gaussian parameters
2. **Starts web server** → http://localhost:7007
3. **Renders in real-time** → as you move camera
4. **Updates instantly** → change rendering settings on the fly

### Viewer Controls

- **Mouse drag**: Rotate camera
- **Scroll**: Zoom in/out
- **Right-click drag**: Pan camera
- **Settings panel**: Adjust rendering quality, show/hide features

### Validation Checklist

✅ **Size Check**
```python
# Measure vase bounds after placement
vase_bounds = vase_points + VASE_POSITION
vase_height = vase_bounds[:, 2].max() - vase_bounds[:, 2].min()
print(f"Vase height: {vase_height:.3f} meters")

# Compare to scene
scene_height = scene_means_np[:, 2].max() - scene_means_np[:, 2].min()
print(f"Scene height: {scene_height:.3f} meters")
print(f"Vase is {vase_height/scene_height*100:.1f}% of scene height")
```

✅ **Position Check**
```python
# Verify vase is on floor
floor_z = np.percentile(scene_means_np[:, 2], 5)
vase_bottom_z = (vase_points + VASE_POSITION)[:, 2].min()
print(f"Floor Z: {floor_z:.3f}")
print(f"Vase bottom Z: {vase_bottom_z:.3f}")
print(f"Difference: {abs(vase_bottom_z - floor_z):.3f}")
```

✅ **Color Check**
```python
# Verify colors were extracted
print(f"Color range: [{colors.min():.3f}, {colors.max():.3f}]")
print(f"Mean color: {colors.mean(axis=0)}")
# Should be in [0, 1] range
```

---

## Advanced Topics

### Optimizing Merged Scene

After merging, you can optionally re-optimize:

```python
# Load merged checkpoint
checkpoint = torch.load("scene_with_vase.ckpt")

# Continue training for a few iterations
# This allows Gaussians to adjust to each other
# Useful if there are lighting mismatches

# Run training (pseudo-code)
optimizer = setup_optimizer(checkpoint)
for i in range(100):  # Just 100 iterations
    loss = render_loss(checkpoint)
    loss.backward()
    optimizer.step()

# Save refined checkpoint
torch.save(checkpoint, "scene_with_vase_refined.ckpt")
```

### Handling Lighting

Gaussian Splatting uses Spherical Harmonics for view-dependent appearance. When injecting objects:

```python
# Option 1: Match scene lighting (copy SH from nearby Gaussians)
nearby_scene_gaussians = find_nearest_k(VASE_POSITION, scene_means_np, k=100)
mean_sh_rest = scene_features_rest[nearby_scene_gaussians].mean(dim=0)
vase_features_rest = mean_sh_rest.unsqueeze(0).expand(num_vase, -1, -1)

# Option 2: Neutral lighting (zero higher-order SH)
vase_features_rest = torch.zeros((num_vase, 15, 3))  # Already doing this

# Option 3: Custom lighting direction
# (Advanced - requires understanding SH basis functions)
```

### Multi-Object Scenes

```python
# Merge multiple objects in one go
objects = [
    load_object("vase.obj", position=[0, 0, floor_z], scale=0.008),
    load_object("lamp.obj", position=[1, 1, table_z], scale=0.015),
    load_object("book.obj", position=[0.5, 0.5, table_z], scale=0.005),
]

# Concatenate all at once
all_means = torch.cat([scene_means] + [obj.means for obj in objects])
all_scales = torch.cat([scene_scales] + [obj.scales for obj in objects])
# ... etc for other parameters

# Single merged checkpoint
save_checkpoint("scene_with_objects.ckpt", all_means, all_scales, ...)
```

---

## Troubleshooting

### Vase Too Big/Small

**Problem:** Vase dominates scene or is invisible

**Solution:**
```python
# Check mesh bounds
print(f"Original mesh bounds: {mesh.bounds}")
print(f"Scene bounds: {scene_means_np.min(axis=0)} to {scene_means_np.max(axis=0)}")

# Calculate appropriate scale
scene_size = scene_means_np.max(axis=0) - scene_means_np.min(axis=0)
mesh_size = mesh.bounds[1] - mesh.bounds[0]
recommended_scale = (scene_size.mean() * 0.1) / mesh_size.mean()  # 10% of scene
print(f"Recommended scale: {recommended_scale:.5f}")
```

### Vase Not Visible

**Problem:** Checkpoint loads but vase doesn't appear

**Possible causes:**
1. **Opacity too low** → Increase `VASE_OPACITY` to 0.95+
2. **Gaussians too small** → Increase `GAUSSIAN_SCALE`
3. **Wrong position** → Check if vase is outside camera view
4. **Too few points** → Increase `NUM_POINTS` to 50,000+

**Debug:**
```python
# Print merged checkpoint stats
print(f"Total Gaussians: {len(merged_means)}")
print(f"Vase Gaussians: {num_vase}")
print(f"Vase position range: {(vase_points + VASE_POSITION).min(axis=0)} to {(vase_points + VASE_POSITION).max(axis=0)}")
print(f"Vase opacity (logit): {opacity_logit:.3f} → {1/(1+np.exp(-opacity_logit)):.3f}")
```

### Colors Don't Match

**Problem:** Vase appears gray or wrong color

**Check:**
```python
# Verify color extraction
print(f"Mesh has vertex colors: {mesh.visual.kind == 'vertex'}")
print(f"Mesh has texture: {mesh.visual.kind == 'texture'}")

# If texture exists
if mesh.visual.kind == 'texture':
    print(f"Material color: {mesh.visual.material.main_color}")
    
# Manually set color if needed
colors = np.tile([0.8, 0.5, 0.3], (len(points), 1))  # Terracotta
```

---

## Summary

**What We Learned:**

1. **Checkpoints** are saved states containing Gaussian parameters
2. **Rendering** composites Gaussians via alpha blending
3. **Camera extraction** gives us specific viewpoints
4. **Mesh → Gaussians** via surface sampling preserves structure
5. **Merging** is simple array concatenation
6. **Placement** can be calculated from scene analysis
7. **Iteration** allows refinement until perfect

**The Power of This Approach:**

- ✅ **Non-destructive**: Original scene unchanged
- ✅ **Flexible**: Easy to add/remove/modify objects
- ✅ **Fast**: No retraining needed
- ✅ **Realistic**: Proper lighting and integration
- ✅ **Scalable**: Add as many objects as needed

**Next Steps:**

1. Try different objects
2. Experiment with placement strategies
3. Create complex scenes with multiple objects
4. Optimize merged scenes for best quality

---

## References

- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Nerfstudio Documentation](https://docs.nerf.studio/)
- [Trimesh Documentation](https://trimsh.org/)
- [PyTorch Checkpoint Format](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

---

**Created:** February 3, 2026  
**Last Updated:** February 3, 2026  
**Author:** DG-3DPlace Team
