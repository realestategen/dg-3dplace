# 3D Gaussian Splatting with Nerfstudio - Complete Guide

## � Quick Start (3 Steps)

### Prerequisites
Ensure Docker and NVIDIA Container Toolkit are installed (see [detailed setup](#quick-start) below if needed).

### Step 1: Extract Frames from Video
```bash
# Create output directory with proper permissions
mkdir -p room/data
chmod -R 777 room/data

# Extract frames at 2 FPS from your video (with GPU support for COLMAP)
sudo docker run --rm \
  --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-process-data video \
  --data /workspace/data/room_01.mp4 \
  --output-dir /workspace/data \
  --num-frames-target 200
```
This creates `room/data/images/` with extracted frames and runs COLMAP reconstruction.

### Step 2: Train 3D Gaussian Splatting Model
```bash
# Create output directory with proper permissions
mkdir -p room/output/my_scene
chmod -R 777 room/output/my_scene

# Train for 7000 iterations (~15 minutes on RTX 4070 Ti)
sudo docker run --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-train splatfacto \
  --data /workspace/data \
  --output-dir /workspace/output/my_scene \
  --max-num-iterations 7000 \
  colmap
```
Output saved to `room/output/my_scene/data/splatfacto/[timestamp]/`

### Step 3: View the 3D Scene
```bash
# Start interactive viewer at http://localhost:7007
sudo docker run --rm -it \
  --gpus all \
  -v $(pwd)/room:/workspace \
  -p 7007:7007 \
  nerfstudio/nerfstudio:latest \
  ns-viewer --load-config /workspace/output/my_scene/data/splatfacto/2026-01-03_223700/config.yml
```
Open your browser to **http://localhost:7007** to interact with the 3D scene.

**That's it!** Continue reading for detailed explanations, export options, and troubleshooting.

---

## �📋 Table of Contents
- [What is 3D Gaussian Splatting?](#what-is-3d-gaussian-splatting)
- [Why Use Docker + Nerfstudio?](#why-use-docker--nerfstudio)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Processing New Videos](#processing-new-videos)
- [Understanding Outputs](#understanding-outputs)
- [Exporting for 3D Software](#exporting-for-3d-software)
- [Troubleshooting](#troubleshooting)

---

## 📁 Project Structure

```
DG-3DPlace/
├── room/
│   ├── data/                    # Input data (images/video) - git ignored
│   │   ├── images/              # Extracted frames or photos
│   │   └── sparse/              # COLMAP reconstruction (auto-generated)
│   ├── output/                  # Training outputs - git ignored
│   │   └── docker_gsplat/       # Checkpoints, configs (~275MB per run)
│   └── scripts/
│       └── extract_frames.py    # Helper to extract video frames
├── README.md                    # This file - complete documentation
└── .gitignore                   # Git ignore patterns
```

**What's ignored in git?**  
- Input videos (`.mp4`, `.avi`, etc.) - Too large for version control
- Image directories (`room/data/images/`) - Can be hundreds of frames
- Training outputs (`room/output/`) - Large checkpoint files (~275MB+)
- COLMAP files (`.bin`, `sparse/`) - Generated artifacts

Only scripts and documentation are version controlled.

**What happened to `gaussian-splatting/` folder?**  
It has been removed. The original 3D Gaussian Splatting repository required CUDA 11.8 and doesn't work with modern GPUs (RTX 40-series). All functionality is now provided through Docker + Nerfstudio with no local compilation needed.

---

## 🎯 What is 3D Gaussian Splatting?

**3D Gaussian Splatting (3DGS)** is a cutting-edge technique for creating photorealistic 3D scenes from 2D images or videos.

### How it Works:
1. **Input**: Multiple photos/video frames of a scene from different angles
2. **COLMAP**: Reconstructs camera positions and creates a sparse 3D point cloud
3. **Training**: Optimizes millions of 3D Gaussian "splats" to represent the scene
4. **Output**: Real-time renderable 3D scene with photorealistic quality

### Each Gaussian Splat Contains:
- **Position** (x, y, z) - Location in 3D space
- **Color** (RGB/SH) - Appearance from different viewing angles
- **Opacity** (α) - Transparency level
- **Scale** (sx, sy, sz) - Size in 3 dimensions
- **Rotation** (quaternion) - Orientation

### Advantages:
✅ **Real-time rendering** (60+ FPS on modern GPUs)  
✅ **Photorealistic quality** comparable to NeRF  
✅ **Fast training** (minutes vs hours for NeRF)  
✅ **Editable** scene representation  

---

## 🐳 Why Use Docker + Nerfstudio?

### The Problem: CUDA Compatibility Hell

The original 3D Gaussian Splatting implementation (`gaussian-splatting/`) has strict requirements:
- **CUDA 11.8** specifically
- **Compute Capability ≤ 8.6** (RTX 30-series and older)

**Our GPU**: NVIDIA RTX 4070 Ti SUPER has **Compute Capability 8.9**, requiring **CUDA 12+**.

### Failed Attempts:
1. ❌ **Original 3DGS + CUDA 11.8** → `__cudaGetKernel` symbol errors (GPU too new)
2. ❌ **Original 3DGS + CUDA 12** → Half-precision compilation failures
3. ❌ **Native gsplat installation** → Missing CUDA headers (thrust, cub, nv/target)

### The Solution: Docker + Nerfstudio

**Nerfstudio** is a modern framework that includes:
- **gsplat**: Modern CUDA 12-compatible Gaussian Splatting implementation
- **Splatfacto**: Optimized training pipeline
- **Pre-built Docker images** with all dependencies
- **Built-in viewer** for visualization
- **Export tools** for various formats

#### Why Docker?
✅ **Pre-compiled binaries** - No compilation headaches  
✅ **Isolated environment** - No conda conflicts  
✅ **GPU passthrough** - Full CUDA 12 support  
✅ **Reproducible** - Same environment for entire team  
✅ **Easy updates** - Just pull new images  

---

## 🚀 Quick Start

### Prerequisites
```bash
# 1. Install Docker
sudo apt update
sudo apt install -y docker.io

# 2. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. Pull Nerfstudio Docker image
sudo docker pull nerfstudio/nerfstudio:latest

# 4. Test GPU access
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Training a Scene (Existing Data)
```bash
# Your data structure should be:
# room/data/
#   ├── images/          # Your photos/frames
#   └── sparse/0/        # COLMAP reconstruction (or will be created)

cd /home/cse_g2/RealEstateGen/DG-3DPlace

# Run training
sudo docker run --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-train splatfacto \
  --data /workspace/data \
  --output-dir /workspace/output/my_scene \
  --max-num-iterations 7000 \
  colmap \
  --colmap-path sparse/0 \
  --images-path images \
  --downscale-factor 1
```

### View the Result
```bash
# Start viewer (with port forwarding)
sudo docker run --rm -it \
  --gpus all \
  -v $(pwd)/room:/workspace \
  -p 7007:7007 \
  nerfstudio/nerfstudio:latest \
  ns-viewer --load-config /workspace/output/my_scene/data/splatfacto/*/config.yml

# Open browser: http://localhost:7007
```

---

## 📹 Processing New Videos

### Method 1: Extract Frames + COLMAP (Recommended)

```bash
# 1. Create project directory
mkdir -p room/new_scene/data/images
cd room/new_scene

# 2. Extract frames from video
sudo docker run --rm \
  -v $(pwd):/workspace \
  nerfstudio/nerfstudio:latest \
  ns-process-data video \
  --data /workspace/input_video.mp4 \
  --output-dir /workspace/data

# This creates:
#   data/
#     ├── images/           # Extracted frames
#     ├── colmap/           # COLMAP reconstruction
#     └── transforms.json   # Camera parameters

# 3. Train on the processed data
sudo docker run --gpus all \
  -v $(pwd):/workspace \
  nerfstudio/nerfstudio:latest \
  ns-train splatfacto \
  --data /workspace/data \
  --output-dir /workspace/output \
  --max-num-iterations 7000 \
  colmap
```

### Method 2: Manual Frame Extraction

```bash
# Use ffmpeg to extract frames
ffmpeg -i input_video.mp4 -qscale:v 1 -qmin 1 -vf fps=2 room/data/images/frame_%04d.jpg

# Then run COLMAP + training
sudo docker run --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-train splatfacto \
  --data /workspace/data \
  --output-dir /workspace/output/my_scene \
  --max-num-iterations 7000 \
  colmap \
  --images-path images \
  --downscale-factor 1
```

### Using the Helper Script

We provide `room/scripts/extract_frames.py` for convenience:

```bash
# Extract frames at 2 FPS
python room/scripts/extract_frames.py \
  --video input_video.mp4 \
  --output room/data/images \
  --fps 2

# Or use the existing script (if available)
```

### Recommended Video Settings:
- **Resolution**: 1080p or higher
- **Frame Rate**: Extract 1-3 frames per second
- **Coverage**: 360° coverage of the scene if possible
- **Lighting**: Consistent lighting throughout
- **Minimum Frames**: 50-100 for small scenes, 150-300 for rooms

---

## 📦 Understanding Outputs

### Directory Structure After Training

```
room/output/my_scene/
└── data/
    └── splatfacto/
        └── 2026-01-03_204011/          # Timestamp of training run
            ├── config.yml               # Training configuration + camera data
            ├── dataparser_transforms.json  # Camera transforms
            └── nerfstudio_models/
                └── step-000006999.ckpt  # Model checkpoint (275MB)
```

### File Types Explained

#### 1. **Checkpoint Files** (`.ckpt`)
- **What**: PyTorch model containing all trained Gaussian parameters
- **Size**: ~275MB for 7000 iterations
- **Contains**: 
  - Positions, colors, scales, rotations, opacities of all Gaussians
  - Optimizer state
  - Training metrics
- **Use**: Load for viewing, rendering, continued training, or export

#### 2. **Config File** (`config.yml`)
- **What**: Complete training configuration
- **Contains**:
  - Model hyperparameters
  - Camera intrinsics/extrinsics
  - Data paths
  - Training settings
- **Use**: Required to load the checkpoint for any operation

#### 3. **Transform Files** (`transforms.json`, `dataparser_transforms.json`)
- **What**: Camera pose information from COLMAP
- **Contains**:
  - Camera position and orientation for each image
  - Intrinsic parameters (focal length, principal point)
  - Image file paths
- **Use**: Understanding scene structure, camera trajectories

### What Can You Do With These Files?

1. **View Interactively**: Load in ns-viewer (as shown above)
2. **Render Images**: Generate novel views from any camera angle
3. **Export**: Convert to standard 3D formats (.ply, .obj, .splat)
4. **Continue Training**: Resume from checkpoint with more iterations
5. **Edit Scene**: Modify Gaussian properties programmatically

---

## 🎨 Exporting for 3D Software

Your trained model is currently in Nerfstudio's native format. Here's how to export it for use in standard 3D software like Blender, Unity, or MeshLab.

### Export Options

#### 1. Point Cloud Export (.ply) - **Best for Blender**

```bash
# Export as colored point cloud
sudo docker run --rm \
  --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-export pointcloud \
  --load-config /workspace/output/docker_gsplat/data/splatfacto/2026-01-03_204011/config.yml \
  --output-dir /workspace/output/exports/pointcloud \
  --num-points 1000000 \
  --remove-outliers True \
  --normal-method open3d

# Output: pointcloud.ply (can open in Blender, MeshLab, CloudCompare)
```

**Import to Blender:**
1. `File > Import > Stanford (.ply)`
2. Select `pointcloud.ply`
3. Adjust point size in viewport shading settings
4. Can convert to mesh using modifiers

#### 2. Gaussian Splat Export (.ply) - **For Web Viewers**

```bash
# Export native Gaussian splat format
sudo docker run --rm \
  --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-export gaussian-splat \
  --load-config /workspace/output/docker_gsplat/data/splatfacto/2026-01-03_204011/config.yml \
  --output-dir /workspace/output/exports/splat

# Output: splat.ply (for web-based 3D Gaussian Splatting viewers)
```

**View Online:**
- Upload to https://antimatter15.com/splat/
- Or use https://github.com/antimatter15/splat for local viewing

#### 3. Mesh Export (.obj) - **For Game Engines**

```bash
# Export as textured mesh using Poisson reconstruction
sudo docker run --rm \
  --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-export poisson \
  --load-config /workspace/output/docker_gsplat/data/splatfacto/2026-01-03_204011/config.yml \
  --output-dir /workspace/output/exports/mesh \
  --num-points 1000000 \
  --depth 10

# Output: mesh.obj + texture files
```

**Import to Unity/Unreal:**
1. Import the `.obj` file
2. Import associated texture maps
3. Create material with textures
4. Apply to mesh

#### 4. Video Rendering

```bash
# Render a camera path video
sudo docker run --rm \
  --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-render camera-path \
  --load-config /workspace/output/docker_gsplat/data/splatfacto/2026-01-03_204011/config.yml \
  --camera-path-filename /workspace/camera_path.json \
  --output-path /workspace/output/exports/video/render.mp4

# Or render interpolated path between training cameras
sudo docker run --rm \
  --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-render interpolate \
  --load-config /workspace/output/docker_gsplat/data/splatfacto/2026-01-03_204011/config.yml \
  --output-path /workspace/output/exports/video/interpolated.mp4
```

### Workflow for Object Insertion/Manipulation

Based on your need to **insert and manipulate 3D objects**, here's the recommended workflow:

#### Option A: Blender Pipeline (Recommended)

```bash
# 1. Export high-density point cloud
sudo docker run --rm --gpus all \
  -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-export pointcloud \
  --load-config /workspace/output/docker_gsplat/data/splatfacto/2026-01-03_204011/config.yml \
  --output-dir /workspace/output/exports/for_blender \
  --num-points 5000000 \
  --remove-outliers True

# 2. Open in Blender
# - Import the .ply file
# - Add 3D objects (furniture, decorations, etc.)
# - Position and scale objects
# - Export combined scene

# 3. (Optional) Re-render with new objects
# - Take screenshots from Blender
# - Re-train Gaussian Splatting with new images
```

#### Option B: Programmatic Editing (Advanced)

For programmatic insertion of objects, you can:
1. Load the checkpoint in Python
2. Add new Gaussians at desired positions
3. Re-optimize local regions
4. Save modified checkpoint

Example script location: `room/scripts/edit_gaussians.py` (to be created)

### Supported 3D Software

| Software | Format | Import Method | Best For |
|----------|--------|---------------|----------|
| **Blender** | .ply, .obj | File > Import | Full 3D editing, animation, rendering |
| **MeshLab** | .ply | File > Import Mesh | Point cloud processing, mesh cleaning |
| **CloudCompare** | .ply | File > Open | Point cloud analysis, measurements |
| **Unity** | .obj, .fbx | Drag & drop | Game development, VR/AR |
| **Unreal Engine** | .obj, .fbx | Import | Game development, architectural viz |
| **3ds Max** | .obj | File > Import | Professional 3D modeling |
| **Maya** | .obj | File > Import | Animation, VFX |

---

## 🔧 Troubleshooting

### Viewer Not Loading
```bash
# Make sure port 7007 is forwarded:
sudo docker run --rm -it --gpus all \
  -v $(pwd)/room:/workspace \
  -p 7007:7007 \  # <-- This line is critical!
  nerfstudio/nerfstudio:latest \
  ns-viewer --load-config /workspace/output/.../config.yml
```

### Out of Memory During Training
```bash
# Reduce image resolution
--downscale-factor 2  # or 4 for very large images

# Reduce batch size (edit config or use)
--pipeline.model.num-random-samples 8192  # default is higher
```

### COLMAP Fails
```bash
# Try with fewer features
--colmap.num-keypoints 8192  # default is higher

# Or use pre-processed transforms
ns-train splatfacto --data /workspace/data nerfstudio-data
```

### Docker Permission Issues
```bash
# Fix output directory permissions
mkdir -p room/output/my_scene
chmod -R 777 room/output/my_scene

# Or run with user mapping
sudo docker run --user $(id -u):$(id -g) ...
```

---

## 📚 Additional Resources

### Official Documentation
- **Nerfstudio**: https://docs.nerf.studio/
- **3D Gaussian Splatting Paper**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **gsplat**: https://github.com/nerfstudio-project/gsplat

### Useful Tools
- **COLMAP**: https://colmap.github.io/
- **Blender**: https://www.blender.org/
- **Gaussian Splat Viewer**: https://antimatter15.com/splat/

### Helper Scripts in This Project

| Script | Location | Purpose |
|--------|----------|---------|
| `extract_frames.py` | `room/scripts/` | Extract frames from video (optional - can use Docker) |

**Note**: All common commands are documented in this README. See the [Quick Reference Commands](#quick-reference-commands) section below.

---

## 🎯 Quick Reference Commands

```bash
# Train new scene
sudo docker run --gpus all -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-train splatfacto --data /workspace/data --output-dir /workspace/output/scene \
  --max-num-iterations 7000 colmap

# View trained scene
sudo docker run --rm -it --gpus all -v $(pwd)/room:/workspace -p 7007:7007 \
  nerfstudio/nerfstudio:latest \
  ns-viewer --load-config /workspace/output/scene/data/splatfacto/*/config.yml

# Export to point cloud
sudo docker run --rm --gpus all -v $(pwd)/room:/workspace \
  nerfstudio/nerfstudio:latest \
  ns-export pointcloud --load-config /workspace/output/scene/data/splatfacto/*/config.yml \
  --output-dir /workspace/output/exports/pointcloud

# Process video
sudo docker run --rm -v $(pwd):/workspace \
  nerfstudio/nerfstudio:latest \
  ns-process-data video --data /workspace/video.mp4 --output-dir /workspace/data
```

---

## 👥 Team Notes

### Current Setup
- **GPU**: NVIDIA RTX 4070 Ti SUPER (Compute 8.9, CUDA 12 required)
- **Method**: Nerfstudio splatfacto (via Docker)
- **Training Time**: ~15 minutes for 7000 iterations
- **Output Size**: ~275MB checkpoint

### Successful Training Example
- **Input**: 180 frames from room video (1074x1910 resolution)
- **COLMAP**: 108,931 sparse points
- **Training**: 7000 iterations
- **Output**: `room/output/docker_gsplat/data/splatfacto/2026-01-03_204011/`

### Why Previous Methods Failed
1. ❌ **Original 3DGS repo** (`gaussian-splatting/`) - Requires CUDA 11.8, incompatible with RTX 4070 Ti SUPER (compute 8.9)
2. ❌ **CUDA 12 environment** - Half-precision compilation errors in native build
3. ❌ **Native gsplat installation** - Missing CUDA headers during compilation
4. ✅ **Docker solution** - Works perfectly with pre-built binaries, no compilation needed

**Note**: The original `gaussian-splatting/` folder has been removed as it's not compatible with modern GPUs. All functionality is provided by Docker + Nerfstudio.

---

**Last Updated**: January 4, 2026  
**Maintained by**: DG-3DPlace Team
