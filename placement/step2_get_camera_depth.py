#!/usr/bin/env python3
"""
Step 2: Load camera parameters and render depth map for the view that produced background.png
"""

import torch
import json
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
import cv2
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

print("=" * 80)
print("      STEP 2: GET CAMERA PARAMETERS & DEPTH MAP")
print("=" * 80)

# ==================== CONFIGURATION ====================
config_path = Path("../room/output/my_scene/data/splatfacto/2026-02-02_124835/config.yml")
local_data_path = Path("../room/data").resolve()
background_img = cv2.imread("background.png")

bg_h, bg_w = background_img.shape[:2]
print(f"\n[TARGET IMAGE]")
print(f"Background.png: {bg_w}x{bg_h}")

# ==================== LOAD CONFIG & CHECKPOINT ====================
print(f"\n[LOADING SCENE]")
print("-" * 80)

config = yaml.load(config_path.open(), Loader=yaml.Loader)
config.data = local_data_path
config.pipeline.datamanager.data = local_data_path

if not hasattr(config.pipeline.model, "camera_optimizer"):
    config.pipeline.model.camera_optimizer = CameraOptimizerConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = config.pipeline.setup(device=device, test_mode="inference")

# Load ORIGINAL checkpoint (without vase)
ckpt_path = config_path.parent / "nerfstudio_models" / "step-000006999.ckpt"
print(f"Loading checkpoint: {ckpt_path.name}...", end=" ")

loaded_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
pipeline.model.eval()

if not hasattr(pipeline.model, 'optimizers'):
    pipeline.model.optimizers = None
if not hasattr(pipeline.model, 'strategy_state'):
    pipeline.model.strategy_state = None
if not hasattr(pipeline.model, 'step'):
    pipeline.model.step = loaded_state["step"]
if not hasattr(pipeline.model, 'info'):
    pipeline.model.info = {}

print("✓")

# ==================== FIND MATCHING CAMERA ====================
print(f"\n[FINDING CAMERA]")
print("-" * 80)

cameras = pipeline.datamanager.train_dataset.cameras
num_cameras = len(cameras)
print(f"Total cameras: {num_cameras}")

# Find camera with matching resolution
camera_idx = None
for idx in range(min(num_cameras, 100)):
    cam_h = int(cameras.height[idx])
    cam_w = int(cameras.width[idx])
    
    if cam_w == bg_w and cam_h == bg_h:
        # Quick similarity check
        camera = cameras[idx : idx + 1]
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        
        rgb = outputs["rgb"].cpu().numpy()
        rgb_img = (rgb * 255).astype(np.uint8)
        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        bg_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
        
        similarity = np.corrcoef(bg_gray.flatten(), rgb_gray.flatten())[0, 1]
        
        if similarity > 0.99:  # Very high similarity
            camera_idx = idx
            print(f"✓ Found matching camera: {idx} (similarity: {similarity:.4f})")
            break

if camera_idx is None:
    print("ERROR: Could not find matching camera!")
    exit(1)

# ==================== GET CAMERA PARAMETERS ====================
print(f"\n[CAMERA PARAMETERS]")
print("-" * 80)

camera = cameras[camera_idx : camera_idx + 1]

fx = float(camera.fx[0])
fy = float(camera.fy[0])
cx = float(camera.cx[0])
cy = float(camera.cy[0])
c2w = camera.camera_to_worlds[0].cpu().numpy()

print(f"Camera Index: {camera_idx}")
print(f"Intrinsics:")
print(f"  fx = {fx:.2f}")
print(f"  fy = {fy:.2f}")
print(f"  cx = {cx:.2f}")
print(f"  cy = {cy:.2f}")
print(f"Resolution: {bg_w}x{bg_h}")

# ==================== RENDER DEPTH MAP ====================
print(f"\n[RENDERING DEPTH MAP]")
print("-" * 80)

with torch.no_grad():
    outputs = pipeline.model.get_outputs_for_camera(camera)

depth_map = outputs["depth"].cpu().numpy().squeeze()
rgb = outputs["rgb"].cpu().numpy()

print(f"✓ Depth map shape: {depth_map.shape}")
print(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}] meters")

# Save depth map
np.save("step2_depth.npy", depth_map)
print(f"✓ Saved: step2_depth.npy")

# Save depth visualization
depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
cv2.imwrite("step2_depth_visualization.png", depth_vis)
print(f"✓ Saved: step2_depth_visualization.png")

# ==================== SAVE CAMERA DATA ====================
print(f"\n[SAVING CAMERA DATA]")
print("-" * 80)

camera_data = {
    "camera_index": int(camera_idx),
    "intrinsics": {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy)
    },
    "resolution": {
        "width": int(bg_w),
        "height": int(bg_h)
    },
    "c2w_matrix": c2w.tolist(),
    "depth_stats": {
        "min": float(depth_map.min()),
        "max": float(depth_map.max()),
        "mean": float(depth_map.mean())
    }
}

with open("step2_camera_data.json", 'w') as f:
    json.dump(camera_data, f, indent=2)

print(f"✓ Saved: step2_camera_data.json")

print("\n" + "=" * 80)
print("✅ STEP 2 COMPLETE")
print("=" * 80)
print(f"\n📷 Camera Index: {camera_idx}")
print(f"📐 Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
print(f"📊 Depth Range: [{depth_map.min():.3f}, {depth_map.max():.3f}] meters")
print(f"\n👉 Please check step2_depth_visualization.png")
print(f"   Does it look like a proper depth map of the scene?")
print("=" * 80)
