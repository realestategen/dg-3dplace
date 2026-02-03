#!/usr/bin/env python3
"""
Step 3: Sample depth at vase location and unproject to 3D world coordinates
"""

import json
import numpy as np
import cv2
from PIL import Image

print("=" * 80)
print("      STEP 3: UNPROJECT TO 3D COORDINATES")
print("=" * 80)

# ==================== LOAD DATA ====================
print(f"\n[LOADING DATA]")
print("-" * 80)

# Load vase detection in camera space
with open("vase_detection_scaled.json", 'r') as f:
    vase_data = json.load(f)

# Load camera parameters
with open("step2_camera_data.json", 'r') as f:
    camera_data = json.load(f)

# Load depth map
depth_map = np.load("step2_depth.npy")

bbox_cam = vase_data['bounding_box_camera']
cx_vase = bbox_cam['center_x']
cy_vase = bbox_cam['center_y']
width_pixels = bbox_cam['width']
height_pixels = bbox_cam['height']

print(f"✓ Vase centroid (camera space): ({cx_vase:.1f}, {cy_vase:.1f})")
print(f"✓ Vase size (pixels): {width_pixels:.1f} × {height_pixels:.1f}")
print(f"✓ Camera intrinsics: fx={camera_data['intrinsics']['fx']:.2f}, fy={camera_data['intrinsics']['fy']:.2f}")
print(f"✓ Depth map shape: {depth_map.shape}")

# ==================== SAMPLE DEPTH ====================
print(f"\n[SAMPLING DEPTH AT VASE LOCATION]")
print("-" * 80)
u = int(round(cx_vase))
v = int(round(cy_vase))

# Sample 5x5 region around centroid for robustness
window_size = 5
half = window_size // 2

v_min = max(0, v - half)
v_max = min(depth_map.shape[0], v + half + 1)
u_min = max(0, u - half)
u_max = min(depth_map.shape[1], u + half + 1)

depth_window = depth_map[v_min:v_max, u_min:u_max]
depth_median = np.median(depth_window)
depth_mean = np.mean(depth_window)
depth_std = np.std(depth_window)

print(f"Pixel location: ({u}, {v})")
print(f"Depth window: {depth_window.shape}")
print(f"  Median: {depth_median:.4f} m")
print(f"  Mean:   {depth_mean:.4f} m")
print(f"  Std:    {depth_std:.4f} m")
print(f"\n✓ Using median depth: {depth_median:.4f} m")

# ==================== UNPROJECT TO CAMERA SPACE ====================
print(f"\n[UNPROJECTING TO CAMERA SPACE]")
print("-" * 80)

fx = camera_data['intrinsics']['fx']
fy = camera_data['intrinsics']['fy']
cx_cam = camera_data['intrinsics']['cx']
cy_cam = camera_data['intrinsics']['cy']

# Pinhole camera model (using exact centroid, not rounded)
x_cam = (cx_vase - cx_cam) * depth_median / fx
y_cam = (cy_vase - cy_cam) * depth_median / fy
z_cam = depth_median

print(f"Camera coordinates:")
print(f"  x_cam = {x_cam:.4f} m")
print(f"  y_cam = {y_cam:.4f} m")
print(f"  z_cam = {z_cam:.4f} m")

# ==================== TRANSFORM TO WORLD SPACE ====================
print(f"\n[TRANSFORMING TO WORLD SPACE]")
print("-" * 80)

c2w = np.array(camera_data['c2w_matrix'])
point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
point_world = c2w @ point_cam

x_world, y_world, z_world = point_world[:3]

print(f"World coordinates:")
print(f"  x = {x_world:.4f} m")
print(f"  y = {y_world:.4f} m")
print(f"  z = {z_world:.4f} m")

# ==================== CALCULATE 3D SIZE ====================
print(f"\n[CALCULATING 3D SIZE]")
print("-" * 80)

# Physical size = (pixel_size / focal_length) * depth
width_3d = (width_pixels / fx) * depth_median
height_3d = (height_pixels / fy) * depth_median

print(f"Vase 3D dimensions:")
print(f"  Width:  {width_3d:.4f} m ({width_3d*100:.1f} cm)")
print(f"  Height: {height_3d:.4f} m ({height_3d*100:.1f} cm)")

# Calculate scale factor relative to vase.obj
# vase.obj is unit scale, we need to scale to match detected size
scale = height_3d  # Use height as primary scale reference

print(f"\n✓ Scale factor for vase.obj: {scale:.4f}")

# ==================== CREATE VISUALIZATION ====================
print(f"\n[CREATING VISUALIZATION]")
print("-" * 80)

# Load background image
bg_img = cv2.imread("background.png")

# Draw depth sampling region
cv2.rectangle(bg_img, (u_min, v_min), (u_max, v_max), (255, 255, 0), 2)  # Cyan box
cv2.circle(bg_img, (u, v), 5, (0, 255, 255), -1)  # Yellow center

# Add text
text = f"Depth: {depth_median:.3f}m"
cv2.putText(bg_img, text, (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

text2 = f"3D: ({x_world:.2f}, {y_world:.2f}, {z_world:.2f})"
cv2.putText(bg_img, text2, (u + 10, v + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# Draw bounding box from scaled detection
x1 = int(round(bbox_cam['x1']))
y1 = int(round(bbox_cam['y1']))
x2 = int(round(bbox_cam['x2']))
y2 = int(round(bbox_cam['y2']))
cv2.rectangle(bg_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

cv2.imwrite("step3_3d_position.png", bg_img)
print(f"✓ Saved: step3_3d_position.png")

# ==================== SAVE 3D PLACEMENT DATA ====================
print(f"\n[SAVING 3D PLACEMENT DATA]")
print("-" * 80)

placement_data = {
    "vase_position_world": {
        "x": float(x_world),
        "y": float(y_world),
        "z": float(z_world)
    },
    "vase_position_camera": {
        "x": float(x_cam),
        "y": float(y_cam),
        "z": float(z_cam)
    },
    "vase_size_3d": {
        "width": float(width_3d),
        "height": float(height_3d)
    },
    "scale_factor": float(scale),
    "depth_sampled": float(depth_median),
    "pixel_location": {
        "u": int(u),
        "v": int(v)
    },
    "detection_2d": {
        "centroid_x": float(cx_vase),
        "centroid_y": float(cy_vase),
        "bbox": {
            "x1": float(bbox_cam['x1']),
            "y1": float(bbox_cam['y1']),
            "x2": float(bbox_cam['x2']),
            "y2": float(bbox_cam['y2'])
        },
        "width_pixels": float(width_pixels),
        "height_pixels": float(height_pixels)
    }
}

with open("step3_3d_placement.json", 'w') as f:
    json.dump(placement_data, f, indent=2)

print(f"✓ Saved: step3_3d_placement.json")

print("\n" + "=" * 80)
print("✅ STEP 3 COMPLETE")
print("=" * 80)
print(f"\n📍 Vase Position (World): ({x_world:.4f}, {y_world:.4f}, {z_world:.4f}) m")
print(f"📏 Vase Size (3D): {width_3d*100:.1f} cm × {height_3d*100:.1f} cm")
print(f"📊 Depth: {depth_median:.4f} m")
print(f"🔍 Scale Factor: {scale:.4f}")
print(f"\n👉 Please check step3_3d_position.png")
print(f"   Cyan box = depth sampling region")
print(f"   Yellow dot = vase centroid")
print(f"   Red box = vase bounding box")
print("=" * 80)
