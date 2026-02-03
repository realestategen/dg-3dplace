#!/usr/bin/env python3
"""
Step 1: Scale YOLO detection from diffusion_added.png back to camera resolution
"""

import json
import cv2

print("=" * 80)
print("      STEP 1: SCALE DETECTION TO CAMERA RESOLUTION")
print("=" * 80)

# Load detection data
with open("vase_detection.json", 'r') as f:
    detection = json.load(f)

# Load images to get dimensions
background = cv2.imread("background.png")
diffusion = cv2.imread("diffusion_added.png")

camera_w, camera_h = background.shape[1], background.shape[0]
diffusion_w, diffusion_h = diffusion.shape[1], diffusion.shape[0]

print(f"\n[IMAGE DIMENSIONS]")
print(f"Camera (background.png): {camera_w}x{camera_h}")
print(f"Diffusion (upscaled): {diffusion_w}x{diffusion_h}")

# Calculate scale factors
scale_x = camera_w / diffusion_w
scale_y = camera_h / diffusion_h

print(f"\n[SCALE FACTORS]")
print(f"Scale X: {scale_x:.6f}")
print(f"Scale Y: {scale_y:.6f}")

# Get detection in diffusion space
bbox = detection['bounding_box']
center_x_diff = bbox['center_x']
center_y_diff = bbox['center_y']
width_diff = bbox['width']
height_diff = bbox['height']

print(f"\n[DETECTION IN DIFFUSION IMAGE]")
print(f"Centroid: ({center_x_diff:.1f}, {center_y_diff:.1f})")
print(f"Size: {width_diff:.1f} x {height_diff:.1f}")

# Scale to camera resolution
center_x_cam = center_x_diff * scale_x
center_y_cam = center_y_diff * scale_y
width_cam = width_diff * scale_x
height_cam = height_diff * scale_y

x1_cam = center_x_cam - width_cam / 2
y1_cam = center_y_cam - height_cam / 2
x2_cam = center_x_cam + width_cam / 2
y2_cam = center_y_cam + height_cam / 2

print(f"\n[SCALED TO CAMERA RESOLUTION]")
print(f"Centroid: ({center_x_cam:.1f}, {center_y_cam:.1f})")
print(f"Bounding Box: [{x1_cam:.1f}, {y1_cam:.1f}] → [{x2_cam:.1f}, {y2_cam:.1f}]")
print(f"Size: {width_cam:.1f} x {height_cam:.1f}")

# Save scaled detection
scaled_detection = {
    "camera_resolution": {
        "width": camera_w,
        "height": camera_h
    },
    "bounding_box_camera": {
        "x1": float(x1_cam),
        "y1": float(y1_cam),
        "x2": float(x2_cam),
        "y2": float(y2_cam),
        "center_x": float(center_x_cam),
        "center_y": float(center_y_cam),
        "width": float(width_cam),
        "height": float(height_cam)
    },
    "original_detection": detection
}

with open("vase_detection_scaled.json", 'w') as f:
    json.dump(scaled_detection, f, indent=2)

# Visualize on background image
background_vis = background.copy()

# Draw bounding box
cv2.rectangle(background_vis, 
              (int(x1_cam), int(y1_cam)), 
              (int(x2_cam), int(y2_cam)), 
              (0, 0, 255), 2)

# Draw centroid
cv2.circle(background_vis, (int(center_x_cam), int(center_y_cam)), 5, (0, 255, 0), -1)
cv2.circle(background_vis, (int(center_x_cam), int(center_y_cam)), 7, (0, 0, 255), 2)

# Add label
label = f"Vase: ({center_x_cam:.0f}, {center_y_cam:.0f})"
cv2.putText(background_vis, label, (int(x1_cam), int(y1_cam) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imwrite("step1_scaled_detection.png", background_vis)

print(f"\n[OUTPUT FILES]")
print(f"✓ vase_detection_scaled.json")
print(f"✓ step1_scaled_detection.png")

print("\n" + "=" * 80)
print("✅ STEP 1 COMPLETE")
print("=" * 80)
print(f"\n📍 Vase centroid in camera space: ({center_x_cam:.1f}, {center_y_cam:.1f})")
print(f"📏 Vase size in camera space: {width_cam:.1f} x {height_cam:.1f} pixels")
print(f"\n👉 Please check step1_scaled_detection.png")
print(f"   Does the red box correctly highlight the vase in background.png?")
print("=" * 80)
