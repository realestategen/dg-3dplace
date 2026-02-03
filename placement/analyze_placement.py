#!/usr/bin/env python3
"""
Analyze the placement by comparing rendered view with diffusion image and YOLO detection.
Debug why the vase appears incorrectly sized/positioned/rotated.
"""

import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 80)
print("                  PLACEMENT ANALYSIS & DEBUGGING")
print("=" * 80)

# Load files
rendered_view = cv2.imread("rendered_view.png")
diffusion_added = cv2.imread("diffusion_added.png")
detection_2d = json.load(open("object_properties.json"))
placement_3d = json.load(open("3d_placement.json"))
depth_map = np.load("depth.npy") if Path("depth.npy").exists() else None

print(f"\n[FILES LOADED]")
print(f"✓ Rendered view: {rendered_view.shape if rendered_view is not None else 'NOT FOUND'}")
print(f"✓ Diffusion image: {diffusion_added.shape if diffusion_added is not None else 'NOT FOUND'}")
print(f"✓ Depth map: {depth_map.shape if depth_map is not None else 'NOT FOUND'}")

# ==================== ANALYZE YOLO DETECTION ====================
print(f"\n[YOLO DETECTION IN DIFFUSION IMAGE]")
print("-" * 80)

bbox_diff = detection_2d['bounding_box_aligned']
center_x_diff = bbox_diff['center_x']
center_y_diff = bbox_diff['center_y']
width_diff = bbox_diff['width']
height_diff = bbox_diff['height']

print(f"Bounding box in diffusion_added.png ({diffusion_added.shape[1]}x{diffusion_added.shape[0]}):")
print(f"  Center: ({center_x_diff:.1f}, {center_y_diff:.1f})")
print(f"  Size: {width_diff:.1f}x{height_diff:.1f}")
print(f"  Rotation: {detection_2d['rotation']['estimated_degrees']:.2f}°")

# Draw detection on diffusion image
diffusion_vis = diffusion_added.copy()
x1 = int(bbox_diff['x1'])
y1 = int(bbox_diff['y1'])
x2 = int(bbox_diff['x2'])
y2 = int(bbox_diff['y2'])
cv2.rectangle(diffusion_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.circle(diffusion_vis, (int(center_x_diff), int(center_y_diff)), 5, (0, 0, 255), -1)
cv2.putText(diffusion_vis, f"YOLO: {width_diff:.0f}x{height_diff:.0f}", 
            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite("analysis_diffusion_detection.png", diffusion_vis)
print(f"✓ Saved: analysis_diffusion_detection.png")

# ==================== SCALE TO RENDERED CAMERA ====================
print(f"\n[SCALING TO RENDERED CAMERA]")
print("-" * 80)

render_h, render_w = rendered_view.shape[:2]
diff_h, diff_w = diffusion_added.shape[:2]

scale_x = render_w / diff_w
scale_y = render_h / diff_h

center_x_render = center_x_diff * scale_x
center_y_render = center_y_diff * scale_y
width_render = width_diff * scale_x
height_render = height_diff * scale_y

print(f"Rendered view size: {render_w}x{render_h}")
print(f"Diffusion size: {diff_w}x{diff_h}")
print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
print(f"\nScaled detection in rendered view:")
print(f"  Center: ({center_x_render:.1f}, {center_y_render:.1f})")
print(f"  Size: {width_render:.1f}x{height_render:.1f}")

# Draw scaled detection on rendered view
rendered_vis = rendered_view.copy()
x1_render = int(center_x_render - width_render/2)
y1_render = int(center_y_render - height_render/2)
x2_render = int(center_x_render + width_render/2)
y2_render = int(center_y_render + height_render/2)

cv2.rectangle(rendered_vis, (x1_render, y1_render), (x2_render, y2_render), (255, 0, 0), 2)
cv2.circle(rendered_vis, (int(center_x_render), int(center_y_render)), 5, (0, 0, 255), -1)
cv2.putText(rendered_vis, f"Expected: {width_render:.0f}x{height_render:.0f}", 
            (x1_render, y1_render-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# ==================== ANALYZE DEPTH ====================
print(f"\n[DEPTH ANALYSIS]")
print("-" * 80)

if depth_map is not None:
    u = int(center_x_render)
    v = int(center_y_render)
    u = max(0, min(u, depth_map.shape[1] - 1))
    v = max(0, min(v, depth_map.shape[0] - 1))
    
    # Sample region around center
    region_size = 10
    v_min = max(0, v - region_size)
    v_max = min(depth_map.shape[0], v + region_size)
    u_min = max(0, u - region_size)
    u_max = min(depth_map.shape[1], u + region_size)
    
    depth_region = depth_map[v_min:v_max, u_min:u_max]
    depth_center = depth_map[v, u]
    depth_median = np.median(depth_region)
    depth_mean = np.mean(depth_region)
    
    print(f"Depth at center pixel ({u}, {v}):")
    print(f"  Center value: {depth_center:.3f}")
    print(f"  Median (10px region): {depth_median:.3f}")
    print(f"  Mean (10px region): {depth_mean:.3f}")
    print(f"  Min/Max in region: [{depth_region.min():.3f}, {depth_region.max():.3f}]")
    
    # Visualize depth with detection overlay
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_vis = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.rectangle(depth_vis, (x1_render, y1_render), (x2_render, y2_render), (255, 255, 255), 2)
    cv2.circle(depth_vis, (u, v), 5, (0, 0, 255), -1)
    cv2.putText(depth_vis, f"Depth: {depth_median:.2f}m", 
                (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite("analysis_depth_with_detection.png", depth_vis)
    print(f"✓ Saved: analysis_depth_with_detection.png")

# ==================== ANALYZE 3D PLACEMENT ====================
print(f"\n[3D PLACEMENT ANALYSIS]")
print("-" * 80)

pos_3d = placement_3d['position_3d']
scale_3d = placement_3d['scale_3d']
camera_info = placement_3d['camera_info']

print(f"Calculated 3D position: [{pos_3d['x']:.3f}, {pos_3d['y']:.3f}, {pos_3d['z']:.3f}]")
print(f"Calculated 3D size: {scale_3d['width']:.3f}m x {scale_3d['height']:.3f}m")
print(f"Used depth: {scale_3d['depth_value']:.3f}m")

# Recompute to verify
fx = camera_info['fx']
fy = camera_info['fy']
cx = camera_info['cx']
cy = camera_info['cy']

print(f"\nCamera intrinsics:")
print(f"  fx={fx:.2f}, fy={fy:.2f}")
print(f"  cx={cx:.2f}, cy={cy:.2f}")

# Recalculate from scratch
u_calc = int(center_x_render)
v_calc = int(center_y_render)
if depth_map is not None:
    depth_calc = depth_map[v_calc, u_calc]
    
    x_cam = (u_calc - cx) * depth_calc / fx
    y_cam = (v_calc - cy) * depth_calc / fy
    z_cam = depth_calc
    
    print(f"\nRecalculated camera space:")
    print(f"  Pixel: ({u_calc}, {v_calc})")
    print(f"  Depth: {depth_calc:.3f}")
    print(f"  Camera coords: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")
    
    # Height calculation
    height_3d_calc = (height_render / fy) * depth_calc
    width_3d_calc = (width_render / fx) * depth_calc
    
    print(f"\nRecalculated 3D size:")
    print(f"  Height: {height_3d_calc:.3f}m (stored: {scale_3d['height']:.3f}m)")
    print(f"  Width: {width_3d_calc:.3f}m (stored: {scale_3d['width']:.3f}m)")
    
    # Check if sizes match reasonably
    if abs(height_3d_calc - scale_3d['height']) > 0.01:
        print(f"  ⚠️  HEIGHT MISMATCH! Difference: {abs(height_3d_calc - scale_3d['height']):.3f}m")

# ==================== LOOK FOR RED VASE IN RENDERED VIEW ====================
print(f"\n[RED OBJECT DETECTION IN RENDERED VIEW]")
print("-" * 80)

# Convert to HSV and look for red
hsv = cv2.cvtColor(rendered_view, cv2.COLOR_BGR2HSV)

# Red has two ranges in HSV
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 | mask2

# Find contours of red regions
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    # Get largest red region
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area > 100:  # Minimum area threshold
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_red_x = x + w/2
        center_red_y = y + h/2
        
        print(f"✓ Found red object in rendered view:")
        print(f"  Position: ({center_red_x:.1f}, {center_red_y:.1f})")
        print(f"  Size: {w}x{h} pixels")
        print(f"  Area: {area:.0f} pixels²")
        
        # Compare with expected position
        dist = np.sqrt((center_red_x - center_x_render)**2 + (center_red_y - center_y_render)**2)
        print(f"\nComparison with expected position:")
        print(f"  Expected: ({center_x_render:.1f}, {center_y_render:.1f})")
        print(f"  Actual: ({center_red_x:.1f}, {center_red_y:.1f})")
        print(f"  Distance: {dist:.1f} pixels")
        
        if dist > 50:
            print(f"  ⚠️  LARGE POSITION ERROR! Vase is {dist:.0f} pixels away from expected location")
        
        # Size comparison
        size_ratio_w = w / width_render
        size_ratio_h = h / height_render
        print(f"\nSize comparison:")
        print(f"  Expected: {width_render:.1f}x{height_render:.1f}")
        print(f"  Actual: {w}x{h}")
        print(f"  Ratio: {size_ratio_w:.2f}x (width), {size_ratio_h:.2f}x (height)")
        
        if size_ratio_h > 1.5 or size_ratio_h < 0.5:
            print(f"  ⚠️  SIZE MISMATCH! Vase is {size_ratio_h:.1f}x the expected height")
        
        # Draw both on visualization
        cv2.rectangle(rendered_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(rendered_vis, (int(center_red_x), int(center_red_y)), 5, (0, 255, 0), -1)
        cv2.putText(rendered_vis, f"Actual: {w}x{h}", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw connection line
        cv2.line(rendered_vis, (int(center_x_render), int(center_y_render)), 
                 (int(center_red_x), int(center_red_y)), (255, 255, 0), 2)
        
        # Calculate corrected depth if position is different
        if dist > 20 and depth_map is not None:
            u_red = int(center_red_x)
            v_red = int(center_red_y)
            u_red = max(0, min(u_red, depth_map.shape[1] - 1))
            v_red = max(0, min(v_red, depth_map.shape[0] - 1))
            
            depth_red_region = depth_map[max(0, v_red-5):min(depth_map.shape[0], v_red+5),
                                        max(0, u_red-5):min(depth_map.shape[1], u_red+5)]
            depth_red = np.median(depth_red_region)
            
            print(f"\n[CORRECTED MAPPING SUGGESTION]")
            print(f"Using actual red vase position ({u_red}, {v_red}):")
            print(f"  Depth at vase: {depth_red:.3f}m")
            
            x_cam_corrected = (u_red - cx) * depth_red / fx
            y_cam_corrected = (v_red - cy) * depth_red / fy
            z_cam_corrected = depth_red
            
            print(f"  Camera space: ({x_cam_corrected:.3f}, {y_cam_corrected:.3f}, {z_cam_corrected:.3f})")
            
            # Transform to world
            c2w = np.array(camera_info['c2w'])
            point_cam = np.array([x_cam_corrected, y_cam_corrected, z_cam_corrected, 1.0])
            point_world = c2w @ point_cam
            
            print(f"  World space: ({point_world[0]:.3f}, {point_world[1]:.3f}, {point_world[2]:.3f})")
            
            # Corrected size
            height_corrected = (h / fy) * depth_red
            width_corrected = (w / fx) * depth_red
            
            print(f"  Corrected 3D size: {width_corrected:.3f}m x {height_corrected:.3f}m")
            
            # Save corrected placement
            corrected_placement = {
                "method": "camera_depth_unprojection_corrected",
                "position_3d": {
                    "x": float(point_world[0]),
                    "y": float(point_world[1]),
                    "z": float(point_world[2])
                },
                "scale_3d": {
                    "height": float(height_corrected),
                    "width": float(width_corrected),
                    "depth_value": float(depth_red)
                },
                "rotation": placement_3d['rotation'],
                "camera_info": camera_info,
                "detection_2d_corrected": {
                    "center_x_px": float(center_red_x),
                    "center_y_px": float(center_red_y),
                    "width_px": float(w),
                    "height_px": float(h)
                }
            }
            
            with open("3d_placement_corrected.json", 'w') as f:
                json.dump(corrected_placement, f, indent=2)
            
            print(f"\n✓ Saved corrected placement: 3d_placement_corrected.json")
else:
    print("✗ No significant red regions found in rendered view")
    print("  Vase may not be visible or is very small")

cv2.imwrite("analysis_rendered_with_both.png", rendered_vis)
print(f"\n✓ Saved: analysis_rendered_with_both.png")
print(f"  Blue box = Expected from YOLO")
print(f"  Green box = Actual red object (if found)")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nCheck these files:")
print(f"  1. analysis_diffusion_detection.png - YOLO detection on original")
print(f"  2. analysis_depth_with_detection.png - Depth map with expected location")
print(f"  3. analysis_rendered_with_both.png - Expected vs Actual comparison")
print(f"\nIf corrected placement was generated:")
print(f"  4. 3d_placement_corrected.json - Use this for next injection")
print("=" * 80)
