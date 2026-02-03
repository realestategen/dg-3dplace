#!/usr/bin/env python3
"""
Detect the added vase by comparing background.png and diffusion_added.png
using YOLO object detection and differential analysis.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import json

print("=" * 80)
print("                  DETECT ADDED VASE")
print("=" * 80)

# ==================== LOAD IMAGES ====================
print(f"\n[STEP 1] Loading Images")
print("-" * 80)

background = cv2.imread("background.png")
diffusion_added = cv2.imread("diffusion_added.png")

if background is None:
    print("ERROR: background.png not found!")
    exit(1)

if diffusion_added is None:
    print("ERROR: diffusion_added.png not found!")
    exit(1)

print(f"✓ Loaded background.png: {background.shape[1]}x{background.shape[0]}")
print(f"✓ Loaded diffusion_added.png: {diffusion_added.shape[1]}x{diffusion_added.shape[0]}")

# ==================== LOAD YOLO MODEL ====================
print(f"\n[STEP 2] Loading YOLO Model")
print("-" * 80)

model = YOLO("yolov8n.pt")  # small model, fast
print(f"✓ Loaded YOLOv8n model")

# ==================== DETECT OBJECTS IN BOTH IMAGES ====================
print(f"\n[STEP 3] Running Object Detection")
print("-" * 80)

print("Running YOLO on background.png...")
results_background = model.predict(background, verbose=False)
boxes_bg = results_background[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
classes_bg = results_background[0].boxes.cls.cpu().numpy()
conf_bg = results_background[0].boxes.conf.cpu().numpy()

print(f"✓ Found {len(boxes_bg)} objects in background")

print("Running YOLO on diffusion_added.png...")
results_diffusion = model.predict(diffusion_added, verbose=False)
boxes_diff = results_diffusion[0].boxes.xyxy.cpu().numpy()
classes_diff = results_diffusion[0].boxes.cls.cpu().numpy()
conf_diff = results_diffusion[0].boxes.conf.cpu().numpy()

print(f"✓ Found {len(boxes_diff)} objects in diffusion image")

# ==================== FIND NEW OBJECTS ====================
print(f"\n[STEP 4] Finding Added Objects")
print("-" * 80)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

# Find objects in diffusion that are NOT in background
new_objects = []
iou_threshold = 0.3

for i, box_diff in enumerate(boxes_diff):
    is_new = True
    
    # Check if this object overlaps significantly with any background object
    for box_bg in boxes_bg:
        iou = calculate_iou(box_diff, box_bg)
        if iou > iou_threshold:
            is_new = False
            break
    
    if is_new:
        new_objects.append({
            'index': i,
            'box': box_diff,
            'class': int(classes_diff[i]),
            'confidence': float(conf_diff[i]),
            'class_name': model.names[int(classes_diff[i])]
        })

print(f"✓ Found {len(new_objects)} new object(s) added in diffusion image")

if len(new_objects) == 0:
    print("⚠️  No new objects detected! The vase might not be detected by YOLO.")
    print("   Try lowering detection confidence threshold or using different model.")
    exit(1)

# ==================== IDENTIFY VASE ====================
print(f"\n[STEP 5] Identifying Vase")
print("-" * 80)

# Look for vase class (class 75 in COCO) or largest new object
vase_object = None

for obj in new_objects:
    print(f"New object {obj['index']}: {obj['class_name']} (class {obj['class']}, conf={obj['confidence']:.2f})")
    
    if obj['class_name'] == 'vase' or obj['class'] == 75:
        vase_object = obj
        print(f"  → Found vase!")
        break

# If no explicit vase detected, take the largest new object
if vase_object is None and len(new_objects) > 0:
    print("\n⚠️  No explicit 'vase' class detected")
    print("   Using largest new object as vase...")
    
    # Find largest by area
    largest_obj = max(new_objects, key=lambda obj: 
                     (obj['box'][2] - obj['box'][0]) * (obj['box'][3] - obj['box'][1]))
    vase_object = largest_obj
    print(f"✓ Selected: {vase_object['class_name']} (largest new object)")

# ==================== CALCULATE VASE PROPERTIES ====================
print(f"\n[STEP 6] Calculating Vase Properties")
print("-" * 80)

x1, y1, x2, y2 = vase_object['box']
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
width = x2 - x1
height = y2 - y1

print(f"Bounding Box:")
print(f"  Top-left: ({x1:.1f}, {y1:.1f})")
print(f"  Bottom-right: ({x2:.1f}, {y2:.1f})")
print(f"  Center: ({center_x:.1f}, {center_y:.1f})")
print(f"  Size: {width:.1f} x {height:.1f} pixels")

# ==================== SAVE RESULTS ====================
print(f"\n[STEP 7] Saving Results")
print("-" * 80)

# Save detection data
detection_data = {
    "vase": {
        "class": vase_object['class_name'],
        "class_id": vase_object['class'],
        "confidence": vase_object['confidence']
    },
    "bounding_box": {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "width": float(width),
        "height": float(height),
        "center_x": float(center_x),
        "center_y": float(center_y)
    },
    "image_info": {
        "width": diffusion_added.shape[1],
        "height": diffusion_added.shape[0]
    }
}

with open("vase_detection.json", 'w') as f:
    json.dump(detection_data, f, indent=2)

print(f"✓ Saved detection data: vase_detection.json")

# Visualize detection
highlighted = diffusion_added.copy()

# Draw bounding box
cv2.rectangle(highlighted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

# Draw center point
cv2.circle(highlighted, (int(center_x), int(center_y)), 8, (0, 255, 0), -1)
cv2.circle(highlighted, (int(center_x), int(center_y)), 10, (0, 0, 255), 2)

# Add label
label = f"Vase: {vase_object['confidence']:.2f}"
cv2.putText(highlighted, label, (int(x1), int(y1) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Add centroid coordinates
coords_text = f"({center_x:.0f}, {center_y:.0f})"
cv2.putText(highlighted, coords_text, (int(center_x) + 15, int(center_y)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Save visualization
cv2.imwrite("vase_detected.png", highlighted)
print(f"✓ Saved visualization: vase_detected.png")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ VASE DETECTION COMPLETE")
print("=" * 80)
print(f"\n🎯 Vase Location:")
print(f"   Centroid: ({center_x:.1f}, {center_y:.1f})")
print(f"   Bounding Box: [{x1:.1f}, {y1:.1f}] → [{x2:.1f}, {y2:.1f}]")
print(f"\n📏 Vase Size:")
print(f"   Width: {width:.1f} pixels")
print(f"   Height: {height:.1f} pixels")
print(f"\n📊 Detection Info:")
print(f"   Class: {vase_object['class_name']}")
print(f"   Confidence: {vase_object['confidence']:.2f}")
print(f"\n💾 Output Files:")
print(f"   vase_detection.json - Detection data")
print(f"   vase_detected.png - Visualization with centroid")
print("=" * 80)
