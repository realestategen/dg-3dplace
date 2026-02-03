#!/usr/bin/env python3
"""
Detect the NEW object (vase) added in diffusion_added.png using YOLOv8.
Compares objects in diffusion vs background to find what was added.
"""

import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import json

print("=" * 80)
print("     DETECT VASE USING YOLO (DIFFERENTIAL DETECTION)")
print("=" * 80)

# ==================== CONFIGURATION ====================
BACKGROUND_IMAGE = Path("background.png")
DIFFUSION_IMAGE = Path("diffusion_added.png")
OUTPUT_JSON = Path("object_properties.json")
MODEL_NAME = "yolov8n.pt"  # Nano model - fast

# Detection parameters
IOU_THRESHOLD = 0.3  # IoU below this = different objects
CONFIDENCE_THRESHOLD = 0.25  # YOLO confidence threshold

print(f"\n[CONFIG]")
print(f"  Background: {BACKGROUND_IMAGE}")
print(f"  Diffusion: {DIFFUSION_IMAGE}")
print(f"  Model: {MODEL_NAME}")
print(f"  IoU threshold: {IOU_THRESHOLD}")

# ==================== LOAD MODEL ====================
print(f"\n[STEP 1] Loading YOLO Model")
print("-" * 80)

model = YOLO(MODEL_NAME)
print(f"✓ Loaded {MODEL_NAME}")

# ==================== DETECT IN BOTH IMAGES ====================
print(f"\n[STEP 2] Running Detection on Both Images")
print("-" * 80)

# Load images
background_img = cv2.imread(str(BACKGROUND_IMAGE))
diffusion_img = cv2.imread(str(DIFFUSION_IMAGE))

# Resize diffusion to match background if needed
if background_img.shape != diffusion_img.shape:
    print(f"  Resizing diffusion {diffusion_img.shape[:2]} → {background_img.shape[:2]}")
    diffusion_img = cv2.resize(diffusion_img, 
                               (background_img.shape[1], background_img.shape[0]))

# Run YOLO on background
print("  Running YOLO on background...")
results_bg = model.predict(background_img, conf=CONFIDENCE_THRESHOLD, verbose=False)
boxes_bg = results_bg[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
classes_bg = results_bg[0].boxes.cls.cpu().numpy()
confs_bg = results_bg[0].boxes.conf.cpu().numpy()

print(f"✓ Background detections: {len(boxes_bg)}")
for i, (box, cls, conf) in enumerate(zip(boxes_bg, classes_bg, confs_bg)):
    class_name = model.names[int(cls)]
    print(f"  {i+1}. {class_name} ({conf:.2f}) at {box.astype(int)}")

# Run YOLO on diffusion
print("\n  Running YOLO on diffusion...")
results_diff = model.predict(diffusion_img, conf=CONFIDENCE_THRESHOLD, verbose=False)
boxes_diff = results_diff[0].boxes.xyxy.cpu().numpy()
classes_diff = results_diff[0].boxes.cls.cpu().numpy()
confs_diff = results_diff[0].boxes.conf.cpu().numpy()

print(f"✓ Diffusion detections: {len(boxes_diff)}")
for i, (box, cls, conf) in enumerate(zip(boxes_diff, classes_diff, confs_diff)):
    class_name = model.names[int(cls)]
    print(f"  {i+1}. {class_name} ({conf:.2f}) at {box.astype(int)}")

# ==================== FIND NEW OBJECT ====================
print(f"\n[STEP 3] Finding New Object (Vase)")
print("-" * 80)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Find objects in diffusion that are NOT in background
new_objects = []

for i, (box_diff, cls_diff, conf_diff) in enumerate(zip(boxes_diff, classes_diff, confs_diff)):
    # Check if this object overlaps significantly with any background object
    is_new = True
    max_iou = 0
    
    for box_bg, cls_bg in zip(boxes_bg, classes_bg):
        # Only compare same class
        if cls_diff == cls_bg:
            iou = calculate_iou(box_diff, box_bg)
            max_iou = max(max_iou, iou)
            if iou > IOU_THRESHOLD:
                is_new = False
                break
    
    if is_new:
        class_name = model.names[int(cls_diff)]
        new_objects.append({
            'box': box_diff,
            'class': class_name,
            'class_id': int(cls_diff),
            'confidence': float(conf_diff),
            'index': i
        })
        print(f"✓ NEW object detected: {class_name} (conf={conf_diff:.2f}, max_iou={max_iou:.2f})")
        print(f"  BBox: {box_diff.astype(int)}")

if len(new_objects) == 0:
    print("❌ No new objects detected!")
    print("   Try lowering IOU_THRESHOLD or CONFIDENCE_THRESHOLD")
    exit(1)

# Select the most confident new object (assume it's the vase)
vase = max(new_objects, key=lambda x: x['confidence'])
print(f"\n✓ Selected as vase: {vase['class']} (confidence={vase['confidence']:.2f})")

# ==================== ANALYZE VASE PROPERTIES ====================
print(f"\n[STEP 4] Analyzing Vase Properties")
print("-" * 80)

box = vase['box']
x1, y1, x2, y2 = box

# Bounding box properties
width = x2 - x1
height = y2 - y1
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
aspect_ratio = width / height

print(f"✓ Bounding box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
print(f"✓ Size: {width:.0f} x {height:.0f} pixels")
print(f"✓ Center: ({center_x:.0f}, {center_y:.0f})")
print(f"✓ Aspect ratio: {aspect_ratio:.2f}")

# Estimate rotation by extracting vase region and using contours
vase_region = diffusion_img[int(y1):int(y2), int(x1):int(x2)]
gray = cv2.cvtColor(vase_region, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) >= 5:  # Need at least 5 points for fitEllipse
        ellipse = cv2.fitEllipse(largest_contour)
        rotation = ellipse[2]  # Angle in degrees
    else:
        rotation = 0.0
else:
    rotation = 0.0

print(f"✓ Estimated rotation: {rotation:.1f}°")

# Scale estimation
img_height, img_width = background_img.shape[:2]
scale_x = width / img_width
scale_y = height / img_height

ASSUMED_ROOM_HEIGHT = 5.0  # meters
estimated_height_m = scale_y * ASSUMED_ROOM_HEIGHT
estimated_width_m = scale_x * ASSUMED_ROOM_HEIGHT

print(f"\n✓ Relative scale:")
print(f"  - Width: {scale_x:.3f} (fraction of image)")
print(f"  - Height: {scale_y:.3f} (fraction of image)")
print(f"\n✓ Estimated 3D size (assuming {ASSUMED_ROOM_HEIGHT}m room):")
print(f"  - Height: {estimated_height_m:.2f} meters")
print(f"  - Width: {estimated_width_m:.2f} meters")

# ==================== SAVE RESULTS ====================
print(f"\n[STEP 5] Saving Results")
print("-" * 80)

results = {
    "detection_method": "YOLOv8_differential",
    "vase": {
        "class": vase['class'],
        "class_id": vase['class_id'],
        "confidence": vase['confidence']
    },
    "bounding_box_aligned": {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "width": float(width),
        "height": float(height),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "aspect_ratio": float(aspect_ratio)
    },
    "rotation": {
        "estimated_degrees": float(rotation)
    },
    "scale": {
        "relative_to_image": {
            "width_fraction": float(scale_x),
            "height_fraction": float(scale_y)
        },
        "estimated_3d_size_meters": {
            "height": float(estimated_height_m),
            "width": float(estimated_width_m),
            "assumed_room_height": ASSUMED_ROOM_HEIGHT
        },
        "pixels": {
            "width": float(width),
            "height": float(height)
        }
    },
    "image_info": {
        "size": [img_width, img_height],
        "iou_threshold": IOU_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved: {OUTPUT_JSON}")

# ==================== VISUALIZATION ====================
print(f"\n[STEP 6] Creating Visualization")
print("-" * 80)

# Draw all detections on diffusion
vis = diffusion_img.copy()

# Draw background objects (gray)
for box_bg in boxes_bg:
    x1_bg, y1_bg, x2_bg, y2_bg = box_bg.astype(int)
    cv2.rectangle(vis, (x1_bg, y1_bg), (x2_bg, y2_bg), (128, 128, 128), 1)

# Draw new objects (yellow)
for obj in new_objects:
    box_obj = obj['box'].astype(int)
    cv2.rectangle(vis, (box_obj[0], box_obj[1]), (box_obj[2], box_obj[3]), (0, 255, 255), 2)

# Draw selected vase (red, thick)
cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
cv2.circle(vis, (int(center_x), int(center_y)), 5, (255, 0, 255), -1)

# Add text
cv2.putText(vis, f"VASE: {vase['class']}", (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.putText(vis, f"{width:.0f}x{height:.0f}px", (int(x1), int(y2) + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(vis, f"Est: {estimated_height_m:.2f}m", (int(x1), int(y2) + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

vis_path = Path("yolo_detection_visualization.png")
cv2.imwrite(str(vis_path), vis)
print(f"✓ Saved: {vis_path}")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("✅ VASE DETECTION COMPLETE")
print("=" * 80)

print(f"\n📦 Detection Summary:")
print(f"  - Object: {vase['class']} (YOLO class)")
print(f"  - Confidence: {vase['confidence']:.2f}")
print(f"  - Center: ({center_x:.0f}, {center_y:.0f})")
print(f"  - Size: {width:.0f} x {height:.0f} pixels")
print(f"  - Rotation: {rotation:.1f}°")
print(f"  - Estimated height: {estimated_height_m:.2f} meters")

print(f"\n📊 Recommended 3D Injection Parameters:")
print(f"  VASE_SCALE = {estimated_height_m / 100:.4f}  # Assuming vase.obj is ~100 units tall")
print(f"  VASE_ROTATION_Z = {rotation:.1f}  # Degrees around Z-axis")

print(f"\n📁 Output Files:")
print(f"  - {OUTPUT_JSON}")
print(f"  - {vis_path}")

print("\n" + "=" * 80)
