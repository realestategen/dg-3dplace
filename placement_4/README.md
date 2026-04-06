# placement_4: PnP-Based Object Placement

This folder contains a PnP-first placement pipeline for inserting an OBJ mesh into a 3D Gaussian Splatting room checkpoint.

## What this script does

`detection_optimized.py` now follows this flow:

1. Detects the target class (default: `vase`) with YOLO.
2. Uses `OBJECT_3D` dimensions (`width, depth, height`) to build 3D cuboid keypoints.
3. Solves object pose from 2D bbox keypoints with OpenCV PnP.
4. Fits a support plane from projected scene Gaussians in the bbox bottom region (no depth-map unprojection).
5. Refines pose for upright orientation and floor/surface contact.
6. Scales + rotates + translates sampled OBJ points, converts them to Gaussians, and writes a new checkpoint.

## Prerequisites

- Run from this directory: `placement_4/`
- Required files present:
  - `cupboard_room.ckpt`
  - `yolov8n.pt`
  - your object mesh, e.g. `vase.obj`
- Python deps (typical): `torch`, `numpy`, `opencv-python`, `matplotlib`, `Pillow`, `ultralytics`, `gsplat`, `scipy`, `psutil`

## Usage

### 1) Camera preview/selection

```bash
python detection_optimized.py
```

This renders camera views into a new `session_YYYYMMDD_HHMMSS/` folder.

### 2) Place object with PnP + OBJECT_3D

```bash
python detection_optimized.py <image_with_object.png> <object.obj> --cam-idx 1 --object-3d 0.22,0.22,0.36
```

Arguments:

- `image_with_object.png`: image where the object is visible (used for YOLO bbox)
- `object.obj`: OBJ mesh to insert
- `--cam-idx`: 1-based camera index used for this placement run
- `--object-3d`: real-world dimensions in meters as `width,depth,height`

Output files go to the new session folder, including:

- `room_with_object.ckpt`
- `vase_detection_bbox.png`
- `vase_pnp_keypoints.png`
- `vase_placement_verification.png`
- `detection_resource_report.txt`

## View the result

```bash
python view_room.py session_YYYYMMDD_HHMMSS/room_with_object.ckpt --port 8080
```

Use the actual session folder created by your run.

## Tips for better initial placement

- Set `--object-3d` close to the real object size; wrong dimensions produce wrong depth/scale.
- Use an image where the detected object is not heavily occluded.
- If class mismatch occurs, change `OBJECT_CLASSNAME` in `detection_optimized.py`.
