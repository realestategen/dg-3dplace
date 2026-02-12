"""
Vase Placement Pipeline — OBJ → Gaussians on Table Surface

This script:
1. Loads room_table_highlighted.ckpt to identify table gaussians (red)
2. Uses YOLO on the diffusion output (diffusion_added.png) to detect
   both vase and table, comparing their 2D bounding boxes
3. Computes the correct 3D scale for the vase relative to the known
   3D table dimensions
4. Converts vase.obj mesh → Gaussian splats (surface sampling)
5. Orients the vase vertically (OBJ Y-up → Scene Z-up)
6. Scales and positions vase gaussians on the table surface
7. Merges vase gaussians with room gaussians
8. Saves the combined scene as a new checkpoint

Scene convention:
  - Z is the vertical (up) axis
  - world_up = [0, 0, 1]
"""

import math
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
ROOM_CKPT = "room.ckpt"                       # original room
TABLE_CKPT = "room_table_highlighted.ckpt"     # with red table gaussians
VASE_OBJ = "vase.obj"
DIFFUSION_IMG = "diffusion_added.png"          # diffusion output: vase on table
OUTPUT_CKPT = "room_with_vase.ckpt"
YOLO_MODEL = "yolov8n.pt"

# SH DC constant
C0 = 0.28209479177387814

# Gaussian sampling
NUM_VASE_GAUSSIANS = 15000         # number of gaussians to create for the vase
VASE_GAUSSIAN_SCALE = -6.0         # log-scale for each vase gaussian (small splats)
VASE_OPACITY_LOGIT = 2.0           # sigmoid(2.0) ≈ 0.88 — solid
DEFAULT_VASE_COLOR = [0.65, 0.45, 0.30]  # warm terracotta/clay color

# Fallback scale ratio (vase height / table height in 2D) if YOLO fails
FALLBACK_VASE_TABLE_HEIGHT_RATIO = 0.35
FALLBACK_VASE_TABLE_WIDTH_RATIO = 0.15

# YOLO class names for detection
VASE_CLASSES = ["vase", "bottle", "cup", "wine glass", "potted plant"]
TABLE_CLASSES = ["dining table", "table", "desk", "bench", "couch", "bed"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════
# 1. Load Table Gaussians (from red-highlighted checkpoint)
# ══════════════════════════════════════════════════════════════════════
def load_table_properties(ckpt_path):
    """Load the highlighted checkpoint and extract the table gaussian properties."""
    print(f"Loading table-highlighted checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]

    means = state["_model.means"].numpy()
    features_dc = state["_model.features_dc"].numpy()
    if features_dc.ndim == 3:
        features_dc = features_dc.squeeze(1)

    # Identify red gaussians = table
    colors = np.clip(C0 * features_dc + 0.5, 0, 1)
    red_mask = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)

    table_means = means[red_mask]
    table_indices = np.where(red_mask)[0]

    table_min = table_means.min(axis=0)
    table_max = table_means.max(axis=0)
    table_center = table_means.mean(axis=0)
    table_extent = table_max - table_min

    # Table surface = top of table = max Z (Z is up)
    # Use percentile to be robust against outliers
    table_top_z = np.percentile(table_means[:, 2], 95)

    # Table surface center (XY at top Z)
    top_mask = table_means[:, 2] > (table_top_z - 0.05)
    if top_mask.sum() > 10:
        surface_center_xy = table_means[top_mask][:, :2].mean(axis=0)
    else:
        surface_center_xy = table_center[:2]

    props = {
        "count": len(table_means),
        "indices": table_indices,
        "means": table_means,
        "center": table_center,
        "min": table_min,
        "max": table_max,
        "extent": table_extent,
        "top_z": table_top_z,
        "surface_center_xy": surface_center_xy,
        "width_x": table_extent[0],
        "depth_y": table_extent[1],
        "height_z": table_extent[2],
    }

    print(f"  Table gaussians:  {props['count']:,}")
    print(f"  Table center:     {np.round(table_center, 4)}")
    print(f"  Table extent:     X={table_extent[0]:.3f}  Y={table_extent[1]:.3f}  Z={table_extent[2]:.3f}")
    print(f"  Table top (Z):    {table_top_z:.4f}")
    print(f"  Surface center:   ({surface_center_xy[0]:.4f}, {surface_center_xy[1]:.4f})")

    return props


# ══════════════════════════════════════════════════════════════════════
# 2. YOLO Detection on Diffusion Output
# ══════════════════════════════════════════════════════════════════════
def detect_vase_and_table(image_path, yolo_model_path):
    """Run YOLO on the diffusion image to find vase & table bounding boxes.

    Returns:
        vase_bbox: (x1, y1, x2, y2) or None
        table_bbox: (x1, y1, x2, y2) or None
        vase_to_table_ratios: dict with width_ratio, height_ratio
    """
    print(f"\nRunning YOLO on: {image_path}")
    model = YOLO(yolo_model_path)
    results = model(image_path, verbose=False)

    vase_det = None
    table_det = None

    all_dets = []
    for r_res in results:
        for box in r_res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det = {"class": cls_name, "confidence": conf, "bbox": (x1, y1, x2, y2)}
            all_dets.append(det)
            print(f"  Detected: {cls_name} (conf={conf:.2f}), bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

            # Best vase
            if cls_name.lower() in [c.lower() for c in VASE_CLASSES]:
                if vase_det is None or conf > vase_det["confidence"]:
                    vase_det = det

            # Best table
            if cls_name.lower() in [c.lower() for c in TABLE_CLASSES]:
                if table_det is None or conf > table_det["confidence"]:
                    table_det = det

    # Compute ratios
    ratios = {
        "width_ratio": FALLBACK_VASE_TABLE_WIDTH_RATIO,
        "height_ratio": FALLBACK_VASE_TABLE_HEIGHT_RATIO,
    }

    if vase_det and table_det:
        vx1, vy1, vx2, vy2 = vase_det["bbox"]
        tx1, ty1, tx2, ty2 = table_det["bbox"]

        vase_w = vx2 - vx1
        vase_h = vy2 - vy1
        table_w = tx2 - tx1
        table_h = ty2 - ty1

        ratios["width_ratio"] = float(vase_w / table_w) if table_w > 0 else FALLBACK_VASE_TABLE_WIDTH_RATIO
        ratios["height_ratio"] = float(vase_h / table_h) if table_h > 0 else FALLBACK_VASE_TABLE_HEIGHT_RATIO

        print(f"\n  Vase/Table width ratio:  {ratios['width_ratio']:.3f}")
        print(f"  Vase/Table height ratio: {ratios['height_ratio']:.3f}")
    else:
        missing = []
        if not vase_det:
            missing.append("vase")
        if not table_det:
            missing.append("table")
        print(f"\n  Could not detect: {', '.join(missing)} — using fallback ratios")
        print(f"  Fallback width ratio:  {ratios['width_ratio']:.3f}")
        print(f"  Fallback height ratio: {ratios['height_ratio']:.3f}")

    # Save annotated image
    _save_detection_viz(image_path, vase_det, table_det, all_dets)

    return vase_det, table_det, ratios


def _save_detection_viz(image_path, vase_det, table_det, all_dets):
    """Save visualization of YOLO detections on diffusion image."""
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)

    for det in all_dets:
        x1, y1, x2, y2 = det["bbox"]
        if det == vase_det:
            color, lw = "lime", 3
        elif det == table_det:
            color, lw = "red", 3
        else:
            color, lw = "yellow", 1

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=lw, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{det['class']} {det['confidence']:.2f}",
                color=color, fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    ax.set_title("YOLO Detections on Diffusion Output\n(Green=Vase, Red=Table, Yellow=Other)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("diffusion_yolo_detections.png", dpi=150)
    plt.close()
    print("  Saved diffusion_yolo_detections.png")


# ══════════════════════════════════════════════════════════════════════
# 3. Compute Target 3D Vase Dimensions
# ══════════════════════════════════════════════════════════════════════
def compute_vase_3d_size(table_props, ratios):
    """Compute target 3D dimensions for the vase based on table size + 2D ratios.

    The 2D image gives us the ratio of vase-to-table in visual space.
    The 3D table dimensions are known from the gaussians.
    We map:
      - 2D width ratio  → 3D XY footprint ratio
      - 2D height ratio → 3D Z (vertical) height ratio

    Returns dict with target_width, target_depth, target_height.
    """
    # The table's horizontal extent in the image roughly corresponds to
    # max(width_x, depth_y) in 3D (depending on viewing angle)
    # Use the average of X/Y for robustness
    table_horizontal_3d = (table_props["width_x"] + table_props["depth_y"]) / 2.0
    table_vertical_3d = table_props["height_z"]

    # Vase dimensions from 2D ratios
    vase_width_3d = ratios["width_ratio"] * table_horizontal_3d
    vase_height_3d = ratios["height_ratio"] * table_vertical_3d

    # Vase is roughly cylindrical → depth ≈ width
    vase_depth_3d = vase_width_3d

    # Sanity bounds: vase shouldn't be larger than half the table
    max_footprint = min(table_props["width_x"], table_props["depth_y"]) * 0.4
    if vase_width_3d > max_footprint:
        scale_down = max_footprint / vase_width_3d
        vase_width_3d *= scale_down
        vase_depth_3d *= scale_down
        print(f"  Clamped vase footprint (too large), factor={scale_down:.3f}")

    # Vase height shouldn't exceed the table height (unrealistic)
    max_height = table_vertical_3d * 0.8
    if vase_height_3d > max_height:
        vase_height_3d = max_height
        print(f"  Clamped vase height to {max_height:.4f}")

    # Minimum size guard
    min_dim = 0.02
    vase_width_3d = max(vase_width_3d, min_dim)
    vase_depth_3d = max(vase_depth_3d, min_dim)
    vase_height_3d = max(vase_height_3d, min_dim)

    target = {
        "width": vase_width_3d,   # X extent
        "depth": vase_depth_3d,   # Y extent
        "height": vase_height_3d, # Z extent (vertical)
    }

    print(f"\n  Target 3D vase dimensions:")
    print(f"    Width  (X): {target['width']:.4f}")
    print(f"    Depth  (Y): {target['depth']:.4f}")
    print(f"    Height (Z): {target['height']:.4f}")

    return target


# ══════════════════════════════════════════════════════════════════════
# 4. Load OBJ Mesh & Sample Surface Points
# ══════════════════════════════════════════════════════════════════════
def load_obj_mesh(obj_path):
    """Load vertices and triangle faces from a Wavefront OBJ file."""
    vertices = []
    faces = []

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Handle f v1 v2 v3 or f v1/vt1 v2/vt2 ... or f v1/vt1/vn1 ...
                face_verts = []
                for p in parts:
                    idx = int(p.split("/")[0]) - 1  # OBJ is 1-indexed
                    face_verts.append(idx)
                # Triangulate if polygon has more than 3 verts (fan triangulation)
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    print(f"  Loaded mesh: {len(vertices)} vertices, {len(faces)} triangles")
    return vertices, faces


def sample_points_on_mesh(vertices, faces, num_points):
    """Uniformly sample points on the surface of a triangle mesh.

    Uses area-weighted random sampling of triangles,
    then uniform barycentric coordinates within each triangle.
    """
    # Compute triangle areas
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()

    if total_area < 1e-12:
        raise ValueError("Mesh has zero surface area!")

    # Probability per triangle
    probs = areas / total_area

    # Sample triangle indices weighted by area
    tri_indices = np.random.choice(len(faces), size=num_points, p=probs)

    # Random barycentric coordinates
    r1 = np.random.rand(num_points)
    r2 = np.random.rand(num_points)
    sqrt_r1 = np.sqrt(r1)

    bary_u = 1 - sqrt_r1
    bary_v = sqrt_r1 * (1 - r2)
    bary_w = sqrt_r1 * r2

    # Interpolate positions
    p0 = vertices[faces[tri_indices, 0]]
    p1 = vertices[faces[tri_indices, 1]]
    p2 = vertices[faces[tri_indices, 2]]

    points = (bary_u[:, None] * p0 +
              bary_v[:, None] * p1 +
              bary_w[:, None] * p2)

    # Compute per-point normals (face normals)
    face_normals = cross[tri_indices]
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = face_normals / norms

    print(f"  Sampled {num_points} points on mesh surface")
    return points.astype(np.float64), normals.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════
# 5. Transform Vase Points → Scene Space
# ══════════════════════════════════════════════════════════════════════
def transform_vase_to_scene(points, normals, target_size, table_props):
    """Transform vase sample points from OBJ space to 3D scene space.

    OBJ space:  Y is up, centered roughly around origin in XZ
    Scene space: Z is up

    Steps:
      1. Center the vase at origin
      2. Rotate: OBJ Y-up → Scene Z-up (rotate -90° around X)
      3. Scale to target dimensions
      4. Translate to table surface center
    """
    pts = points.copy()
    nrm = normals.copy()

    # ── Step 1: Center at origin ──
    obj_min = pts.min(axis=0)
    obj_max = pts.max(axis=0)
    obj_center = (obj_min + obj_max) / 2.0
    # Center XZ, but put bottom at Y=0
    pts[:, 0] -= obj_center[0]  # center X
    pts[:, 2] -= obj_center[2]  # center Z
    pts[:, 1] -= obj_min[1]      # bottom at Y=0

    obj_extent = obj_max - obj_min  # before centering
    print(f"  OBJ extent: X={obj_extent[0]:.2f}, Y(height)={obj_extent[1]:.2f}, Z={obj_extent[2]:.2f}")

    # ── Step 2: Rotate Y-up → Z-up ──
    # Rotation: (X, Y, Z)_obj → (X, -Z, Y)_scene
    # This maps OBJ Y → Scene Z (both "up")
    # and OBJ Z → Scene -Y (so it faces correctly)
    pts_scene = np.column_stack([pts[:, 0], -pts[:, 2], pts[:, 1]])
    nrm_scene = np.column_stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]])

    # ── Step 3: Scale to target dimensions ──
    # After rotation, scene extent:
    #   X = OBJ X width
    #   Y = OBJ Z depth
    #   Z = OBJ Y height (up)
    current_extent = pts_scene.max(axis=0) - pts_scene.min(axis=0)
    current_extent = np.maximum(current_extent, 1e-8)

    scale_factors = np.array([
        target_size["width"] / current_extent[0],
        target_size["depth"] / current_extent[1],
        target_size["height"] / current_extent[2],
    ])

    # Use uniform scale to preserve aspect ratio
    # Pick the minimum to ensure it fits within target bounds
    uniform_scale = scale_factors.min()
    print(f"  Scale factors: {scale_factors}")
    print(f"  Using uniform scale: {uniform_scale:.6f}")

    pts_scene *= uniform_scale
    # normals don't need scaling (they're directions)

    # ── Step 4: Position on table surface ──
    # After scaling, put vase bottom at table top Z
    vase_bottom_z = pts_scene[:, 2].min()
    pts_scene[:, 2] -= vase_bottom_z  # bottom at Z=0
    pts_scene[:, 2] += table_props["top_z"]  # lift to table surface

    # Center XY on table surface center
    vase_center_xy = pts_scene[:, :2].mean(axis=0)
    pts_scene[:, 0] += table_props["surface_center_xy"][0] - vase_center_xy[0]
    pts_scene[:, 1] += table_props["surface_center_xy"][1] - vase_center_xy[1]

    final_min = pts_scene.min(axis=0)
    final_max = pts_scene.max(axis=0)
    print(f"  Final vase position:")
    print(f"    X: {final_min[0]:.4f} to {final_max[0]:.4f}")
    print(f"    Y: {final_min[1]:.4f} to {final_max[1]:.4f}")
    print(f"    Z: {final_min[2]:.4f} to {final_max[2]:.4f}")
    print(f"    Center: {pts_scene.mean(axis=0)}")

    return pts_scene, nrm_scene


# ══════════════════════════════════════════════════════════════════════
# 6. Create Vase Gaussians
# ══════════════════════════════════════════════════════════════════════
def create_vase_gaussians(points, normals, target_size, table_props, color=None):
    """Create Gaussian splat parameters for the vase from sampled points.

    Returns dict with means, scales, quats, features_dc, features_rest, opacities
    matching the checkpoint format.
    """
    if color is None:
        color = np.array(DEFAULT_VASE_COLOR)
    else:
        color = np.array(color)

    N = len(points)
    print(f"\n  Creating {N} vase Gaussians...")

    # ── Means ──
    means = torch.tensor(points, dtype=torch.float32)

    # ── Scales ──
    # Adaptive scale based on vase size and gaussian count:
    # each gaussian should cover roughly (total_volume / N) ^ (1/3)
    vase_vol = target_size["width"] * target_size["depth"] * target_size["height"]
    adaptive_radius = (vase_vol / N) ** (1.0 / 3.0) * 1.5
    log_scale = math.log(max(adaptive_radius, 1e-7))
    print(f"  Adaptive gaussian radius: {adaptive_radius:.6f} (log: {log_scale:.3f})")

    scales = torch.full((N, 3), log_scale, dtype=torch.float32)

    # ── Quaternions ──
    # Identity quaternion (wxyz) — the vase is already correctly oriented
    quats = torch.zeros(N, 4, dtype=torch.float32)
    quats[:, 0] = 1.0  # w=1, x=y=z=0

    # ── Colors (SH DC) ──
    # Convert RGB [0,1] to SH DC space: sh = (color - 0.5) / C0
    sh_color = (color - 0.5) / C0
    features_dc = torch.tensor(sh_color, dtype=torch.float32).unsqueeze(0).expand(N, -1)
    features_dc = features_dc.unsqueeze(1)  # shape (N, 1, 3) to match checkpoint

    # ── Features rest (higher-order SH) ──
    # Match the shape from the checkpoint — typically (N, K, 3) where K=15 for SH degree 3
    features_rest = torch.zeros(N, 15, 3, dtype=torch.float32)

    # ── Opacities (logit space) ──
    opacities = torch.full((N, 1), VASE_OPACITY_LOGIT, dtype=torch.float32)

    gaussians = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "opacities": opacities,
    }

    print(f"  Vase gaussians created:")
    print(f"    means:        {means.shape}")
    print(f"    scales:       {scales.shape} (log-scale={log_scale:.3f})")
    print(f"    quats:        {quats.shape}")
    print(f"    features_dc:  {features_dc.shape}")
    print(f"    features_rest:{features_rest.shape}")
    print(f"    opacities:    {opacities.shape} (logit={VASE_OPACITY_LOGIT:.1f})")

    return gaussians


# ══════════════════════════════════════════════════════════════════════
# 7. Merge Vase into Room Checkpoint
# ══════════════════════════════════════════════════════════════════════
def merge_and_save(room_ckpt_path, vase_gaussians, output_path):
    """Load the original room checkpoint, append vase gaussians, save."""
    print(f"\nLoading original room: {room_ckpt_path}")
    ckpt = torch.load(room_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]

    room_n = state["_model.means"].shape[0]
    vase_n = vase_gaussians["means"].shape[0]
    total_n = room_n + vase_n
    print(f"  Room gaussians:  {room_n:,}")
    print(f"  Vase gaussians:  {vase_n:,}")
    print(f"  Total:           {total_n:,}")

    # Get features_rest shape from room
    room_rest = state["_model.features_rest"]
    rest_channels = room_rest.shape[1] if room_rest.ndim == 3 else 0

    # Adjust vase features_rest to match room shape
    if room_rest.ndim == 3 and rest_channels != vase_gaussians["features_rest"].shape[1]:
        vase_gaussians["features_rest"] = torch.zeros(
            vase_n, rest_channels, 3, dtype=torch.float32
        )
        print(f"  Adjusted features_rest to {rest_channels} channels")

    # Adjust features_dc shape to match room
    room_dc = state["_model.features_dc"]
    if room_dc.ndim != vase_gaussians["features_dc"].ndim:
        if room_dc.ndim == 2:
            vase_gaussians["features_dc"] = vase_gaussians["features_dc"].squeeze(1)
        else:
            vase_gaussians["features_dc"] = vase_gaussians["features_dc"].unsqueeze(1)
        print(f"  Adjusted features_dc shape to match room: {vase_gaussians['features_dc'].shape}")

    # Adjust opacities shape
    room_op = state["_model.opacities"]
    if room_op.ndim == 1:
        vase_gaussians["opacities"] = vase_gaussians["opacities"].squeeze(-1)

    # Concatenate all gaussian fields
    state["_model.means"] = torch.cat([
        state["_model.means"], vase_gaussians["means"]
    ], dim=0)

    state["_model.scales"] = torch.cat([
        state["_model.scales"], vase_gaussians["scales"]
    ], dim=0)

    state["_model.quats"] = torch.cat([
        state["_model.quats"], vase_gaussians["quats"]
    ], dim=0)

    state["_model.features_dc"] = torch.cat([
        state["_model.features_dc"], vase_gaussians["features_dc"]
    ], dim=0)

    state["_model.features_rest"] = torch.cat([
        state["_model.features_rest"], vase_gaussians["features_rest"]
    ], dim=0)

    state["_model.opacities"] = torch.cat([
        state["_model.opacities"], vase_gaussians["opacities"]
    ], dim=0)

    # Save
    torch.save(ckpt, output_path)
    print(f"\n  Saved combined scene: {output_path}")
    print(f"  Total gaussians: {total_n:,}")

    return total_n


# ══════════════════════════════════════════════════════════════════════
# 8. Verification Render
# ══════════════════════════════════════════════════════════════════════
def verification_render(output_ckpt, table_props):
    """Render verification views of the combined scene."""
    from detection import SceneCamera, render_gaussians, make_camera_from_config

    print("\n--- Verification Render ---")
    ckpt = torch.load(output_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]

    means = state["_model.means"].numpy()
    scales = state["_model.scales"].numpy()
    quats = state["_model.quats"].numpy()
    features_dc = state["_model.features_dc"].numpy()
    opacities_raw = state["_model.opacities"].numpy()

    # Camera looking at table center
    scene_center = table_props["center"]
    fov_rad = math.radians(60.0)

    # Generate 4 views around the table
    angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    orbit_r = 1.5  # close to table

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, azimuth in enumerate(angles):
        cam = make_camera_from_config(
            scene_center, orbit_r, 0.5, azimuth,
            cos_axis=0, sin_axis=1, fixed_axis=2,
            world_up_vec=[0, 0, 1],
            fov_rad=fov_rad, w=1280, h=720,
        )
        img, _ = render_gaussians(means, scales, quats, features_dc, opacities_raw, cam)
        axes[i].imshow(img)
        axes[i].set_title(f"View {i + 1} — {math.degrees(azimuth):.0f}°")
        axes[i].axis("off")

        # Also save individual
        Image.fromarray((img * 255).astype(np.uint8)).save(f"vase_placed_view_{i}.png")

    plt.suptitle("Vase Placed on Table — Multi-Angle Verification", fontsize=14)
    plt.tight_layout()
    plt.savefig("vase_placement_verification.png", dpi=150)
    plt.close()
    print("  Saved vase_placement_verification.png")
    print("  Saved 4 individual view images")


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  VASE PLACEMENT PIPELINE")
    print("  OBJ → Scaled Gaussians → Table Surface → Combined Checkpoint")
    print("=" * 70)

    # ── Step 1: Get table properties from highlighted checkpoint ──
    print("\n" + "─" * 60)
    print("STEP 1: Load Table Properties")
    print("─" * 60)
    table_props = load_table_properties(TABLE_CKPT)

    # ── Step 2: YOLO detection on diffusion output ──
    print("\n" + "─" * 60)
    print("STEP 2: YOLO Detection on Diffusion Output")
    print("─" * 60)
    vase_det, table_det, ratios = detect_vase_and_table(DIFFUSION_IMG, YOLO_MODEL)

    # ── Step 3: Compute target 3D vase size ──
    print("\n" + "─" * 60)
    print("STEP 3: Compute Target 3D Vase Dimensions")
    print("─" * 60)
    target_size = compute_vase_3d_size(table_props, ratios)

    # ── Step 4: Load vase mesh and sample surface ──
    print("\n" + "─" * 60)
    print("STEP 4: Load Vase Mesh & Sample Surface Points")
    print("─" * 60)
    vertices, faces = load_obj_mesh(VASE_OBJ)
    points, normals = sample_points_on_mesh(vertices, faces, NUM_VASE_GAUSSIANS)

    # ── Step 5: Transform vase to scene space ──
    print("\n" + "─" * 60)
    print("STEP 5: Transform Vase → Scene Space")
    print("─" * 60)
    scene_points, scene_normals = transform_vase_to_scene(
        points, normals, target_size, table_props
    )

    # ── Step 6: Create vase gaussians ──
    print("\n" + "─" * 60)
    print("STEP 6: Create Vase Gaussians")
    print("─" * 60)
    vase_gaussians = create_vase_gaussians(
        scene_points, scene_normals, target_size, table_props
    )

    # ── Step 7: Merge and save ──
    print("\n" + "─" * 60)
    print("STEP 7: Merge Vase into Room & Save")
    print("─" * 60)
    total = merge_and_save(ROOM_CKPT, vase_gaussians, OUTPUT_CKPT)

    # ── Step 8: Verification ──
    print("\n" + "─" * 60)
    print("STEP 8: Verification Renders")
    print("─" * 60)
    try:
        verification_render(OUTPUT_CKPT, table_props)
    except Exception as e:
        print(f"  Verification render failed (non-critical): {e}")
        print("  You can still view with: python view_room.py room_with_vase.ckpt")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Table properties:")
    print(f"    Gaussians:  {table_props['count']:,}")
    print(f"    3D extent:  X={table_props['width_x']:.3f}  Y={table_props['depth_y']:.3f}  Z={table_props['height_z']:.3f}")
    print(f"    Top surface Z: {table_props['top_z']:.4f}")
    print(f"\n  YOLO 2D ratios (vase/table):")
    print(f"    Width:  {ratios['width_ratio']:.3f}")
    print(f"    Height: {ratios['height_ratio']:.3f}")
    print(f"\n  Vase 3D target:")
    print(f"    Width:  {target_size['width']:.4f}")
    print(f"    Depth:  {target_size['depth']:.4f}")
    print(f"    Height: {target_size['height']:.4f}")
    print(f"\n  Output: {OUTPUT_CKPT} ({total:,} total gaussians)")
    print(f"\n  View with: python view_room.py {OUTPUT_CKPT} --port 8080")


if __name__ == "__main__":
    main()
