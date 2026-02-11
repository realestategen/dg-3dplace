"""
Table Detection & Gaussian Identification for Object Placement

Target Object Prompt: "A round table separated into 4 parts with blue and white boards"

This script:
1. Loads the room.ckpt 3DGS scene
2. Generates multi-angle cameras around the scene center
3. Renders the scene with gsplat from each viewpoint
4. Uses YOLO to detect the target table in 2D
5. Projects Gaussians to find which ones belong to the table
6. Colors identified table Gaussians bright red
7. Saves modified checkpoint for viewing
"""

import math
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
import gsplat
from gsplat import rasterization

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
TARGET_PROMPT = "A round table separated into 4 parts with blue and white boards"
CKPT_PATH = "room.ckpt"
OUTPUT_PATH = "room_table_highlighted.ckpt"
RENDER_W, RENDER_H = 1280, 720
NUM_CAMERAS = 5
FOV_DEG = 60.0
ORBIT_SCALE = 0.04          # fraction of scene extent for orbit radius
CAMERA_HEIGHT_OFFSET = 0.0  # keep horizontal view
OPACITY_THRESHOLD = 0.1
HEIGHT_TOLERANCE = 0.15      # for surface filtering
DEVICE = "cuda"

# SH DC constant
C0 = 0.28209479177387814

# YOLO table-like class names
TABLE_CLASSES = ["dining table", "table", "desk", "bench", "couch", "bed"]


# ══════════════════════════════════════════════════════════════════════
# SceneCamera
# ══════════════════════════════════════════════════════════════════════
class SceneCamera:
    """Unified camera for gsplat rendering & Gaussian projection.

    Handles OpenGL → OpenCV conversion consistently.
    All downstream code uses camera.w2c (OpenCV convention) and camera.project().
    """

    def __init__(self, position, wxyz, fov_rad, width, height):
        self.position = np.array(position, dtype=np.float64)
        self.wxyz = np.array(wxyz, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.fov_rad = float(fov_rad)

        # Intrinsics from vertical FOV
        self.fy = (height / 2) / np.tan(fov_rad / 2)
        self.fx = self.fy  # square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Camera-to-World (OpenGL: -Z forward, Y up)
        quat_xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
        rot = R.from_quat(quat_xyzw).as_matrix()
        self.c2w = np.eye(4, dtype=np.float64)
        self.c2w[:3, :3] = rot
        self.c2w[:3, 3] = self.position

        # World-to-Camera — OpenCV convention (+Z forward, Y down)
        w2c_gl = np.linalg.inv(self.c2w)
        self.w2c = w2c_gl.copy()
        self.w2c[1, :] *= -1
        self.w2c[2, :] *= -1

    def project(self, points_3d):
        """Project world points → 2D pixel coordinates.

        Returns:
            u, v  – pixel coordinates (invalid → -1)
            z     – depth in OpenCV space (positive = in front)
            valid – boolean mask
        """
        pts = np.asarray(points_3d, dtype=np.float64)
        N = len(pts)
        pts_h = np.column_stack([pts, np.ones(N)])

        cam_pts = (self.w2c @ pts_h.T).T  # (N, 4)
        x, y, z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]

        valid = z > 0.1

        u = np.full(N, -1.0)
        v = np.full(N, -1.0)
        u[valid] = self.fx * x[valid] / z[valid] + self.cx
        v[valid] = self.fy * y[valid] / z[valid] + self.cy

        return u, v, z, valid

    def get_K(self):
        """3×3 intrinsic matrix."""
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float64,
        )


# ══════════════════════════════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════════════════════════════
def render_gaussians(means, scales, quats, features_dc, opacities, camera, device=DEVICE):
    """Render Gaussians using gsplat from a SceneCamera."""
    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    quats_t = torch.tensor(quats, dtype=torch.float32, device=device)

    fdc = features_dc.copy()
    if fdc.ndim == 3:
        fdc = fdc.squeeze(1)
    colors_rgb = np.clip(C0 * fdc + 0.5, 0, 1)
    colors_t = torch.tensor(colors_rgb, dtype=torch.float32, device=device)

    ops = opacities.copy().squeeze()
    if ops.min() < 0:
        ops = 1 / (1 + np.exp(-ops))
    opacities_t = torch.tensor(ops, dtype=torch.float32, device=device)

    viewmat = torch.tensor(camera.w2c, dtype=torch.float32, device=device)
    K = torch.tensor(camera.get_K(), dtype=torch.float32, device=device)

    renders, alphas, meta = rasterization(
        means=means_t,
        quats=quats_t / quats_t.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales_t),
        opacities=opacities_t,
        colors=colors_t,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=camera.width,
        height=camera.height,
        sh_degree=None,
        backgrounds=torch.ones(1, 3, device=device),
    )

    rgb = renders[0].cpu().numpy()
    return np.clip(rgb, 0, 1), alphas[0].cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════
def make_camera_from_config(scene_center, orbit_radius, height_offset, azimuth,
                            cos_axis, sin_axis, fixed_axis, world_up_vec, fov_rad, w, h):
    """Build a SceneCamera from a config.
    cos_axis/sin_axis/fixed_axis are 0=X, 1=Y, 2=Z.
    The fixed_axis gets scene_center[fixed_axis] + height_offset.
    The other two get scene_center[axis] + orbit_radius * cos/sin(azimuth).
    """
    pos = np.zeros(3)
    pos[cos_axis]   = scene_center[cos_axis]   + orbit_radius * math.cos(azimuth)
    pos[sin_axis]   = scene_center[sin_axis]   + orbit_radius * math.sin(azimuth)
    pos[fixed_axis] = scene_center[fixed_axis] + height_offset

    forward = scene_center - pos
    forward = forward / np.linalg.norm(forward)

    wup = np.array(world_up_vec, dtype=np.float64)
    right = np.cross(forward, wup)
    if np.linalg.norm(right) < 1e-6:
        # Fallback if forward is parallel to world_up
        fallback = np.array([0.0, 0.0, 1.0]) if wup[2] < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, fallback)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    rot_matrix = np.column_stack([right, up, -forward])
    rot_obj = R.from_matrix(rot_matrix)
    quat_xyzw = rot_obj.as_quat()
    wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return SceneCamera(position=pos, wxyz=wxyz, fov_rad=fov_rad, width=w, height=h)


def main():
    print(f"Target object: {TARGET_PROMPT}\n")

    # ── 1. Load checkpoint ──────────────────────────────────────────
    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    state = ckpt["pipeline"]
    means = state["_model.means"].numpy()
    features_dc = state["_model.features_dc"].numpy()
    scales = state["_model.scales"].numpy()
    quats = state["_model.quats"].numpy()
    opacities_raw = state["_model.opacities"].numpy()

    if features_dc.ndim == 3:
        features_dc_squeezed = features_dc.squeeze(1)
    else:
        features_dc_squeezed = features_dc

    opacities = (1 / (1 + np.exp(-opacities_raw))).squeeze()

    print(f"Loaded {len(means):,} Gaussians")
    print(f"Position range: {means.min(axis=0)} to {means.max(axis=0)}")
    print(f"Scene center: {means.mean(axis=0)}")

    # ── 2. Scene stats ──────────────────────────────────────────────
    vis_mask = opacities > OPACITY_THRESHOLD
    vis_means = means[vis_mask]
    scene_center = vis_means.mean(axis=0)
    scene_extent = vis_means.max(axis=0) - vis_means.min(axis=0)

    print(f"Scene center:  {np.round(scene_center, 3)}")
    print(f"Scene extent:  {np.round(scene_extent, 3)}")

    # ── 3. Generate cameras — B06 config: orbit XY, fix Z, up=[0,0,1] ──
    orbit_radius = float(np.linalg.norm(scene_extent)) * ORBIT_SCALE
    camera_height_offset = 0.3

    print(f"Orbit radius:  {orbit_radius:.3f}")

    azimuth_angles = np.linspace(0, 2 * math.pi, NUM_CAMERAS, endpoint=False)
    fov_rad = math.radians(FOV_DEG)

    cameras = []
    for i, azimuth in enumerate(azimuth_angles):
        cam = make_camera_from_config(
            scene_center, orbit_radius, camera_height_offset, azimuth,
            cos_axis=0, sin_axis=1, fixed_axis=2,  # orbit XY, fix Z
            world_up_vec=[0, 0, 1],
            fov_rad=fov_rad, w=RENDER_W, h=RENDER_H,
        )
        cameras.append(cam)
        angle_deg = math.degrees(azimuth)
        print(f"  Camera {i + 1}: azimuth={angle_deg:.0f}°, pos={np.round(cam.position, 3)}")

    print(f"Generated {len(cameras)} cameras around scene center")

    # ── 4. Render all camera angles ─────────────────────────────────
    print(f"\n--- Rendering ({gsplat.__version__}) ---")
    rendered_images = []
    for i, cam in enumerate(cameras):
        print(f"  Rendering camera {i + 1}/{len(cameras)}...")
        img, alpha = render_gaussians(means, scales, quats, features_dc, opacities_raw, cam)
        rendered_images.append(img)
        Image.fromarray((img * 255).astype(np.uint8)).save(f"camera_view_{i}.png")
        coverage = 100 * (alpha > 0.5).sum() / alpha.size
        print(f"    alpha coverage: {coverage:.1f}%")

    print(f"Saved {len(cameras)} camera views.")

    # Save grid of renders
    ncols = min(3, len(rendered_images))
    nrows = math.ceil(len(rendered_images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    for idx, img in enumerate(rendered_images):
        r, c = divmod(idx, ncols)
        axes[r][c].imshow(img)
        axes[r][c].set_title(f"Camera {idx + 1} — {math.degrees(azimuth_angles[idx]):.0f}°")
        axes[r][c].axis("off")
    for idx in range(len(rendered_images), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    plt.suptitle("Multi-Angle Renders of Scene Center", fontsize=14)
    plt.tight_layout()
    plt.savefig("renders_grid.png", dpi=150)
    plt.close()
    print("Saved renders_grid.png")

    # ── 5. YOLO detection ───────────────────────────────────────────
    print("\n--- YOLO detection ---")
    model = YOLO("yolov8n.pt")

    best_det = None
    best_cam_idx = None
    all_detections = {}

    for i, img in enumerate(rendered_images):
        img_path = f"camera_view_{i}.png"
        results = model(img_path, verbose=False)

        dets = []
        for r_res in results:
            for box in r_res.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                det = {"class": cls_name, "confidence": conf, "bbox": (x1, y1, x2, y2)}
                dets.append(det)

                if cls_name.lower() in TABLE_CLASSES:
                    if best_det is None or conf > best_det["confidence"]:
                        best_det = det
                        best_cam_idx = i

        all_detections[i] = dets
        det_summary = ", ".join(f"{d['class']}({d['confidence']:.2f})" for d in dets) or "nothing"
        print(f"  Camera {i + 1}: {det_summary}")

    print(f"\n{'=' * 50}")
    if best_det:
        print(f"Best table detection: Camera {best_cam_idx + 1}")
        print(f"  {best_det['class']} (conf={best_det['confidence']:.2f}), bbox={best_det['bbox']}")
    else:
        print("No table detected in any view. Will use manual bbox fallback.")

    # Save YOLO annotated grid
    ncols = min(3, len(rendered_images))
    nrows = math.ceil(len(rendered_images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_2d(np.array(axes).reshape(nrows, ncols))
    for i in range(len(rendered_images)):
        r, c = divmod(i, ncols)
        results = model(f"camera_view_{i}.png", verbose=False)
        axes[r][c].imshow(results[0].plot())
        marker = " * BEST" if i == best_cam_idx else ""
        axes[r][c].set_title(f"Camera {i + 1}{marker}")
        axes[r][c].axis("off")
    for idx in range(len(rendered_images), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    plt.suptitle("YOLO Detections on All Camera Angles", fontsize=14)
    plt.tight_layout()
    plt.savefig("yolo_detections_grid.png", dpi=150)
    plt.close()
    print("Saved yolo_detections_grid.png")

    # ── 6. Select best camera & bbox ────────────────────────────────
    if best_det and best_cam_idx is not None:
        camera = cameras[best_cam_idx]
        rendered_img = rendered_images[best_cam_idx]
        TABLE_BBOX = best_det["bbox"]
        print(f"\nUsing Camera {best_cam_idx + 1}")
        print(f"Table: {best_det['class']} (conf={best_det['confidence']:.2f})")
        print(f"Bounding box: {TABLE_BBOX}")
    else:
        camera = cameras[0]
        rendered_img = rendered_images[0]
        TABLE_BBOX = (200, 250, 450, 400)
        best_cam_idx = 0
        print("\nNo table detected — using Camera 1 with manual bbox")
        print(f"Manual bbox: {TABLE_BBOX}")

    Image.fromarray((rendered_img * 255).astype(np.uint8)).save("camera_view.png")
    print("Saved selected view as camera_view.png")

    # Save bbox visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rendered_img)
    x1, y1, x2, y2 = TABLE_BBOX
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)
    ax.set_title(f"Table Detection: {TABLE_BBOX}")
    ax.axis("off")
    plt.savefig("table_bbox.png", dpi=150)
    plt.close()
    print("Saved table_bbox.png")

    # ── 7. Identify table Gaussians ─────────────────────────────────
    print("\n--- Identifying table Gaussians ---")
    u_all, v_all, z_all, valid_all = camera.project(means)

    x1, y1, x2, y2 = TABLE_BBOX
    in_table_bbox = (
        valid_all
        & (u_all >= x1)
        & (u_all <= x2)
        & (v_all >= y1)
        & (v_all <= y2)
        & (opacities > OPACITY_THRESHOLD)
    )

    table_indices = np.where(in_table_bbox)[0]
    print(f"Gaussians in table bbox: {len(table_indices):,} / {len(means):,}")

    # Analyze table Gaussians
    table_means = means[table_indices]
    table_heights = table_means[:, 1]

    print(f"Table Gaussian positions:")
    print(f"  X range: {table_means[:, 0].min():.2f} to {table_means[:, 0].max():.2f}")
    print(f"  Y range: {table_means[:, 1].min():.2f} to {table_means[:, 1].max():.2f}")
    print(f"  Z range: {table_means[:, 2].min():.2f} to {table_means[:, 2].max():.2f}")

    # Height histogram
    plt.figure(figsize=(10, 4))
    plt.hist(table_heights, bins=50)
    plt.xlabel("Y (height)")
    plt.ylabel("Count")
    plt.title("Height Distribution of Table Gaussians")
    plt.savefig("table_height_hist.png", dpi=150)
    plt.close()
    print("Saved table_height_hist.png")

    # Surface filtering (optional)
    hist, bins = np.histogram(table_heights, bins=50)
    peak_idx = np.argmax(hist)
    surface_height = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    print(f"Detected table surface height: {surface_height:.2f}")

    surface_mask = np.abs(table_means[:, 1] - surface_height) < HEIGHT_TOLERANCE
    surface_indices = table_indices[surface_mask]
    print(f"Table surface Gaussians: {len(surface_indices):,}")

    # Use all table indices (change to surface_indices for just the top)
    SELECTED_INDICES = table_indices
    print(f"\nWill color {len(SELECTED_INDICES):,} Gaussians red")

    # ── 8. Color table Gaussians red ────────────────────────────────
    print("\n--- Modifying checkpoint ---")
    ckpt_modified = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_mod = ckpt_modified["pipeline"]

    features_dc_mod = state_mod["_model.features_dc"].clone()
    print(f"features_dc shape: {features_dc_mod.shape}")

    red_color = np.array([1.0, 0.0, 0.0])
    red_sh = (red_color - 0.5) / C0
    red_sh_tensor = torch.tensor(red_sh, dtype=features_dc_mod.dtype)
    print(f"Red color in SH space: {red_sh}")

    if features_dc_mod.ndim == 3:  # Shape (N, 1, 3)
        features_dc_mod[SELECTED_INDICES, 0, :] = red_sh_tensor
    else:  # Shape (N, 3)
        features_dc_mod[SELECTED_INDICES, :] = red_sh_tensor

    state_mod["_model.features_dc"] = features_dc_mod
    print(f"Modified {len(SELECTED_INDICES):,} Gaussians to red")

    # ── 9. Save ─────────────────────────────────────────────────────
    torch.save(ckpt_modified, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")

    # ── 10. Verification render ─────────────────────────────────────
    print("\n--- Verification render ---")
    features_dc_viz = features_dc.copy()
    if features_dc_viz.ndim == 3:
        features_dc_viz = features_dc_viz.squeeze(1)

    colors_viz = np.clip(C0 * features_dc_viz + 0.5, 0, 1)
    colors_viz[SELECTED_INDICES] = [1.0, 0.0, 0.0]

    features_dc_mod_viz = (colors_viz - 0.5) / C0
    if features_dc.ndim == 3:
        features_dc_mod_viz = features_dc_mod_viz[:, np.newaxis, :]

    rendered_mod, _ = render_gaussians(
        means, scales, quats, features_dc_mod_viz, opacities_raw, camera
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(rendered_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(rendered_mod)
    axes[1].set_title(f"Table Highlighted (Red) - {len(SELECTED_INDICES):,} Gaussians")
    axes[1].axis("off")
    plt.suptitle(f"Target: {TARGET_PROMPT}", fontsize=12)
    plt.tight_layout()
    plt.savefig("verification.png", dpi=150)
    plt.close()
    print("Saved verification.png")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTarget Object: {TARGET_PROMPT}")
    print(f"\nSelected Camera: {best_cam_idx + 1} / {len(cameras)}")
    print(f"  Position:  {np.round(camera.position, 3)}")
    print(f"  FOV:       {np.degrees(camera.fov_rad):.1f} deg")
    print(f"  Resolution: {camera.width} x {camera.height}")
    print(f"\nTable Detection:")
    print(f"  2D Bounding Box: {TABLE_BBOX}")
    print(f"  Gaussians identified: {len(SELECTED_INDICES):,}")
    if len(SELECTED_INDICES) > 0:
        sel_means = means[SELECTED_INDICES]
        print(f"  3D Center: {np.round(sel_means.mean(axis=0), 3)}")
        print(f"  3D Bounds: {np.round(sel_means.min(axis=0), 3)} -> {np.round(sel_means.max(axis=0), 3)}")
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"\nTo view: python view_room.py {OUTPUT_PATH} --port 8080")


if __name__ == "__main__":
    main()