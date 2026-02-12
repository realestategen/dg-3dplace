

import math
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
import gsplat
from gsplat import rasterization

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

import os
import datetime

CKPT_PATH = "room.ckpt"
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

# Configurable object class for detection
OBJECT_CLASSNAME = "chair"  # Change to "chair" or any other class as needed

# Session folder for outputs
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = f"session_{SESSION_TIMESTAMP}"
os.makedirs(SESSION_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SESSION_DIR, "room_with_object.ckpt")


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



# ══════════════════════════════════════════════════════════════════════
# User-driven camera selection and rendering
# ══════════════════════════════════════════════════════════════════════
def select_camera_and_render():
    print("\n--- Camera Selection & Rendering ---")
    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    means = state["_model.means"].numpy()
    features_dc = state["_model.features_dc"].numpy()
    scales = state["_model.scales"].numpy()
    quats = state["_model.quats"].numpy()
    opacities_raw = state["_model.opacities"].numpy()
    opacities = (1 / (1 + np.exp(-opacities_raw))).squeeze()

    vis_mask = opacities > OPACITY_THRESHOLD
    vis_means = means[vis_mask]
    scene_center = vis_means.mean(axis=0)
    scene_extent = vis_means.max(axis=0) - vis_means.min(axis=0)

    orbit_radius = float(np.linalg.norm(scene_extent)) * ORBIT_SCALE
    camera_height_offset = 0.3
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

    # Render all camera angles
    rendered_images = []
    for i, cam in enumerate(cameras):
        print(f"  Rendering camera {i + 1}/{len(cameras)}...")
        img, alpha = render_gaussians(means, scales, quats, features_dc, opacities_raw, cam)
        rendered_images.append(img)
        Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(SESSION_DIR, f"camera_view_{i}.png"))
        coverage = 100 * (alpha > 0.5).sum() / alpha.size
        print(f"    alpha coverage: {coverage:.1f}%")

    print(f"Saved {len(cameras)} camera views.")

    # Save grid of renders
    ncols = min(3, len(rendered_images))
    nrows = math.ceil(len(rendered_images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_2d(np.array(axes).reshape(nrows, ncols))
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
    plt.savefig(os.path.join(SESSION_DIR, "renders_grid.png"), dpi=150)
    plt.close()
    print(f"Saved renders_grid.png in {SESSION_DIR}")

    # User selects camera
    cam_idx = int(input(f"Select camera index (1-{len(cameras)}): ")) - 1
    if cam_idx < 0 or cam_idx >= len(cameras):
        print("Invalid index. Defaulting to 0.")
        cam_idx = 0
    selected_img = rendered_images[cam_idx]
    Image.fromarray((selected_img * 255).astype(np.uint8)).save(os.path.join(SESSION_DIR, "selected_camera_view.png"))
    print(f"Saved selected view as selected_camera_view.png (camera {cam_idx + 1}) in {SESSION_DIR}")

# ══════════════════════════════════════════════════════════════════════
# Vase addition via unprojection
# ══════════════════════════════════════════════════════════════════════


def add_object_to_scene(object_image_path, object_obj_path):
    from ultralytics import YOLO
    print("\n--- Object Detection & Highlighting ---")
    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    means = state["_model.means"].numpy()
    features_dc = state["_model.features_dc"].numpy()
    scales = state["_model.scales"].numpy()
    quats = state["_model.quats"].numpy()
    opacities_raw = state["_model.opacities"].numpy()
    opacities = (1 / (1 + np.exp(-opacities_raw))).squeeze()

    # Detect object in image
    model = YOLO("yolov8n.pt")
    results = model(object_image_path, verbose=False)
    object_det = None
    for r_res in results:
        for box in r_res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            if OBJECT_CLASSNAME.lower() in cls_name.lower():
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                object_det = {"class": cls_name, "confidence": conf, "bbox": (x1, y1, x2, y2)}
                break
        if object_det:
            break
    if not object_det:
        print(f"No {OBJECT_CLASSNAME} detected in image.")
        return
    print(f"{OBJECT_CLASSNAME.capitalize()} detected: {object_det['class']} (conf={object_det['confidence']:.2f}), bbox={object_det['bbox']}")

    # Save detection visualization
    img = Image.open(object_image_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    x1, y1, x2, y2 = object_det["bbox"]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"{OBJECT_CLASSNAME.capitalize()} Detection: {object_det['bbox']}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SESSION_DIR, f"{OBJECT_CLASSNAME}_detection_bbox.png"), dpi=150)
    plt.close()
    print(f"Saved {OBJECT_CLASSNAME}_detection_bbox.png in {SESSION_DIR}")

    # Unproject object bbox to 3D and highlight Gaussians
    cam_idx = int(input(f"Enter camera index used for {OBJECT_CLASSNAME} image (1-{NUM_CAMERAS}): ")) - 1
    if cam_idx < 0 or cam_idx >= NUM_CAMERAS:
        print("Invalid index. Defaulting to 0.")
        cam_idx = 0
    vis_mask = opacities > OPACITY_THRESHOLD
    vis_means = means[vis_mask]
    scene_center = vis_means.mean(axis=0)
    scene_extent = vis_means.max(axis=0) - vis_means.min(axis=0)
    orbit_radius = float(np.linalg.norm(scene_extent)) * ORBIT_SCALE
    camera_height_offset = 0.3
    azimuth_angles = np.linspace(0, 2 * math.pi, NUM_CAMERAS, endpoint=False)
    fov_rad = math.radians(FOV_DEG)
    cam = make_camera_from_config(
        scene_center, orbit_radius, camera_height_offset, azimuth_angles[cam_idx],
        cos_axis=0, sin_axis=1, fixed_axis=2, world_up_vec=[0, 0, 1],
        fov_rad=fov_rad, w=RENDER_W, h=RENDER_H,
    )

    u_all, v_all, z_all, valid_all = cam.project(means)
    in_object_bbox = (
        valid_all
        & (u_all >= x1)
        & (u_all <= x2)
        & (v_all >= y1)
        & (v_all <= y2)
        & (opacities > OPACITY_THRESHOLD)
    )
    object_indices = np.where(in_object_bbox)[0]
    print(f"Gaussians in {OBJECT_CLASSNAME} bbox: {len(object_indices):,} / {len(means):,}")

    # 1. Save checkpoint with red-highlighted detected gaussians (for verification)
    C0 = 0.28209479177387814
    red_color = np.array([1.0, 0.0, 0.0])
    red_sh = (red_color - 0.5) / C0
    features_dc_mod = features_dc.copy()
    if features_dc_mod.ndim == 3:
        features_dc_mod[object_indices, 0, :] = red_sh
    else:
        features_dc_mod[object_indices, :] = red_sh
    state["_model.features_dc"] = torch.tensor(features_dc_mod)
    torch.save(ckpt, os.path.join(SESSION_DIR, f"room_with_{OBJECT_CLASSNAME}_highlighted.ckpt"))
    print(f"Saved: room_with_{OBJECT_CLASSNAME}_highlighted.ckpt (red highlight only) in {SESSION_DIR}")

    # Verification render
    features_dc_viz = features_dc.copy()
    if features_dc_viz.ndim == 3:
        features_dc_viz = features_dc_viz.squeeze(1)
    colors_viz = np.clip(C0 * features_dc_viz + 0.5, 0, 1)
    colors_viz[object_indices] = [1.0, 0.0, 0.0]
    features_dc_mod_viz = (colors_viz - 0.5) / C0
    if features_dc.ndim == 3:
        features_dc_mod_viz = features_dc_mod_viz[:, np.newaxis, :]
    rendered_mod, _ = render_gaussians(
        means, scales, quats, features_dc_mod_viz, opacities_raw, cam
    )
    Image.fromarray((rendered_mod * 255).astype(np.uint8)).save(os.path.join(SESSION_DIR, f"{OBJECT_CLASSNAME}_highlighted_verification.png"))
    print(f"Saved {OBJECT_CLASSNAME}_highlighted_verification.png in {SESSION_DIR}")

    # 2. Add vase gaussians to the original checkpoint (no red highlight)
    print(f"\n--- Generating and integrating {OBJECT_CLASSNAME} gaussians from OBJ ---")
    if len(object_indices) == 0:
        print(f"No gaussians detected for {OBJECT_CLASSNAME} placement, skipping integration.")
        return
    # Reload original checkpoint (no red highlight)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    # Compute placement
    target_means = means[object_indices]
    target_min = target_means.min(axis=0)
    target_max = target_means.max(axis=0)
    target_center = target_means.mean(axis=0)
    target_extent = target_max - target_min
    # Clamp scale to min(X, Y) to avoid oversize
    scale = min(target_extent[0], target_extent[1])
    translation = target_center
    rotation = np.eye(3)
    num_gaussians = 15000
    color = [0.1, 0.3, 1.0]  # bright blue for visibility
    C0 = 0.28209479177387814
    print(f"Target region center: {target_center}, extent: {target_extent}, scale (clamped): {scale}")
    print(f"Scene means min: {means.min(axis=0)}, max: {means.max(axis=0)}, center: {means.mean(axis=0)}")
    # OBJ mesh loading and sampling
    def load_obj_mesh(obj_path):
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
                    face_verts = []
                    for p in parts:
                        idx = int(p.split("/")[0]) - 1
                        face_verts.append(idx)
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int64)
        return vertices, faces
    def sample_points_on_mesh(vertices, faces, num_points):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        total_area = areas.sum()
        probs = areas / total_area
        tri_indices = np.random.choice(len(faces), size=num_gaussians, p=probs)
        r1 = np.random.rand(num_gaussians)
        r2 = np.random.rand(num_gaussians)
        sqrt_r1 = np.sqrt(r1)
        bary_u = 1 - sqrt_r1
        bary_v = sqrt_r1 * (1 - r2)
        bary_w = sqrt_r1 * r2
        p0 = vertices[faces[tri_indices, 0]]
        p1 = vertices[faces[tri_indices, 1]]
        p2 = vertices[faces[tri_indices, 2]]
        points = (bary_u[:, None] * p0 + bary_v[:, None] * p1 + bary_w[:, None] * p2)
        face_normals = cross[tri_indices]
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = face_normals / norms
        return points.astype(np.float64), normals.astype(np.float64)
    vertices, faces = load_obj_mesh(object_obj_path)
    print(f"Loaded OBJ: {len(vertices)} vertices, {len(faces)} faces")
    points, normals = sample_points_on_mesh(vertices, faces, num_gaussians)
    print(f"Sampled {len(points)} points from mesh")
    # Center, rotate, scale, translate
    obj_min = points.min(axis=0)
    obj_max = points.max(axis=0)
    obj_center = (obj_min + obj_max) / 2.0
    print(f"OBJ center: {obj_center}, min: {obj_min}, max: {obj_max}")
    points -= obj_center
    pts_scene = np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])
    SCALE_FACTOR = 0.4
    pts_scene *= (scale * SCALE_FACTOR) / (obj_max - obj_min).max()
    pts_scene += translation
    print(f"Object points after transform: min {pts_scene.min(axis=0)}, max {pts_scene.max(axis=0)}, mean {pts_scene.mean(axis=0)}")
    means_object = torch.tensor(pts_scene, dtype=torch.float32)
    adaptive_radius = (np.ptp(pts_scene, axis=0).prod() / num_gaussians) ** (1.0 / 3.0) * 1.5
    log_scale = math.log(max(adaptive_radius, 1e-7))
    scales_object = torch.full((num_gaussians, 3), log_scale, dtype=torch.float32)
    quats_object = torch.zeros(num_gaussians, 4, dtype=torch.float32)
    quats_object[:, 0] = 1.0
    sh_color = (np.array(color) - 0.5) / C0
    features_dc_object = torch.tensor(sh_color, dtype=torch.float32).unsqueeze(0).expand(num_gaussians, -1)
    features_dc_object = features_dc_object.unsqueeze(1)
    features_rest_object = torch.zeros(num_gaussians, 15, 3, dtype=torch.float32)
    opacities_object = torch.full((num_gaussians, 1), 5.0, dtype=torch.float32)  # high opacity for visibility
    features_dc = state["_model.features_dc"]
    opacities_raw = state["_model.opacities"]
    if features_dc.ndim == 2:
        features_dc_object = features_dc_object.squeeze(1)
    if opacities_raw.ndim == 1:
        opacities_object = opacities_object.squeeze(-1)
    n_before = state["_model.means"].shape[0]
    state["_model.means"] = torch.cat([state["_model.means"], means_object], dim=0)
    state["_model.scales"] = torch.cat([state["_model.scales"], scales_object], dim=0)
    state["_model.quats"] = torch.cat([state["_model.quats"], quats_object], dim=0)
    state["_model.features_dc"] = torch.cat([state["_model.features_dc"], features_dc_object], dim=0)
    if "features_rest" in state:
        state["_model.features_rest"] = torch.cat([state["_model.features_rest"], features_rest_object], dim=0)
    state["_model.opacities"] = torch.cat([state["_model.opacities"], opacities_object], dim=0)
    n_after = state["_model.means"].shape[0]
    print(f"Gaussians before: {n_before}, after: {n_after} (added {n_after-n_before})")
    torch.save(ckpt, OUTPUT_PATH)
    print(f"Object gaussians generated and integrated. Saved to {OUTPUT_PATH}")

# ══════════════════════════════════════════════════════════════════════
# Vase Gaussian Integration
# ══════════════════════════════════════════════════════════════════════
def integrate_vase_gaussians(vase_gaussian_dict):
    import torch
    ckpt = torch.load(OUTPUT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    # Adjust shapes if needed
    features_dc = state["_model.features_dc"]
    opacities = state["_model.opacities"]
    if features_dc.ndim == 2:
        vase_gaussian_dict["features_dc"] = vase_gaussian_dict["features_dc"].squeeze(1)
    if opacities.ndim == 1:
        vase_gaussian_dict["opacities"] = vase_gaussian_dict["opacities"].squeeze(-1)
    # Merge
    state["_model.means"] = torch.cat([state["_model.means"], vase_gaussian_dict["means"]], dim=0)
    state["_model.scales"] = torch.cat([state["_model.scales"], vase_gaussian_dict["scales"]], dim=0)
    state["_model.quats"] = torch.cat([state["_model.quats"], vase_gaussian_dict["quats"]], dim=0)
    state["_model.features_dc"] = torch.cat([state["_model.features_dc"], vase_gaussian_dict["features_dc"]], dim=0)
    if "features_rest" in state:
        state["_model.features_rest"] = torch.cat([state["_model.features_rest"], vase_gaussian_dict["features_rest"]], dim=0)
    state["_model.opacities"] = torch.cat([state["_model.opacities"], vase_gaussian_dict["opacities"]], dim=0)
    torch.save(ckpt, "room_with_vase_final.ckpt")
    print("Saved: room_with_vase_final.ckpt (vase integrated)")

"""
Optimized Gaussian Placement for Vase Addition

This script is split into two main functions:
1. select_camera_and_render():
    - Loads the room.ckpt 3DGS scene
    - Generates multi-angle cameras around the scene center
    - Lets the user select a camera angle interactively
    - Renders the scene from the selected angle and saves the image

2. add_object_to_scene():
    - Takes a user-supplied image with a vase added (via diffusion model)
    - Uses YOLO to detect the vase in the image
    - Unprojects the vase's detected 2D location to 3D, finds corresponding Gaussians in room.ckpt
    - Computes scale and rotation
    - Adds vase Gaussians to the scene and saves the new checkpoint
"""

# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    print("\n--- Pipeline Commands ---")
    print(f"1. Camera selection: python detection_optimized.py")
    print(f"2. Detection & addition: python detection_optimized.py <image_with_object.png> <object.obj>")
    print(f"3. Viewer: python view_room.py {OUTPUT_PATH} --port 8080")
    if len(sys.argv) == 1:
        select_camera_and_render()
    elif len(sys.argv) == 3:
        object_image_path = sys.argv[1]
        object_obj_path = sys.argv[2]
        add_object_to_scene(object_image_path, object_obj_path)
    else:
        print("Usage:")
        print("  python detection_optimized.py           # Camera selection and render")
        print("  python detection_optimized.py <image_with_object.png> <object.obj>   # Object addition")
        
        
        
# python view_room.py /home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260213_004047/room_with_object.ckpt --port 8080