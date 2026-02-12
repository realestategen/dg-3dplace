
"""
Optimized Gaussian Placement for Vase Addition

This script is split into two main functions:
1. select_camera_and_render():
    - Loads the room.ckpt 3DGS scene
    - Generates multi-angle cameras around the scene center
    - Lets the user select a camera angle interactively
    - Renders the scene from the selected angle and saves the image

2. add_vase_to_scene():
    - Takes a user-supplied image with a vase added (via diffusion model)
    - Uses YOLO to detect the vase in the image
    - Unprojects the vase's detected 2D location to 3D, finds corresponding Gaussians in room.ckpt
    - Computes scale and rotation
    - Adds vase Gaussians to the scene and saves the new checkpoint
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
import gsplat
from gsplat import rasterization

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
CKPT_PATH = "room.ckpt"
OUTPUT_PATH = "room_with_vase.ckpt"
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
        Image.fromarray((img * 255).astype(np.uint8)).save(f"camera_view_{i}.png")
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
    plt.savefig("renders_grid.png", dpi=150)
    plt.close()
    print("Saved renders_grid.png")

    # User selects camera
    cam_idx = int(input(f"Select camera index (1-{len(cameras)}): ")) - 1
    if cam_idx < 0 or cam_idx >= len(cameras):
        print("Invalid index. Defaulting to 0.")
        cam_idx = 0
    selected_img = rendered_images[cam_idx]
    Image.fromarray((selected_img * 255).astype(np.uint8)).save("selected_camera_view.png")
    print(f"Saved selected view as selected_camera_view.png (camera {cam_idx + 1})")

# ══════════════════════════════════════════════════════════════════════
# Vase addition via unprojection
# ══════════════════════════════════════════════════════════════════════


def add_vase_to_scene(vase_image_path, _):
    from ultralytics import YOLO
    print("\n--- Vase Detection & Highlighting ---")
    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    means = state["_model.means"].numpy()
    features_dc = state["_model.features_dc"].numpy()
    scales = state["_model.scales"].numpy()
    quats = state["_model.quats"].numpy()
    opacities_raw = state["_model.opacities"].numpy()
    opacities = (1 / (1 + np.exp(-opacities_raw))).squeeze()

    # Detect vase in image
    model = YOLO("yolov8n.pt")
    results = model(vase_image_path, verbose=False)
    vase_det = None
    for r_res in results:
        for box in r_res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            if "vase" in cls_name.lower():
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                vase_det = {"class": cls_name, "confidence": conf, "bbox": (x1, y1, x2, y2)}
                break
        if vase_det:
            break
    if not vase_det:
        print("No vase detected in image.")
        return
    print(f"Vase detected: {vase_det['class']} (conf={vase_det['confidence']:.2f}), bbox={vase_det['bbox']}")

    # Save detection visualization
    img = Image.open(vase_image_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    x1, y1, x2, y2 = vase_det["bbox"]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"Vase Detection: {vase_det['bbox']}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("vase_detection_bbox.png", dpi=150)
    plt.close()
    print("Saved vase_detection_bbox.png")

    # Unproject vase bbox to 3D and highlight Gaussians
    cam_idx = int(input(f"Enter camera index used for vase image (1-{NUM_CAMERAS}): ")) - 1
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
    in_vase_bbox = (
        valid_all
        & (u_all >= x1)
        & (u_all <= x2)
        & (v_all >= y1)
        & (v_all <= y2)
        & (opacities > OPACITY_THRESHOLD)
    )
    vase_indices = np.where(in_vase_bbox)[0]
    print(f"Gaussians in vase bbox: {len(vase_indices):,} / {len(means):,}")

    # Color those Gaussians red
    C0 = 0.28209479177387814
    red_color = np.array([1.0, 0.0, 0.0])
    red_sh = (red_color - 0.5) / C0
    features_dc_mod = features_dc.copy()
    if features_dc_mod.ndim == 3:
        features_dc_mod[vase_indices, 0, :] = red_sh
    else:
        features_dc_mod[vase_indices, :] = red_sh
    state["_model.features_dc"] = torch.tensor(features_dc_mod)
    torch.save(ckpt, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")

    # Verification render
    features_dc_viz = features_dc.copy()
    if features_dc_viz.ndim == 3:
        features_dc_viz = features_dc_viz.squeeze(1)
    colors_viz = np.clip(C0 * features_dc_viz + 0.5, 0, 1)
    colors_viz[vase_indices] = [1.0, 0.0, 0.0]
    features_dc_mod_viz = (colors_viz - 0.5) / C0
    if features_dc.ndim == 3:
        features_dc_mod_viz = features_dc_mod_viz[:, np.newaxis, :]
    rendered_mod, _ = render_gaussians(
        means, scales, quats, features_dc_mod_viz, opacities_raw, cam
    )
    Image.fromarray((rendered_mod * 255).astype(np.uint8)).save("vase_highlighted_verification.png")
    print("Saved vase_highlighted_verification.png")

# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        select_camera_and_render()
    elif len(sys.argv) == 3:
        vase_image_path = sys.argv[1]
        vase_ckpt_path = sys.argv[2]
        add_vase_to_scene(vase_image_path, vase_ckpt_path)
    else:
        print("Usage:")
        print("  python detection_optimized.py           # Camera selection and render")
        print("  python detection_optimized.py vase_image.png vase_ckpt.pt   # Vase addition")