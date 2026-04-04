
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
import time
import psutil

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

import os
import datetime
from gemini_image_gen import generate_diffusion_image_with_gemini

CKPT_PATH = "bench_park.ckpt"
RENDER_W, RENDER_H = 1280, 720
NUM_CAMERAS = 15
FOV_DEG = 60.0
ORBIT_SCALE = 0.008       # fraction of scene extent for orbit radius
CAMERA_HEIGHT_OFFSET = 0.0  # keep horizontal view
OPACITY_THRESHOLD = 0.1
HEIGHT_TOLERANCE = 0.15      # for surface filtering
DEVICE = "cuda"
# SH DC constant
C0 = 0.28209479177387814

# Configurable object class for detection
OBJECT_CLASSNAME = "car"  # Change to "chair" or any other class as needed
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-image-preview")

# Session folder for outputs
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = f"session_{SESSION_TIMESTAMP}"
os.makedirs(SESSION_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SESSION_DIR, "room_with_object.ckpt")
CAMERA_STATE_PATH = os.path.join(SESSION_DIR, "selected_camera_state.pt")


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

    selected_cam = cameras[cam_idx]
    selected_azimuth = float(azimuth_angles[cam_idx])
    camera_state = {
        "cam_idx": int(cam_idx),
        "azimuth_rad": selected_azimuth,
        "azimuth_deg": float(math.degrees(selected_azimuth)),
        "intrinsics": selected_cam.get_K(),
        "extrinsics_w2c": selected_cam.w2c,
        "c2w": selected_cam.c2w,
        "position": selected_cam.position,
        "wxyz": selected_cam.wxyz,
        "render_width": int(selected_cam.width),
        "render_height": int(selected_cam.height),
        "fov_rad": float(selected_cam.fov_rad),
        "scene_center": scene_center,
        "scene_extent": scene_extent,
        "orbit_radius": float(orbit_radius),
        "camera_height_offset": float(camera_height_offset),
        "num_cameras": int(NUM_CAMERAS),
    }
    torch.save(camera_state, CAMERA_STATE_PATH)

    selected_img = rendered_images[cam_idx]
    selected_view_path = os.path.join(SESSION_DIR, "selected_camera_view.png")
    Image.fromarray((selected_img * 255).astype(np.uint8)).save(selected_view_path)
    print(f"Saved selected view as selected_camera_view.png (camera {cam_idx + 1}) in {SESSION_DIR}")
    print(f"Saved selected camera metadata to {CAMERA_STATE_PATH}")

    camera_state["selected_view_path"] = selected_view_path
    torch.save(camera_state, CAMERA_STATE_PATH)

    return camera_state


# ══════════════════════════════════════════════════════════════════════
# Vase addition via unprojection
# ══════════════════════════════════════════════════════════════════════


def _select_best_component(mask, prompt_box=None, min_area=120, max_area_frac=0.35):
    """Keep the most likely object component and suppress spill/noise regions."""
    from scipy import ndimage

    if mask.sum() == 0:
        return mask

    h, w = mask.shape
    total_px = h * w
    labeled, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return mask

    best_label = None
    best_score = -1e9
    cx_box = cy_box = None
    if prompt_box is not None:
        x1, y1, x2, y2 = prompt_box
        cx_box = 0.5 * (x1 + x2)
        cy_box = 0.5 * (y1 + y2)

    for label in range(1, num_labels + 1):
        comp = labeled == label
        area = int(comp.sum())
        if area < min_area:
            continue
        if area > int(max_area_frac * total_px):
            continue

        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())

        score = float(area)
        if prompt_box is not None:
            bx1, by1, bx2, by2 = map(float, prompt_box)
            overlap = (
                (xs >= bx1) & (xs <= bx2) &
                (ys >= by1) & (ys <= by2)
            ).sum() / max(1, area)
            dist = np.hypot(cx - cx_box, cy - cy_box)
            score = score * (1.0 + 2.0 * overlap) - 0.8 * dist

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return np.zeros_like(mask, dtype=bool)
    return (labeled == best_label)


def build_added_object_mask(base_image_path, edited_image_path, diff_threshold=0.05):
    """Build a binary mask for pixels added or changed by Gemini.

    Returns:
        mask: bool array of shape (H, W)
        diff_map: float array of per-pixel max-channel differences
    """
    from scipy import ndimage

    base_img = np.asarray(Image.open(base_image_path).convert("RGB"), dtype=np.float32) / 255.0
    edited_img = np.asarray(Image.open(edited_image_path).convert("RGB"), dtype=np.float32) / 255.0

    if base_img.shape != edited_img.shape:
        raise ValueError(
            f"Image shapes must match for differencing. base={base_img.shape}, edited={edited_img.shape}"
        )

    abs_diff = np.abs(edited_img - base_img)
    diff_map = abs_diff.max(axis=2)

    # Use both absolute floor threshold and adaptive high-percentile gate.
    adaptive_thr = max(diff_threshold, float(np.percentile(diff_map, 93)))
    mask = diff_map > adaptive_thr

    # Clean small isolated pixels and connect nearby regions.
    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=structure)
    mask = ndimage.binary_fill_holes(mask)

    mask = _select_best_component(mask, prompt_box=None, min_area=120, max_area_frac=0.35)

    return mask.astype(bool), diff_map


def detect_object_masks_with_yolo_seg(image_path, object_classname):
    """Run YOLO segmentation and return candidate masks for the requested class."""
    from ultralytics import YOLO

    model = YOLO("yolov8n-seg.pt")
    results = model(image_path, verbose=False)

    candidates = []
    for res in results:
        if res.masks is None or res.boxes is None:
            continue

        masks = res.masks.data.cpu().numpy() > 0.5
        for i, box in enumerate(res.boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if object_classname.lower() not in cls_name.lower():
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            candidates.append(
                {
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "mask": masks[i],
                }
            )
    return candidates


def detect_prompt_box_with_owlv2(image_path, object_prompt, score_threshold=0.08):
    """Optional text-guided detection with OWLv2.

    Returns (x1, y1, x2, y2) in pixel space or None if unavailable/not found.
    """
    try:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
    except Exception:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        text_queries = [[f"a photo of a {object_prompt}", object_prompt]]
        inputs = processor(text=text_queries, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
        res0 = results[0]
        if len(res0["scores"]) == 0:
            return None

        scores = res0["scores"].detach().cpu().numpy()
        boxes = res0["boxes"].detach().cpu().numpy()
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) < score_threshold:
            return None

        x1, y1, x2, y2 = boxes[best_idx]
        return float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return None


def refine_mask_with_sam(image_path, coarse_mask, prompt_box=None):
    """Optional SAM refinement inside region hinted by coarse mask and optional prompt box.

    Returns refined bool mask or None if SAM is unavailable/fails.
    """
    try:
        from transformers import SamModel, SamProcessor
        from scipy import ndimage
    except Exception:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        h, w = coarse_mask.shape

        ys, xs = np.where(coarse_mask)
        if len(xs) == 0:
            return None

        cx1, cy1 = xs.min(), ys.min()
        cx2, cy2 = xs.max(), ys.max()

        if prompt_box is not None:
            px1, py1, px2, py2 = prompt_box
            x1 = int(max(0, min(cx1, px1)))
            y1 = int(max(0, min(cy1, py1)))
            x2 = int(min(w - 1, max(cx2, px2)))
            y2 = int(min(h - 1, max(cy2, py2)))
        else:
            x1, y1, x2, y2 = int(cx1), int(cy1), int(cx2), int(cy2)

        pad = 8
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)

        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        inputs = processor(
            images=image,
            input_boxes=[[[x1, y1, x2, y2]]],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=True)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        sam_masks = masks[0][0].numpy()  # (num_masks, H, W)
        iou_scores = outputs.iou_scores[0, 0].detach().cpu().numpy()
        best_idx = int(np.argmax(iou_scores))
        sam_mask = sam_masks[best_idx] > 0

        # Keep SAM very close to change map to suppress subtle global style changes.
        coarse_dilated = ndimage.binary_dilation(coarse_mask, iterations=5)
        fused = sam_mask & coarse_dilated

        # If intersection is too small, grow from coarse signal rather than trusting full SAM mask.
        if fused.sum() < max(100, int(0.03 * max(1, coarse_mask.sum()))):
            tighter = ndimage.binary_dilation(coarse_mask, iterations=2)
            fused = sam_mask & tighter

        structure = np.ones((3, 3), dtype=bool)
        fused = ndimage.binary_opening(fused, structure=structure)
        fused = ndimage.binary_closing(fused, structure=structure)
        fused = ndimage.binary_fill_holes(fused)

        fused = _select_best_component(fused, prompt_box=prompt_box, min_area=100, max_area_frac=0.30)

        return fused.astype(bool)
    except Exception:
        return None


def add_object_to_scene(
    object_image_path,
    object_obj_path,
    camera_state_path=CAMERA_STATE_PATH,
    object_classname=OBJECT_CLASSNAME,
):
    import time
    timings = {}
    t_total_start = time.time()
    from ultralytics import YOLO
    print("\n--- Object Detection & Highlighting ---")
    # Start timing and resource tracking
    t_start = time.time()
    process = psutil.Process()
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss
    gpu_mem_start = None
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_start = torch.cuda.memory_allocated()
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    means = state["_model.means"].numpy()
    features_dc = state["_model.features_dc"].numpy()
    scales = state["_model.scales"].numpy()
    quats = state["_model.quats"].numpy()
    opacities_raw = state["_model.opacities"].numpy()
    opacities = (1 / (1 + np.exp(-opacities_raw))).squeeze()

    # Detect object in edited image with YOLO bbox
    t_yolo_start = time.time()
    model = YOLO("yolov8n.pt")
    results = model(object_image_path, verbose=False)
    t_yolo_end = time.time()
    timings['YOLO detection'] = t_yolo_end - t_yolo_start

    object_det = None
    for r_res in results:
        for box in r_res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            if object_classname.lower() in cls_name.lower():
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                object_det = {"class": cls_name, "confidence": conf, "bbox": (x1, y1, x2, y2)}
                break
        if object_det:
            break

    if not object_det:
        print(f"No {object_classname} detected in image.")
        return

    print(
        f"{object_classname.capitalize()} detected: {object_det['class']} "
        f"(conf={object_det['confidence']:.2f}), bbox={object_det['bbox']}"
    )

    # Save bbox visualization
    img = Image.open(object_image_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    x1, y1, x2, y2 = object_det["bbox"]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"{object_classname.capitalize()} Detection: {object_det['bbox']}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SESSION_DIR, f"{object_classname}_detection_bbox.png"), dpi=150)
    plt.close()
    print(f"Saved {object_classname}_detection_bbox.png in {SESSION_DIR}")

    if not os.path.exists(camera_state_path):
        print(f"Camera state file not found: {camera_state_path}")
        return

    camera_state = torch.load(camera_state_path, map_location="cpu", weights_only=False)
    selected_view_path = camera_state.get("selected_view_path", os.path.join(SESSION_DIR, "selected_camera_view.png"))
    if not os.path.exists(selected_view_path):
        print(f"Selected view file not found: {selected_view_path}")
        return

    # Unprojection & 3D detection timing
    t_unproj_start = time.time()
    cam = SceneCamera(
        position=camera_state["position"],
        wxyz=camera_state["wxyz"],
        fov_rad=float(camera_state["fov_rad"]),
        width=int(camera_state["render_width"]),
        height=int(camera_state["render_height"]),
    )
    cam_idx = int(camera_state["cam_idx"])
    print(f"Using selected camera {cam_idx + 1} from {camera_state_path}")

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
    t_unproj_end = time.time()
    timings['Unprojection & 3D detection'] = t_unproj_end - t_unproj_start
    t_highlight_start = time.time()
    print(f"Gaussians in {object_classname} bbox: {len(object_indices):,} / {len(means):,}")

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
    torch.save(ckpt, os.path.join(SESSION_DIR, f"room_with_{object_classname}_highlighted.ckpt"))
    print(f"Saved: room_with_{object_classname}_highlighted.ckpt (red highlight only) in {SESSION_DIR}")

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
    Image.fromarray((rendered_mod * 255).astype(np.uint8)).save(os.path.join(SESSION_DIR, f"{object_classname}_highlighted_verification.png"))
    print(f"Saved {object_classname}_highlighted_verification.png in {SESSION_DIR}")
    t_highlight_end = time.time()
    timings['Highlighting & ckpt'] = t_highlight_end - t_highlight_start
    t_vase_start = time.time()

    # 2. Add vase gaussians to the original checkpoint (no red highlight)
    print(f"\n--- Generating and integrating {object_classname} gaussians from OBJ ---")
    if len(object_indices) == 0:
        print(f"No gaussians detected for {object_classname} placement, skipping integration.")
        # End timing and write report even if failed
        t_end = time.time()
        cpu_end = process.cpu_times()
        mem_end = process.memory_info().rss
        gpu_mem_end = None
        if torch.cuda.is_available():
            gpu_mem_end = torch.cuda.memory_allocated()
        report_path = os.path.join(SESSION_DIR, "detection_resource_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Detection Resource Report\n========================\n")
            f.write(f"Status: No gaussians detected for {object_classname}\n")
            f.write(f"Elapsed time (s): {t_end - t_start:.2f}\n")
            f.write(f"CPU user time (s): {cpu_end.user - cpu_start.user:.2f}\n")
            f.write(f"CPU system time (s): {cpu_end.system - cpu_start.system:.2f}\n")
            f.write(f"Memory usage (MB): {(mem_end - mem_start) / 1024 / 1024:.2f}\n")
            if gpu_mem_start is not None and gpu_mem_end is not None:
                f.write(f"GPU: {gpu_name}\n")
                f.write(f"GPU memory used (MB): {(gpu_mem_end - gpu_mem_start) / 1024 / 1024:.2f}\n")
        print(f"Resource report saved to {report_path}")
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
        # End timing and write resource report after final ckpt is created
        t_end = time.time()
        cpu_end = process.cpu_times()
        mem_end = process.memory_info().rss
        gpu_mem_end = None
        if torch.cuda.is_available():
            gpu_mem_end = torch.cuda.memory_allocated()
        report_path = os.path.join(SESSION_DIR, "detection_resource_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Detection Resource Report\n========================\n")
            f.write(f"Status: Success\n")
            f.write(f"Elapsed time (s): {t_end - t_start:.2f}\n")
            f.write(f"CPU user time (s): {cpu_end.user - cpu_start.user:.2f}\n")
            f.write(f"CPU system time (s): {cpu_end.system - cpu_start.system:.2f}\n")
            f.write(f"Memory usage (MB): {(mem_end - mem_start) / 1024 / 1024:.2f}\n")
            if gpu_mem_start is not None and gpu_mem_end is not None:
                f.write(f"GPU: {gpu_name}\n")
                f.write(f"GPU memory used (MB): {(gpu_mem_end - gpu_mem_start) / 1024 / 1024:.2f}\n")
        print(f"Resource report saved to {report_path}")
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
    t_vase_end = time.time()
    timings['Vase integration'] = t_vase_end - t_vase_start
    # End timing and write resource report after final ckpt is created
    t_end = time.time()
    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss
    gpu_mem_end = None
    if torch.cuda.is_available():
        gpu_mem_end = torch.cuda.memory_allocated()
    report_path = os.path.join(SESSION_DIR, "detection_resource_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Detection Resource Report\n========================\n")
        f.write(f"Status: Success\n\n")
        f.write("| Stage | Time (s) |\n|---|---:|\n")
        for k, v in timings.items():
            f.write(f"| {k} | {v:.2f} |\n")
        f.write(f"| Total | {t_end - t_total_start:.2f} |\n\n")
        f.write(f"CPU user time (s): {cpu_end.user - cpu_start.user:.2f}\n")
        f.write(f"CPU system time (s): {cpu_end.system - cpu_start.system:.2f}\n")
        f.write(f"Memory usage (MB): {(mem_end - mem_start) / 1024 / 1024:.2f}\n")
        if gpu_mem_start is not None and gpu_mem_end is not None:
            f.write(f"GPU: {gpu_name}\n")
            f.write(f"GPU memory used (MB): {(gpu_mem_end - gpu_mem_start) / 1024 / 1024:.2f}\n")
    print(f"Resource report saved to {report_path}")

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
    - Takes the selected camera view and the Gemini-edited image
    - Uses YOLO to detect requested object in edited image
    - Uses YOLO bounding box for 3D gaussian filtering
    - Unprojects bounding-box pixels to 3D, finds corresponding Gaussians in room.ckpt
    - Computes scale and rotation
    - Adds vase Gaussians to the scene and saves the new checkpoint
"""

# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    print("\n--- Unified Pipeline (Single Runtime / Single Session) ---")
    print("1) Generate candidate camera views")
    print("2) Choose one camera interactively")
    print("3) Save camera metadata (.pt) inside this session folder")
    print("4) Generate diffusion-added image with Gemini API + continue with object OBJ in same runtime")
    print(f"Viewer command after completion: python view_room.py {OUTPUT_PATH} --port 8080")

    select_camera_and_render()

    if len(sys.argv) == 3:
        object_prompt = sys.argv[1]
        object_obj_path = sys.argv[2]
    else:
        object_prompt = input("Enter wanted object (e.g. car/chair/vase): ").strip()
        object_obj_path = input("Enter object OBJ path: ").strip()

    if not object_prompt:
        print("Wanted object cannot be empty.")
        sys.exit(1)
    if not os.path.exists(object_obj_path):
        print(f"OBJ path does not exist: {object_obj_path}")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = input("Enter Gemini API key: ").strip()
    if not api_key:
        print("Gemini API key is required.")
        sys.exit(1)

    selected_view_path = os.path.join(SESSION_DIR, "selected_camera_view.png")
    generated_image_path = os.path.join(SESSION_DIR, "gemini_diffusion_added.png")

    model_to_use = os.environ.get("GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL

    try:
        generate_diffusion_image_with_gemini(
            api_key=api_key,
            input_image_path=selected_view_path,
            object_prompt=object_prompt,
            output_image_path=generated_image_path,
            width=RENDER_W,
            height=RENDER_H,
            model=model_to_use,
        )
    except Exception as e:
        print(f"Failed to generate diffusion-added image with Gemini: {e}")
        sys.exit(1)

    add_object_to_scene(
        generated_image_path,
        object_obj_path,
        CAMERA_STATE_PATH,
        object_prompt,
    )
        
        
        
# python view_room.py /home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260213_004047/room_with_object.ckpt --port 8080