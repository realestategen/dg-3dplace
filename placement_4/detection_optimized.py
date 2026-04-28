
import math
import re
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
import sys
import importlib
import subprocess
import shutil
import importlib.util
import inspect
from gemini_image_gen import generate_diffusion_image_with_gemini


def _load_two_d_three_d_module():
    module_path = os.path.join(os.path.dirname(__file__), "2d_3d.py")
    spec = importlib.util.spec_from_file_location("two_d_three_d", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_glb_to_gaussians():
    module_path = os.path.join(os.path.dirname(__file__), "glb_to_gaussians.py")
    spec = importlib.util.spec_from_file_location("glb_to_gs", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.glb_to_gaussians


_TWO_D_THREE_D_MODULE = _load_two_d_three_d_module()
generate_obj_from_prompt_image = _TWO_D_THREE_D_MODULE.generate_obj_from_prompt_image
generate_obj_from_cutout_image = _TWO_D_THREE_D_MODULE.generate_obj_from_cutout_image
glb_to_gaussians = _load_glb_to_gaussians()

CKPT_PATH = "ckpt/bench_park.ckpt"
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
REQUIRE_GEMINI_CUTOUT = True

# Session folder for outputs
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_SESSION_DIR = f"session_{SESSION_TIMESTAMP}"
SESSION_DIR = os.path.abspath(DEFAULT_SESSION_DIR)
OUTPUT_PATH = os.path.join(SESSION_DIR, "room_with_object.ckpt")
CAMERA_STATE_PATH = os.path.join(SESSION_DIR, "selected_camera_state.pt")
OPTIMIZATION_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "DG_3DPlace_Optimization")
)
OPTIMIZATION_CONDA_ENV = os.environ.get("OPTIMIZATION_CONDA_ENV", "dg3d_optimize").strip()


def _configure_session_paths(session_dir, create=False):
    """Set global session paths so the rest of the pipeline stays unchanged."""
    global SESSION_DIR, OUTPUT_PATH, CAMERA_STATE_PATH
    SESSION_DIR = os.path.abspath(session_dir)
    if create:
        os.makedirs(SESSION_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(SESSION_DIR, "room_with_object.ckpt")
    CAMERA_STATE_PATH = os.path.join(SESSION_DIR, "selected_camera_state.pt")


def _find_latest_session_dir(base_dir):
    candidates = []
    if not os.path.isdir(base_dir):
        return None
    for name in os.listdir(base_dir):
        if not name.startswith("session_"):
            continue
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            candidates.append(full)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _write_object_prompt(session_dir, object_prompt):
    prompt_path = os.path.join(session_dir, "object_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write((object_prompt or "").strip() + "\n")


def _read_object_prompt(session_dir):
    prompt_path = os.path.join(session_dir, "object_prompt.txt")
    if not os.path.exists(prompt_path):
        return ""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


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


def detect_prompt_box_with_owlv2(image_path, object_prompt, score_threshold=0.06):
    """Text-guided detection with OWLv2 using full prompt + internal simplified variants.

    Returns dict with bbox/score/query or None if unavailable/not found.
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

        prompt_raw = (object_prompt or "").strip()
        prompt_l = prompt_raw.lower()

        stop_words = {
            "a", "an", "the", "on", "in", "at", "near", "next", "to", "of", "with",
            "and", "or", "under", "over", "behind", "front", "left", "right",
            "red", "blue", "green", "yellow", "white", "black", "brown", "gray",
            "small", "large", "big",
        }
        known_targets = ["car", "bench", "vase", "laptop", "chair", "table", "sofa", "plant", "bottle"]

        query_variants = []
        for q in [prompt_raw, f"a photo of {prompt_raw}"]:
            q = q.strip()
            if q and q not in query_variants:
                query_variants.append(q)

        for obj in known_targets:
            if re.search(rf"\b{obj}\b", prompt_l):
                for q in [obj, f"a photo of a {obj}"]:
                    if q not in query_variants:
                        query_variants.append(q)

        prompt_tokens = [t for t in re.split(r"[^a-z0-9]+", prompt_l) if len(t) >= 3 and t not in stop_words]
        for tok in prompt_tokens[:3]:
            if tok not in query_variants:
                query_variants.append(tok)

        text_queries = [query_variants]
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
        labels = res0["labels"].detach().cpu().numpy() if "labels" in res0 else np.zeros(len(scores), dtype=np.int64)
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) < score_threshold:
            return None

        x1, y1, x2, y2 = boxes[best_idx]
        label_idx = int(labels[best_idx]) if len(labels) > best_idx else 0
        matched_query = query_variants[label_idx] if 0 <= label_idx < len(query_variants) else prompt_raw
        return {
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "score": float(scores[best_idx]),
            "query": matched_query,
        }
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


def _save_black_white_mask(mask_bool, output_path):
    """Save binary mask as black/white PNG."""
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(output_path)
    return output_path


def _extract_cutout_silhouette(cutout_image_path):
    """Extract object silhouette from Gemini cutout image."""
    from scipy import ndimage

    if not cutout_image_path or not os.path.exists(cutout_image_path):
        return None

    img_rgba = Image.open(cutout_image_path).convert("RGBA")
    arr = np.asarray(img_rgba, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # Prefer alpha if present; otherwise remove near-white background.
    if (alpha < 250).any():
        mask = alpha > 10
    else:
        max_c = rgb.max(axis=2)
        min_c = rgb.min(axis=2)
        near_white = (max_c >= 245) & ((max_c - min_c) <= 12)
        mask = ~near_white

    if mask.sum() == 0:
        return None

    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=structure)
    mask = ndimage.binary_fill_holes(mask)
    mask = _select_best_component(mask, prompt_box=None, min_area=40, max_area_frac=0.98)
    if mask.sum() == 0:
        return None

    # Crop to the tight silhouette bounds so scale matching uses true object size.
    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    tight = mask[y1:y2 + 1, x1:x2 + 1]
    return tight.astype(bool)


def _match_cutout_inside_bbox(edited_image_path, prompt_box, cutout_image_path):
    """Match cutout silhouette inside detected bbox and return full-res mask."""
    from scipy import ndimage

    cutout_mask = _extract_cutout_silhouette(cutout_image_path)
    if cutout_mask is None:
        raise RuntimeError("Could not extract silhouette from gemini_object_cutout.png")

    edited_img = np.asarray(Image.open(edited_image_path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = edited_img.shape[:2]
    edited_gray = edited_img.mean(axis=2)

    cutout_rgb = np.asarray(Image.open(cutout_image_path).convert("RGB"), dtype=np.float32) / 255.0
    cutout_gray = cutout_rgb.mean(axis=2)

    x1, y1, x2, y2 = prompt_box
    bx1 = int(max(0, min(w - 1, np.floor(x1))))
    by1 = int(max(0, min(h - 1, np.floor(y1))))
    bx2 = int(max(0, min(w, np.ceil(x2))))
    by2 = int(max(0, min(h, np.ceil(y2))))
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    ch, cw = cutout_mask.shape
    fit_scale = min(bw / max(1, cw), bh / max(1, ch))
    scale_candidates = [fit_scale * s for s in (0.82, 0.9, 1.0, 1.1, 1.2)]

    best = None
    best_score = -1e9

    for scale in scale_candidates:
        tw = max(8, int(round(cw * scale)))
        th = max(8, int(round(ch * scale)))
        if tw > bw or th > bh:
            continue

        mask_rs = np.asarray(
            Image.fromarray((cutout_mask.astype(np.uint8) * 255), mode="L").resize((tw, th), Image.NEAREST),
            dtype=np.uint8,
        ) > 0
        if mask_rs.sum() < 50:
            continue

        gray_rs = np.asarray(
            Image.fromarray((cutout_gray * 255).astype(np.uint8), mode="L").resize((tw, th), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        cx = bx1 + (bw - tw) // 2
        cy = by1 + (bh - th) // 2
        dx_lim = max(6, int(0.15 * bw))
        dy_lim = max(6, int(0.15 * bh))
        step = 2

        for oy in range(max(by1, cy - dy_lim), min(by2 - th, cy + dy_lim) + 1, step):
            for ox in range(max(bx1, cx - dx_lim), min(bx2 - tw, cx + dx_lim) + 1, step):
                patch = edited_gray[oy:oy + th, ox:ox + tw]
                m = mask_rs
                pv = patch[m]
                cv = gray_rs[m]
                if pv.size < 40:
                    continue

                pv_std = float(pv.std()) + 1e-6
                cv_std = float(cv.std()) + 1e-6
                pv_n = (pv - float(pv.mean())) / pv_std
                cv_n = (cv - float(cv.mean())) / cv_std
                ncc = float((pv_n * cv_n).mean())

                # Prefer solutions nearer bbox center when scores are similar.
                dist_penalty = 0.002 * np.hypot((ox + tw / 2) - (bx1 + bw / 2), (oy + th / 2) - (by1 + bh / 2))
                score = ncc - float(dist_penalty)
                if score > best_score:
                    best_score = score
                    best = (ox, oy, tw, th, mask_rs)

    if best is None:
        # Fallback: centered fit into bbox.
        tw = max(8, int(round(cw * fit_scale)))
        th = max(8, int(round(ch * fit_scale)))
        tw = min(tw, bw)
        th = min(th, bh)
        mask_rs = np.asarray(
            Image.fromarray((cutout_mask.astype(np.uint8) * 255), mode="L").resize((tw, th), Image.NEAREST),
            dtype=np.uint8,
        ) > 0
        ox = bx1 + max(0, (bw - tw) // 2)
        oy = by1 + max(0, (bh - th) // 2)
    else:
        ox, oy, tw, th, mask_rs = best

    out = np.zeros((h, w), dtype=bool)
    out[oy:oy + th, ox:ox + tw] = mask_rs

    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[by1:by2, bx1:bx2] = True
    out = out & bbox_mask

    structure = np.ones((3, 3), dtype=bool)
    out = ndimage.binary_opening(out, structure=structure)
    out = ndimage.binary_closing(out, structure=structure)
    out = ndimage.binary_fill_holes(out)
    out = _select_best_component(out, prompt_box=prompt_box, min_area=80, max_area_frac=0.30)

    if out.sum() == 0:
        raise RuntimeError("Matched cutout mask became empty inside bbox")
    return out.astype(bool)


def save_added_object_mask_for_session(
    edited_image_path,
    session_dir,
    prompt_box,
    cutout_image_path,
    output_name="added_object_mask.png",
):
    """Save final full-resolution mask using detected bbox + Gemini cutout matching."""
    output_path = os.path.join(session_dir, output_name)
    final_mask = _match_cutout_inside_bbox(
        edited_image_path=edited_image_path,
        prompt_box=prompt_box,
        cutout_image_path=cutout_image_path,
    )
    _save_black_white_mask(final_mask, output_path)
    return output_path


def add_object_to_scene(
    object_image_path,
    object_obj_path=None,
    camera_state_path=CAMERA_STATE_PATH,
    detection_target=OBJECT_CLASSNAME,
):
    import time
    timings = {}
    t_total_start = time.time()
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

    object_det = None
    detection_target = (detection_target or "").strip()
    detection_label = infer_detection_target_from_prompt(detection_target)

    # OWLv2-only open-vocabulary detection from the rich prompt.
    t_owl_start = time.time()
    owl_det = detect_prompt_box_with_owlv2(object_image_path, detection_target, score_threshold=0.06)
    t_owl_end = time.time()
    timings['OWLv2 detection'] = t_owl_end - t_owl_start
    if owl_det is not None:
        x1, y1, x2, y2 = owl_det["bbox"]
        object_det = {
            "class": detection_label,
            "confidence": float(owl_det["score"]),
            "bbox": (x1, y1, x2, y2),
            "source": "owlv2",
            "query": owl_det.get("query", detection_target),
        }

    if not object_det:
        print(f"No object detected for rich prompt: '{detection_target}'.")
        return

    print(
        f"Detected via {object_det.get('source', 'unknown')}: "
        f"class={object_det['class']}, query='{object_det.get('query', detection_target)}', "
        f"conf={object_det['confidence']:.2f}, bbox={object_det['bbox']}"
    )

    color_mesh_path = object_obj_path
    gemini_cutout_path = None
    gemini_cutout_processed_path = None

    # Generate OBJ automatically from detected bbox + prompt if path wasn't provided.
    if not object_obj_path:
        auto_obj_path = os.path.join(SESSION_DIR, "generated_object.obj")
        try:
            gen_result = generate_obj_from_prompt_image(
                image_path=object_image_path,
                prompt=detection_target,
                output_obj_path=auto_obj_path,
                bbox=object_det["bbox"],
                session_dir=SESSION_DIR,
                require_gemini_cutout=REQUIRE_GEMINI_CUTOUT,
                api_key=api_key,
            )
            object_obj_path = gen_result["output_obj_path"]
            color_mesh_path = gen_result.get("output_color_mesh_path") or gen_result.get("output_glb_path") or object_obj_path
            print(f"Generated object OBJ: {object_obj_path}")
            if color_mesh_path and color_mesh_path != object_obj_path:
                print(f"Using textured mesh for colors: {color_mesh_path}")
            if gen_result.get("textured_output_path"):
                print(f"Generated textured mesh: {gen_result['textured_output_path']}")
            if gen_result.get("mtl_path"):
                print(f"Generated MTL: {gen_result['mtl_path']}")
            if gen_result.get("albedo_path"):
                print(f"Generated albedo texture: {gen_result['albedo_path']}")
            if gen_result.get("texture_error"):
                print(f"Texture generation warning: {gen_result['texture_error']}")
            if gen_result.get("object_only_png_path"):
                print(f"Saved object-only PNG: {gen_result['object_only_png_path']}")
            if gen_result.get("gemini_object_cutout_path"):
                print(f"Saved Gemini cutout PNG: {gen_result['gemini_object_cutout_path']}")
                gemini_cutout_path = gen_result["gemini_object_cutout_path"]
            if gen_result.get("gemini_object_cleaned_path"):
                print(f"Saved Gemini cleaned PNG: {gen_result['gemini_object_cleaned_path']}")
                gemini_cutout_processed_path = gen_result["gemini_object_cleaned_path"]
        except Exception as e:
            print(f"Failed to generate OBJ from detected image region: {e}")
            return

    if color_mesh_path is None:
        color_mesh_path = object_obj_path

    if not gemini_cutout_path:
        session_cutout = os.path.join(SESSION_DIR, "gemini_object_cutout.png")
        if os.path.exists(session_cutout):
            gemini_cutout_path = session_cutout

    if not gemini_cutout_processed_path:
        session_processed = os.path.join(SESSION_DIR, "gemini_object_cutout_processed.png")
        if os.path.exists(session_processed):
            gemini_cutout_processed_path = session_processed
        else:
            session_cleaned = os.path.join(SESSION_DIR, "gemini_object_cleaned.png")
            if os.path.exists(session_cleaned):
                gemini_cutout_processed_path = session_cleaned

    # Save bbox visualization
    img = Image.open(object_image_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    x1, y1, x2, y2 = object_det["bbox"]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"{detection_label.capitalize()} Detection ({object_det.get('source', 'detector')}): {object_det['bbox']}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SESSION_DIR, f"{detection_label}_detection_bbox.png"), dpi=150)
    plt.close()
    print(f"Saved {detection_label}_detection_bbox.png in {SESSION_DIR}")

    if not os.path.exists(camera_state_path):
        print(f"Camera state file not found: {camera_state_path}")
        return

    camera_state = torch.load(camera_state_path, map_location="cpu", weights_only=False)
    selected_view_path = camera_state.get("selected_view_path", os.path.join(SESSION_DIR, "selected_camera_view.png"))
    if not os.path.exists(selected_view_path):
        print(f"Selected view file not found: {selected_view_path}")
        return

    # Additional artifact: full-resolution black/white mask from bbox + Gemini cutout.
    cutout_for_mask = None
    if gemini_cutout_processed_path and os.path.exists(gemini_cutout_processed_path):
        cutout_for_mask = gemini_cutout_processed_path
    elif gemini_cutout_path and os.path.exists(gemini_cutout_path):
        cutout_for_mask = gemini_cutout_path

    if cutout_for_mask and os.path.exists(cutout_for_mask):
        try:
            mask_path = save_added_object_mask_for_session(
                edited_image_path=object_image_path,
                session_dir=SESSION_DIR,
                prompt_box=object_det["bbox"],
                cutout_image_path=cutout_for_mask,
                output_name="added_object_mask.png",
            )
            print(f"Saved added object mask (black/white): {mask_path}")
        except Exception as e:
            print(f"Warning: failed to save added object mask from bbox+cutout: {e}")
    else:
        print("Warning: Gemini cutout processed/original file not found; skipping added_object_mask.png")

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
    print(f"Gaussians in {detection_label} bbox: {len(object_indices):,} / {len(means):,}")

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
    torch.save(ckpt, os.path.join(SESSION_DIR, f"room_with_{detection_label}_highlighted.ckpt"))
    print(f"Saved: room_with_{detection_label}_highlighted.ckpt (red highlight only) in {SESSION_DIR}")

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
    Image.fromarray((rendered_mod * 255).astype(np.uint8)).save(os.path.join(SESSION_DIR, f"{detection_label}_highlighted_verification.png"))
    print(f"Saved {detection_label}_highlighted_verification.png in {SESSION_DIR}")
    t_highlight_end = time.time()
    timings['Highlighting & ckpt'] = t_highlight_end - t_highlight_start
    t_vase_start = time.time()

    # 2. Add vase gaussians to the original checkpoint (no red highlight)
    print(f"\n--- Generating and integrating {detection_label} gaussians from textured mesh ---")
    if len(object_indices) == 0:
        print(f"No gaussians detected for {detection_label} placement, skipping integration.")
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
            f.write(f"Status: No gaussians detected for {detection_label}\n")
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

    # Estimate a support surface height from the detected gaussians.
    # Use a low-percentile of the Z distribution to prefer the supporting surface
    # (e.g. tabletop or ground) rather than an object top inside the bbox.
    try:
        support_z = float(np.percentile(target_means[:, 2], 20))
    except Exception:
        support_z = float(target_center[2])

    # Translate horizontally to the detected region center; let glb_to_gaussians
    # align the object's bottom to `support_z` when adding gaussians.
    translation = np.array([float(target_center[0]), float(target_center[1]), 0.0], dtype=np.float32)
    num_gaussians = 100000
    print(f"Target region center: {target_center}, extent: {target_extent}, scale (clamped): {scale}")
    print(f"Scene means min: {means.min(axis=0)}, max: {means.max(axis=0)}, center: {means.mean(axis=0)}")
    mesh_for_color = color_mesh_path or object_obj_path
    if mesh_for_color is None:
        print("No mesh path available for object Gaussian generation.")
        return

    if not mesh_for_color.lower().endswith(".glb"):
        print(f"Textured GLB not found; got '{mesh_for_color}'. Falling back to previous path may reduce color fidelity.")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        native_train_script_path = os.path.join(script_dir, "train_object_gs_native.py")
        user_train_template = os.environ.get("OBJECT_GS_TRAIN_CMD_TEMPLATE", "").strip()
        train_num_gaussians = int(os.environ.get("OBJECT_GS_TRAIN_NUM_GAUSSIANS", "30000"))
        train_num_gaussians = max(20000, min(50000, train_num_gaussians))
        trainer_cmd_template = user_train_template or (
            f"python {native_train_script_path} --mesh {{mesh_path}} --images {{images_dir}} "
            f"--camera-json {{camera_json}} --sparse {{sparse_dir}} --output {{output_dir}} "
            f"--steps {{steps}} --num-gaussians {train_num_gaussians}"
        )

        use_train_mode = True

        if use_train_mode:
            try:
                object_gaussians = glb_to_gaussians(
                    glb_path=mesh_for_color,
                    num_gaussians=num_gaussians,
                    target_scale=float(scale),
                    scale_factor=0.4,
                    rotation=None,
                    translation=translation,
                    support_z=support_z,
                    opacity_logit=5.0,
                    run_render_colmap=True,
                    work_dir=os.path.join(SESSION_DIR, "glb_colmap_gs"),
                    conversion_mode="train",
                    blender_exe=os.environ.get("BLENDER_EXE", "").strip() or None,
                    colmap_exe=os.environ.get("COLMAP_EXE", "").strip() or None,
                    trainer_cmd_template=trainer_cmd_template,
                    num_views=int(os.environ.get("OBJECT_GS_NUM_VIEWS", "48")),
                    render_size=int(os.environ.get("OBJECT_GS_RENDER_SIZE", "768")),
                    train_steps=int(os.environ.get("OBJECT_GS_TRAIN_STEPS", "3000")),
                )
                print("Using train-mode GLB->GS conversion.")
            except Exception as train_err:
                print(f"[!] Train-mode conversion failed: {train_err}")
                print("[!] Falling back to sample-mode conversion.")
                object_gaussians = glb_to_gaussians(
                    glb_path=mesh_for_color,
                    num_gaussians=num_gaussians,
                    target_scale=float(scale),
                    scale_factor=0.4,
                    rotation=None,
                    translation=translation,
                    support_z=support_z,
                    opacity_logit=5.0,
                    run_render_colmap=False,
                    work_dir=os.path.join(SESSION_DIR, "glb_colmap_gs"),
                    conversion_mode="sample",
                )
        else:
            object_gaussians = glb_to_gaussians(
                glb_path=mesh_for_color,
                num_gaussians=num_gaussians,
                target_scale=float(scale),
                scale_factor=0.4,
                rotation=None,
                translation=translation,
                support_z=support_z,
                opacity_logit=5.0,
                run_render_colmap=False,
                work_dir=os.path.join(SESSION_DIR, "glb_colmap_gs"),
                conversion_mode="sample",
            )

        means_object = object_gaussians["means"]
        scales_object = object_gaussians["scales"]
        quats_object = object_gaussians["quats"]
        features_dc_object = object_gaussians["features_dc"]
        features_rest_object = object_gaussians["features_rest"]
        opacities_object = object_gaussians["opacities"]
        print(f"Generated {means_object.shape[0]} textured object gaussians from GLB pipeline.")
    except Exception as e:
        print(f"Failed converting GLB to gaussians: {e}")
        return

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
    num_object_gaussians = int(n_after - n_before)
    print(f"Gaussians before: {n_before}, after: {n_after} (added {n_after-n_before})")
    # Persist split metadata so optimization can recover exact object/background partition.
    ckpt["num_object_gaussians"] = int(num_object_gaussians)
    torch.save(ckpt, OUTPUT_PATH)
    print(f"Object gaussians generated and integrated. Saved to {OUTPUT_PATH}")
    t_vase_end = time.time()
    timings['Vase integration'] = t_vase_end - t_vase_start
    
    # Render final view from saved camera angle to validate object placement
    t_final_render_start = time.time()
    final_view_path = render_final_view_with_saved_camera(OUTPUT_PATH, camera_state_path, SESSION_DIR)
    t_final_render_end = time.time()
    if final_view_path:
        timings['Final render'] = t_final_render_end - t_final_render_start

    # Optional refinement step using optimization module with existing session artifacts.
    t_opt_start = time.time()
    optimized_ckpt_path = run_post_placement_optimization(
        initial_ckpt_path=OUTPUT_PATH,
        camera_state_path=camera_state_path,
        session_dir=SESSION_DIR,
        num_object_gaussians=num_object_gaussians,
    )
    t_opt_end = time.time()
    if optimized_ckpt_path:
        timings['Post-placement optimization'] = t_opt_end - t_opt_start
        t_opt_render_start = time.time()
        optimized_view_path = render_final_view_with_saved_camera(
            optimized_ckpt_path,
            camera_state_path,
            SESSION_DIR,
            output_name="final_view_with_object_optimized.png",
        )
        t_opt_render_end = time.time()
        if optimized_view_path:
            timings['Optimized final render'] = t_opt_render_end - t_opt_render_start
    
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


def render_final_view_with_saved_camera(ckpt_path, camera_state_path, session_dir, output_name="final_view_with_object.png"):
    """Render the final integrated scene from the saved user-selected camera angle.
    
    This allows instant validation of object placement without running view_room.py.
    """
    if not os.path.exists(camera_state_path):
        print(f"[!] Camera state not found at {camera_state_path}, skipping final render.")
        return None
    
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found at {ckpt_path}, skipping final render.")
        return None
    
    try:
        print("\n--- Rendering Final View with Object ---")
        # Load checkpoint and camera state
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["pipeline"]
        camera_state = torch.load(camera_state_path, map_location="cpu", weights_only=False)
        
        # Extract Gaussian parameters
        means = state["_model.means"].numpy()
        scales = state["_model.scales"].numpy()
        quats = state["_model.quats"].numpy()
        features_dc = state["_model.features_dc"].numpy()
        opacities_raw = state["_model.opacities"].numpy()
        
        # Reconstruct camera from saved state
        cam_data = camera_state
        cam = SceneCamera(
            position=cam_data["position"],
            wxyz=cam_data["wxyz"],
            fov_rad=cam_data["fov_rad"],
            width=cam_data["render_width"],
            height=cam_data["render_height"]
        )
        
        # Render
        img, alpha = render_gaussians(means, scales, quats, features_dc, opacities_raw, cam)
        
        # Save result
        output_path = os.path.join(session_dir, output_name)
        Image.fromarray((img * 255).astype(np.uint8)).save(output_path)
        print(f"Saved final view (with object) to: {output_path}")
        
        return output_path
    except Exception as e:
        print(f"[!] Final render failed: {e}")
        return None


def _load_post_placement_optimization_modules():
    """Dynamically import optimization modules without modifying their source files."""
    if not os.path.isdir(OPTIMIZATION_ROOT):
        raise FileNotFoundError(f"Optimization folder not found: {OPTIMIZATION_ROOT}")

    if OPTIMIZATION_ROOT not in sys.path:
        sys.path.insert(0, OPTIMIZATION_ROOT)

    run_refinement_module = importlib.import_module("run_refinement")
    camera_utils_module = importlib.import_module("src.utils.camera_utils")

    if not hasattr(run_refinement_module, "run_refinement"):
        raise AttributeError("run_refinement.py does not expose run_refinement")
    if not hasattr(camera_utils_module, "load_scout_camera"):
        raise AttributeError("camera_utils does not expose load_scout_camera")

    return run_refinement_module.run_refinement, camera_utils_module.load_scout_camera


def _run_post_placement_optimization_in_conda_env(
    env_name,
    initial_ckpt_path,
    camera_state_path,
    target_img_path,
    mask_path,
    num_object_gaussians,
    optimized_output_path,
):
    """Run optimization in a separate conda env to use env-specific dependencies."""
    if not env_name:
        return False

    conda_exe = os.environ.get("CONDA_EXE", "").strip() or shutil.which("conda")
    if not conda_exe:
        print("[!] conda executable not found on PATH; cannot use external optimization env.")
        return False

    launcher_code = (
        "import os, sys\n"
        "import inspect\n"
        f"optimization_root = {OPTIMIZATION_ROOT!r}\n"
        "if optimization_root not in sys.path:\n"
        "    sys.path.insert(0, optimization_root)\n"
        "from run_refinement import run_refinement\n"
        "from src.utils.camera_utils import load_scout_camera\n"
        f"camera = load_scout_camera({camera_state_path!r})\n"
        "kwargs = dict(\n"
        f"    ckpt_path={initial_ckpt_path!r},\n"
        f"    target_img_path={target_img_path!r},\n"
        f"    mask_path={mask_path!r},\n"
        "    scout_camera_data=camera,\n"
        f"    output_path={optimized_output_path!r},\n"
        ")\n"
        "sig = inspect.signature(run_refinement)\n"
        "if 'num_object_gaussians' in sig.parameters:\n"
        f"    kwargs['num_object_gaussians'] = {int(num_object_gaussians)}\n"
        "run_refinement(**kwargs)\n"
    )

    cmd = [
        conda_exe,
        "run",
        "-n",
        env_name,
        "python",
        "-c",
        launcher_code,
    ]

    print(f"Running optimization via conda env: {env_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(f"conda optimization process failed with code {result.returncode}")

    return True


def run_post_placement_optimization(initial_ckpt_path, camera_state_path, session_dir, num_object_gaussians):
    """Run optimization refinement in the same runtime on session artifacts."""
    selected_view_path = os.path.join(session_dir, "selected_camera_view.png")
    target_img_path = os.path.join(session_dir, "gemini_diffusion_added.png")
    mask_path = os.path.join(session_dir, "added_object_mask.png")
    optimized_output_path = os.path.join(session_dir, "room_with_object_optimized.ckpt")

    required_paths = [
        ("initial checkpoint", initial_ckpt_path),
        ("camera state", camera_state_path),
        ("selected camera view", selected_view_path),
        ("diffusion-added target image", target_img_path),
        ("added object mask", mask_path),
    ]
    missing = [f"{name}: {path}" for name, path in required_paths if not os.path.exists(path)]
    if missing:
        print("[!] Skipping optimization because required files are missing:")
        for item in missing:
            print(f"    - {item}")
        return None

    if int(num_object_gaussians) <= 0:
        print("[!] Skipping optimization because num_object_gaussians is not positive.")
        return None

    try:
        run_refinement, load_scout_camera = _load_post_placement_optimization_modules()
    except Exception as e:
        print(f"[!] Failed to import optimization modules: {e}")
        run_refinement = None
        load_scout_camera = None

    try:
        print("\n--- Running Post-Placement Optimization ---")
        print(f"Using checkpoint: {initial_ckpt_path}")
        print(f"Using camera state: {camera_state_path}")
        print(f"Using target image: {target_img_path}")
        print(f"Using mask image: {mask_path}")
        print(f"Object gaussian count for split: {int(num_object_gaussians)}")

        ran_in_conda = False
        if OPTIMIZATION_CONDA_ENV:
            try:
                ran_in_conda = _run_post_placement_optimization_in_conda_env(
                    env_name=OPTIMIZATION_CONDA_ENV,
                    initial_ckpt_path=initial_ckpt_path,
                    camera_state_path=camera_state_path,
                    target_img_path=target_img_path,
                    mask_path=mask_path,
                    num_object_gaussians=num_object_gaussians,
                    optimized_output_path=optimized_output_path,
                )
            except Exception as conda_err:
                print(f"[!] Conda-env optimization failed ({OPTIMIZATION_CONDA_ENV}): {conda_err}")
                print("[!] Falling back to current runtime environment for optimization.")

        if not ran_in_conda:
            if run_refinement is None or load_scout_camera is None:
                print("[!] Optimization modules unavailable in current runtime; skipping optimization.")
                return None
            real_camera = load_scout_camera(camera_state_path)
            refinement_kwargs = {
                "ckpt_path": initial_ckpt_path,
                "target_img_path": target_img_path,
                "mask_path": mask_path,
                "scout_camera_data": real_camera,
                "output_path": optimized_output_path,
            }
            if "num_object_gaussians" in inspect.signature(run_refinement).parameters:
                refinement_kwargs["num_object_gaussians"] = int(num_object_gaussians)
            run_refinement(**refinement_kwargs)

        if not os.path.exists(optimized_output_path):
            print(f"[!] Optimization did not produce expected output: {optimized_output_path}")
            return None

        print(f"Post-placement optimization finished. Saved to {optimized_output_path}")
        return optimized_output_path
    except Exception as e:
        print(f"[!] Post-placement optimization failed: {e}")
        return None


def infer_detection_target_from_prompt(object_prompt):
    """Infer a compact detection keyword from a rich Gemini prompt."""
    prompt_l = (object_prompt or "").lower()
    known_targets = [
        "car",
        "bench",
        "vase",
        "laptop",
        "chair",
        "table",
        "sofa",
        "plant",
        "bottle",
    ]
    for name in known_targets:
        if re.search(rf"\b{name}\b", prompt_l):
            return name

    tokens = [t for t in re.split(r"[^a-z0-9]+", prompt_l) if len(t) >= 3]
    if len(tokens) > 0:
        return tokens[-1]
    return OBJECT_CLASSNAME


def run_from_session_cutout(
    session_dir,
    object_prompt,
    ckpt_path,
    generated_image_path="",
    cutout_image_path="",
    hunyuan_conda_env="hunyuan",
):
    """Resume from the cutout stage and run placement + optimization without Gemini APIs."""
    _configure_session_paths(session_dir, create=True)

    global CKPT_PATH
    CKPT_PATH = ckpt_path

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    camera_state_path = os.path.join(SESSION_DIR, "selected_camera_state.pt")
    if not os.path.exists(camera_state_path):
        raise FileNotFoundError(f"Camera state not found in session: {camera_state_path}")

    target_image = (generated_image_path or "").strip()
    if not target_image:
        target_image = os.path.join(SESSION_DIR, "gemini_diffusion_added.png")
    target_image = os.path.abspath(target_image)
    if not os.path.exists(target_image):
        raise FileNotFoundError(
            "Diffusion-added image is required for downstream steps and was not found at "
            f"{target_image}. Use --generated-image-path to override."
        )

    cutout_candidates = []
    if (cutout_image_path or "").strip():
        cutout_candidates.append(cutout_image_path.strip())
    cutout_candidates.extend(
        [
            os.path.join(SESSION_DIR, "gemini_object_cutout_processed.png"),
            os.path.join(SESSION_DIR, "gemini_object_cleaned.png"),
            os.path.join(SESSION_DIR, "gemini_object_cutout.png"),
        ]
    )

    selected_cutout = None
    for candidate in cutout_candidates:
        candidate_abs = os.path.abspath(candidate)
        if os.path.exists(candidate_abs):
            selected_cutout = candidate_abs
            break
    if selected_cutout is None:
        raise FileNotFoundError(
            "No cutout image found in session. Expected one of gemini_object_cutout_processed.png, "
            "gemini_object_cleaned.png, gemini_object_cutout.png, or pass --cutout-image-path."
        )

    generated_obj_path = os.path.join(SESSION_DIR, "generated_object.obj")
    print("\n--- Resume Mode: Cutout -> 3D -> GS -> Placement -> Optimization ---")
    print(f"Using session: {SESSION_DIR}")
    print(f"Using cutout: {selected_cutout}")
    print(f"Using diffusion-added image: {target_image}")
    print(f"Using checkpoint: {CKPT_PATH}")

    gen_result = generate_obj_from_cutout_image(
        cutout_image_path=selected_cutout,
        output_obj_path=generated_obj_path,
        session_dir=SESSION_DIR,
        conda_env=hunyuan_conda_env,
    )
    mesh_for_scene = (
        gen_result.get("output_color_mesh_path")
        or gen_result.get("output_glb_path")
        or gen_result.get("output_obj_path")
    )
    if not mesh_for_scene:
        raise RuntimeError("Cutout-to-3D completed but no mesh output path was returned.")

    prompt_for_detection = (object_prompt or "").strip() or OBJECT_CLASSNAME
    _write_object_prompt(SESSION_DIR, prompt_for_detection)

    add_object_to_scene(
        object_image_path=target_image,
        object_obj_path=mesh_for_scene,
        camera_state_path=camera_state_path,
        detection_target=prompt_for_detection,
    )

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
    - Uses OWLv2 open-vocabulary detection with the same rich prompt used for Gemini
    - Uses detected bounding box for 3D gaussian filtering
    - Unprojects bounding-box pixels to 3D, finds corresponding Gaussians in room.ckpt
    - Computes scale and rotation
    - Adds vase Gaussians to the scene and saves the new checkpoint
"""

# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the 3D placement pipeline.")
    parser.add_argument(
        "object_prompt",
        nargs="?",
        help="Rich edit prompt for the object to add.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=CKPT_PATH,
        help=f"Input checkpoint path to load (default: {CKPT_PATH}).",
    )
    parser.add_argument(
        "--resume-from-cutout",
        action="store_true",
        help="Resume from saved cutout onward (no Gemini API calls).",
    )
    parser.add_argument(
        "--session-dir",
        default="",
        help="Existing session directory to resume from. If omitted with --resume-from-cutout, latest session_* is used.",
    )
    parser.add_argument(
        "--generated-image-path",
        default="",
        help="Override path to diffusion-added image for placement (default: <session>/gemini_diffusion_added.png).",
    )
    parser.add_argument(
        "--cutout-image-path",
        default="",
        help="Override path to existing cutout image (default: processed/cleaned/cutout in session).",
    )
    parser.add_argument(
        "--evaluate",
        nargs="?",
        const="__ALL__",
        default=None,
        help=(
            "Run evaluation after pipeline. "
            "If omitted no evaluation is run. "
            "If provided with no value evaluates all sessions; "
            "or pass a session folder path to evaluate that session."
        ),
    )
    parser.add_argument(
        "--hunyuan-conda-env",
        default=os.environ.get("HUNYUAN_CONDA_ENV", "hunyuan"),
        help="Conda env used for Hunyuan3D step1/step2 during resume mode.",
    )
    args = parser.parse_args()

    CKPT_PATH = args.ckpt_path

    if args.resume_from_cutout:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if (args.session_dir or "").strip():
            session_to_use = os.path.abspath(args.session_dir.strip())
        else:
            session_to_use = _find_latest_session_dir(script_dir)

        if not session_to_use:
            print("No session directory provided and no session_* folder found in placement_4.")
            sys.exit(1)

        if not os.path.isdir(session_to_use):
            print(f"Session directory not found: {session_to_use}")
            sys.exit(1)

        prompt_for_resume = (args.object_prompt or "").strip() or _read_object_prompt(session_to_use)
        if not prompt_for_resume:
            prompt_for_resume = input("Enter object prompt for OWLv2 detection: ").strip()
        if not prompt_for_resume:
            print("Object prompt is required for resume mode.")
            sys.exit(1)

        try:
            run_from_session_cutout(
                session_dir=session_to_use,
                object_prompt=prompt_for_resume,
                ckpt_path=CKPT_PATH,
                generated_image_path=args.generated_image_path,
                cutout_image_path=args.cutout_image_path,
                hunyuan_conda_env=args.hunyuan_conda_env,
            )
        except Exception as e:
            print(f"Resume mode failed: {e}")
            sys.exit(1)

        print(f"Viewer command after completion: python view_room.py {OUTPUT_PATH} --port 8080")
        sys.exit(0)

    # If evaluation flag set and user only wanted to evaluate, run and exit
    eval_flag = (args.evaluate is not None)
    if eval_flag and (not args.object_prompt) and not args.resume_from_cutout:
        # user invoked `--evaluate` in default run; run evaluation(s) and exit
        try:
            from evaluate_sessions import evaluate_session, evaluate_all_sessions
        except Exception as e:
            print(f"Evaluation helper import failed: {e}")
            sys.exit(1)

        if args.evaluate == "__ALL__":
            results = evaluate_all_sessions(initial_ckpt=CKPT_PATH)

            for s, txt in results.items():
                session_report_path = os.path.join(os.path.abspath(s), "detection_resource_report.txt")
                os.makedirs(os.path.dirname(session_report_path), exist_ok=True)
                with open(session_report_path, "a") as fh:
                    fh.write(f"\n--- Evaluation run: {datetime.datetime.now().isoformat()} ---\n")
                    fh.write(f"=== Session: {s} ===\n")
                    fh.write(txt)
                    fh.write("\n\n")
                print(f"Evaluation appended to {session_report_path}")

            print("Evaluation complete for all sessions.")
            sys.exit(0)
        else:
            # treat args.evaluate as a session path if provided
            sess = os.path.abspath(args.evaluate)
            try:
                outtxt = evaluate_session(sess, initial_ckpt=CKPT_PATH)
            except Exception as e:
                outtxt = f"Evaluation failed: {e}"
            report_path = os.path.join(sess, "detection_resource_report.txt")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "a") as fh:
                fh.write(f"\n--- Evaluation run: {datetime.datetime.now().isoformat()} ---\n")
                fh.write(f"Session: {sess}\n")
                fh.write(outtxt)
                fh.write("\n\n")
            print(f"Evaluation complete for session {sess}. Report appended to {report_path}")
            sys.exit(0)

    _configure_session_paths(DEFAULT_SESSION_DIR, create=True)
    print("\n--- Unified Pipeline (Single Runtime / Single Session) ---")
    print("1) Generate candidate camera views")
    print("2) Choose one camera interactively")
    print("3) Save camera metadata (.pt) inside this session folder")
    print("4) Generate diffusion-added image with Gemini API")
    print("5) Detect object bbox and auto-generate OBJ via 2d_3d.py")
    print(f"Session directory: {SESSION_DIR}")
    print(f"Viewer command after completion: python view_room.py {OUTPUT_PATH} --port 8080")

    select_camera_and_render()

    if args.object_prompt:
        object_prompt = args.object_prompt
    else:
        object_prompt = input("Enter rich edit prompt (e.g. 'a red car near the bench'): ").strip()
    object_obj_path = None

    if not object_prompt:
        print("Edit prompt cannot be empty.")
        sys.exit(1)
    _write_object_prompt(SESSION_DIR, object_prompt)
    print("No OBJ path provided, generating OBJ from prompt.")

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

# python detection_optimized.py "a computer" --resume-from-cutout

# python detection_optimized.py --evaluate

# python detection_optimized.py --evaluate 