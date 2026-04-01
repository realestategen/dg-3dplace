import os
import math
import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from gsplat import rasterization

# ══════════════════════════════════════════════════════════════════════
# Configuration & Paths
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input Checkpoints (Update these to your actual ckpt paths)
INITIAL_CKPT = os.path.join(BASE_DIR, "data", "checkpoints", "initial_scene.ckpt")
FINAL_CKPT = os.path.join(BASE_DIR, "data", "checkpoints", "final_scene.ckpt")

# Output Directories
MULTI_VIEW_INIT_DIR = os.path.join(BASE_DIR, "data", "2d_images", "multi_view", "initial")
MULTI_VIEW_FINAL_DIR = os.path.join(BASE_DIR, "data", "2d_images", "multi_view", "final")

# Single View Outputs (For original metrics)
SINGLE_INIT_OUT = os.path.join(BASE_DIR, "data", "2d_images", "initial_scene_render.png")
SINGLE_FINAL_OUT = os.path.join(BASE_DIR, "data", "2d_images", "final_scene_render.png")

# Rendering Parameters
RENDER_W, RENDER_H = 1280, 720
NUM_CAMERAS = 10           # Number of multi-view angles to generate
FOV_DEG = 60.0
ORBIT_SCALE = 0.008        # Fraction of scene extent for orbit radius
CAMERA_HEIGHT_OFFSET = 0.3 # Keep horizontal view
OPACITY_THRESHOLD = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814

# ══════════════════════════════════════════════════════════════════════
# Camera & Rendering Engine
# ══════════════════════════════════════════════════════════════════════
class SceneCamera:
    def __init__(self, position, wxyz, fov_rad, width, height):
        self.position = np.array(position, dtype=np.float64)
        self.wxyz = np.array(wxyz, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.fov_rad = float(fov_rad)

        self.fy = (height / 2) / np.tan(fov_rad / 2)
        self.fx = self.fy 
        self.cx = width / 2.0
        self.cy = height / 2.0

        quat_xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
        rot = R.from_quat(quat_xyzw).as_matrix()
        self.c2w = np.eye(4, dtype=np.float64)
        self.c2w[:3, :3] = rot
        self.c2w[:3, 3] = self.position

        w2c_gl = np.linalg.inv(self.c2w)
        self.w2c = w2c_gl.copy()
        self.w2c[1, :] *= -1
        self.w2c[2, :] *= -1

    def get_K(self):
        return np.array([
            [self.fx, 0, self.cx], 
            [0, self.fy, self.cy], 
            [0, 0, 1]
        ], dtype=np.float64)

def make_camera_from_config(scene_center, orbit_radius, height_offset, azimuth,
                            cos_axis, sin_axis, fixed_axis, world_up_vec, fov_rad, w, h):
    pos = np.zeros(3)
    pos[cos_axis]   = scene_center[cos_axis]   + orbit_radius * math.cos(azimuth)
    pos[sin_axis]   = scene_center[sin_axis]   + orbit_radius * math.sin(azimuth)
    pos[fixed_axis] = scene_center[fixed_axis] + height_offset

    forward = scene_center - pos
    forward = forward / np.linalg.norm(forward)

    wup = np.array(world_up_vec, dtype=np.float64)
    right = np.cross(forward, wup)
    if np.linalg.norm(right) < 1e-6:
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

def render_gaussians(means, scales, quats, features_dc, opacities_raw, camera, device=DEVICE):
    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    quats_t = torch.tensor(quats, dtype=torch.float32, device=device)

    fdc = features_dc.copy()
    if fdc.ndim == 3:
        fdc = fdc.squeeze(1)
    colors_rgb = np.clip(C0 * fdc + 0.5, 0, 1)
    colors_t = torch.tensor(colors_rgb, dtype=torch.float32, device=device)

    ops = opacities_raw.copy().squeeze()
    if ops.min() < 0:
        ops = 1 / (1 + np.exp(-ops))
    opacities_t = torch.tensor(ops, dtype=torch.float32, device=device)

    viewmat = torch.tensor(camera.w2c, dtype=torch.float32, device=device)
    K = torch.tensor(camera.get_K(), dtype=torch.float32, device=device)

    # 1. Render without the backgrounds parameter to bypass the gsplat shape assertion
    renders, alphas, _ = rasterization(
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
    )
    
    # 2. Extract the single image and its alpha (transparency) mask
    rgb = renders[0]
    alpha = alphas[0]
    
    # 3. Mathematically apply a solid white background using the alpha mask
    white_bg = torch.ones_like(rgb)
    blended_rgb = rgb + (1.0 - alpha) * white_bg
    
    return np.clip(blended_rgb.cpu().numpy(), 0, 1)

def extract_tensors(ckpt_path):
    print(f"Loading weights from {os.path.basename(ckpt_path)}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    return (
        state["_model.means"].numpy(),
        state["_model.scales"].numpy(),
        state["_model.quats"].numpy(),
        state["_model.features_dc"].numpy(),
        state["_model.opacities"].numpy()
    )

# ══════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(MULTI_VIEW_INIT_DIR, exist_ok=True)
    os.makedirs(MULTI_VIEW_FINAL_DIR, exist_ok=True)

    if not os.path.exists(INITIAL_CKPT) or not os.path.exists(FINAL_CKPT):
        print(f"ERROR: Checkpoints not found.\nPlease ensure your .ckpt files are located at:\n{INITIAL_CKPT}\n{FINAL_CKPT}")
        return

    # 1. Load Initial Scene to determine camera trajectory
    means, scales, quats, features_dc, opacities_raw = extract_tensors(INITIAL_CKPT)
    
    # Calculate scene bounds based on visible gaussians
    opacities_sig = (1 / (1 + np.exp(-opacities_raw))).squeeze()
    vis_mask = opacities_sig > OPACITY_THRESHOLD
    vis_means = means[vis_mask]
    scene_center = vis_means.mean(axis=0)
    scene_extent = vis_means.max(axis=0) - vis_means.min(axis=0)
    
    orbit_radius = float(np.linalg.norm(scene_extent)) * ORBIT_SCALE
    azimuth_angles = np.linspace(0, 2 * math.pi, NUM_CAMERAS, endpoint=False)
    fov_rad = math.radians(FOV_DEG)

    cameras = []
    for azimuth in azimuth_angles:
        cam = make_camera_from_config(
            scene_center, orbit_radius, CAMERA_HEIGHT_OFFSET, azimuth,
            cos_axis=0, sin_axis=1, fixed_axis=2, 
            world_up_vec=[0, 0, 1], fov_rad=fov_rad, w=RENDER_W, h=RENDER_H
        )
        cameras.append(cam)

    # 2. Render Initial Scene
    print(f"\n--- Rendering Initial Scene ({NUM_CAMERAS} views) ---")
    for i, cam in enumerate(cameras):
        img = render_gaussians(means, scales, quats, features_dc, opacities_raw, cam)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.save(os.path.join(MULTI_VIEW_INIT_DIR, f"view_{i:03d}.png"))
        
        # Save view_000 as the base single-view render
        if i == 0:
            img_pil.save(SINGLE_INIT_OUT)
        print(f"Rendered initial view {i:03d}")

    # Clear memory
    del means, scales, quats, features_dc, opacities_raw
    torch.cuda.empty_cache()

    # 3. Render Final Scene (Using identical camera configs)
    print(f"\n--- Rendering Final Scene ({NUM_CAMERAS} views) ---")
    means_f, scales_f, quats_f, features_dc_f, opacities_raw_f = extract_tensors(FINAL_CKPT)

    for i, cam in enumerate(cameras):
        img = render_gaussians(means_f, scales_f, quats_f, features_dc_f, opacities_raw_f, cam)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.save(os.path.join(MULTI_VIEW_FINAL_DIR, f"view_{i:03d}.png"))
        
        # Save view_000 as the base single-view render
        if i == 0:
            img_pil.save(SINGLE_FINAL_OUT)
        print(f"Rendered final view {i:03d}")

    print("\n✅ All multi-view and single-view renders completed successfully.")
    print("You can now run 'python run_evaluation.py'.")

if __name__ == "__main__":
    main()