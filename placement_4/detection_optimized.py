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
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

import os
import datetime

CKPT_PATH = "cupboard_room.ckpt"
RENDER_W, RENDER_H = 1280, 720
NUM_CAMERAS = 15
FOV_DEG = 60.0
ORBIT_SCALE = 0.06       # fraction of scene extent for orbit radius
CAMERA_HEIGHT_OFFSET = 0.0  # keep horizontal view
OPACITY_THRESHOLD = 0.1
HEIGHT_TOLERANCE = 0.15      # for surface filtering
DEVICE = "cuda"
# SH DC constant
C0 = 0.28209479177387814

# Configurable object class for detection
OBJECT_CLASSNAME = "vase"  # Change to "chair" or any other class as needed

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


def render_depth_map(means, scales, quats, opacities, camera, device=DEVICE):
    """Render depth map from Gaussians using gsplat.
    
    Returns:
        depth_map: (H, W) array with depth values in camera space
        alpha_map: (H, W) accumulation weights
    """
    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    quats_t = torch.tensor(quats, dtype=torch.float32, device=device)
    
    ops = opacities.copy().squeeze()
    if ops.min() < 0:
        ops = 1 / (1 + np.exp(-ops))
    opacities_t = torch.tensor(ops, dtype=torch.float32, device=device)
    
    viewmat = torch.tensor(camera.w2c, dtype=torch.float32, device=device)
    K = torch.tensor(camera.get_K(), dtype=torch.float32, device=device)
    
    # Compute depths in camera space for coloring
    pts_h = torch.cat([means_t, torch.ones(len(means_t), 1, device=device)], dim=1)
    cam_pts = (viewmat @ pts_h.T).T
    depths = cam_pts[:, 2:3].expand(-1, 3)  # Use depth as RGB for rendering
    
    renders, alphas, meta = rasterization(
        means=means_t,
        quats=quats_t / quats_t.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales_t),
        opacities=opacities_t,
        colors=depths,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=camera.width,
        height=camera.height,
        sh_degree=None,
        backgrounds=torch.zeros(1, 3, device=device),
    )
    
    depth_map = renders[0, :, :, 0].cpu().numpy()
    alpha_map = alphas[0].cpu().numpy()
    
    return depth_map, alpha_map


# ══════════════════════════════════════════════════════════════════════
# Optimized Detection Functions
# ══════════════════════════════════════════════════════════════════════

def extract_2d_keypoints(bbox, num_points=20, include_edges=True):
    """Extract 2D keypoints from detection bounding box.
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box
        num_points: Number of points to sample
        include_edges: Whether to include edge points for better PnP
        
    Returns:
        keypoints: (N, 2) array of (u, v) pixel coordinates
        types: list of keypoint types ('corner', 'edge', 'center', 'grid')
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    keypoints = []
    types = []
    
    # Corners (4 points)
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    keypoints.extend(corners)
    types.extend(['corner'] * 4)
    
    # Center point
    keypoints.append((cx, cy))
    types.append('center')
    
    if include_edges:
        # Edge midpoints (4 points)
        edge_mids = [(cx, y1), (cx, y2), (x1, cy), (x2, cy)]
        keypoints.extend(edge_mids)
        types.extend(['edge'] * 4)
        
        # Bottom edge points (important for surface contact)
        bottom_points = [(x1 + w * 0.25, y2), (x1 + w * 0.75, y2)]
        keypoints.extend(bottom_points)
        types.extend(['bottom'] * 2)
    
    # Grid points inside bbox
    remaining = num_points - len(keypoints)
    if remaining > 0:
        grid_size = int(np.ceil(np.sqrt(remaining)))
        for i in range(grid_size):
            for j in range(grid_size):
                if len(keypoints) >= num_points:
                    break
                u = x1 + w * (i + 0.5) / grid_size
                v = y1 + h * (j + 0.5) / grid_size
                keypoints.append((u, v))
                types.append('grid')
    
    return np.array(keypoints[:num_points]), types[:num_points]


def unproject_to_3d(keypoints_2d, depth_map, camera, sample_radius=3):
    """Unproject 2D keypoints to 3D using depth map.
    
    Args:
        keypoints_2d: (N, 2) array of (u, v) pixel coordinates
        depth_map: (H, W) depth values
        camera: SceneCamera object
        sample_radius: Radius for depth sampling (median filter)
        
    Returns:
        points_3d: (N, 3) world coordinates
        depths: (N,) sampled depth values
        valid: (N,) boolean mask for valid points
    """
    N = len(keypoints_2d)
    points_3d = np.zeros((N, 3))
    depths = np.zeros(N)
    valid = np.zeros(N, dtype=bool)
    
    H, W = depth_map.shape
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy
    
    # Get inverse of w2c for camera-to-world transform
    c2w_cv = np.linalg.inv(camera.w2c)
    
    for i, (u, v) in enumerate(keypoints_2d):
        u_int, v_int = int(round(u)), int(round(v))
        
        # Bounds check
        if not (sample_radius <= u_int < W - sample_radius and 
                sample_radius <= v_int < H - sample_radius):
            continue
        
        # Sample depth with median filter for robustness
        depth_window = depth_map[
            v_int - sample_radius : v_int + sample_radius + 1,
            u_int - sample_radius : u_int + sample_radius + 1
        ]
        
        # Filter out zeros/invalid depths
        valid_depths = depth_window[depth_window > 0.1]
        if len(valid_depths) == 0:
            continue
        
        depth = np.median(valid_depths)
        depths[i] = depth
        
        # Unproject to camera space (OpenCV convention)
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        # Transform to world space
        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_world = c2w_cv @ pt_cam
        
        points_3d[i] = pt_world[:3]
        valid[i] = True
    
    return points_3d, depths, valid


def estimate_pose_pnp(keypoints_2d, points_3d, camera, use_ransac=True):
    """Estimate object pose using OpenCV solvePnP.
    
    Args:
        keypoints_2d: (N, 2) image points
        points_3d: (N, 3) corresponding 3D points (in object frame)
        camera: SceneCamera object
        use_ransac: Whether to use RANSAC for robustness
        
    Returns:
        rvec: rotation vector (Rodriguez)
        tvec: translation vector
        success: boolean indicating success
        inliers: indices of inliers (if RANSAC)
    """
    K = camera.get_K().astype(np.float64)
    dist_coeffs = np.zeros(5)  # Assuming no distortion
    
    # Filter valid correspondences
    valid = np.all(np.isfinite(points_3d), axis=1) & np.all(np.isfinite(keypoints_2d), axis=1)
    
    if valid.sum() < 4:
        return None, None, False, None
    
    obj_pts = points_3d[valid].astype(np.float64)
    img_pts = keypoints_2d[valid].astype(np.float64)
    
    # Check for coplanar points (use IPPE method)
    pts_centered = obj_pts - obj_pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts_centered)
    is_coplanar = s[-1] / s[0] < 0.01 if len(s) > 0 and s[0] > 0 else False
    
    if use_ransac:
        try:
            if is_coplanar:
                # Use IPPE for coplanar points (more stable)
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE
                )
            else:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            return rvec, tvec, success, inliers
        except cv2.error:
            pass
    
    # Fallback to iterative method
    try:
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return rvec, tvec, success, None
    except cv2.error:
        return None, None, False, None


def identify_supporting_surface(points_3d, depths, bbox, depth_map, camera, 
                                 height_tolerance=0.05):
    """Identify the supporting surface (table/floor) for object placement.
    
    Uses the bottom region of the detection to find the surface depth.
    
    Args:
        points_3d: Unprojected 3D points from bbox
        depths: Corresponding depth values
        bbox: (x1, y1, x2, y2) detection bbox
        depth_map: Full depth map
        camera: SceneCamera object
        height_tolerance: Tolerance for surface plane detection
        
    Returns:
        surface_depth: Depth of the supporting surface
        surface_normal: Estimated surface normal
        surface_height: Z-coordinate of the surface in world space
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = depth_map.shape
    
    # Focus on bottom 20% of the bbox (where object meets surface)
    bottom_margin = int((y2 - y1) * 0.8)
    bottom_region_y1 = min(y1 + bottom_margin, y2 - 5)
    
    # Clamp to image bounds
    bottom_region_y1 = max(0, min(bottom_region_y1, H - 1))
    y2_clamped = max(0, min(y2, H))
    x1_clamped = max(0, min(x1, W - 1))
    x2_clamped = max(1, min(x2, W))
    
    if bottom_region_y1 >= y2_clamped or x1_clamped >= x2_clamped:
        # Fallback: use full bbox center
        bottom_depths = depth_map[y1:y2, x1:x2]
    else:
        bottom_depths = depth_map[bottom_region_y1:y2_clamped, x1_clamped:x2_clamped]
    
    # Get the maximum depth in bottom region (farthest point = surface)
    valid_bottom_depths = bottom_depths[bottom_depths > 0.1]
    
    if len(valid_bottom_depths) == 0:
        # Fallback
        valid_depths = depths[depths > 0.1]
        surface_depth = np.max(valid_depths) if len(valid_depths) > 0 else 1.0
    else:
        # Surface is the farthest point in bottom region (max depth)
        # Use 90th percentile to avoid outliers
        surface_depth = np.percentile(valid_bottom_depths, 90)
    
    # Unproject surface point to get world coordinates
    cx_bbox = (x1 + x2) / 2
    cy_surface = y2 - 5  # Near bottom of bbox
    
    c2w_cv = np.linalg.inv(camera.w2c)
    x_cam = (cx_bbox - camera.cx) * surface_depth / camera.fx
    y_cam = (cy_surface - camera.cy) * surface_depth / camera.fy
    z_cam = surface_depth
    
    pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    pt_world = c2w_cv @ pt_cam
    surface_height = pt_world[2]
    
    # Estimate surface normal (assume roughly horizontal)
    surface_normal = np.array([0.0, 0.0, 1.0])
    
    return surface_depth, surface_normal, surface_height


def compute_empty_volume(points_3d, depths, surface_depth, bbox, camera,
                          object_height_estimate=None):
    """Compute the 3D empty volume where the object should be placed.
    
    The current method detects gaussians on the surface. This computes
    the volume ABOVE the surface where the object would occupy space.
    
    Args:
        points_3d: Unprojected 3D points
        depths: Depth values for these points  
        surface_depth: Depth of the supporting surface
        bbox: Detection bounding box
        camera: SceneCamera
        object_height_estimate: Optional estimated height of object
        
    Returns:
        volume_center: 3D center of the empty volume
        volume_extent: (width, depth, height) of the volume
        volume_bounds: (min_pt, max_pt) in world coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # Get valid 3D points
    valid = np.all(np.isfinite(points_3d), axis=1) & (depths > 0.1)
    valid_pts = points_3d[valid]
    valid_depths = depths[valid]
    
    if len(valid_pts) == 0:
        # Fallback: estimate from bbox center
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        fallback_depth = surface_depth - 0.1
        c2w_cv = np.linalg.inv(camera.w2c)
        x_cam = (cx - camera.cx) * fallback_depth / camera.fx
        y_cam = (cy - camera.cy) * fallback_depth / camera.fy
        pt_cam = np.array([x_cam, y_cam, fallback_depth, 1.0])
        center = (c2w_cv @ pt_cam)[:3]
        return center, np.array([0.1, 0.1, 0.2]), (center - 0.1, center + 0.1)
    
    # Find depth range: object is CLOSER than surface (smaller depth)
    min_depth = np.min(valid_depths)
    
    # Object occupies space from min_depth to surface_depth
    object_depth_range = surface_depth - min_depth
    
    if object_height_estimate is None:
        # Estimate from bbox aspect ratio and depth range
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        # Use depth range as proxy for object depth/height
        object_height_estimate = object_depth_range
    
    # Compute volume bounds in world coordinates
    min_pt = np.min(valid_pts, axis=0)
    max_pt = np.max(valid_pts, axis=0)
    
    # The actual volume is ABOVE the detected points
    # Shift z-range upward to represent empty space
    volume_min = min_pt.copy()
    volume_max = max_pt.copy()
    
    # The detected points are ON the surface, object goes UP from there
    volume_min[2] = min_pt[2]  # Bottom of object (surface level)
    volume_max[2] = max_pt[2] + object_height_estimate  # Top of object
    
    volume_center = (volume_min + volume_max) / 2
    volume_extent = volume_max - volume_min
    
    return volume_center, volume_extent, (volume_min, volume_max)


def filter_gaussians_by_depth(means, opacities, camera, bbox, depth_map,
                               surface_depth, depth_margin=0.05):
    """Filter Gaussians using depth-aware selection.
    
    Instead of selecting ALL gaussians in the 2D bbox (which includes
    the surface), this selects only gaussians that are:
    1. Within the 2D bbox
    2. CLOSER than the supporting surface (in front of it)
    3. Within a reasonable depth range for the object
    
    Args:
        means: (N, 3) Gaussian positions
        opacities: (N,) Gaussian opacities
        camera: SceneCamera
        bbox: (x1, y1, x2, y2) detection box
        depth_map: Rendered depth map
        surface_depth: Depth of supporting surface
        depth_margin: Margin for depth filtering
        
    Returns:
        object_indices: Indices of Gaussians belonging to the object region
        surface_indices: Indices of Gaussians on the surface (to exclude)
    """
    x1, y1, x2, y2 = bbox
    
    # Project all Gaussians
    u_all, v_all, z_all, proj_valid = camera.project(means)
    
    # Initial 2D filtering (within bbox)
    in_bbox = (
        proj_valid
        & (u_all >= x1) & (u_all <= x2)
        & (v_all >= y1) & (v_all <= y2)
        & (opacities > OPACITY_THRESHOLD)
    )
    
    # Depth-based filtering
    # Object gaussians should be CLOSER than surface (smaller z)
    min_object_depth = np.min(z_all[in_bbox]) if in_bbox.any() else 0.1
    
    # Object occupies: [min_depth, surface_depth - margin]
    in_object_depth = (
        in_bbox
        & (z_all < surface_depth - depth_margin)  # In front of surface
        & (z_all > min_object_depth - depth_margin)  # Behind object front
    )
    
    # Surface gaussians: AT or BEHIND surface depth
    on_surface = (
        in_bbox
        & (z_all >= surface_depth - depth_margin)
    )
    
    object_indices = np.where(in_object_depth)[0]
    surface_indices = np.where(on_surface)[0]
    
    return object_indices, surface_indices


def compute_optimal_placement(means, object_indices, surface_indices, 
                               surface_height, camera):
    """Compute optimal 3D placement for the object.
    
    Instead of placing at the center of detected gaussians (which are
    on the surface), compute the placement position ABOVE the surface.
    
    Args:
        means: All Gaussian positions
        object_indices: Indices of object-region gaussians
        surface_indices: Indices of surface gaussians  
        surface_height: Z-coordinate of the surface
        camera: SceneCamera
        
    Returns:
        placement_center: 3D position for object center
        placement_scale: Scale factor for the object
        rotation_matrix: 3x3 rotation matrix for object orientation
    """
    if len(object_indices) == 0:
        # Fallback: use surface gaussians and offset upward
        if len(surface_indices) == 0:
            return None, None, None
        target_means = means[surface_indices]
    else:
        target_means = means[object_indices]
    
    # Compute bounding box of detected region
    target_min = target_means.min(axis=0)
    target_max = target_means.max(axis=0)
    target_center = target_means.mean(axis=0)
    target_extent = target_max - target_min
    
    # Scale: use minimum of X, Y extents (for cylindrical objects)
    horizontal_scale = min(target_extent[0], target_extent[1])
    
    # Placement center: XY from detection, Z offset above surface
    placement_center = target_center.copy()
    
    # If we got surface gaussians, offset upward
    if len(object_indices) == 0 and len(surface_indices) > 0:
        # Object bottom should be at surface height
        estimated_height = horizontal_scale * 1.5  # Assume aspect ratio
        placement_center[2] = surface_height + estimated_height / 2
    
    # Default rotation: identity (upright)
    rotation_matrix = np.eye(3)
    
    return placement_center, horizontal_scale, rotation_matrix


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


def add_object_to_scene(object_image_path, object_obj_path, cam_idx=None):
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

    # Detect object in image
    # YOLO detection timing
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

    # Unprojection & 3D detection timing
    t_unproj_start = time.time()
    if cam_idx is None:
        cam_idx = 0
    if cam_idx < 0 or cam_idx >= NUM_CAMERAS:
        print("Invalid camera index. Defaulting to 0.")
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

    # ═══════════════════════════════════════════════════════════════════
    # OPTIMIZED DETECTION: Depth-aware Gaussian filtering with PnP
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Optimized Depth-Aware Detection ---")
    
    # Step 1: Render depth map from scene Gaussians
    t_depth_start = time.time()
    depth_map, alpha_map = render_depth_map(means, scales, quats, opacities_raw, cam)
    print(f"  Depth map rendered: shape={depth_map.shape}, range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")
    
    # Save depth visualization
    valid_depth = depth_map[alpha_map.squeeze() > 0.1]
    if len(valid_depth) > 0:
        depth_norm = (depth_map - valid_depth.min()) / (valid_depth.max() - valid_depth.min() + 1e-6)
    else:
        depth_norm = depth_map / (depth_map.max() + 1e-6)
    depth_vis = (plt.cm.viridis(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(os.path.join(SESSION_DIR, f"{OBJECT_CLASSNAME}_depth_map.png"))
    print(f"  Saved depth visualization: {OBJECT_CLASSNAME}_depth_map.png")
    t_depth_end = time.time()
    timings['Depth map rendering'] = t_depth_end - t_depth_start
    
    # Step 2: Extract 2D keypoints from detection bbox
    t_keypoint_start = time.time()
    keypoints_2d, kp_types = extract_2d_keypoints(object_det["bbox"], num_points=25, include_edges=True)
    print(f"  Extracted {len(keypoints_2d)} keypoints from bbox")
    
    # Step 3: Unproject keypoints to 3D using depth map
    points_3d, depths, valid_kp = unproject_to_3d(keypoints_2d, depth_map, cam, sample_radius=5)
    valid_count = valid_kp.sum()
    print(f"  Unprojected {valid_count}/{len(keypoints_2d)} keypoints to 3D")
    
    if valid_count < 4:
        print("  WARNING: Insufficient valid keypoints for PnP, using fallback method")
    t_keypoint_end = time.time()
    timings['Keypoint extraction & unprojection'] = t_keypoint_end - t_keypoint_start
    
    # Step 4: Identify supporting surface (table/floor)
    t_surface_start = time.time()
    surface_depth, surface_normal, surface_height = identify_supporting_surface(
        points_3d, depths, object_det["bbox"], depth_map, cam
    )
    print(f"  Surface detection: depth={surface_depth:.4f}, height_z={surface_height:.4f}")
    t_surface_end = time.time()
    timings['Surface detection'] = t_surface_end - t_surface_start
    
    # Step 5: Pose estimation using PnP (for orientation refinement)
    t_pnp_start = time.time()
    if valid_count >= 4:
        # Create object-frame points (centered at origin)
        valid_pts_3d = points_3d[valid_kp]
        obj_frame_pts = valid_pts_3d - valid_pts_3d.mean(axis=0)
        
        rvec, tvec, pnp_success, inliers = estimate_pose_pnp(
            keypoints_2d[valid_kp], obj_frame_pts, cam, use_ransac=True
        )
        
        if pnp_success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            print(f"  PnP pose estimation: SUCCESS (inliers: {len(inliers) if inliers is not None else 'N/A'})")
        else:
            rotation_matrix = np.eye(3)
            print(f"  PnP pose estimation: FAILED, using identity rotation")
    else:
        rotation_matrix = np.eye(3)
        print(f"  PnP skipped: insufficient keypoints")
    t_pnp_end = time.time()
    timings['PnP pose estimation'] = t_pnp_end - t_pnp_start
    
    # Step 6: Depth-aware Gaussian filtering (OPTIMIZED)
    t_filter_start = time.time()
    object_indices, surface_indices = filter_gaussians_by_depth(
        means, opacities, cam, object_det["bbox"], depth_map, 
        surface_depth, depth_margin=0.03
    )
    print(f"  Depth-filtered Gaussians:")
    print(f"    Object region: {len(object_indices):,} gaussians (in front of surface)")
    print(f"    Surface region: {len(surface_indices):,} gaussians (on/behind surface)")
    
    # Step 7: Compute empty volume for placement
    volume_center, volume_extent, volume_bounds = compute_empty_volume(
        points_3d, depths, surface_depth, object_det["bbox"], cam
    )
    print(f"  Computed placement volume:")
    print(f"    Center: [{volume_center[0]:.4f}, {volume_center[1]:.4f}, {volume_center[2]:.4f}]")
    print(f"    Extent: [{volume_extent[0]:.4f}, {volume_extent[1]:.4f}, {volume_extent[2]:.4f}]")
    
    # Step 8: Optimal placement computation
    placement_center, placement_scale, placement_rotation = compute_optimal_placement(
        means, object_indices, surface_indices, surface_height, cam
    )
    
    if placement_center is not None:
        print(f"  Optimal placement:")
        print(f"    Center: [{placement_center[0]:.4f}, {placement_center[1]:.4f}, {placement_center[2]:.4f}]")
        print(f"    Scale: {placement_scale:.4f}")
    t_filter_end = time.time()
    timings['Depth filtering & placement'] = t_filter_end - t_filter_start
    
    # Use optimized indices (object_indices only, excluding surface)
    # For highlighting, combine both for visualization comparison
    all_bbox_indices = np.union1d(object_indices, surface_indices)
    
    t_unproj_end = time.time()
    timings['Unprojection & 3D detection'] = t_unproj_end - t_unproj_start
    t_highlight_start = time.time()
    print(f"\nTotal Gaussians in {OBJECT_CLASSNAME} region:")
    print(f"  Optimized (object only): {len(object_indices):,}")
    print(f"  Original (all in bbox): {len(all_bbox_indices):,}")
    print(f"  Reduction: {100*(1 - len(object_indices)/max(1,len(all_bbox_indices))):.1f}%")

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

    # Verification render - Optimized detection (object gaussians in red)
    features_dc_viz = features_dc.copy()
    if features_dc_viz.ndim == 3:
        features_dc_viz = features_dc_viz.squeeze(1)
    colors_viz = np.clip(C0 * features_dc_viz + 0.5, 0, 1)
    colors_viz[object_indices] = [1.0, 0.0, 0.0]  # Red for object region
    features_dc_mod_viz = (colors_viz - 0.5) / C0
    if features_dc.ndim == 3:
        features_dc_mod_viz = features_dc_mod_viz[:, np.newaxis, :]
    rendered_mod, _ = render_gaussians(
        means, scales, quats, features_dc_mod_viz, opacities_raw, cam
    )
    Image.fromarray((rendered_mod * 255).astype(np.uint8)).save(os.path.join(SESSION_DIR, f"{OBJECT_CLASSNAME}_highlighted_verification.png"))
    print(f"Saved {OBJECT_CLASSNAME}_highlighted_verification.png in {SESSION_DIR}")
    
    # Enhanced comparison visualization: Object (red) vs Surface (blue)
    if len(surface_indices) > 0:
        features_dc_compare = features_dc.copy()
        if features_dc_compare.ndim == 3:
            features_dc_compare = features_dc_compare.squeeze(1)
        colors_compare = np.clip(C0 * features_dc_compare + 0.5, 0, 1)
        colors_compare[object_indices] = [1.0, 0.0, 0.0]    # Red: object region (front)
        colors_compare[surface_indices] = [0.0, 0.5, 1.0]   # Blue: surface region (back)
        features_compare_viz = (colors_compare - 0.5) / C0
        if features_dc.ndim == 3:
            features_compare_viz = features_compare_viz[:, np.newaxis, :]
        rendered_compare, _ = render_gaussians(
            means, scales, quats, features_compare_viz, opacities_raw, cam
        )
        Image.fromarray((rendered_compare * 255).astype(np.uint8)).save(
            os.path.join(SESSION_DIR, f"{OBJECT_CLASSNAME}_depth_separation.png")
        )
        print(f"Saved {OBJECT_CLASSNAME}_depth_separation.png (red=object, blue=surface)")
    
    t_highlight_end = time.time()
    timings['Highlighting & ckpt'] = t_highlight_end - t_highlight_start
    t_vase_start = time.time()

    # 2. Add vase gaussians to the original checkpoint (no red highlight)
    print(f"\n--- Generating and integrating {OBJECT_CLASSNAME} gaussians from OBJ ---")
    if len(object_indices) == 0:
        print(f"No gaussians detected for {OBJECT_CLASSNAME} placement, skipping integration.")
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
            f.write(f"Status: No gaussians detected for {OBJECT_CLASSNAME}\n")
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
    
    # ═══════════════════════════════════════════════════════════════════
    # OPTIMIZED PLACEMENT: Use depth-aware computed values
    # ═══════════════════════════════════════════════════════════════════
    
    # Use optimized placement if available, fallback to original method
    if placement_center is not None and placement_scale is not None:
        # Use optimized depth-aware placement
        translation = placement_center
        scale = placement_scale
        target_center = placement_center
        target_extent = volume_extent
        print(f"  Using OPTIMIZED placement (depth-aware)")
    else:
        # Fallback to original method
        target_means = means[object_indices]
        target_min = target_means.min(axis=0)
        target_max = target_means.max(axis=0)
        target_center = target_means.mean(axis=0)
        target_extent = target_max - target_min
        scale = min(target_extent[0], target_extent[1])
        translation = target_center
        print(f"  Using FALLBACK placement (original method)")
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
    elif len(sys.argv) == 3 or len(sys.argv) == 4:
        object_image_path = sys.argv[1]
        object_obj_path = sys.argv[2]
        cam_idx = int(sys.argv[3]) - 1 if len(sys.argv) == 4 else 0
        add_object_to_scene(object_image_path, object_obj_path, cam_idx)
    else:
        print("Usage:")
        print("  python detection_optimized.py           # Camera selection and render")
        print("  python detection_optimized.py <image_with_object.png> <object.obj>   # Object addition")
        
        
        
# python view_room.py /home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260213_004047/room_with_object.ckpt --port 8080
