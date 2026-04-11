import math
import os
import subprocess
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh

C0 = 0.28209479177387814


def _as_scene(glb_path: str) -> trimesh.Scene:
    loaded = trimesh.load(glb_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    scene = trimesh.Scene()
    scene.add_geometry(loaded)
    return scene


def _material_image(material) -> Optional[np.ndarray]:
    if material is None:
        return None

    # Common trimesh material layouts for GLB/OBJ+MTL.
    candidates = []
    for attr in ("image", "baseColorTexture", "base_color_texture", "albedoTexture"):
        if hasattr(material, attr):
            candidates.append(getattr(material, attr))

    for c in candidates:
        if c is None:
            continue
        if hasattr(c, "image") and c.image is not None:
            return np.asarray(c.image.convert("RGB") if hasattr(c.image, "convert") else c.image)
        if hasattr(c, "convert"):
            return np.asarray(c.convert("RGB"))
        arr = np.asarray(c)
        if arr.size > 0:
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] >= 3:
                return arr[..., :3]

    return None


def _sample_texture_bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    tex = texture
    if tex.dtype != np.float32:
        tex = tex.astype(np.float32)
    if tex.max() > 1.0:
        tex = tex / 255.0

    h, w = tex.shape[:2]
    uv = np.clip(uv, 0.0, 1.0)

    x = uv[:, 0] * (w - 1)
    y = (1.0 - uv[:, 1]) * (h - 1)

    x0 = np.floor(x).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (x - x0).reshape(-1, 1)
    wy = (y - y0).reshape(-1, 1)

    c00 = tex[y0, x0, :3]
    c01 = tex[y0, x1, :3]
    c10 = tex[y1, x0, :3]
    c11 = tex[y1, x1, :3]

    return (1.0 - wx) * (1.0 - wy) * c00 + wx * (1.0 - wy) * c01 + (1.0 - wx) * wy * c10 + wx * wy * c11


def _sample_mesh_points_colors(mesh: trimesh.Trimesh, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    area_sum = float(areas.sum())
    if area_sum <= 0.0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    face_idx = np.random.choice(len(faces), size=n, p=(areas / area_sum))

    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    sr1 = np.sqrt(r1)
    b0 = 1.0 - sr1
    b1 = sr1 * (1.0 - r2)
    b2 = sr1 * r2
    bary = np.stack([b0, b1, b2], axis=1)

    f = faces[face_idx]
    p0 = verts[f[:, 0]]
    p1 = verts[f[:, 1]]
    p2 = verts[f[:, 2]]
    pts = b0[:, None] * p0 + b1[:, None] * p1 + b2[:, None] * p2

    colors = None
    visual = mesh.visual

    face_colors = np.asarray(getattr(visual, "face_colors", np.array([])))
    if face_colors.size > 0 and len(face_colors) >= len(faces):
        colors = face_colors[face_idx, :3].astype(np.float32)
        if colors.max() > 1.0:
            colors /= 255.0

    if colors is None:
        vertex_colors = np.asarray(getattr(visual, "vertex_colors", np.array([])))
        if vertex_colors.size > 0 and len(vertex_colors) >= len(verts):
            vc = vertex_colors[f, :3].astype(np.float32)
            if vc.max() > 1.0:
                vc /= 255.0
            colors = b0[:, None] * vc[:, 0] + b1[:, None] * vc[:, 1] + b2[:, None] * vc[:, 2]

    if colors is None:
        uv = np.asarray(getattr(visual, "uv", np.array([])))
        tex = _material_image(getattr(visual, "material", None))
        if uv.size > 0 and tex is not None and len(uv) >= len(verts):
            uv_tri = uv[f]
            uv_s = b0[:, None] * uv_tri[:, 0] + b1[:, None] * uv_tri[:, 1] + b2[:, None] * uv_tri[:, 2]
            colors = _sample_texture_bilinear(tex, uv_s)

    if colors is None:
        colors = np.tile(np.array([0.65, 0.65, 0.65], dtype=np.float32), (n, 1))

    return pts.astype(np.float32), np.clip(colors.astype(np.float32), 0.0, 1.0)


def _extract_points_colors_from_glb(glb_path: str, num_gaussians: int) -> Tuple[np.ndarray, np.ndarray]:
    scene = _as_scene(glb_path)
    meshes = scene.dump(concatenate=False)
    meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
    if len(meshes) == 0:
        raise RuntimeError(f"No mesh geometry found in GLB: {glb_path}")

    areas = np.array([float(m.area) for m in meshes], dtype=np.float64)
    area_sum = float(areas.sum())
    if area_sum <= 0:
        areas = np.ones(len(meshes), dtype=np.float64)
        area_sum = float(areas.sum())

    counts = np.maximum(1, np.floor(num_gaussians * areas / area_sum).astype(np.int64))
    counts[-1] += int(num_gaussians - counts.sum())

    all_pts = []
    all_cols = []
    for mesh, n in zip(meshes, counts.tolist()):
        pts, cols = _sample_mesh_points_colors(mesh, int(n))
        if len(pts) == 0:
            continue
        all_pts.append(pts)
        all_cols.append(cols)

    if not all_pts:
        raise RuntimeError(f"Failed sampling points/colors from GLB: {glb_path}")

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)

    if len(points) > num_gaussians:
        sel = np.random.choice(len(points), size=num_gaussians, replace=False)
        points = points[sel]
        colors = colors[sel]

    return points, colors


def _run_render_colmap_stage(glb_path: str, work_dir: str, num_views: int = 24) -> None:
    images_dir = os.path.join(work_dir, "renders")
    os.makedirs(images_dir, exist_ok=True)
    db_path = os.path.join(work_dir, "colmap.db")
    sparse_dir = os.path.join(work_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    # Render stage placeholder: this function reserves the pipeline stage and filesystem layout.
    # If your environment has a custom renderer, drop it in here to generate images_dir/*.png.
    if len([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]) == 0:
        return

    colmap_bin = "colmap"
    try:
        subprocess.run([colmap_bin, "help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        return

    subprocess.run(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            db_path,
            "--image_path",
            images_dir,
            "--ImageReader.single_camera",
            "1",
        ],
        check=False,
    )
    subprocess.run([colmap_bin, "exhaustive_matcher", "--database_path", db_path], check=False)
    subprocess.run(
        [
            colmap_bin,
            "mapper",
            "--database_path",
            db_path,
            "--image_path",
            images_dir,
            "--output_path",
            sparse_dir,
        ],
        check=False,
    )


def glb_to_gaussians(
    glb_path: str,
    num_gaussians: int = 15000,
    target_scale: Optional[float] = None,
    scale_factor: float = 0.4,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    opacity_logit: float = 5.0,
    run_render_colmap: bool = False,
    work_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB not found: {glb_path}")

    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(glb_path), "glb_to_gs")
    os.makedirs(work_dir, exist_ok=True)

    if run_render_colmap:
        _run_render_colmap_stage(glb_path, work_dir)

    points, colors = _extract_points_colors_from_glb(glb_path, num_gaussians)

    obj_min = points.min(axis=0)
    obj_max = points.max(axis=0)
    obj_center = (obj_min + obj_max) / 2.0
    points = points - obj_center

    # Match existing room coordinate conversion used by detection_optimized.
    pts_scene = np.column_stack([points[:, 0], -points[:, 2], points[:, 1]]).astype(np.float32)

    extent_max = float(max((obj_max - obj_min).max(), 1e-6))
    if target_scale is not None:
        pts_scene *= float(target_scale * scale_factor / extent_max)

    if rotation is not None:
        pts_scene = np.dot(pts_scene, np.asarray(rotation, dtype=np.float32).T)

    if translation is not None:
        pts_scene += np.asarray(translation, dtype=np.float32)

    means = torch.tensor(pts_scene, dtype=torch.float32)

    adaptive_radius = (np.ptp(pts_scene, axis=0).prod() / max(1, len(pts_scene))) ** (1.0 / 3.0) * 1.5
    log_scale = math.log(max(float(adaptive_radius), 1e-7))
    scales = torch.full((len(pts_scene), 3), log_scale, dtype=torch.float32)

    quats = torch.zeros(len(pts_scene), 4, dtype=torch.float32)
    quats[:, 0] = 1.0

    sh_color = (np.clip(colors, 0.0, 1.0) - 0.5) / C0
    features_dc = torch.tensor(sh_color, dtype=torch.float32).unsqueeze(1)
    features_rest = torch.zeros(len(pts_scene), 15, 3, dtype=torch.float32)
    opacities = torch.full((len(pts_scene), 1), float(opacity_logit), dtype=torch.float32)

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "opacities": opacities,
    }


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert textured GLB into Gaussian tensors")
    parser.add_argument("--glb", required=True, help="Path to textured GLB")
    parser.add_argument("--out", required=True, help="Output .pt file path")
    parser.add_argument("--num-gaussians", type=int, default=15000)
    parser.add_argument("--target-scale", type=float, default=None)
    parser.add_argument("--scale-factor", type=float, default=0.4)
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=0.0)
    parser.add_argument("--tz", type=float, default=0.0)
    parser.add_argument("--run-render-colmap", action="store_true")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    translation = np.array([args.tx, args.ty, args.tz], dtype=np.float32)
    gs = glb_to_gaussians(
        glb_path=args.glb,
        num_gaussians=args.num_gaussians,
        target_scale=args.target_scale,
        scale_factor=args.scale_factor,
        translation=translation,
        run_render_colmap=args.run_render_colmap,
        work_dir=args.work_dir,
    )
    torch.save(gs, args.out)
    print(f"Saved Gaussian tensors: {args.out}")


if __name__ == "__main__":
    _cli()
