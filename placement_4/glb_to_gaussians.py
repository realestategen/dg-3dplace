import math
import os
import shlex
import subprocess
import tempfile
import json
import shutil
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import trimesh

C0 = 0.28209479177387814


def _run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + (result.stdout or "")
            + "\nSTDERR:\n"
            + (result.stderr or "")
        )


def _find_exe(explicit_path: Optional[str], candidates: List[str], label: str) -> str:
    if explicit_path:
        found = shutil.which(explicit_path)
        if found:
            return found
        if os.path.exists(explicit_path):
            return explicit_path

    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found

    raise RuntimeError(f"{label} executable not found. Please provide explicit path.")


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


def _write_blender_render_script(
    script_path: str,
    mesh_path: str,
    images_dir: str,
    camera_json_path: str,
    num_views: int,
    image_size: int,
) -> None:
    script = f'''
import bpy
import os
import json
import math
from mathutils import Vector

mesh_path = r"{os.path.abspath(mesh_path)}"
images_dir = r"{os.path.abspath(images_dir)}"
camera_json_path = r"{os.path.abspath(camera_json_path)}"
num_views = {int(num_views)}
image_size = {int(image_size)}

bpy.ops.wm.read_factory_settings(use_empty=True)

ext = os.path.splitext(mesh_path)[1].lower()
if ext == ".obj":
    bpy.ops.wm.obj_import(filepath=mesh_path)
elif ext in [".glb", ".gltf"]:
    bpy.ops.import_scene.gltf(filepath=mesh_path)
else:
    raise RuntimeError(f"Unsupported mesh extension: {{ext}}")

objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not objs:
    raise RuntimeError("No mesh objects loaded.")

mins = [1e9, 1e9, 1e9]
maxs = [-1e9, -1e9, -1e9]
for o in objs:
    for v in o.bound_box:
        p = o.matrix_world @ Vector(v)
        for i in range(3):
            mins[i] = min(mins[i], p[i])
            maxs[i] = max(maxs[i], p[i])

center = Vector([(mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5, (mins[2] + maxs[2]) * 0.5])
extent = max(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2], 1e-6)
radius = 2.2 * extent

cam_data = bpy.data.cameras.new("OrbitCam")
cam = bpy.data.objects.new("OrbitCam", cam_data)
bpy.context.scene.collection.objects.link(cam)

light_data = bpy.data.lights.new(name="KeyLight", type="AREA")
light = bpy.data.objects.new(name="KeyLight", object_data=light_data)
light.location = (2.0 * extent, -2.0 * extent, 2.0 * extent)
light_data.energy = 1500
bpy.context.scene.collection.objects.link(light)

scene = bpy.context.scene
scene.camera = cam
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.render.resolution_x = image_size
scene.render.resolution_y = image_size
scene.render.image_settings.file_format = "PNG"

os.makedirs(images_dir, exist_ok=True)

meta = {{"frames": []}}
meta["camera"] = {{
    "width": int(image_size),
    "height": int(image_size),
    "angle_x": float(cam_data.angle_x),
    "angle_y": float(cam_data.angle_y),
}}

for i in range(num_views):
    az = 2.0 * math.pi * i / num_views
    el = math.radians(15.0 if i % 2 == 0 else -15.0)

    x = center.x + radius * math.cos(az) * math.cos(el)
    y = center.y + radius * math.sin(az) * math.cos(el)
    z = center.z + radius * math.sin(el)

    cam.location = (x, y, z)
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    out_path = os.path.join(images_dir, f"view_{{i:04d}}.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    meta["frames"].append({{
        "image_path": out_path,
        "camera_world_matrix": [list(row) for row in cam.matrix_world]
    }})

with open(camera_json_path, "w") as f:
    json.dump(meta, f, indent=2)
'''
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


def _render_synthetic_views(
    mesh_path: str,
    work_dir: str,
    num_views: int,
    image_size: int,
    blender_exe: Optional[str],
) -> Dict[str, str]:
    blender_bin = _find_exe(blender_exe, ["blender", "/usr/bin/blender", "/snap/bin/blender"], "Blender")

    render_dir = os.path.join(work_dir, "renders")
    images_dir = os.path.join(render_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    camera_json = os.path.join(render_dir, "blender_cameras.json")

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
        script_path = tf.name

    _write_blender_render_script(
        script_path=script_path,
        mesh_path=mesh_path,
        images_dir=images_dir,
        camera_json_path=camera_json,
        num_views=num_views,
        image_size=image_size,
    )

    try:
        _run_cmd([blender_bin, "-b", "-P", script_path])
    finally:
        try:
            os.remove(script_path)
        except Exception:
            pass

    pngs = [p for p in os.listdir(images_dir) if p.lower().endswith(".png")]
    if len(pngs) == 0:
        raise RuntimeError("Blender rendering produced no images.")

    return {
        "images_dir": images_dir,
        "camera_json": camera_json,
    }


def _run_colmap(images_dir: str, work_dir: str, colmap_exe: Optional[str]) -> Dict[str, str]:
    colmap_bin = _find_exe(colmap_exe, ["colmap"], "COLMAP")
    xvfb_run = shutil.which("xvfb-run")
    db_path = os.path.join(work_dir, "colmap.db")
    sparse_dir = os.path.join(work_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    colmap_prefix: List[str] = []
    if xvfb_run:
        colmap_prefix = [xvfb_run, "-a"]

    _run_cmd(
        colmap_prefix
        + [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            db_path,
            "--image_path",
            images_dir,
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.use_gpu",
            "0",
        ]
    )
    _run_cmd(
        colmap_prefix
        + [
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            db_path,
            "--SiftMatching.use_gpu",
            "0",
        ]
    )
    _run_cmd(
        colmap_prefix
        + [
            colmap_bin,
            "mapper",
            "--database_path",
            db_path,
            "--image_path",
            images_dir,
            "--output_path",
            sparse_dir,
        ]
    )

    return {
        "database_path": db_path,
        "sparse_dir": sparse_dir,
    }


def _normalize_trained_gaussian_dict(obj: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    alias = {
        "means": ["means", "_model.means"],
        "scales": ["scales", "_model.scales"],
        "quats": ["quats", "rotations", "_model.quats"],
        "features_dc": ["features_dc", "_model.features_dc"],
        "features_rest": ["features_rest", "_model.features_rest"],
        "opacities": ["opacities", "_model.opacities"],
    }

    out: Dict[str, torch.Tensor] = {}
    for key, options in alias.items():
        tensor = None
        for opt in options:
            if opt in obj:
                tensor = obj[opt]
                break
        if tensor is None:
            raise RuntimeError(f"Missing trained gaussian key: {key}")
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        out[key] = tensor

    if out["features_dc"].dim() == 2:
        out["features_dc"] = out["features_dc"].unsqueeze(1)
    if out["opacities"].dim() == 1:
        out["opacities"] = out["opacities"].unsqueeze(1)
    if out["features_rest"].dim() == 2 and out["features_rest"].shape[1] == 45:
        n = out["features_rest"].shape[0]
        out["features_rest"] = out["features_rest"].reshape(n, 15, 3)

    return out


def _run_external_training(
    work_dir: str,
    mesh_path: str,
    images_dir: str,
    camera_json: str,
    sparse_dir: Optional[str],
    train_steps: int,
    num_gaussians: int,
    trainer_cmd_template: Optional[str],
) -> Dict[str, torch.Tensor]:
    if not trainer_cmd_template:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        trainer_script = os.path.join(script_dir, "train_object_gs.py")
        raise RuntimeError(
            "trainer_cmd_template is required for train mode. "
            f"Example: python {trainer_script} --images {{images_dir}} --sparse {{sparse_dir}} --output {{output_dir}} --steps {{steps}}"
        )

    output_dir = os.path.join(work_dir, "trained_gs")
    os.makedirs(output_dir, exist_ok=True)

    cmd_text = trainer_cmd_template.format(
        mesh_path=os.path.abspath(mesh_path),
        images_dir=os.path.abspath(images_dir),
        camera_json=os.path.abspath(camera_json),
        sparse_dir=os.path.abspath(sparse_dir) if sparse_dir else "",
        output_dir=os.path.abspath(output_dir),
        steps=int(train_steps),
        num_gaussians=int(num_gaussians),
    )
    _run_cmd(shlex.split(cmd_text))

    candidate_files = [
        os.path.join(output_dir, "gaussians.pt"),
        os.path.join(output_dir, "object_gaussians.pt"),
        os.path.join(output_dir, "final_gaussians.pt"),
    ]
    selected = None
    for cand in candidate_files:
        if os.path.exists(cand):
            selected = cand
            break
    if selected is None:
        raise RuntimeError(f"No gaussian tensor output found in {output_dir}")

    loaded = torch.load(selected, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        loaded = loaded["pipeline"]
    if not isinstance(loaded, dict):
        raise RuntimeError("Trainer output must be a dict of gaussian tensors")

    return _normalize_trained_gaussian_dict(loaded)


def _apply_transform_to_gaussians(
    gs: Dict[str, torch.Tensor],
    target_scale: Optional[float],
    scale_factor: float,
    rotation: Optional[np.ndarray],
    translation: Optional[np.ndarray],
    support_z: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    means = gs["means"].detach().cpu().numpy().astype(np.float32)
    scales = gs["scales"].detach().cpu().numpy().astype(np.float32)

    obj_min = means.min(axis=0)
    obj_max = means.max(axis=0)
    obj_center = (obj_min + obj_max) / 2.0
    means = means - obj_center

    means = np.column_stack([means[:, 0], -means[:, 2], means[:, 1]]).astype(np.float32)

    extent_max = float(max((obj_max - obj_min).max(), 1e-6))
    if target_scale is not None:
        mul = float(target_scale * scale_factor / extent_max)
        means *= mul
        scales += math.log(max(mul, 1e-8))

    if rotation is not None:
        means = np.dot(means, np.asarray(rotation, dtype=np.float32).T)

    if translation is not None:
        means += np.asarray(translation, dtype=np.float32)

    # If a support surface height is provided, lift the object so its
    # lowest gaussian sits exactly at support_z (prevents sinking).
    if support_z is not None and means.size > 0:
        min_z = float(means[:, 2].min())
        shift = float(support_z - min_z)
        means[:, 2] += shift

    out = dict(gs)
    out["means"] = torch.tensor(means, dtype=torch.float32)
    out["scales"] = torch.tensor(scales, dtype=torch.float32)
    return out
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
    num_gaussians: int = 100000,
    target_scale: Optional[float] = None,
    scale_factor: float = 0.4,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    support_z: Optional[float] = None,
    opacity_logit: float = 5.0,
    run_render_colmap: bool = False,
    work_dir: Optional[str] = None,
    conversion_mode: str = "sample",
    blender_exe: Optional[str] = None,
    colmap_exe: Optional[str] = None,
    trainer_cmd_template: Optional[str] = None,
    num_views: int = 48,
    render_size: int = 768,
    train_steps: int = 3000,
) -> Dict[str, torch.Tensor]:
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB not found: {glb_path}")

    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(glb_path), "glb_to_gs")
    os.makedirs(work_dir, exist_ok=True)

    mode = (conversion_mode or "sample").strip().lower()

    if mode == "train":
        render_info = _render_synthetic_views(
            mesh_path=glb_path,
            work_dir=work_dir,
            num_views=int(num_views),
            image_size=int(render_size),
            blender_exe=blender_exe,
        )

        sparse_dir = None
        if run_render_colmap:
            colmap_info = _run_colmap(
                images_dir=render_info["images_dir"],
                work_dir=work_dir,
                colmap_exe=colmap_exe,
            )
            sparse_dir = colmap_info["sparse_dir"]

        trained = _run_external_training(
            work_dir=work_dir,
            mesh_path=glb_path,
            images_dir=render_info["images_dir"],
            camera_json=render_info["camera_json"],
            sparse_dir=sparse_dir,
            train_steps=int(train_steps),
            num_gaussians=int(num_gaussians),
            trainer_cmd_template=trainer_cmd_template,
        )

        transformed = _apply_transform_to_gaussians(
            trained,
            target_scale=target_scale,
            scale_factor=scale_factor,
            rotation=rotation,
            translation=translation,
            support_z=support_z,
        )

        ops = transformed["opacities"]
        if ops.min() >= 0.0 and ops.max() <= 1.0:
            p = torch.clamp(ops, 1e-5, 1.0 - 1e-5)
            transformed["opacities"] = torch.log(p / (1.0 - p))

        return transformed

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

    # Align object bottom to support surface if requested.
    if support_z is not None and pts_scene.size > 0:
        min_z = float(pts_scene[:, 2].min())
        shift = float(support_z - min_z)
        pts_scene[:, 2] += shift

    means = torch.tensor(pts_scene, dtype=torch.float32)

    adaptive_radius = (np.ptp(pts_scene, axis=0).prod() / max(1, len(pts_scene))) ** (1.0 / 3.0) * 0.35
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
    parser.add_argument("--conversion-mode", choices=["sample", "train"], default="sample")
    parser.add_argument("--blender-exe", default=None)
    parser.add_argument("--colmap-exe", default=None)
    parser.add_argument("--trainer-cmd-template", default=None)
    parser.add_argument("--num-views", type=int, default=48)
    parser.add_argument("--render-size", type=int, default=768)
    parser.add_argument("--train-steps", type=int, default=3000)
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
        conversion_mode=args.conversion_mode,
        blender_exe=args.blender_exe,
        colmap_exe=args.colmap_exe,
        trainer_cmd_template=args.trainer_cmd_template,
        num_views=args.num_views,
        render_size=args.render_size,
        train_steps=args.train_steps,
    )
    torch.save(gs, args.out)
    print(f"Saved Gaussian tensors: {args.out}")


if __name__ == "__main__":
    _cli()
