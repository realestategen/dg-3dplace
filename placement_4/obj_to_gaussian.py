"""
OBJ to Gaussian Splat Converter

This script:
- Loads a Wavefront OBJ mesh
- Samples points on the surface (area-weighted)
- Assigns color, scale, and orientation
- Outputs Gaussian parameters for integration into a checkpoint
"""

import numpy as np
import torch
import math

# SH DC constant
C0 = 0.28209479177387814


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
    tri_indices = np.random.choice(len(faces), size=num_points, p=probs)
    r1 = np.random.rand(num_points)
    r2 = np.random.rand(num_points)
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


def obj_to_gaussians(obj_path, num_gaussians=15000, color=[0.65, 0.45, 0.30], scale=None, rotation=None, translation=None):
    vertices, faces = load_obj_mesh(obj_path)
    points, normals = sample_points_on_mesh(vertices, faces, num_gaussians)

    # Center, rotate, scale, translate
    obj_min = points.min(axis=0)
    obj_max = points.max(axis=0)
    obj_center = (obj_min + obj_max) / 2.0
    points[:, 0] -= obj_center[0]
    points[:, 2] -= obj_center[2]
    points[:, 1] -= obj_min[1]
    pts_scene = np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])
    nrm_scene = np.column_stack([normals[:, 0], -normals[:, 2], normals[:, 1]])

    # Apply scale
    if scale is not None:
        pts_scene *= scale
    # Apply rotation
    if rotation is not None:
        pts_scene = np.dot(pts_scene, rotation.T)
    # Apply translation
    if translation is not None:
        pts_scene += translation

    # Create Gaussians
    means = torch.tensor(pts_scene, dtype=torch.float32)
    adaptive_radius = (np.ptp(pts_scene, axis=0).prod() / num_gaussians) ** (1.0 / 3.0) * 1.5
    log_scale = math.log(max(adaptive_radius, 1e-7))
    scales = torch.full((num_gaussians, 3), log_scale, dtype=torch.float32)
    quats = torch.zeros(num_gaussians, 4, dtype=torch.float32)
    quats[:, 0] = 1.0
    sh_color = (np.array(color) - 0.5) / C0
    features_dc = torch.tensor(sh_color, dtype=torch.float32).unsqueeze(0).expand(num_gaussians, -1)
    features_dc = features_dc.unsqueeze(1)
    features_rest = torch.zeros(num_gaussians, 15, 3, dtype=torch.float32)
    opacities = torch.full((num_gaussians, 1), 2.0, dtype=torch.float32)

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "opacities": opacities,
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python obj_to_gaussian.py vase.obj [num_gaussians] [R G B] [scale] [rotation_matrix_flat] [translation_x translation_y translation_z]")
        sys.exit(1)
    obj_path = sys.argv[1]
    num_gaussians = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    color = [float(c) for c in sys.argv[3:6]] if len(sys.argv) > 5 else [0.65, 0.45, 0.30]
    scale = float(sys.argv[6]) if len(sys.argv) > 6 else None
    rotation = np.array([float(r) for r in sys.argv[7:16]]).reshape(3, 3) if len(sys.argv) > 15 else None
    translation = np.array([float(t) for t in sys.argv[16:19]]) if len(sys.argv) > 18 else None
    gaussians = obj_to_gaussians(obj_path, num_gaussians, color, scale, rotation, translation)
    print("Gaussians ready. Use in checkpoint integration.")
