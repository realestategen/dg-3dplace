"""Convert a 3DGS .ckpt checkpoint to a standard 3DGS .ply file.

The .ckpt stores all Gaussian parameters under the key 'pipeline' with
sub-keys:  _model.means  _model.scales  _model.quats  _model.features_dc
           _model.features_rest  _model.opacities

The output PLY uses the standard 3DGS vertex layout expected by every
web viewer (GaussianSplats3D, gsplat.js, SuperSplat, etc.):
  x y z  nx ny nz  f_dc_0 f_dc_1 f_dc_2  f_rest_0…f_rest_44
  opacity  scale_0 scale_1 scale_2  rot_0 rot_1 rot_2 rot_3
"""

import numpy as np
import torch
import plyfile
import os


def ckpt_to_ply(ckpt_path: str, ply_path: str) -> str:
    """Export a pipeline .ckpt to a 3DGS-standard binary PLY.

    Returns the path of the written PLY file.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]

    def _np(key):
        t = state[key]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return t.detach().float().cpu().numpy()

    means       = _np("_model.means")          # (N, 3)
    scales      = _np("_model.scales")         # (N, 3) log
    quats       = _np("_model.quats")          # (N, 4) w x y z
    opacities   = _np("_model.opacities").reshape(-1)   # (N,) logit
    features_dc = _np("_model.features_dc")    # (N,1,3) or (N,3)

    if features_dc.ndim == 3:
        features_dc = features_dc.squeeze(1)   # (N, 3)

    N = len(means)
    if "_model.features_rest" in state:
        features_rest = _np("_model.features_rest")  # (N,15,3) or (N,45)
        if features_rest.ndim == 3:
            features_rest = features_rest.reshape(features_rest.shape[0], -1)
        n_rest_cols = features_rest.shape[1] if features_rest.ndim == 2 else 45
        if len(features_rest) != N:
            # Scene and object gaussians may have different SH degrees in merged ckpt
            print(f"[ckpt_to_ply] features_rest row count ({len(features_rest)}) "
                  f"!= means count ({N}), padding/truncating to match")
            if len(features_rest) < N:
                pad = np.zeros((N - len(features_rest), n_rest_cols), dtype=np.float32)
                features_rest = np.concatenate([features_rest, pad], axis=0)
            else:
                features_rest = features_rest[:N]
    else:
        n_rest_cols = 45
        features_rest = np.zeros((N, n_rest_cols), dtype=np.float32)

    normals = np.zeros((N, 3), dtype=np.float32)

    # Build the dtype expected by standard 3DGS viewers
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    n_rest = features_rest.shape[1]
    for i in range(n_rest):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    vertex = np.empty(N, dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertex["nx"], vertex["ny"], vertex["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    vertex["f_dc_0"] = features_dc[:, 0]
    vertex["f_dc_1"] = features_dc[:, 1]
    vertex["f_dc_2"] = features_dc[:, 2]
    for i in range(n_rest):
        vertex[f"f_rest_{i}"] = features_rest[:, i]
    vertex["opacity"] = opacities
    vertex["scale_0"] = scales[:, 0]
    vertex["scale_1"] = scales[:, 1]
    vertex["scale_2"] = scales[:, 2]
    vertex["rot_0"] = quats[:, 0]
    vertex["rot_1"] = quats[:, 1]
    vertex["rot_2"] = quats[:, 2]
    vertex["rot_3"] = quats[:, 3]

    os.makedirs(os.path.dirname(os.path.abspath(ply_path)), exist_ok=True)
    el = plyfile.PlyElement.describe(vertex, "vertex")
    plyfile.PlyData([el], text=False).write(ply_path)
    size_mb = os.path.getsize(ply_path) / 1024 / 1024
    print(f"Exported {N:,} Gaussians → {ply_path}  ({size_mb:.1f} MB)")
    return ply_path
