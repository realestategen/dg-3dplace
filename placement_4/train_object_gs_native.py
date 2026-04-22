import argparse
import json
import math
import os
import random

import numpy as np
from PIL import Image
import torch
from gsplat import rasterization

from glb_to_gaussians import C0, _extract_points_colors_from_glb


def _load_cameras(camera_json_path, device):
    with open(camera_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cam = data.get("camera", {})
    width = int(cam.get("width", 1024))
    height = int(cam.get("height", 1024))

    angle_x = float(cam.get("angle_x", math.radians(50.0)))
    angle_y = float(cam.get("angle_y", angle_x))

    fx = (width * 0.5) / math.tan(max(angle_x * 0.5, 1e-6))
    fy = (height * 0.5) / math.tan(max(angle_y * 0.5, 1e-6))
    cx = width * 0.5
    cy = height * 0.5

    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    frames = []
    for fr in data.get("frames", []):
        path = fr.get("image_path")
        if not path or not os.path.exists(path):
            continue
        c2w = np.asarray(fr["camera_world_matrix"], dtype=np.float32)
        if c2w.shape != (4, 4):
            continue

        w2c_gl = np.linalg.inv(c2w)
        w2c = w2c_gl.copy()
        # OpenGL camera to OpenCV convention used by gsplat wrapper.
        w2c[1, :] *= -1.0
        w2c[2, :] *= -1.0

        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        frames.append(
            {
                "image": torch.tensor(img, dtype=torch.float32, device=device),
                "w2c": torch.tensor(w2c, dtype=torch.float32, device=device),
            }
        )

    if len(frames) == 0:
        raise RuntimeError(f"No valid frames found in {camera_json_path}")

    return frames, K, width, height


def _init_gaussians(mesh_path, num_gaussians, device):
    pts, cols = _extract_points_colors_from_glb(mesh_path, int(num_gaussians))
    if len(pts) == 0:
        raise RuntimeError("Could not sample mesh points for initialization")

    means = torch.nn.Parameter(torch.tensor(pts, dtype=torch.float32, device=device))

    extent = np.ptp(pts, axis=0)
    base = max(float(np.max(extent)) / 220.0, 1e-4)
    scales = torch.nn.Parameter(torch.full((len(pts), 3), math.log(base), dtype=torch.float32, device=device))

    quats = torch.zeros((len(pts), 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0

    sh = (np.clip(cols, 0.0, 1.0) - 0.5) / C0
    features_dc = torch.nn.Parameter(torch.tensor(sh, dtype=torch.float32, device=device).unsqueeze(1))

    opacities = torch.nn.Parameter(torch.full((len(pts), 1), 3.0, dtype=torch.float32, device=device))

    return means, scales, quats, features_dc, opacities


def main():
    ap = argparse.ArgumentParser(description="Native object Gaussian trainer from GLB synthetic views")
    ap.add_argument("--mesh", required=True, help="Path to textured GLB")
    ap.add_argument("--images", required=True, help="Rendered images directory")
    ap.add_argument("--camera-json", required=True, help="Blender camera metadata JSON")
    ap.add_argument("--sparse", default="", help="COLMAP sparse path (unused, kept for compatibility)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--num-gaussians", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, K, width, height = _load_cameras(args.camera_json, device)

    means, scales, quats, features_dc, opacities = _init_gaussians(
        mesh_path=args.mesh,
        num_gaussians=int(args.num_gaussians),
        device=device,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": args.lr},
            {"params": [scales], "lr": args.lr * 0.35},
            {"params": [features_dc], "lr": args.lr * 0.6},
            {"params": [opacities], "lr": args.lr * 0.25},
        ]
    )

    for step in range(1, int(args.steps) + 1):
        fr = frames[np.random.randint(0, len(frames))]
        target = fr["image"]
        viewmat = fr["w2c"]

        qn = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
        scales_pos = torch.exp(torch.clamp(scales, min=-8.0, max=2.0))
        alpha = torch.sigmoid(opacities).squeeze(-1)

        renders, _, _ = rasterization(
            means=means,
            quats=qn,
            scales=scales_pos,
            opacities=alpha,
            colors=features_dc.squeeze(1) * C0 + 0.5,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=None,
            backgrounds=torch.ones(3, device=device),
        )

        pred = torch.clamp(renders[0], 0.0, 1.0)
        loss_l1 = torch.mean(torch.abs(pred - target))
        loss_scale_reg = 1e-4 * torch.mean(torch.relu(scales + 7.0) ** 2)
        loss = loss_l1 + loss_scale_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == 1 or step == int(args.steps):
            print(f"step {step}/{args.steps} loss={float(loss.item()):.6f}")

    with torch.no_grad():
        keep = torch.sigmoid(opacities).squeeze(-1) > 0.08
        if int(keep.sum().item()) < max(5000, int(args.num_gaussians * 0.25)):
            keep = torch.sigmoid(opacities).squeeze(-1) > 0.02

        means_o = means.detach().cpu()[keep.cpu()]
        scales_o = scales.detach().cpu()[keep.cpu()]
        quats_o = quats.detach().cpu()[keep.cpu()]
        fdc_o = features_dc.detach().cpu()[keep.cpu()]
        ops_o = opacities.detach().cpu()[keep.cpu()]

        features_rest = torch.zeros((means_o.shape[0], 15, 3), dtype=torch.float32)
        out = {
            "means": means_o.float(),
            "scales": scales_o.float(),
            "quats": quats_o.float(),
            "features_dc": fdc_o.float(),
            "features_rest": features_rest,
            "opacities": ops_o.float(),
        }

    out_path = os.path.join(args.output, "gaussians.pt")
    torch.save(out, out_path)
    print(f"Saved trained object gaussians: {out_path} (count={out['means'].shape[0]})")


if __name__ == "__main__":
    main()
