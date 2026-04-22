import argparse
import os
import shlex
import subprocess
from typing import Optional, Tuple

import numpy as np
import torch

REQUIRED_KEYS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
C0 = 0.28209479177387814


def run_cmd(text: str) -> None:
    cmd = shlex.split(text)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Trainer command failed.\n"
            + "CMD: "
            + text
            + "\nSTDOUT:\n"
            + (p.stdout or "")
            + "\nSTDERR:\n"
            + (p.stderr or "")
        )


def normalize(obj):
    if "pipeline" in obj and isinstance(obj["pipeline"], dict):
        st = obj["pipeline"]
        mapped = {
            "means": st.get("_model.means"),
            "scales": st.get("_model.scales"),
            "quats": st.get("_model.quats"),
            "features_dc": st.get("_model.features_dc"),
            "features_rest": st.get("_model.features_rest"),
            "opacities": st.get("_model.opacities"),
        }
        if all(v is not None for v in mapped.values()):
            obj = mapped

    for k in REQUIRED_KEYS:
        if k not in obj:
            raise RuntimeError(f"Missing key in trainer output: {k}")
        if not isinstance(obj[k], torch.Tensor):
            obj[k] = torch.tensor(obj[k])

    if obj["features_dc"].dim() == 2:
        obj["features_dc"] = obj["features_dc"].unsqueeze(1)
    if obj["opacities"].dim() == 1:
        obj["opacities"] = obj["opacities"].unsqueeze(1)
    if obj["features_rest"].dim() == 2 and obj["features_rest"].shape[1] == 45:
        n = obj["features_rest"].shape[0]
        obj["features_rest"] = obj["features_rest"].reshape(n, 15, 3)

    return obj


def _run_best_effort(cmd) -> bool:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode == 0
    except Exception:
        return False


def _find_points3d_txt(sparse_dir: str) -> Optional[str]:
    direct = os.path.join(sparse_dir, "points3D.txt")
    if os.path.exists(direct):
        return direct

    nested = os.path.join(sparse_dir, "0", "points3D.txt")
    if os.path.exists(nested):
        return nested

    # If only binary model exists, try converting it to text with COLMAP.
    model_in = os.path.join(sparse_dir, "0")
    if os.path.isdir(model_in):
        txt_out = os.path.join(sparse_dir, "txt")
        os.makedirs(txt_out, exist_ok=True)
        colmap_bin = os.environ.get("COLMAP_EXE", "").strip() or "colmap"
        ok = _run_best_effort(
            [
                colmap_bin,
                "model_converter",
                "--input_path",
                model_in,
                "--output_path",
                txt_out,
                "--output_type",
                "TXT",
            ]
        )
        if ok:
            converted = os.path.join(txt_out, "points3D.txt")
            if os.path.exists(converted):
                return converted

    return None


def _load_colmap_points(points3d_txt: str) -> Tuple[np.ndarray, np.ndarray]:
    xyzs = []
    rgbs = []
    with open(points3d_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
            except Exception:
                continue
            xyzs.append([x, y, z])
            rgbs.append([r / 255.0, g / 255.0, b / 255.0])

    if len(xyzs) == 0:
        raise RuntimeError(f"No valid 3D points parsed from {points3d_txt}")

    return np.asarray(xyzs, dtype=np.float32), np.asarray(rgbs, dtype=np.float32)


def _bootstrap_gaussians_from_sparse(sparse_dir: str) -> dict:
    points3d_txt = _find_points3d_txt(sparse_dir)
    if not points3d_txt:
        raise RuntimeError("COLMAP sparse model not found for fallback bootstrap")

    xyz, rgb = _load_colmap_points(points3d_txt)
    n = xyz.shape[0]

    extent = xyz.max(axis=0) - xyz.min(axis=0)
    base_sigma = max(float(extent.max()) / 250.0, 1e-3)
    log_scale = float(np.log(base_sigma))

    means = torch.tensor(xyz, dtype=torch.float32)
    scales = torch.full((n, 3), log_scale, dtype=torch.float32)
    quats = torch.zeros((n, 4), dtype=torch.float32)
    quats[:, 0] = 1.0
    features_dc = torch.tensor((rgb - 0.5) / C0, dtype=torch.float32).unsqueeze(1)
    features_rest = torch.zeros((n, 15, 3), dtype=torch.float32)
    opacities = torch.full((n, 1), 4.0, dtype=torch.float32)

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "opacities": opacities,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Wrapper to normalize external GS trainer output")
    ap.add_argument("--images", required=True, help="Rendered images folder")
    ap.add_argument("--sparse", default="", help="COLMAP sparse folder (optional)")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument(
        "--trainer-cmd-template",
        default=os.environ.get("OBJECT_GS_INNER_TRAIN_CMD_TEMPLATE", ""),
        help="External trainer command template using placeholders: {images}, {sparse}, {output}, {steps}",
    )
    ap.add_argument(
        "--trainer-output-candidate",
        default=os.environ.get("OBJECT_GS_TRAINER_OUTPUT", ""),
        help="Path to trainer-produced gaussian tensor file (optional).",
    )
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.trainer_cmd_template:
        cmd = args.trainer_cmd_template.format(
            images=os.path.abspath(args.images),
            sparse=os.path.abspath(args.sparse) if args.sparse else "",
            output=os.path.abspath(args.output),
            steps=int(args.steps),
        )
        run_cmd(cmd)

    candidates = []
    if args.trainer_output_candidate:
        candidates.append(args.trainer_output_candidate)
    candidates.extend(
        [
            os.path.join(args.output, "gaussians.pt"),
            os.path.join(args.output, "object_gaussians.pt"),
            os.path.join(args.output, "final_gaussians.pt"),
            os.path.join(args.output, "model.pt"),
        ]
    )

    src = None
    for c in candidates:
        if c and os.path.exists(c):
            src = c
            break

    if src is None:
        # If no external trainer output is produced, bootstrap from COLMAP sparse points.
        obj = _bootstrap_gaussians_from_sparse(args.sparse)
        print("No trainer output found; bootstrapped gaussians from COLMAP sparse points.")
    else:
        obj = torch.load(src, map_location="cpu", weights_only=False)
        obj = normalize(obj)

    out_path = os.path.join(args.output, "gaussians.pt")
    torch.save(obj, out_path)
    print(f"Saved normalized gaussians: {out_path}")


if __name__ == "__main__":
    main()
