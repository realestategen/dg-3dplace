"""Ablation runner: run optimization variants and collect quantitative metrics.

This script copies a completed session into per-variant folders, sets
environment toggles for each ablation, attempts to run the post-placement
optimization step in-process, then runs the quantitative evaluator and
writes results to `ablation_results.csv`.

Enhancements in this runner:
- Expanded ablation variants (depth init, contact loss, support estimation).
- Records reproducibility metadata: seed and git commit hash.
- Computes contact error via DG_3DPlace_Evaluation.metrics.contact_error.
- Writes an expanded CSV with `clip_dir_image`, `clip_dir_text`, `dino_sim`,
  `contact_err`, `seed`, and `git_commit`.

Notes:
- This relies on `run_post_placement_optimization` from `detection_optimized.py`.
- The optimization implementation should respect environment flags such as
  `USE_DEPTH_INIT`, `USE_CONTACT_LOSS`, `USE_SUPPORT_RANSAC`, `USE_MASK_BCE`,
  `USE_COM_LOSS` for the ablations to take effect.
"""
import os
import sys
import shutil
import argparse
import datetime
import csv
import subprocess
import random
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

# Define a richer set of ablation variants. Each variant maps to environment
# toggles expected by the optimization code.
VARIANTS = {
    "full": {"USE_DEPTH_INIT": "1", "USE_CONTACT_LOSS": "1", "USE_SUPPORT_RANSAC": "0", "USE_MASK_BCE": "1", "USE_COM_LOSS": "1"},
    "no_depth": {"USE_DEPTH_INIT": "0", "USE_CONTACT_LOSS": "1", "USE_SUPPORT_RANSAC": "0", "USE_MASK_BCE": "1", "USE_COM_LOSS": "1"},
    "no_contact": {"USE_DEPTH_INIT": "1", "USE_CONTACT_LOSS": "0", "USE_SUPPORT_RANSAC": "0", "USE_MASK_BCE": "1", "USE_COM_LOSS": "1"},
    "support_ransac": {"USE_DEPTH_INIT": "1", "USE_CONTACT_LOSS": "1", "USE_SUPPORT_RANSAC": "1", "USE_MASK_BCE": "1", "USE_COM_LOSS": "1"},
    "no_mask_bce": {"USE_DEPTH_INIT": "1", "USE_CONTACT_LOSS": "1", "USE_SUPPORT_RANSAC": "0", "USE_MASK_BCE": "0", "USE_COM_LOSS": "1"},
}


def _git_commit_hash():
    try:
        root = ROOT
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def copy_session(src_session, dst_session):
    """Copy session, but use symlinks for large files to save disk space."""
    if os.path.exists(dst_session):
        shutil.rmtree(dst_session)
    
    os.makedirs(dst_session, exist_ok=True)
    
    # Large binary files to symlink instead of copy
    large_file_extensions = {'.ckpt', '.obj', '.glb', '.pth', '.pt'}
    
    for entry in os.listdir(src_session):
        src_path = os.path.join(src_session, entry)
        dst_path = os.path.join(dst_session, entry)
        
        if os.path.isdir(src_path):
            # Recursively copy directories
            shutil.copytree(src_path, dst_path, symlinks=True)
        elif os.path.isfile(src_path):
            # Check file size and extension
            file_size_mb = os.path.getsize(src_path) / (1024 * 1024)
            _, ext = os.path.splitext(src_path)
            
            # Symlink large files or binary files
            if file_size_mb > 10 or ext.lower() in large_file_extensions:
                os.symlink(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        else:
            # Copy other file types
            shutil.copy2(src_path, dst_path)


def _extract_num_object_gaussians(ckpt_path):
    """Extract num_object_gaussians from checkpoint file."""
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict):
            if 'num_object_gaussians' in ckpt:
                return int(ckpt['num_object_gaussians'])
    except Exception as e:
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'num_object_gaussians' in ckpt:
                return int(ckpt['num_object_gaussians'])
        except Exception as e2:
            print(f"[warning] Failed to extract num_object_gaussians from {ckpt_path}: {e2}")
    return 0


def run_variant(src_session, variant_name, initial_ckpt=None, seed=42):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.abspath(f"{src_session}_ablation_{variant_name}_{ts}")
    copy_session(src_session, dst)

    # write ablation config for record
    cfg = VARIANTS[variant_name]
    cfg_path = os.path.join(dst, "ablation_config.txt")
    with open(cfg_path, "w") as fh:
        meta = {"variant": variant_name, "timestamp": ts, "seed": seed}
        fh.write(json.dumps(meta) + "\n")
        for k, v in cfg.items():
            fh.write(f"{k}={v}\n")

    # set reproducible seed for this process and record an env var
    os.environ["AB_SEED"] = str(seed)
    random.seed(seed)

    # set env toggles for this process
    old_env = {k: os.environ.get(k) for k in cfg.keys()}
    os.environ.update(cfg)

    # attempt to run post-placement optimization in-process
    out_ckpt = None
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import detection_optimized as det
        camera_state = os.path.join(dst, "selected_camera_state.pt")
        # choose ckpt: optimized if exists else original
        ckpt = initial_ckpt or det.CKPT_PATH
        # Extract num_object_gaussians from the checkpoint file itself
        num_object_gaussians = _extract_num_object_gaussians(ckpt)
        if int(num_object_gaussians) <= 0:
            override = os.environ.get("AB_OVERRIDE_NUM_GAUSSIANS", "128").strip()
            try:
                num_object_gaussians = int(override)
                print(f"[info] Falling back to num_object_gaussians={num_object_gaussians}")
            except Exception:
                pass
        try:
            out_ckpt = det.run_post_placement_optimization(
                initial_ckpt_path=ckpt,
                camera_state_path=camera_state,
                session_dir=dst,
                num_object_gaussians=num_object_gaussians,
            )
        except Exception as e:
            out_ckpt = None
            print(f"Variant {variant_name}: optimization failed: {e}")
    finally:
        # restore env
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return dst, out_ckpt


def append_results(csv_path, rows, fieldnames=None):
    if not rows:
        return
    exists = os.path.exists(csv_path)
    dirn = os.path.dirname(csv_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _compute_contact_err(session_dir, ckpt_path=None):
    try:
        from DG_3DPlace_Evaluation.metrics.contact_error import compute_contact_error
        return compute_contact_error(session_dir, ckpt_path)
    except Exception as e:
        print(f"Contact error computation failed for {session_dir}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None, help="Completed session folder to ablate")
    parser.add_argument("--all", action="store_true", help="Ablate all session_* folders")
    parser.add_argument("--start-session", default=None, help="First session folder name or path (inclusive)")
    parser.add_argument("--end-session", default=None, help="Last session folder name or path (inclusive)")
    parser.add_argument("--initial-ckpt", default=None, help="Initial checkpoint path to use for optimization")
    parser.add_argument("--override-gaussians", type=int, default=128, help="Fallback num_object_gaussians when metadata extraction fails")
    parser.add_argument("--out", default=os.path.join(HERE, "ablation_results.csv"))
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility (each variant will use seed+idx)")
    args = parser.parse_args()

    os.environ["AB_OVERRIDE_NUM_GAUSSIANS"] = str(args.override_gaussians)

    # Resolve which sessions to ablate
    sessions = []
    if args.all:
        sessions = sorted([p for p in os.listdir(HERE) if p.startswith("session_") and os.path.isdir(os.path.join(HERE, p))])
        sessions = [os.path.join(HERE, s) for s in sessions]
    elif args.start_session or args.end_session:
        all_sessions = sorted(
            [p for p in os.listdir(HERE) if p.startswith("session_") and os.path.isdir(os.path.join(HERE, p))]
        )
        all_sessions = [os.path.join(HERE, s) for s in all_sessions]

        def _resolve_session(value):
            if not value:
                return None
            return os.path.abspath(value) if os.path.isabs(value) or os.path.isdir(value) else os.path.join(HERE, value)

        start_session = _resolve_session(args.start_session)
        end_session = _resolve_session(args.end_session)
        if not start_session or not end_session:
            print("Both --start-session and --end-session are required for range mode.")
            sys.exit(1)
        if start_session not in all_sessions:
            print(f"Start session not found in placement_4 session folders: {start_session}")
            sys.exit(1)
        if end_session not in all_sessions:
            print(f"End session not found in placement_4 session folders: {end_session}")
            sys.exit(1)

        start_idx = all_sessions.index(start_session)
        end_idx = all_sessions.index(end_session)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        sessions = all_sessions[start_idx : end_idx + 1]
    elif args.session:
        sessions = [os.path.abspath(args.session)]
    else:
        print("Specify --session, --all, or --start-session/--end-session")
        sys.exit(1)

    rows = []
    git_commit = _git_commit_hash()
    for session_num, src in enumerate(sessions):
        if not os.path.isdir(src):
            print(f"Session not found: {src}, skipping")
            continue

        print(f"\n=== Ablating session {session_num + 1}/{len(sessions)}: {os.path.basename(src)} ===")
        
        for idx, name in enumerate(sorted(VARIANTS.keys())):
            seed = args.seed + idx
            print(f"  Running variant: {name} (seed={seed})")
            dst, out_ckpt = run_variant(src, name, initial_ckpt=args.initial_ckpt, seed=seed)

            # run quantitative evaluation on produced session
            try:
                from evaluate_quantitative import compute_metrics_for_session
                metrics = compute_metrics_for_session(dst, initial_ckpt=args.initial_ckpt)
            except Exception as e:
                print(f"    Evaluation for {dst} failed: {e}")
                metrics = {"clip_dir_image": None, "clip_dir_text": None, "dino_sim": None}

            contact_err = _compute_contact_err(dst, ckpt_path=out_ckpt or args.initial_ckpt)

            row = {
                "variant": name,
                "session": dst,
                "clip_dir_image": metrics.get("clip_dir_image"),
                "clip_dir_text": metrics.get("clip_dir_text"),
                "dino_sim": metrics.get("dino_sim"),
                "contact_err": contact_err,
                "seed": seed,
                "git_commit": git_commit,
            }
            rows.append(row)

    fieldnames = ["variant", "session", "clip_dir_image", "clip_dir_text", "dino_sim", "contact_err", "seed", "git_commit"]
    append_results(args.out, rows, fieldnames=fieldnames)
    print(f"\nAblation finished. Results appended to {args.out}")


if __name__ == "__main__":
    main()
