"""Ablation runner: run optimization variants and collect quantitative metrics.

This script copies a completed session into per-variant folders, sets
environment toggles for each ablation, attempts to run the post-placement
optimization step in-process, then runs the quantitative evaluator and
writes results to `ablation_results.csv`.

Notes:
- This relies on `run_post_placement_optimization` from `detection_optimized.py`.
- The optimization implementation must respect environment flags
  `USE_DEPTH_INIT` and `USE_CONTACT_LOSS` for the ablations to take effect.
  If it doesn't, ablation still runs but effects may be unchanged.
"""
import os
import sys
import shutil
import argparse
import datetime
import csv

HERE = os.path.dirname(os.path.abspath(__file__))


VARIANTS = {
    "full": {"USE_DEPTH_INIT": "1", "USE_CONTACT_LOSS": "1"},
    "no_depth": {"USE_DEPTH_INIT": "0", "USE_CONTACT_LOSS": "1"},
    "no_contact": {"USE_DEPTH_INIT": "1", "USE_CONTACT_LOSS": "0"},
}


def copy_session(src_session, dst_session):
    if os.path.exists(dst_session):
        shutil.rmtree(dst_session)
    shutil.copytree(src_session, dst_session)


def run_variant(src_session, variant_name, initial_ckpt=None):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.abspath(f"{src_session}_ablation_{variant_name}_{ts}")
    copy_session(src_session, dst)

    # write ablation config for record
    cfg = VARIANTS[variant_name]
    cfg_path = os.path.join(dst, "ablation_config.txt")
    with open(cfg_path, "w") as fh:
        for k, v in cfg.items():
            fh.write(f"{k}={v}\n")

    # set env toggles for this process
    old_env = {k: os.environ.get(k) for k in cfg.keys()}
    os.environ.update(cfg)

    # attempt to run post-placement optimization in-process
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        import detection_optimized as det
        camera_state = os.path.join(dst, "selected_camera_state.pt")
        # choose ckpt: optimized if exists else original
        ckpt = initial_ckpt or det.CKPT_PATH
        try:
            out_ckpt = det.run_post_placement_optimization(
                initial_ckpt_path=ckpt,
                camera_state_path=camera_state,
                session_dir=dst,
                num_object_gaussians=int(det.__dict__.get("num_object_gaussians", 0) or 0),
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

    return dst


def append_results(csv_path, rows, fieldnames=None):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Completed session folder to ablate")
    parser.add_argument("--initial-ckpt", default=None, help="Initial checkpoint path to use for optimization")
    parser.add_argument("--out", default=os.path.join(HERE, "ablation_results.csv"))
    args = parser.parse_args()

    src = os.path.abspath(args.session)
    if not os.path.isdir(src):
        print(f"Session not found: {src}")
        sys.exit(1)

    rows = []
    for name in VARIANTS.keys():
        print(f"Running variant: {name}")
        dst = run_variant(src, name, initial_ckpt=args.initial_ckpt)

        # run quantitative evaluation on produced session
        try:
            from evaluate_quantitative import compute_metrics_for_session
            metrics = compute_metrics_for_session(dst, initial_ckpt=args.initial_ckpt)
        except Exception as e:
            metrics = {"clip_dir": None, "dino_sim": None, "contact_err": None}
            print(f"Evaluation for {dst} failed: {e}")

        row = {
            "variant": name,
            "session": dst,
            "clip_dir": metrics.get("clip_dir"),
            "dino_sim": metrics.get("dino_sim"),
            "contact_err": metrics.get("contact_err"),
        }
        rows.append(row)

    append_results(args.out, rows, fieldnames=["variant", "session", "clip_dir", "dino_sim", "contact_err"])
    print(f"Ablation finished. Results appended to {args.out}")


if __name__ == "__main__":
    main()
