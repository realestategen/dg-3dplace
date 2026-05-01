"""Quantitative evaluation runner for DG-3DPlace sessions.

Computes CLIP directional similarity and DINO similarity for a given session
range and appends results to a CSV.
"""
import os
import sys
import argparse
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
EVAL_ROOT = os.path.join(ROOT, "DG_3DPlace_Evaluation")
EVAL_DATA_DIR = os.path.join(EVAL_ROOT, "data", "2d_images")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

def compute_metrics_for_session(session_dir, initial_ckpt=None):
    session_dir = os.path.abspath(session_dir)
    # Prepare evaluation inputs (render + copy) using existing helper
    try:
        import evaluate_sessions as eval_wrap
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate_sessions",
            os.path.join(HERE, "evaluate_sessions.py"),
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load evaluate_sessions.py")
        eval_wrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_wrap)

    # Run the preparation and rendering (best-effort)
    try:
        eval_wrap.evaluate_session(session_dir, initial_ckpt=initial_ckpt)
    except Exception:
        # proceed: metrics functions will check for files
        pass

    # Paths expected by DG_3DPlace_Evaluation metrics
    path_initial = os.path.join(EVAL_DATA_DIR, "initial_scene_render.png")
    path_diffusion = os.path.join(EVAL_DATA_DIR, "diffusion_guided.png")
    path_final = os.path.join(EVAL_DATA_DIR, "final_scene_render.png")

    results = {"session": session_dir, "clip_dir": None, "dino_sim": None}

    # CLIP directional (normalized)
    try:
        from DG_3DPlace_Evaluation.metrics.clip_metric import clip_directional_similarity_normalized
        if os.path.exists(path_initial) and os.path.exists(path_final) and os.path.exists(path_diffusion):
            results["clip_dir"] = float(clip_directional_similarity_normalized(path_initial, path_final, path_diffusion))
        else:
            print(f"[quant] Missing CLIP inputs for {session_dir}")
    except Exception as e:
        print(f"[quant] CLIP metric failed for {session_dir}: {e}")

    # DINO similarity
    try:
        from DG_3DPlace_Evaluation.metrics.dino_metric import dino_similarity
        if os.path.exists(path_final) and os.path.exists(path_diffusion):
            results["dino_sim"] = float(dino_similarity(path_final, path_diffusion))
        else:
            print(f"[quant] Missing DINO inputs for {session_dir}")
    except Exception as e:
        print(f"[quant] DINO metric failed for {session_dir}: {e}")

    return results


def append_csv(csv_path, row, fieldnames=None):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None, help="Session folder to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all session_* folders")
    parser.add_argument(
        "--start-session",
        default=None,
        help="First session folder name or path to include in a range (inclusive).",
    )
    parser.add_argument(
        "--end-session",
        default=None,
        help="Last session folder name or path to include in a range (inclusive).",
    )
    parser.add_argument("--initial-ckpt", default=None, help="Initial ckpt path to use for evaluation")
    parser.add_argument("--out", default=os.path.join(HERE, "quant_eval_results.csv"), help="CSV output path")
    args = parser.parse_args()

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

    for s in sessions:
        print(f"Computing metrics for {s}...")
        res = compute_metrics_for_session(s, initial_ckpt=args.initial_ckpt)
        append_csv(args.out, res, fieldnames=["session", "clip_dir", "dino_sim"])
        print("Done:", res)


if __name__ == "__main__":
    main()


# python evaluate_quantitative.py --start-session session_20260427_224918 --end-session session_20260429_010039

# python evaluate_quantitative.py --all