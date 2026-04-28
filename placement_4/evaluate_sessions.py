"""Wrapper to run DG_3DPlace_Evaluation for one or many sessions.

Usage:
  - import and call `evaluate_session(session_dir, initial_ckpt=None)`
  - run as CLI: `python evaluate_sessions.py [session_dir]` (no arg => evaluate all session_* folders)

This script copies session artifacts into the evaluation/data locations, invokes
the render + evaluation scripts, captures their output, and returns it.
"""
import os
import sys
import shutil
import glob
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
EVAL_ROOT = os.path.join(ROOT, "DG_3DPlace_Evaluation")
EVAL_DATA_DIR = os.path.join(EVAL_ROOT, "data", "2d_images")
EVAL_CHECKPOINTS_DIR = os.path.join(EVAL_ROOT, "data", "checkpoints")


def _safe_copy(src, dst):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        return False


def evaluate_session(session_dir, initial_ckpt=None):
    """Evaluate a single session. Returns a tuple (session_dir, stdout_str).

    This will:
      - copy session selected view -> evaluation initial_scene_render.png
      - copy session gemini diffusion image -> diffusion_guided.png
      - copy initial_ckpt -> data/checkpoints/initial_scene.ckpt (if provided)
      - copy session final checkpoint (room_with_object_optimized.ckpt if present,
        else room_with_object.ckpt) -> data/checkpoints/final_scene.ckpt
      - run render_multi_view.py then run run_evaluation.py capturing stdout
    """
    session_dir = os.path.abspath(session_dir)
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"Session not found: {session_dir}")

    os.makedirs(EVAL_DATA_DIR, exist_ok=True)
    os.makedirs(EVAL_CHECKPOINTS_DIR, exist_ok=True)

    # map session artifacts to evaluation expected filenames
    selected_view = os.path.join(session_dir, "selected_camera_view.png")
    diffusion_img = os.path.join(session_dir, "gemini_diffusion_added.png")
    optimized_ckpt = os.path.join(session_dir, "room_with_object_optimized.ckpt")
    final_ckpt = os.path.join(session_dir, "room_with_object.ckpt")

    target_initial = os.path.join(EVAL_DATA_DIR, "initial_scene_render.png")
    target_diffusion = os.path.join(EVAL_DATA_DIR, "diffusion_guided.png")
    target_final = os.path.join(EVAL_DATA_DIR, "final_scene_render.png")

    cp_log = []
    if os.path.exists(selected_view):
        _safe_copy(selected_view, target_initial)
        cp_log.append(f"copied selected view -> {target_initial}")
    else:
        cp_log.append(f"missing selected view: {selected_view}")

    if os.path.exists(diffusion_img):
        _safe_copy(diffusion_img, target_diffusion)
        cp_log.append(f"copied diffusion image -> {target_diffusion}")
    else:
        cp_log.append(f"missing diffusion image: {diffusion_img}")

    # copy checkpoints
    if initial_ckpt and os.path.exists(initial_ckpt):
        try:
            shutil.copy2(initial_ckpt, os.path.join(EVAL_CHECKPOINTS_DIR, "initial_scene.ckpt"))
            cp_log.append(f"copied initial ckpt -> initial_scene.ckpt")
        except Exception as e:
            cp_log.append(f"failed to copy initial ckpt: {e}")
    else:
        cp_log.append(f"initial ckpt not provided or missing: {initial_ckpt}")

    chosen_final_ckpt = optimized_ckpt if os.path.exists(optimized_ckpt) else final_ckpt
    if os.path.exists(chosen_final_ckpt):
        try:
            shutil.copy2(chosen_final_ckpt, os.path.join(EVAL_CHECKPOINTS_DIR, "final_scene.ckpt"))
            cp_log.append(f"copied session final ckpt -> final_scene.ckpt")
        except Exception as e:
            cp_log.append(f"failed to copy final ckpt: {e}")
    else:
        cp_log.append(f"missing final ckpt (tried optimized then final): {chosen_final_ckpt}")

    # Run the renderer (will produce final_scene_render.png if checkpoints exist)
    out_lines = []
    out_lines.append(f"Evaluation started: {datetime.datetime.now().isoformat()}")
    out_lines.append(f"Session: {session_dir}")
    out_lines.extend(cp_log)

    # Run render_multi_view (may be heavy) -- best-effort
    render_script = os.path.join(EVAL_ROOT, "render_multi_view.py")
    eval_script = os.path.join(EVAL_ROOT, "run_evaluation.py")

    if os.path.exists(render_script):
        try:
            proc = subprocess.run([sys.executable, render_script], capture_output=True, text=True)
            out_lines.append("--- render_multi_view.py output ---")
            out_lines.append(proc.stdout or "")
            if proc.stderr:
                out_lines.append("--- render_multi_view.py stderr ---")
                out_lines.append(proc.stderr)
        except Exception as e:
            out_lines.append(f"Failed to run render_multi_view.py: {e}")
    else:
        out_lines.append("render_multi_view.py not found; skipping render step")

    # Run the evaluation runner and capture its stdout
    if os.path.exists(eval_script):
        try:
            proc = subprocess.run([sys.executable, eval_script], capture_output=True, text=True)
            out_lines.append("--- run_evaluation.py output ---")
            out_lines.append(proc.stdout or "")
            if proc.stderr:
                out_lines.append("--- run_evaluation.py stderr ---")
                out_lines.append(proc.stderr)
        except Exception as e:
            out_lines.append(f"Failed to run run_evaluation.py: {e}")
    else:
        out_lines.append("run_evaluation.py not found; cannot run evaluation")

    out_lines.append(f"Evaluation finished: {datetime.datetime.now().isoformat()}")
    return "\n".join(out_lines)


def evaluate_all_sessions(initial_ckpt=None):
    pattern = os.path.join(HERE, "session_*")
    sessions = sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])
    results = {}
    for s in sessions:
        try:
            results[s] = evaluate_session(s, initial_ckpt=initial_ckpt)
        except Exception as e:
            results[s] = f"Evaluation failed: {e}"
    return results


if __name__ == "__main__":
    # CLI: optional session dir; if omitted evaluate all sessions
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", default=None, help="Session directory to evaluate (optional)")
    parser.add_argument("--initial-ckpt", default=None, help="Initial ckpt path to use for evaluation")
    args = parser.parse_args()

    if args.session:
        out = evaluate_session(args.session, initial_ckpt=args.initial_ckpt)
        print(out)
        sys.exit(0)

    all_res = evaluate_all_sessions(initial_ckpt=args.initial_ckpt)
    for s, txt in all_res.items():
        print(f"=== Session: {s} ===")
        print(txt)
        print("\n\n")
