import os
import csv
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = [
    os.path.join(HERE, 'session_20260427_214420'),
    os.path.join(HERE, 'session_20260427_215829'),
]
OUT = os.path.join(HERE, 'ablation_results_2sessions.csv')
FIELDNAMES = ["variant", "session", "clip_dir_image", "clip_dir_text", "dino_sim", "contact_err", "seed", "git_commit"]


def git_commit_hash(root):
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def compute_contact_err(session_dir, ckpt_path=None):
    try:
        from DG_3DPlace_Evaluation.metrics.contact_error import compute_contact_error
        return compute_contact_error(session_dir, ckpt_path)
    except Exception as e:
        print(f"Contact error computation failed for {session_dir}: {e}")
        return None


def main():
    rows = []
    git_commit = git_commit_hash(os.path.join(HERE, '..'))
    for s in SESSIONS:
        if not os.path.isdir(s):
            print(f"Session missing, skipping: {s}")
            continue
        print(f"Computing metrics for {s}")
        try:
            sys.path.insert(0, HERE)
            from evaluate_quantitative import compute_metrics_for_session
            metrics = compute_metrics_for_session(s)
        except Exception as e:
            print(f"Evaluation failed for {s}: {e}")
            metrics = {"clip_dir_image": None, "clip_dir_text": None, "dino_sim": None}

        contact_err = compute_contact_err(s)
        row = {
            "variant": "base",
            "session": s,
            "clip_dir_image": metrics.get("clip_dir_image"),
            "clip_dir_text": metrics.get("clip_dir_text"),
            "dino_sim": metrics.get("dino_sim"),
            "contact_err": contact_err,
            "seed": 42,
            "git_commit": git_commit,
        }
        rows.append(row)

    # write CSV
    dirn = os.path.dirname(OUT)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    exists = os.path.exists(OUT)
    with open(OUT, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote results to {OUT}")


if __name__ == '__main__':
    main()
