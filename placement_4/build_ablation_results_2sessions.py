import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "ablation_results_2sessions.csv")

BASE_SESSIONS = [
    "session_20260427_214420",
    "session_20260427_215829",
]
VARIANT_ORDER = ["full", "no_contact", "no_depth", "no_mask_bce", "support_ransac"]
VARIANT_MARKERS = {
    "full": "ablation_full_",
    "no_contact": "ablation_no_contact_",
    "no_depth": "ablation_no_depth_",
    "no_mask_bce": "ablation_no_mask_bce_",
    "support_ransac": "ablation_support_ransac_",
}

FIELDNAMES = [
    "session_group",
    "parent_session",
    "session_kind",
    "variant",
    "session_dir",
    "optimized_ckpt",
    "clip_dir_image",
    "clip_dir_text",
    "dino_sim",
    "contact_err",
    "seed",
    "git_commit",
]


def git_commit_hash(root):
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def compute_metrics(session_dir):
    sys.path.insert(0, HERE)
    from evaluate_quantitative import compute_metrics_for_session
    return compute_metrics_for_session(session_dir)


def compute_contact_err(session_dir, ckpt_path=None):
    from DG_3DPlace_Evaluation.metrics.contact_error import compute_contact_error
    return compute_contact_error(session_dir, ckpt_path)


def find_optimized_ckpt(session_dir):
    candidates = [
        os.path.join(session_dir, "room_with_object_optimized.ckpt"),
        os.path.join(session_dir, "ckpt", "room.ckpt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def infer_variant(session_name, base_name):
    if session_name == base_name:
        return "base"
    for variant, marker in VARIANT_MARKERS.items():
        if session_name.startswith(f"{base_name}_{marker}"):
            return variant
    return "unknown"


def main():
    git_commit = git_commit_hash(ROOT)
    rows = []

    for base_name in BASE_SESSIONS:
        base_dir = os.path.join(HERE, base_name)
        if not os.path.isdir(base_dir):
            print(f"Missing base session: {base_dir}")
            continue

        sessions = [base_dir]
        for variant in VARIANT_ORDER:
            prefix = f"{base_name}_ablation_{variant}_"
            matches = [
                os.path.join(HERE, name)
                for name in sorted(os.listdir(HERE))
                if name.startswith(prefix) and os.path.isdir(os.path.join(HERE, name))
            ]
            if matches:
                sessions.append(matches[-1])
            else:
                print(f"Missing ablation session for {base_name} / {variant}")

        for session_dir in sessions:
            session_name = os.path.basename(session_dir)
            variant = infer_variant(session_name, base_name)
            session_kind = "base" if variant == "base" else "ablation"
            print(f"Computing metrics for {session_name} [{variant}]")
            try:
                metrics = compute_metrics(session_dir)
            except Exception as e:
                print(f"Evaluation failed for {session_dir}: {e}")
                metrics = {"clip_dir_image": None, "clip_dir_text": None, "dino_sim": None}

            ckpt_path = find_optimized_ckpt(session_dir)
            try:
                contact_err = compute_contact_err(session_dir, ckpt_path=ckpt_path)
            except Exception as e:
                print(f"Contact metric failed for {session_dir}: {e}")
                contact_err = None

            rows.append({
                "session_group": base_name,
                "parent_session": base_name,
                "session_kind": session_kind,
                "variant": variant,
                "session_dir": session_dir,
                "optimized_ckpt": ckpt_path,
                "clip_dir_image": metrics.get("clip_dir_image"),
                "clip_dir_text": metrics.get("clip_dir_text"),
                "dino_sim": metrics.get("dino_sim"),
                "contact_err": contact_err,
                "seed": 42 if variant == "base" else None,
                "git_commit": git_commit,
            })

    if not rows:
        print("No rows collected; nothing to write.")
        return

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
