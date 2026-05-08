"""Quantitative evaluation runner for DG-3DPlace sessions.

Computes CLIP directional similarity and DINO similarity for both the
unoptimized final render and the optimized final render in a session, then
rewrites the CSV with descriptive columns and per-metric gains.
"""
import os
import sys
import argparse
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
LEGACY_FIELDNAMES = ["session", "clip_dir_image", "clip_dir_text", "dino_sim"]
CSV_FIELDNAMES = [
    "session_dir",
    "unoptimized_final_view",
    "optimized_final_view",
    "unoptimized_clip_directional_image",
    "unoptimized_clip_directional_text",
    "unoptimized_dino_similarity",
    "optimized_clip_directional_image",
    "optimized_clip_directional_text",
    "optimized_dino_similarity",
    "clip_directional_image_gain",
    "clip_directional_text_gain",
    "dino_similarity_gain",
]

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _first_existing_path(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _metric_gain(optimized_value, unoptimized_value):
    if optimized_value is None or unoptimized_value is None:
        return None
    return float(optimized_value) - float(unoptimized_value)


def _empty_result_row(session_dir=""):
    row = {field: None for field in CSV_FIELDNAMES}
    row["session_dir"] = session_dir
    return row

def _extract_object_prompt_from_report(session_dir):
    """Extract object_prompt from detection_resource_report.txt if available."""
    report_path = os.path.join(session_dir, "detection_resource_report.txt")
    if not os.path.exists(report_path):
        return None
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("object_prompt:"):
                    # Extract everything after "object_prompt: "
                    prompt = line.split("object_prompt:", 1)[1].strip()
                    if prompt:
                        return prompt
    except Exception:
        pass
    
    return None


def _compute_metrics_for_variant(session_dir, variant_name, final_view_path, path_initial, path_diffusion, text_source="", text_target=""):
    row = {
        f"{variant_name}_final_view": final_view_path,
        f"{variant_name}_clip_directional_image": None,
        f"{variant_name}_clip_directional_text": None,
        f"{variant_name}_dino_similarity": None,
    }

    if not final_view_path or not os.path.exists(final_view_path):
        print(f"[quant] Missing {variant_name} final view for {session_dir}: {final_view_path}")
        return row

    try:
        from DG_3DPlace_Evaluation.metrics.clip_metric import clip_directional_similarity_normalized
        if os.path.exists(path_initial) and os.path.exists(path_diffusion):
            row[f"{variant_name}_clip_directional_image"] = float(
                clip_directional_similarity_normalized(path_initial, final_view_path, path_diffusion)
            )
        else:
            print(f"[quant] Missing CLIP image inputs for {session_dir} ({variant_name})")
    except Exception as e:
        print(f"[quant] CLIP image metric failed for {session_dir} ({variant_name}): {e}")

    try:
        from DG_3DPlace_Evaluation.metrics.clip_metric import clip_directional_similarity_standard

        txt_src = text_source or "A photo of the room"
        txt_tgt = text_target or _extract_object_prompt_from_report(session_dir)

        if not txt_tgt:
            for name in ["object_prompt.txt", "target_prompt.txt", "diffusion_prompt.txt", "edit_prompt.txt", "prompt.txt"]:
                prompt_path = os.path.join(session_dir, name)
                if os.path.exists(prompt_path) and os.path.isfile(prompt_path):
                    txt = open(prompt_path, "r", encoding="utf-8").read().strip()
                    if txt:
                        txt_tgt = txt
                        break

        if txt_src and txt_tgt and os.path.exists(path_initial):
            row[f"{variant_name}_clip_directional_text"] = float(
                clip_directional_similarity_standard(path_initial, final_view_path, txt_src, txt_tgt)
            )
        else:
            if not txt_tgt:
                print(f"[quant] Missing text prompts for CLIP text-image metric for {session_dir} ({variant_name})")
    except Exception as e:
        print(f"[quant] CLIP text-image metric failed for {session_dir} ({variant_name}): {e}")

    try:
        from DG_3DPlace_Evaluation.metrics.dino_metric import dino_similarity
        if os.path.exists(path_diffusion):
            row[f"{variant_name}_dino_similarity"] = float(dino_similarity(final_view_path, path_diffusion))
        else:
            print(f"[quant] Missing DINO inputs for {session_dir} ({variant_name})")
    except Exception as e:
        print(f"[quant] DINO metric failed for {session_dir} ({variant_name}): {e}")

    return row


def compute_metrics_for_session(session_dir, text_source="", text_target=""):
    session_dir = os.path.abspath(session_dir)
    path_initial = os.path.join(session_dir, "selected_camera_view.png")
    path_diffusion = os.path.join(session_dir, "gemini_diffusion_added.png")
    path_unoptimized = _first_existing_path(
        os.path.join(session_dir, "final_view_with_object.png"),
        os.path.join(session_dir, "final_scene_render.png"),
    )
    path_optimized = _first_existing_path(
        os.path.join(session_dir, "final_view_with_object_optimized.png"),
        os.path.join(session_dir, "room_with_object_optimized.ckpt"),
    )

    results = _empty_result_row(session_dir)
    results["unoptimized_final_view"] = path_unoptimized
    results["optimized_final_view"] = path_optimized

    unoptimized_metrics = _compute_metrics_for_variant(
        session_dir,
        "unoptimized",
        path_unoptimized,
        path_initial,
        path_diffusion,
        text_source=text_source,
        text_target=text_target,
    )
    optimized_metrics = _compute_metrics_for_variant(
        session_dir,
        "optimized",
        path_optimized,
        path_initial,
        path_diffusion,
        text_source=text_source,
        text_target=text_target,
    )

    results.update(unoptimized_metrics)
    results.update(optimized_metrics)
    results["clip_directional_image_gain"] = _metric_gain(
        results["optimized_clip_directional_image"], results["unoptimized_clip_directional_image"]
    )
    results["clip_directional_text_gain"] = _metric_gain(
        results["optimized_clip_directional_text"], results["unoptimized_clip_directional_text"]
    )
    results["dino_similarity_gain"] = _metric_gain(
        results["optimized_dino_similarity"], results["unoptimized_dino_similarity"]
    )

    return results


def _normalize_row(row):
    normalized = _empty_result_row(row.get("session_dir", ""))
    for field in CSV_FIELDNAMES:
        if field in row and row[field] not in (None, ""):
            normalized[field] = row[field]
    return normalized


def _load_existing_rows(csv_path):
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r", newline="") as fh:
        first_line = fh.readline().strip()
        fh.seek(0)

        if not first_line:
            return []

        first_token = first_line.split(",", 1)[0]
        if first_token in CSV_FIELDNAMES:
            reader = csv.DictReader(fh)
            return [_normalize_row(row) for row in reader]

        reader = csv.reader(fh)
        rows = []
        for raw_row in reader:
            if not raw_row:
                continue
            legacy = dict(zip(LEGACY_FIELDNAMES, raw_row))
            row = _empty_result_row(legacy.get("session", ""))
            row["optimized_clip_directional_image"] = legacy.get("clip_dir_image")
            row["optimized_clip_directional_text"] = legacy.get("clip_dir_text")
            row["optimized_dino_similarity"] = legacy.get("dino_sim")
            rows.append(row)
        return rows


def _write_rows(csv_path, rows):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDNAMES})


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
    parser.add_argument("--text-source", default="", help="Source text prompt (optional)")
    parser.add_argument("--text-target", default="", help="Target text prompt (optional)")
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

    existing_rows = _load_existing_rows(args.out)
    rows_by_session = {}
    ordered_sessions = []

    for row in existing_rows:
        session_dir = row.get("session_dir")
        if session_dir and session_dir not in rows_by_session:
            rows_by_session[session_dir] = row
            ordered_sessions.append(session_dir)

    for s in sessions:
        print(f"Computing metrics for {s}...")
        res = compute_metrics_for_session(s, text_source=args.text_source, text_target=args.text_target)
        if s not in rows_by_session:
            ordered_sessions.append(s)
        rows_by_session[s] = res
        print("Done:", res)

    _write_rows(args.out, [rows_by_session[session_dir] for session_dir in ordered_sessions])


if __name__ == "__main__":
    main()


# python evaluate_quantitative.py --start-session session_20260427_214420 --end-session session_20260429_010039

# python evaluate_quantitative.py --all