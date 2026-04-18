#!/usr/bin/env python3
"""Automated paper-grade experiment runner for DG-3DPlace.

This script runs placement_4/detection_optimized.py repeatedly with hardcoded prompts,
collects timing/quality metrics, and generates report-ready plots, grids, and tables.
"""

from __future__ import annotations

import argparse
import csv
import errno
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from prompts import BASELINE_PROMPT, PROMPT_SPECS, SOURCE_TEXT


SESSION_RE = re.compile(r"session_\d{8}_\d{6}")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+\|\s+Total:\s+([0-9.]+)\s+\|\s+MASK:\s+([0-9.]+)\s+\|\s+RGB:\s+([0-9.]+)")


@dataclass
class RunArtifacts:
    session_dir: str
    selected_view: Optional[str]
    diffusion: Optional[str]
    bbox: Optional[str]
    added_mask: Optional[str]
    final_pre: Optional[str]
    final_post: Optional[str]
    highlighted: Optional[str]
    resource_report: Optional[str]


@dataclass
class EvalResult:
    clip_directional_similarity: Optional[float]
    clip_text_directional_similarity: Optional[float]
    dino_similarity: Optional[float]
    background_ssim: Optional[float]
    errors: Dict[str, str]


def _is_no_space_error(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC


def _best_effort_output(action_name: str, fn, *args, **kwargs) -> bool:
    """Run non-critical output actions without aborting the whole experiment run.

    Returns True on success, False if skipped due to disk-full condition.
    """
    try:
        fn(*args, **kwargs)
        return True
    except Exception as exc:
        if _is_no_space_error(exc):
            print(f"[WARN] Skipping {action_name}: no space left on device")
            return False
        raise


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_json_from_stdout(stdout: str) -> Dict:
    # The eval env may print library logs before JSON; parse the last JSON object.
    begin = stdout.find("{")
    end = stdout.rfind("}")
    if begin == -1 or end == -1 or end < begin:
        raise ValueError("Could not parse JSON block from eval worker output")
    return json.loads(stdout[begin : end + 1])


def _safe_float(x: Optional[float]) -> str:
    return "" if x is None else f"{float(x):.6f}"


def _discover_session_dir(placement_dir: str, seen_before: set[str], fallback_start_time: float) -> Optional[str]:
    candidates = []
    for path in glob.glob(os.path.join(placement_dir, "session_*")):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        if not SESSION_RE.fullmatch(name):
            continue
        if name in seen_before:
            continue
        mtime = os.path.getmtime(path)
        if mtime >= fallback_start_time - 60.0:
            candidates.append((mtime, path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _collect_artifacts(session_dir: str) -> RunArtifacts:
    def p(name: str) -> Optional[str]:
        path = os.path.join(session_dir, name)
        return path if os.path.exists(path) else None

    return RunArtifacts(
        session_dir=session_dir,
        selected_view=p("selected_camera_view.png"),
        diffusion=p("gemini_diffusion_added.png"),
        bbox=_first_existing(
            [
                p("car_detection_bbox.png"),
                p("bench_detection_bbox.png"),
                p("vase_detection_bbox.png"),
                p("chair_detection_bbox.png"),
                p("table_detection_bbox.png"),
                p("sofa_detection_bbox.png"),
                p("plant_detection_bbox.png"),
                p("bottle_detection_bbox.png"),
            ]
        ),
        added_mask=p("added_object_mask.png"),
        final_pre=p("final_view_with_object.png"),
        final_post=p("final_view_with_object_optimized.png"),
        highlighted=_first_existing(
            [
                p("car_highlighted_verification.png"),
                p("bench_highlighted_verification.png"),
                p("vase_highlighted_verification.png"),
                p("chair_highlighted_verification.png"),
                p("table_highlighted_verification.png"),
                p("sofa_highlighted_verification.png"),
                p("plant_highlighted_verification.png"),
                p("bottle_highlighted_verification.png"),
            ]
        ),
        resource_report=p("detection_resource_report.txt"),
    )


def _detect_object_label_from_artifacts(artifacts: RunArtifacts) -> str:
    if artifacts.bbox:
        name = os.path.basename(artifacts.bbox)
        if name.endswith("_detection_bbox.png"):
            return name[: -len("_detection_bbox.png")]
    if artifacts.highlighted:
        name = os.path.basename(artifacts.highlighted)
        if name.endswith("_highlighted_verification.png"):
            return name[: -len("_highlighted_verification.png")]
    return "object"


def _build_run_data_from_existing_session(session_dir: str) -> Dict:
    artifacts = _collect_artifacts(session_dir)
    stage_times = _parse_resource_report(artifacts.resource_report)

    total = stage_times.get("Total")
    optimization = stage_times.get("Post-placement optimization")

    # Existing session reports do not expose separate Gemini/Hunyuan durations.
    gemini_total = None
    hunyuan_total = None
    placement_including = total
    placement_excluding = None
    if placement_including is not None:
        placement_excluding = placement_including

    object_label = _detect_object_label_from_artifacts(artifacts)

    return {
        "returncode": 0,
        "prompt_id": os.path.basename(session_dir),
        "prompt": f"[existing session] {os.path.basename(session_dir)}",
        "target_text": f"A real-estate scene with added {object_label}",
        "run_total_s": total,
        "placement_total_including_gen_s": placement_including,
        "placement_total_excluding_gen_s": placement_excluding,
        "gemini_total_s": gemini_total,
        "hunyuan_total_s": hunyuan_total,
        "optimization_total_s": optimization,
        "stage_times": stage_times,
        "markers": {},
        "line_events": [],
        "losses": [],
        "artifacts": artifacts,
    }


def _list_existing_sessions(placement_dir: str, limit_sessions: int = 0) -> List[str]:
    sessions = [
        path
        for path in glob.glob(os.path.join(placement_dir, "session_*"))
        if os.path.isdir(path) and SESSION_RE.fullmatch(os.path.basename(path))
    ]
    sessions.sort(key=lambda p: os.path.getmtime(p))
    if limit_sessions > 0:
        sessions = sessions[-limit_sessions:]
    return sessions


def _first_existing(paths: List[Optional[str]]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _parse_resource_report(path: Optional[str]) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}

    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) != 2:
                continue
            if parts[0].lower() in {"stage", "---"}:
                continue
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return out


def _call_eval_worker(
    project_root: str,
    eval_env: str,
    initial_img: str,
    diffusion_img: str,
    final_img: str,
    source_text: str,
    target_text: str,
) -> EvalResult:
    worker = os.path.join(project_root, "experiments", "paper_automation", "eval_worker.py")
    cmd = [
        "conda",
        "run",
        "-n",
        eval_env,
        "python",
        worker,
        "--project-root",
        project_root,
        "--initial",
        initial_img,
        "--diffusion",
        diffusion_img,
        "--final",
        final_img,
        "--source-text",
        source_text,
        "--target-text",
        target_text,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return EvalResult(None, None, None, None, {"worker": (proc.stderr or proc.stdout).strip()})

    try:
        data = _parse_json_from_stdout(proc.stdout)
    except Exception as exc:
        return EvalResult(None, None, None, None, {"worker": f"JSON parse failure: {exc}"})

    return EvalResult(
        clip_directional_similarity=data.get("clip_directional_similarity"),
        clip_text_directional_similarity=data.get("clip_text_directional_similarity"),
        dino_similarity=data.get("dino_similarity"),
        background_ssim=data.get("background_ssim"),
        errors=data.get("errors", {}) or {},
    )


def _copy_if_exists(src: Optional[str], dst: str) -> None:
    if src and os.path.exists(src):
        shutil.copy2(src, dst)


def _make_storyboard(output_path: str, title: str, labeled_images: List[Tuple[str, Optional[str]]]) -> None:
    valid = [(label, path) for label, path in labeled_images if path and os.path.exists(path)]
    if not valid:
        return

    n = len(valid)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes_arr[i // cols, i % cols]
        if i < n:
            label, path = valid[i]
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_title(label)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def _make_cross_run_grid(output_path: str, run_rows: List[Dict]) -> None:
    if not run_rows:
        return

    cols = [
        ("Selected", "selected_view"),
        ("Diffusion", "diffusion"),
        ("Placement", "final_pre"),
        ("Optimized", "final_post"),
    ]
    rows = len(run_rows)

    fig, axes = plt.subplots(rows, len(cols), figsize=(4.2 * len(cols), 2.8 * rows))
    axes_arr = np.atleast_2d(axes)

    for r, row in enumerate(run_rows):
        for c, (col_title, key) in enumerate(cols):
            ax = axes_arr[r, c]
            path = row.get(key)
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
                if r == 0:
                    ax.set_title(col_title)
                ax.set_ylabel(row.get("run_label", ""), rotation=90)
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                if r == 0:
                    ax.set_title(col_title)
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_timing_bars(output_path: str, rows: List[Dict]) -> None:
    if not rows:
        return

    labels = [r["run_label"] for r in rows]
    x = np.arange(len(rows))

    placement_inc = np.array([r.get("placement_total_including_gen_s", np.nan) for r in rows], dtype=float)
    placement_exc = np.array([r.get("placement_total_excluding_gen_s", np.nan) for r in rows], dtype=float)
    optimization = np.array([r.get("optimization_total_s", np.nan) for r in rows], dtype=float)

    w = 0.26
    plt.figure(figsize=(max(9, len(rows) * 1.2), 5))
    plt.bar(x - w, placement_inc, width=w, label="Placement (incl. Gemini+Hunyuan)")
    plt.bar(x, placement_exc, width=w, label="Placement (excl. Gemini+Hunyuan)")
    plt.bar(x + w, optimization, width=w, label="Optimization")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Seconds")
    plt.title("Timing Comparison Across Runs")
    handles, labels_for_legend = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def _plot_metric_delta(output_path: str, rows: List[Dict]) -> None:
    if not rows:
        return

    metrics = [
        "clip_directional_similarity",
        "clip_text_directional_similarity",
        "dino_similarity",
        "background_ssim",
    ]

    metric_means = []
    for metric in metrics:
        deltas = []
        for row in rows:
            pre = row.get(f"{metric}_pre")
            post = row.get(f"{metric}_post")
            if pre is not None and post is not None:
                deltas.append(float(post) - float(pre))
        metric_means.append(float(np.mean(deltas)) if deltas else np.nan)

    x = np.arange(len(metrics))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, metric_means)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, [m.replace("_", "\n") for m in metrics])
    plt.ylabel("Average (post - pre)")
    plt.title("Average Optimization Delta Across Metrics")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def _plot_loss_curves(output_path: str, loss_curves: Dict[str, List[Dict]]) -> None:
    if not loss_curves:
        return

    plt.figure(figsize=(9, 5))
    for run_label, samples in loss_curves.items():
        if not samples:
            continue
        xs = [int(s["epoch"]) for s in samples]
        ys = [float(s["total"]) for s in samples]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=run_label)

    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Post-Placement Optimization Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def _plot_success_rate(output_path: str, rows: List[Dict]) -> None:
    if not rows:
        return

    total = len(rows)
    success = sum(1 for r in rows if r.get("returncode") == 0)
    eval_ok = sum(1 for r in rows if r.get("eval_available"))

    labels = ["Run success", "Eval success"]
    values = [success / total if total else 0.0, eval_ok / total if total else 0.0]

    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, values, color=["#2b8a3e", "#1c7ed6"])
    plt.ylim(0, 1.05)
    plt.ylabel("Rate")
    plt.title("Success Rates Across Automated Runs")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def _plot_failure_taxonomy(output_path: str, rows: List[Dict]) -> None:
    if not rows:
        return

    taxonomy = {
        "pipeline_failure": 0,
        "eval_failure": 0,
        "successful": 0,
    }
    for row in rows:
        if row.get("returncode") == 0:
            if row.get("eval_available"):
                taxonomy["successful"] += 1
            else:
                taxonomy["eval_failure"] += 1
        else:
            taxonomy["pipeline_failure"] += 1

    labels = list(taxonomy.keys())
    values = [taxonomy[k] for k in labels]

    plt.figure(figsize=(7, 4.2))
    bars = plt.bar(labels, values, color=["#e03131", "#f08c00", "#2f9e44"])
    plt.ylabel("Count")
    plt.title("Failure / Success Taxonomy")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def _plot_ablation_bars(output_path: str, rows: List[Dict]) -> None:
    if not rows:
        return

    categories = ["Placement time", "Optimization time", "CLIP text", "DINO", "SSIM"]
    values = []

    placement_gap = []
    opt_vals = []
    clip_vals = []
    dino_vals = []
    ssim_vals = []
    for row in rows:
        inc = row.get("placement_total_including_gen_s")
        exc = row.get("placement_total_excluding_gen_s")
        if inc is not None and exc is not None:
            placement_gap.append(float(inc) - float(exc))
        if row.get("optimization_total_s") is not None:
            opt_vals.append(float(row["optimization_total_s"]))
        if row.get("clip_text_directional_similarity_post") is not None and row.get("clip_text_directional_similarity_pre") is not None:
            clip_vals.append(float(row["clip_text_directional_similarity_post"]) - float(row["clip_text_directional_similarity_pre"]))
        if row.get("dino_similarity_post") is not None and row.get("dino_similarity_pre") is not None:
            dino_vals.append(float(row["dino_similarity_post"]) - float(row["dino_similarity_pre"]))
        if row.get("background_ssim_post") is not None and row.get("background_ssim_pre") is not None:
            ssim_vals.append(float(row["background_ssim_post"]) - float(row["background_ssim_pre"]))

    values = [
        float(np.mean(placement_gap)) if placement_gap else np.nan,
        float(np.mean(opt_vals)) if opt_vals else np.nan,
        float(np.mean(clip_vals)) if clip_vals else np.nan,
        float(np.mean(dino_vals)) if dino_vals else np.nan,
        float(np.mean(ssim_vals)) if ssim_vals else np.nan,
    ]

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(categories, values, color=["#495057", "#1864ab", "#5f3dc4", "#0b7285", "#2f9e44"])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Average delta / time")
    plt.title("Ablation-Oriented Summary Bars")
    plt.xticks(rotation=18, ha="right")
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        plt.text(bar.get_x() + bar.get_width() / 2, value + (0.02 if value >= 0 else -0.08), f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def _write_csv(path: str, rows: List[Dict], headers: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_one(
    project_root: str,
    placement_dir: str,
    prompt_spec: Dict,
    run_idx: int,
    camera_index: int,
    gemini_api_key: str,
) -> Dict:
    seen_sessions = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(placement_dir, "session_*"))
        if os.path.isdir(p)
    }

    cmd = [sys.executable, "detection_optimized.py", prompt_spec["prompt"]]
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = gemini_api_key

    run_start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=placement_dir,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("Failed to open stdin/stdout for detection_optimized subprocess")

    # detection_optimized asks for camera index first. Send once and let script consume it.
    proc.stdin.write(f"{camera_index}\n")
    proc.stdin.flush()

    line_events = []
    markers: Dict[str, float] = {}
    losses: List[Dict] = []

    def set_marker(name: str, now_t: float) -> None:
        if name not in markers:
            markers[name] = now_t

    while True:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            break
        if not line:
            continue

        now_t = time.time()
        line = line.rstrip("\n")
        line_events.append({"t": now_t, "line": line})
        print(f"[run {run_idx:02d}] {line}")

        if "Saved selected camera metadata to" in line:
            set_marker("gemini_image_start", now_t)
        if "Gemini edited image saved to" in line:
            set_marker("gemini_image_end", now_t)

        if "Generating Gemini object cutout" in line:
            set_marker("gemini_cutout_start", now_t)
        if "Gemini cutout saved to:" in line:
            set_marker("gemini_cutout_end", now_t)

        if "Running Hunyuan3D step1" in line:
            set_marker("hunyuan_start", now_t)
        if "Painted mesh completed in" in line:
            set_marker("hunyuan_end", now_t)

        if "Object gaussians generated and integrated" in line:
            set_marker("placement_done", now_t)

        if "--- Running Post-Placement Optimization ---" in line:
            set_marker("optimization_start", now_t)
        if "Post-placement optimization finished" in line:
            set_marker("optimization_end", now_t)

        m = EPOCH_RE.search(line)
        if m:
            losses.append(
                {
                    "epoch": int(m.group(1)),
                    "total": float(m.group(2)),
                    "mask": float(m.group(3)),
                    "rgb": float(m.group(4)),
                }
            )

    rc = proc.wait()
    run_end = time.time()

    session_dir = _discover_session_dir(placement_dir, seen_sessions, run_start)
    if session_dir is None:
        session_names = []
        for ev in line_events:
            session_names.extend(SESSION_RE.findall(ev["line"]))
        session_names = [s for s in session_names if os.path.isdir(os.path.join(placement_dir, s))]
        if session_names:
            session_dir = os.path.join(placement_dir, session_names[-1])

    if session_dir is None:
        raise RuntimeError("Could not determine session directory for run")

    artifacts = _collect_artifacts(session_dir)
    stage_times = _parse_resource_report(artifacts.resource_report)

    gemini_image = _duration(markers, "gemini_image_start", "gemini_image_end")
    gemini_cutout = _duration(markers, "gemini_cutout_start", "gemini_cutout_end")
    gemini_total = _sum_durations([gemini_image, gemini_cutout])
    hunyuan_total = _duration(markers, "hunyuan_start", "hunyuan_end")

    placement_total = _duration_value(run_start, markers.get("placement_done"))
    if placement_total is None:
        placement_total = stage_times.get("Total")

    placement_excluding = None
    if placement_total is not None:
        placement_excluding = placement_total - (gemini_total or 0.0) - (hunyuan_total or 0.0)
        if placement_excluding < 0:
            placement_excluding = 0.0

    optimization_total = _duration(markers, "optimization_start", "optimization_end")
    if optimization_total is None:
        optimization_total = stage_times.get("Post-placement optimization")

    return {
        "returncode": rc,
        "prompt_id": prompt_spec["id"],
        "prompt": prompt_spec["prompt"],
        "target_text": prompt_spec["target_text"],
        "run_total_s": run_end - run_start,
        "placement_total_including_gen_s": placement_total,
        "placement_total_excluding_gen_s": placement_excluding,
        "gemini_total_s": gemini_total,
        "hunyuan_total_s": hunyuan_total,
        "optimization_total_s": optimization_total,
        "stage_times": stage_times,
        "markers": markers,
        "line_events": line_events,
        "losses": losses,
        "artifacts": artifacts,
    }


def _duration(markers: Dict[str, float], a: str, b: str) -> Optional[float]:
    if a not in markers or b not in markers:
        return None
    return max(0.0, markers[b] - markers[a])


def _duration_value(start: float, end: Optional[float]) -> Optional[float]:
    if end is None:
        return None
    return max(0.0, end - start)


def _sum_durations(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid))


def _maybe_eval(
    project_root: str,
    eval_env: str,
    artifacts: RunArtifacts,
    target_text: str,
) -> Tuple[Optional[EvalResult], Optional[EvalResult]]:
    if not artifacts.selected_view or not artifacts.diffusion:
        return None, None

    pre = None
    post = None

    if artifacts.final_pre and os.path.exists(artifacts.final_pre):
        pre = _call_eval_worker(
            project_root=project_root,
            eval_env=eval_env,
            initial_img=artifacts.selected_view,
            diffusion_img=artifacts.diffusion,
            final_img=artifacts.final_pre,
            source_text=SOURCE_TEXT,
            target_text=target_text,
        )

    final_post = artifacts.final_post or artifacts.final_pre
    if final_post and os.path.exists(final_post):
        post = _call_eval_worker(
            project_root=project_root,
            eval_env=eval_env,
            initial_img=artifacts.selected_view,
            diffusion_img=artifacts.diffusion,
            final_img=final_post,
            source_text=SOURCE_TEXT,
            target_text=target_text,
        )

    return pre, post


def _write_run_outputs(root_out: str, run_label: str, run_data: Dict, eval_pre, eval_post) -> Dict:
    run_dir = os.path.join(root_out, "runs", run_label)
    _mkdir(run_dir)

    artifacts: RunArtifacts = run_data["artifacts"]

    _copy_if_exists(artifacts.selected_view, os.path.join(run_dir, "selected_camera_view.png"))
    _copy_if_exists(artifacts.diffusion, os.path.join(run_dir, "gemini_diffusion_added.png"))
    _copy_if_exists(artifacts.bbox, os.path.join(run_dir, "detection_bbox.png"))
    _copy_if_exists(artifacts.added_mask, os.path.join(run_dir, "added_object_mask.png"))
    _copy_if_exists(artifacts.final_pre, os.path.join(run_dir, "final_view_with_object.png"))
    _copy_if_exists(artifacts.final_post, os.path.join(run_dir, "final_view_with_object_optimized.png"))
    _copy_if_exists(artifacts.highlighted, os.path.join(run_dir, "highlighted_verification.png"))
    _copy_if_exists(artifacts.resource_report, os.path.join(run_dir, "detection_resource_report.txt"))

    with open(os.path.join(run_dir, "raw_log.json"), "w", encoding="utf-8") as f:
        json.dump(run_data["line_events"], f, indent=2)

    with open(os.path.join(run_dir, "optimization_loss.json"), "w", encoding="utf-8") as f:
        json.dump(run_data["losses"], f, indent=2)

    run_meta = {
        "run_label": run_label,
        "returncode": run_data["returncode"],
        "prompt_id": run_data["prompt_id"],
        "prompt": run_data["prompt"],
        "target_text": run_data["target_text"],
        "session_dir": run_data["artifacts"].session_dir,
        "timing": {
            "run_total_s": run_data["run_total_s"],
            "placement_total_including_gen_s": run_data["placement_total_including_gen_s"],
            "placement_total_excluding_gen_s": run_data["placement_total_excluding_gen_s"],
            "gemini_total_s": run_data["gemini_total_s"],
            "hunyuan_total_s": run_data["hunyuan_total_s"],
            "optimization_total_s": run_data["optimization_total_s"],
        },
        "stage_times": run_data["stage_times"],
        "eval_pre": eval_pre.__dict__ if eval_pre else None,
        "eval_post": eval_post.__dict__ if eval_post else None,
    }

    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    storyboard_path = os.path.join(run_dir, "storyboard.png")
    _make_storyboard(
        storyboard_path,
        title=f"{run_label}: {run_data['prompt_id']}",
        labeled_images=[
            ("Selected view", artifacts.selected_view),
            ("Gemini diffusion", artifacts.diffusion),
            ("Detection bbox", artifacts.bbox),
            ("Added object mask", artifacts.added_mask),
            ("Initial placement", artifacts.final_pre),
            ("After optimization", artifacts.final_post or artifacts.final_pre),
        ],
    )

    return {
        "run_dir": run_dir,
        "storyboard": storyboard_path if os.path.exists(storyboard_path) else None,
        "selected_view": artifacts.selected_view,
        "diffusion": artifacts.diffusion,
        "final_pre": artifacts.final_pre,
        "final_post": artifacts.final_post or artifacts.final_pre,
    }


def _build_report(root_out: str, rows: List[Dict]) -> None:
    if not rows:
        return

    numeric_cols = [
        "run_total_s",
        "placement_total_including_gen_s",
        "placement_total_excluding_gen_s",
        "gemini_total_s",
        "hunyuan_total_s",
        "optimization_total_s",
        "clip_directional_similarity_pre",
        "clip_text_directional_similarity_pre",
        "dino_similarity_pre",
        "background_ssim_pre",
        "clip_directional_similarity_post",
        "clip_text_directional_similarity_post",
        "dino_similarity_post",
        "background_ssim_post",
    ]

    stats = {}
    for col in numeric_cols:
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        if vals:
            stats[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    total = len(rows)
    success = sum(1 for r in rows if r.get("returncode") == 0)
    eval_ok = sum(1 for r in rows if r.get("eval_available"))

    report_path = os.path.join(root_out, "report_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# DG-3DPlace Automated Paper Experiment Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("## Run summary\n")
        f.write(f"- Total runs: {total}\n")
        f.write(f"- Successful runs: {success}\n")
        f.write(f"- Evaluated runs: {eval_ok}\n\n")

        f.write("## Core outputs\n")
        f.write("- per_run_summary.csv\n")
        f.write("- per_stage_timings.csv\n")
        f.write("- ablation_bars.png\n")
        f.write("- success_rate.png\n")
        f.write("- failure_taxonomy.png\n")
        f.write("- timing_comparison.png\n")
        f.write("- metric_delta_post_minus_pre.png\n")
        f.write("- optimization_loss_curves.png\n")
        f.write("- cross_run_visual_grid.png\n\n")

        f.write("## Aggregate statistics\n")
        for key, st in sorted(stats.items()):
            f.write(
                f"- {key}: mean={st['mean']:.4f}, std={st['std']:.4f}, min={st['min']:.4f}, max={st['max']:.4f}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated DG-3DPlace paper experiments")
    parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    parser.add_argument("--placement-dir", default="placement_4")
    parser.add_argument("--eval-env", default="dg3d_eval")
    parser.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--camera-mode", choices=["fixed", "random", "cycle"], default="fixed")
    parser.add_argument("--camera-index", type=int, default=2, help="1-based camera index for fixed mode")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means use all hardcoded prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--use-existing-sessions",
        action="store_true",
        help="Do not run new generations. Aggregate experiments from existing placement_4/session_* folders.",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=0,
        help="When using existing sessions, keep only the newest N sessions (0 means all).",
    )
    return parser.parse_args()


def _pick_camera(mode: str, fixed_index: int, run_idx: int, rng: random.Random, num_cameras: int = 15) -> int:
    if mode == "fixed":
        return max(1, min(num_cameras, fixed_index))
    if mode == "cycle":
        return 1 + ((run_idx - 1) % num_cameras)
    return rng.randint(1, num_cameras)


def main() -> int:
    args = parse_args()

    if not args.use_existing_sessions and not args.gemini_api_key.strip():
        print("ERROR: Provide --gemini-api-key (or set GEMINI_API_KEY)")
        return 1

    project_root = os.path.abspath(args.project_root)
    placement_dir = os.path.join(project_root, args.placement_dir)
    if not os.path.isdir(placement_dir):
        print(f"ERROR: placement dir not found: {placement_dir}")
        return 1

    prompts = [BASELINE_PROMPT] + list(PROMPT_SPECS)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    out_root = args.output_root.strip() or os.path.join(
        project_root,
        "experiments",
        "paper_automation",
        "results",
        f"run_{_ts()}",
    )
    _mkdir(out_root)
    _mkdir(os.path.join(out_root, "runs"))
    _mkdir(os.path.join(out_root, "figures"))
    _mkdir(os.path.join(out_root, "tables"))

    rng = random.Random(args.seed)

    all_rows = []
    stage_rows = []
    loss_curves: Dict[str, List[Dict]] = {}
    visual_rows = []

    if args.use_existing_sessions:
        existing_sessions = _list_existing_sessions(placement_dir=placement_dir, limit_sessions=args.limit_sessions)
        print(f"Output root: {out_root}")
        print(f"Using existing sessions: {len(existing_sessions)}")
        if not existing_sessions:
            print("ERROR: No existing session_* folders found.")
            return 1

        for idx, session_dir in enumerate(existing_sessions, start=1):
            run_label = f"existing_{idx:03d}_{os.path.basename(session_dir)}"
            print("\n" + "=" * 90)
            print(f"Processing {run_label}")

            run_data = _build_run_data_from_existing_session(session_dir)

            eval_pre = None
            eval_post = None
            if not args.skip_eval:
                eval_pre, eval_post = _maybe_eval(
                    project_root=project_root,
                    eval_env=args.eval_env,
                    artifacts=run_data["artifacts"],
                    target_text=run_data["target_text"],
                )

            eval_available = bool(eval_pre and eval_post and not eval_pre.errors and not eval_post.errors)

            paths_out = _write_run_outputs(
                root_out=out_root,
                run_label=run_label,
                run_data=run_data,
                eval_pre=eval_pre,
                eval_post=eval_post,
            )

            for stage_name, stage_value in sorted(run_data["stage_times"].items()):
                stage_rows.append(
                    {
                        "run_label": run_label,
                        "prompt_id": run_data["prompt_id"],
                        "stage": stage_name,
                        "time_s": stage_value,
                    }
                )

            all_rows.append(
                {
                    "run_label": run_label,
                    "prompt_id": run_data["prompt_id"],
                    "prompt": run_data["prompt"],
                    "target_text": run_data["target_text"],
                    "camera_index": "",
                    "returncode": run_data["returncode"],
                    "session_dir": run_data["artifacts"].session_dir,
                    "run_total_s": run_data["run_total_s"],
                    "placement_total_including_gen_s": run_data["placement_total_including_gen_s"],
                    "placement_total_excluding_gen_s": run_data["placement_total_excluding_gen_s"],
                    "gemini_total_s": run_data["gemini_total_s"],
                    "hunyuan_total_s": run_data["hunyuan_total_s"],
                    "optimization_total_s": run_data["optimization_total_s"],
                    "clip_directional_similarity_pre": eval_pre.clip_directional_similarity if eval_pre else None,
                    "clip_text_directional_similarity_pre": eval_pre.clip_text_directional_similarity if eval_pre else None,
                    "dino_similarity_pre": eval_pre.dino_similarity if eval_pre else None,
                    "background_ssim_pre": eval_pre.background_ssim if eval_pre else None,
                    "clip_directional_similarity_post": eval_post.clip_directional_similarity if eval_post else None,
                    "clip_text_directional_similarity_post": eval_post.clip_text_directional_similarity if eval_post else None,
                    "dino_similarity_post": eval_post.dino_similarity if eval_post else None,
                    "background_ssim_post": eval_post.background_ssim if eval_post else None,
                    "eval_pre_errors": json.dumps(eval_pre.errors) if eval_pre else "",
                    "eval_post_errors": json.dumps(eval_post.errors) if eval_post else "",
                    "eval_available": eval_available,
                    "error": "",
                }
            )

            loss_curves[run_label] = run_data["losses"]
            visual_rows.append(
                {
                    "run_label": run_label,
                    "selected_view": paths_out["selected_view"],
                    "diffusion": paths_out["diffusion"],
                    "final_pre": paths_out["final_pre"],
                    "final_post": paths_out["final_post"],
                }
            )
    else:
        print(f"Output root: {out_root}")
        print(f"Total prompts: {len(prompts)}")

        for idx, prompt_spec in enumerate(prompts, start=1):
            run_label = f"run_{idx:02d}_{prompt_spec['id']}"
            camera_index = _pick_camera(args.camera_mode, args.camera_index, idx, rng)
            print("\n" + "=" * 90)
            print(f"Starting {run_label}")
            print(f"Prompt: {prompt_spec['prompt']}")
            print(f"Camera index: {camera_index}")

            try:
                run_data = _run_one(
                    project_root=project_root,
                    placement_dir=placement_dir,
                    prompt_spec=prompt_spec,
                    run_idx=idx,
                    camera_index=camera_index,
                    gemini_api_key=args.gemini_api_key,
                )
            except Exception as exc:
                print(f"Run failed for {run_label}: {exc}")
                all_rows.append(
                    {
                        "run_label": run_label,
                        "prompt_id": prompt_spec["id"],
                        "prompt": prompt_spec["prompt"],
                        "camera_index": camera_index,
                        "returncode": -1,
                        "error": str(exc),
                    }
                )
                continue

            eval_pre = None
            eval_post = None
            if not args.skip_eval:
                eval_pre, eval_post = _maybe_eval(
                    project_root=project_root,
                    eval_env=args.eval_env,
                    artifacts=run_data["artifacts"],
                    target_text=prompt_spec["target_text"],
                )

            eval_available = bool(eval_pre and eval_post and not eval_pre.errors and not eval_post.errors)

            paths_out = _write_run_outputs(
                root_out=out_root,
                run_label=run_label,
                run_data=run_data,
                eval_pre=eval_pre,
                eval_post=eval_post,
            )

            for stage_name, stage_value in sorted(run_data["stage_times"].items()):
                stage_rows.append(
                    {
                        "run_label": run_label,
                        "prompt_id": prompt_spec["id"],
                        "stage": stage_name,
                        "time_s": stage_value,
                    }
                )

            row = {
                "run_label": run_label,
                "prompt_id": prompt_spec["id"],
                "prompt": prompt_spec["prompt"],
                "target_text": prompt_spec["target_text"],
                "camera_index": camera_index,
                "returncode": run_data["returncode"],
                "session_dir": run_data["artifacts"].session_dir,
                "run_total_s": run_data["run_total_s"],
                "placement_total_including_gen_s": run_data["placement_total_including_gen_s"],
                "placement_total_excluding_gen_s": run_data["placement_total_excluding_gen_s"],
                "gemini_total_s": run_data["gemini_total_s"],
                "hunyuan_total_s": run_data["hunyuan_total_s"],
                "optimization_total_s": run_data["optimization_total_s"],
                "clip_directional_similarity_pre": eval_pre.clip_directional_similarity if eval_pre else None,
                "clip_text_directional_similarity_pre": eval_pre.clip_text_directional_similarity if eval_pre else None,
                "dino_similarity_pre": eval_pre.dino_similarity if eval_pre else None,
                "background_ssim_pre": eval_pre.background_ssim if eval_pre else None,
                "clip_directional_similarity_post": eval_post.clip_directional_similarity if eval_post else None,
                "clip_text_directional_similarity_post": eval_post.clip_text_directional_similarity if eval_post else None,
                "dino_similarity_post": eval_post.dino_similarity if eval_post else None,
                "background_ssim_post": eval_post.background_ssim if eval_post else None,
                "eval_pre_errors": json.dumps(eval_pre.errors) if eval_pre else "",
                "eval_post_errors": json.dumps(eval_post.errors) if eval_post else "",
                "eval_available": eval_available,
                "error": "",
            }
            all_rows.append(row)

            loss_curves[run_label] = run_data["losses"]
            visual_rows.append(
                {
                    "run_label": run_label,
                    "selected_view": paths_out["selected_view"],
                    "diffusion": paths_out["diffusion"],
                    "final_pre": paths_out["final_pre"],
                    "final_post": paths_out["final_post"],
                }
            )

    # Save tabular outputs
    summary_headers = [
        "run_label",
        "prompt_id",
        "prompt",
        "target_text",
        "camera_index",
        "returncode",
        "session_dir",
        "run_total_s",
        "placement_total_including_gen_s",
        "placement_total_excluding_gen_s",
        "gemini_total_s",
        "hunyuan_total_s",
        "optimization_total_s",
        "clip_directional_similarity_pre",
        "clip_text_directional_similarity_pre",
        "dino_similarity_pre",
        "background_ssim_pre",
        "clip_directional_similarity_post",
        "clip_text_directional_similarity_post",
        "dino_similarity_post",
        "background_ssim_post",
        "eval_available",
        "eval_pre_errors",
        "eval_post_errors",
        "error",
    ]
    _write_csv(os.path.join(out_root, "tables", "per_run_summary.csv"), all_rows, summary_headers)

    stage_headers = ["run_label", "prompt_id", "stage", "time_s"]
    _write_csv(os.path.join(out_root, "tables", "per_stage_timings.csv"), stage_rows, stage_headers)

    success_rows = [r for r in all_rows if r.get("returncode") == 0]

    _best_effort_output(
        "timing_comparison.png",
        _plot_timing_bars,
        os.path.join(out_root, "figures", "timing_comparison.png"),
        success_rows,
    )
    _best_effort_output(
        "metric_delta_post_minus_pre.png",
        _plot_metric_delta,
        os.path.join(out_root, "figures", "metric_delta_post_minus_pre.png"),
        success_rows,
    )
    _best_effort_output(
        "optimization_loss_curves.png",
        _plot_loss_curves,
        os.path.join(out_root, "figures", "optimization_loss_curves.png"),
        loss_curves,
    )
    _best_effort_output(
        "cross_run_visual_grid.png",
        _make_cross_run_grid,
        os.path.join(out_root, "figures", "cross_run_visual_grid.png"),
        visual_rows,
    )
    _best_effort_output(
        "success_rate.png",
        _plot_success_rate,
        os.path.join(out_root, "figures", "success_rate.png"),
        all_rows,
    )
    _best_effort_output(
        "failure_taxonomy.png",
        _plot_failure_taxonomy,
        os.path.join(out_root, "figures", "failure_taxonomy.png"),
        all_rows,
    )
    _best_effort_output(
        "ablation_bars.png",
        _plot_ablation_bars,
        os.path.join(out_root, "figures", "ablation_bars.png"),
        success_rows,
    )

    _best_effort_output("report_summary.md", _build_report, out_root, success_rows)

    config_dump = {
        "project_root": project_root,
        "placement_dir": placement_dir,
        "eval_env": args.eval_env,
        "camera_mode": args.camera_mode,
        "camera_index": args.camera_index,
        "seed": args.seed,
        "prompt_count": len(prompts),
        "skip_eval": args.skip_eval,
        "output_root": out_root,
        "generated_at": datetime.now().isoformat(),
    }
    _best_effort_output(
        "run_config.json",
        _write_json,
        os.path.join(out_root, "run_config.json"),
        config_dump,
    )

    print("\nCompleted automation run.")
    print(f"Summary table: {os.path.join(out_root, 'tables', 'per_run_summary.csv')}")
    print(f"Figures dir:   {os.path.join(out_root, 'figures')}")
    print(f"Report:        {os.path.join(out_root, 'report_summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python run_experiments.py --gemini-api-key "KEY YO" --camera-mode random --eval-env dg3d_eval