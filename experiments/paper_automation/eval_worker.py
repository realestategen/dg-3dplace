#!/usr/bin/env python3
"""Run DG_3DPlace_Evaluation metrics on arbitrary image paths.

This script is intended to run inside the dg3d_eval conda environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _append_eval_path(project_root: str) -> None:
    eval_root = os.path.join(project_root, "DG_3DPlace_Evaluation")
    if eval_root not in sys.path:
        sys.path.insert(0, eval_root)


def _safe_metric(fn, *args):
    try:
        value = fn(*args)
        return float(value), ""
    except Exception as exc:
        return None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute DG_3DPlace metrics for one run")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--diffusion", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--source-text", required=True)
    parser.add_argument("--target-text", required=True)
    args = parser.parse_args()

    _append_eval_path(args.project_root)

    from metrics import (
        calculate_ssim,
        clip_directional_similarity,
        clip_text_directional_similarity,
        dino_similarity,
    )

    results = {
        "clip_directional_similarity": None,
        "clip_text_directional_similarity": None,
        "dino_similarity": None,
        "background_ssim": None,
        "errors": {},
    }

    cdir, err = _safe_metric(
        clip_directional_similarity,
        args.initial,
        args.final,
        args.diffusion,
    )
    if err:
        results["errors"]["clip_directional_similarity"] = err
    results["clip_directional_similarity"] = cdir

    ctext, err = _safe_metric(
        clip_text_directional_similarity,
        args.initial,
        args.final,
        args.source_text,
        args.target_text,
    )
    if err:
        results["errors"]["clip_text_directional_similarity"] = err
    results["clip_text_directional_similarity"] = ctext

    dino, err = _safe_metric(dino_similarity, args.final, args.diffusion)
    if err:
        results["errors"]["dino_similarity"] = err
    results["dino_similarity"] = dino

    ssim_val, err = _safe_metric(calculate_ssim, args.initial, args.final)
    if err:
        results["errors"]["background_ssim"] = err
    results["background_ssim"] = ssim_val

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
