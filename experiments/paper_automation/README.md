# DG-3DPlace Paper Automation

This folder provides a standalone automation pipeline for research-paper experiments.

It runs the existing placement pipeline end-to-end (without modifying any current project files), then creates:
- Per-run visual storyboards
- Cross-run visual comparison grids
- Timing tables, including two placement totals:
  - including Gemini + Hunyuan object creation
  - excluding Gemini + Hunyuan object creation
- Optimization loss curves
- Quality metrics from DG_3DPlace_Evaluation (run in conda env `dg3d_eval`)

## Files
- `run_experiments.py`: batch orchestrator
- `prompts.py`: hardcoded prompt list for automation
- `eval_worker.py`: metric worker executed in `dg3d_eval`

## Usage
From project root:

```bash
python experiments/paper_automation/run_experiments.py \
  --gemini-api-key "YOUR_GEMINI_KEY" \
  --camera-mode random \
  --eval-env dg3d_eval
```

Optional controls:

```bash
python experiments/paper_automation/run_experiments.py \
  --gemini-api-key "YOUR_GEMINI_KEY" \
  --camera-mode fixed \
  --camera-index 2 \
  --max-prompts 4 \
  --seed 123
```

## Output structure
A timestamped output folder is created at:

`experiments/paper_automation/results/run_YYYYMMDD_HHMMSS`

Inside:
- `tables/per_run_summary.csv`
- `tables/per_stage_timings.csv`
- `figures/timing_comparison.png`
- `figures/metric_delta_post_minus_pre.png`
- `figures/optimization_loss_curves.png`
- `figures/cross_run_visual_grid.png`
- `runs/<run_label>/storyboard.png`
- `report_summary.md`

## Notes
- The script calls `placement_4/detection_optimized.py` as-is and automates the interactive camera selection by writing the camera index to stdin.
- It sets `GEMINI_API_KEY` for each subprocess so you only provide the key once.
- No files under `DG_3DPlace_Evaluation` are changed.
