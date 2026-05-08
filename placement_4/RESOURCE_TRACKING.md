# Resource Usage Tracking

## Overview
The pipeline now records **per-stage CPU, GPU, and memory usage** for every detection session. This data is automatically captured and stored in `detection_resource_report.txt` for analysis and visualization.

## What's Tracked

Each pipeline stage records:
- **Elapsed time** (s): Cumulative time from pipeline start
- **CPU User time** (s): User-space CPU usage delta
- **CPU System time** (s): Kernel-space CPU usage delta  
- **Memory Delta** (MB): Change in RAM usage from start
- **GPU Memory Delta** (MB): Change in VRAM usage from start (NVIDIA only)

## Stages Monitored

1. **OWLv2 detection** - Object detection via vision-language model
2. **Unprojection & 3D detection** - Project 2D bounding box to 3D space
3. **Highlighting & ckpt** - Visualize detected region, save checkpoint
4. **Vase integration** - Generate and merge object Gaussians
5. **Final render** - Render the scene from saved camera angle
6. **Post-placement optimization** - Refine object placement
7. **Optimized final render** - Render optimized result

## Report Format

Each `detection_resource_report.txt` now contains:

### 1. Timing Summary (Table)
```
| Stage | Time (s) |
|---|---:|
| OWLv2 detection | 3.42 |
| ... | ... |
| Total | 79.61 |
```

### 2. Overall Resource Usage
```
CPU user time (s): 145.23
CPU system time (s): 8.47
Peak memory delta (MB): 2847.56
GPU: NVIDIA RTX 4090
Peak GPU memory delta (MB): 11523.67
```

### 3. Per-Stage CSV (for graphing)
```
Stage,Elapsed(s),CPU_User(s),CPU_System(s),Mem_Delta(MB),GPU_Mem_Delta(MB)
OWLv2 detection,3.42,12.45,0.89,145.23,512.34
Unprojection & 3D detection,3.60,2.15,0.12,156.78,512.34
...
```

## Parsing & Visualization

### Option 1: Python Script
Use the built-in `plot_resource_usage.py` script to generate plots:

```bash
# Single session
python plot_resource_usage.py --session session_20260427_214420

# Multiple sessions
python plot_resource_usage.py --sessions session_A session_B session_C

# All sessions
python plot_resource_usage.py --all

# Custom output directory
python plot_resource_usage.py --all --out ./plots
```

This generates PNG plots showing:
- Stage duration (time)
- CPU usage per stage
- Memory usage per stage
- GPU memory usage per stage

### Option 2: Parse CSV Manually
```python
import csv

with open("session_XXX/detection_resource_report.txt", "r") as f:
    lines = f.readlines()
    
# Find CSV section
csv_start = next(i for i, line in enumerate(lines) if "Per-Stage Resource Usage (CSV)" in line)
csv_lines = lines[csv_start + 1:]

reader = csv.DictReader(csv_lines)
for row in reader:
    stage = row['Stage']
    elapsed = float(row['Elapsed(s)'])
    cpu_user = float(row['CPU_User(s)'])
    mem = float(row['Mem_Delta(MB)'])
    gpu_mem = float(row['GPU_Mem_Delta(MB)'])
    print(f"{stage}: {elapsed:.2f}s, CPU={cpu_user:.2f}s, RAM={mem:.1f}MB, VRAM={gpu_mem:.1f}MB")
```

### Option 3: Use pandas
```python
import pandas as pd

# Extract CSV section
with open("session_XXX/detection_resource_report.txt", "r") as f:
    lines = f.readlines()
csv_start = next(i for i, line in enumerate(lines) if "Per-Stage Resource Usage (CSV)" in line) + 1

df = pd.read_csv(io.StringIO(''.join(lines[csv_start:])))
print(df)

# Plot
df.set_index('Stage').plot(kind='barh')
```

## Example Use Cases

1. **Identify bottlenecks**: Which stage uses most GPU memory?
2. **Compare sessions**: Which object placement took longer?
3. **Trend analysis**: How do metrics scale with object complexity?
4. **Hardware utilization**: Are we maxing out GPU VRAM?
5. **Optimization impact**: Does post-placement optimization use more resources?

## Notes

- All metrics are **deltas from pipeline start**, not absolute system values
- GPU tracking requires NVIDIA GPU and CUDA
- CPU times reflect actual usage (user + system mode)
- Memory values are approximate and may fluctuate
- CSV format is easy to parse with standard Python tools (csv, pandas)
