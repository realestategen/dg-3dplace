"""Plot GPU/CPU resource usage from detection_resource_report.txt files.

Usage:
    python plot_resource_usage.py --session session_20260427_214420
    python plot_resource_usage.py --all
    python plot_resource_usage.py --sessions session_A session_B session_C
"""
import os
import sys
import csv
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


HERE = os.path.dirname(os.path.abspath(__file__))


def parse_resource_csv(report_path):
    """Parse the CSV section from a resource report.
    
    Returns dict: {stage_name -> {metric -> value}}
    """
    if not os.path.exists(report_path):
        return None
    
    data = {}
    in_csv_section = False
    headers = None
    
    try:
        with open(report_path, "r") as f:
            for line in f:
                line = line.strip()
                
                if line.startswith("Per-Stage Resource Usage (CSV)"):
                    in_csv_section = True
                    continue
                
                if not in_csv_section or not line:
                    continue
                
                if not headers:
                    headers = [h.strip() for h in line.split(",")]
                    continue
                
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < len(headers):
                    continue
                
                row = dict(zip(headers, parts))
                stage = row.get("Stage", "").strip()
                if stage:
                    data[stage] = {
                        "elapsed_s": float(row.get("Elapsed(s)", 0)),
                        "cpu_user_s": float(row.get("CPU_User(s)", 0)),
                        "cpu_system_s": float(row.get("CPU_System(s)", 0)),
                        "mem_mb": float(row.get("Mem_Delta(MB)", 0)),
                        "gpu_mem_mb": float(row.get("GPU_Mem_Delta(MB)", 0)),
                    }
    except Exception as e:
        print(f"Error parsing {report_path}: {e}")
        return None
    
    return data if data else None


def plot_resource_usage(session_dir, output_dir=None):
    """Plot resource usage for a single session."""
    if output_dir is None:
        output_dir = session_dir
    
    report_path = os.path.join(session_dir, "detection_resource_report.txt")
    data = parse_resource_csv(report_path)
    
    if not data:
        print(f"No resource data found in {report_path}")
        return
    
    stages = list(data.keys())
    elapsed = [data[s]["elapsed_s"] for s in stages]
    cpu_user = [data[s]["cpu_user_s"] for s in stages]
    cpu_system = [data[s]["cpu_system_s"] for s in stages]
    mem_mb = [data[s]["mem_mb"] for s in stages]
    gpu_mem_mb = [data[s]["gpu_mem_mb"] for s in stages]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Resource Usage: {os.path.basename(session_dir)}", fontsize=16, fontweight="bold")
    
    # Elapsed time
    ax = axes[0, 0]
    ax.barh(stages, elapsed, color="steelblue")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_title("Stage Duration")
    ax.grid(axis="x", alpha=0.3)
    
    # CPU usage
    ax = axes[0, 1]
    x = np.arange(len(stages))
    width = 0.35
    ax.bar(x - width/2, cpu_user, width, label="CPU User", color="orange")
    ax.bar(x + width/2, cpu_system, width, label="CPU System", color="coral")
    ax.set_ylabel("Time (s)")
    ax.set_title("CPU Usage per Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Memory usage
    ax = axes[1, 0]
    ax.barh(stages, mem_mb, color="green", alpha=0.7, label="RAM")
    ax.set_xlabel("Memory Delta (MB)")
    ax.set_title("Memory Usage per Stage")
    ax.grid(axis="x", alpha=0.3)
    
    # GPU memory usage
    ax = axes[1, 1]
    ax.barh(stages, gpu_mem_mb, color="purple", alpha=0.7)
    ax.set_xlabel("GPU Memory Delta (MB)")
    ax.set_title("GPU Memory Usage per Stage")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f"resource_usage_{os.path.basename(session_dir)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot resource usage from session reports")
    parser.add_argument("--session", default=None, help="Single session to plot")
    parser.add_argument("--all", action="store_true", help="Plot all session_* folders")
    parser.add_argument("--sessions", nargs="+", default=[], help="Multiple sessions to plot")
    parser.add_argument("--out", default=None, help="Output directory for plots")
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
        sys.exit(1)
    
    sessions = []
    if args.all:
        sessions = sorted([
            os.path.join(HERE, d) for d in os.listdir(HERE)
            if d.startswith("session_") and os.path.isdir(os.path.join(HERE, d))
        ])
    elif args.session:
        session_path = args.session
        if not os.path.isabs(session_path):
            session_path = os.path.join(HERE, session_path)
        sessions = [os.path.abspath(session_path)]
    elif args.sessions:
        for s in args.sessions:
            session_path = s if os.path.isabs(s) else os.path.join(HERE, s)
            sessions.append(os.path.abspath(session_path))
    else:
        print("Specify --session, --all, or --sessions")
        sys.exit(1)
    
    output_dir = args.out or os.path.join(HERE, "resource_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    for session in sessions:
        if os.path.isdir(session):
            print(f"Plotting {os.path.basename(session)}...")
            plot_resource_usage(session, output_dir)
        else:
            print(f"Session not found: {session}")
    
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
