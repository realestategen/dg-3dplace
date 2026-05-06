import subprocess
import threading
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Scene

SUDO_PASS = "tuf@cseg2"
DATA_DIR = Path(__file__).parent.parent / "data"
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _get_file_logger(scene_id: int) -> logging.Logger:
    logger = logging.getLogger(f"scene.{scene_id}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    log_file = LOGS_DIR / f"scene_{scene_id}.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def _run_docker(cmd: list[str], workspace: Path, log_cb) -> int:
    full_cmd = ["sudo", "-S", "docker", "run", "--runtime=nvidia", "--rm",
                "-v", f"{workspace}:/workspace",
                "nerfstudio/nerfstudio:latest"] + cmd

    proc = subprocess.Popen(
        full_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc.stdin.write(SUDO_PASS + "\n")
    proc.stdin.flush()
    proc.stdin.close()

    for line in proc.stdout:
        log_cb(line.rstrip())

    proc.wait()
    return proc.returncode


def _update_scene(scene_id: int, **kwargs):
    db: Session = SessionLocal()
    try:
        scene = db.get(Scene, scene_id)
        if scene:
            for k, v in kwargs.items():
                setattr(scene, k, v)
            scene.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


def _append_log(scene_id: int, line: str):
    db: Session = SessionLocal()
    try:
        scene = db.get(Scene, scene_id)
        if scene:
            current = scene.log or ""
            lines = current.split("\n") if current else []
            lines.append(line)
            # keep last 500 lines to avoid unbounded growth
            scene.log = "\n".join(lines[-500:])
            scene.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


def process_scene(scene_id: int, video_path: str):
    workspace = DATA_DIR / "scenes" / str(scene_id)
    workspace.mkdir(parents=True, exist_ok=True)
    os.chmod(workspace, 0o777)

    dest_video = workspace / "input_video.mp4"
    shutil.copy2(video_path, dest_video)
    os.chmod(dest_video, 0o777)

    (workspace / "data").mkdir(exist_ok=True)
    os.chmod(workspace / "data", 0o777)

    file_logger = _get_file_logger(scene_id)

    def log(line: str):
        file_logger.info(line)
        _append_log(scene_id, line)

    try:
        # Step 1: extract frames + COLMAP
        _update_scene(scene_id, status="processing_frames")
        log("=== Extracting frames and running COLMAP ===")
        rc = _run_docker([
            "ns-process-data", "video",
            "--data", "/workspace/input_video.mp4",
            "--output-dir", "/workspace/data",
            "--matching-method", "sequential",
            "--colmap-cmd-extra-args",
            "sequential_matcher: --SiftMatching.use_gpu 0",
        ], workspace, log)

        if rc != 0:
            _update_scene(scene_id, status="failed")
            log(f"ns-process-data failed with exit code {rc}")
            return

        subprocess.run(["sudo", "-S", "chmod", "-R", "777", str(workspace)],
                       input=SUDO_PASS + "\n", text=True, capture_output=True)

        # Step 2: train splatfacto
        _update_scene(scene_id, status="training")
        log("=== Training splatfacto ===")
        rc = _run_docker([
            "ns-train", "splatfacto",
            "--data", "/workspace/data",
            "--output-dir", "/workspace/output",
            "--max-num-iterations", "7000",
            "--viewer.quit-on-train-completion", "True",
            "colmap",
            "--colmap-path", "colmap/sparse/0",
            "--images-path", "images",
            "--downscale-factor", "1",
        ], workspace, log)

        if rc != 0:
            _update_scene(scene_id, status="failed")
            log(f"ns-train failed with exit code {rc}")
            return

        subprocess.run(["sudo", "-S", "chmod", "-R", "777", str(workspace)],
                       input=SUDO_PASS + "\n", text=True, capture_output=True)

        # find config.yml
        configs = list((workspace / "output").rglob("config.yml"))
        if not configs:
            _update_scene(scene_id, status="failed")
            log("No config.yml found after training")
            return
        config_rel = configs[0].relative_to(workspace)

        # Step 3: export .splat
        _update_scene(scene_id, status="exporting")
        log("=== Exporting gaussian splat ===")
        rc = _run_docker([
            "ns-export", "gaussian-splat",
            "--load-config", f"/workspace/{config_rel}",
            "--output-dir", "/workspace/output/splat_export",
        ], workspace, log)

        subprocess.run(["sudo", "-S", "chmod", "-R", "777", str(workspace)],
                       input=SUDO_PASS + "\n", text=True, capture_output=True)

        if rc != 0:
            _update_scene(scene_id, status="failed")
            log(f"ns-export failed with exit code {rc}")
            return

        splat_file = workspace / "output" / "splat_export" / "splat.ply"
        if not splat_file.exists():
            _update_scene(scene_id, status="failed")
            log("splat.ply not found after export")
            return

        _update_scene(scene_id, status="done", splat_path=str(splat_file))
        log("=== Done ===")

    except Exception as e:
        _update_scene(scene_id, status="failed")
        msg = f"Unexpected error: {e}"
        file_logger.exception(msg)
        _append_log(scene_id, msg)


def start_processing(scene_id: int, video_path: str):
    t = threading.Thread(target=process_scene, args=(scene_id, video_path), daemon=True)
    t.start()
