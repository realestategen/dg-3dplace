import cv2
import argparse
from pathlib import Path
import sys

def process_video(video_path: Path, output_dir: Path, target_fps: int = 2):
    """
    Reads a video file and saves frames to an 'input' folder.
    
    Args:
        video_path (Path): Path to the source video.
        output_dir (Path): Root data folder (images will be saved to output_dir/input).
        target_fps (int): Number of frames to save per second of video.
    """
    if not video_path.exists():
        print(f"[Error] Video not found: {video_path}")
        sys.exit(1)

    # 3DGS expects images in a folder named 'input'
    input_folder = output_dir / "input"
    input_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_hop = round(video_fps / target_fps)
    
    print(f"Processing: {video_path.name}")
    print(f"Duration: {total_frames/video_fps:.2f}s | Saving ~{target_fps} fps")

    curr_frame = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if curr_frame % frame_hop == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            save_path = input_folder / frame_name
            cv2.imwrite(str(save_path), frame)
            saved_count += 1
            print(f"Saved: {frame_name}", end='\r')

        curr_frame += 1

    cap.release()
    print(f"\n[Success] Extracted {saved_count} frames to {input_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for 3DGS")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", "-o", type=str, default="../data", help="Output data directory")
    parser.add_argument("--fps", type=int, default=2, help="Target FPS extraction rate")
    
    args = parser.parse_args()
    process_video(Path(args.video), Path(args.output), args.fps)