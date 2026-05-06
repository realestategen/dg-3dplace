import os
import shutil
import subprocess
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from database import get_db
from models import Video

router = APIRouter(prefix="/api/videos", tags=["videos"])

VIDEOS_DIR = Path(__file__).parent.parent.parent / "data" / "videos"
THUMBS_DIR = Path(__file__).parent.parent.parent / "data" / "thumbnails"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
THUMBS_DIR.mkdir(parents=True, exist_ok=True)


def _generate_thumbnail(video_path: Path, thumb_path: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", "00:00:01",
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", "scale=480:-1",
                "-q:v", "3",
                str(thumb_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return thumb_path.exists()
    except Exception:
        return False


@router.post("")
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, "Only video files are accepted")

    dest = VIDEOS_DIR / file.filename
    counter = 1
    stem = Path(file.filename).stem
    suffix = Path(file.filename).suffix
    while dest.exists():
        dest = VIDEOS_DIR / f"{stem}_{counter}{suffix}"
        counter += 1

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size = os.path.getsize(dest)

    # generate thumbnail with ffmpeg
    thumb_path = THUMBS_DIR / f"{dest.stem}.jpg"
    thumbnail_path = str(thumb_path) if _generate_thumbnail(dest, thumb_path) else None

    video = Video(name=stem, filename=dest.name, path=str(dest),
                  size=size, thumbnail_path=thumbnail_path)
    db.add(video)
    db.commit()
    db.refresh(video)
    return _serialize(video)


@router.get("")
def list_videos(db: Session = Depends(get_db)):
    videos = db.query(Video).order_by(Video.created_at.desc()).all()
    return [_serialize(v) for v in videos]


@router.get("/{video_id}")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    return _serialize(video)


@router.get("/{video_id}/stream")
def stream_video(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    return FileResponse(video.path, media_type="video/mp4")


@router.get("/{video_id}/thumbnail")
def get_thumbnail(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video or not video.thumbnail_path:
        raise HTTPException(404, "Thumbnail not available")
    return FileResponse(video.thumbnail_path, media_type="image/jpeg")


@router.delete("/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    for p in [video.path, video.thumbnail_path]:
        if p and Path(p).exists():
            Path(p).unlink()
    db.delete(video)
    db.commit()
    return {"ok": True}


def _serialize(v: Video):
    return {
        "id": v.id,
        "name": v.name,
        "filename": v.filename,
        "size": v.size,
        "has_thumbnail": v.thumbnail_path is not None,
        "created_at": v.created_at.isoformat(),
    }
