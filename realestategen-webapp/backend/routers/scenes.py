from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db
from models import Scene, Video
from processing import start_processing

router = APIRouter(prefix="/api/scenes", tags=["scenes"])

SCENES_DIR = Path(__file__).parent.parent.parent / "data" / "scenes"


class CreateSceneBody(BaseModel):
    video_id: int
    name: str = ""


@router.post("")
def create_scene(body: CreateSceneBody, db: Session = Depends(get_db)):
    video_id = body.video_id
    name = body.name or f"scene_{video_id}"

    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    workspace = SCENES_DIR / "placeholder"  # real path set after insert
    scene = Scene(
        video_id=video_id,
        name=name,
        status="pending",
        workspace_path="",
    )
    db.add(scene)
    db.commit()
    db.refresh(scene)

    workspace = SCENES_DIR / str(scene.id)
    scene.workspace_path = str(workspace)
    db.commit()

    start_processing(scene.id, video.path)
    return _serialize(scene)


@router.get("")
def list_scenes(db: Session = Depends(get_db)):
    scenes = db.query(Scene).order_by(Scene.created_at.desc()).all()
    return [_serialize(s) for s in scenes]


@router.get("/{scene_id}")
def get_scene(scene_id: int, db: Session = Depends(get_db)):
    scene = db.get(Scene, scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")
    return _serialize(scene)


@router.get("/{scene_id}/status")
def get_status(scene_id: int, db: Session = Depends(get_db)):
    scene = db.get(Scene, scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")
    log_lines = (scene.log or "").split("\n")[-50:]
    return {
        "id": scene.id,
        "status": scene.status,
        "log_tail": log_lines,
        "splat_path": scene.splat_path,
    }


@router.get("/{scene_id}/splat")
def download_splat(scene_id: int, db: Session = Depends(get_db)):
    scene = db.get(Scene, scene_id)
    if not scene or scene.status != "done" or not scene.splat_path:
        raise HTTPException(404, "Splat not ready")
    return FileResponse(scene.splat_path, media_type="application/octet-stream",
                        filename=f"scene_{scene_id}.ply")


def _serialize(s: Scene):
    return {
        "id": s.id,
        "video_id": s.video_id,
        "name": s.name,
        "status": s.status,
        "splat_path": s.splat_path,
        "created_at": s.created_at.isoformat(),
        "updated_at": s.updated_at.isoformat(),
    }
