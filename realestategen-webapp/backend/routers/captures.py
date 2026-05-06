import base64
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db
from models import Capture, Scene

router = APIRouter(prefix="/api/captures", tags=["captures"])

CAPTURES_DIR = Path(__file__).parent.parent.parent / "data" / "captured_images"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


class SaveCaptureBody(BaseModel):
    scene_id: int
    image_data: str


@router.post("")
def save_capture(body: SaveCaptureBody, db: Session = Depends(get_db)):
    scene_id = body.scene_id
    image_data = body.image_data

    scene = db.get(Scene, scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")

    # strip data:image/png;base64, prefix
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"scene_{scene_id}_{ts}.png"
    dest = CAPTURES_DIR / filename

    with open(dest, "wb") as f:
        f.write(base64.b64decode(image_data))

    capture = Capture(scene_id=scene_id, filename=filename, path=str(dest))
    db.add(capture)
    db.commit()
    db.refresh(capture)
    return _serialize(capture)


@router.get("/scene/{scene_id}")
def list_captures(scene_id: int, db: Session = Depends(get_db)):
    captures = (db.query(Capture)
                .filter(Capture.scene_id == scene_id)
                .order_by(Capture.created_at.desc())
                .all())
    return [_serialize(c) for c in captures]


@router.get("/{capture_id}/image")
def get_capture_image(capture_id: int, db: Session = Depends(get_db)):
    capture = db.get(Capture, capture_id)
    if not capture:
        raise HTTPException(404, "Capture not found")
    return FileResponse(capture.path, media_type="image/png")


def _serialize(c: Capture):
    return {
        "id": c.id,
        "scene_id": c.scene_id,
        "filename": c.filename,
        "created_at": c.created_at.isoformat(),
    }
