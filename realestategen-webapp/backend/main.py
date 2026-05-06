from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from database import engine, Base
import models  # ensure models are registered
from routers import videos, scenes, captures

Base.metadata.create_all(bind=engine)

app = FastAPI(title="RealEstateGen DG-3DGS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router)
app.include_router(scenes.router)
app.include_router(captures.router)

# serve captured images statically
CAPTURES_DIR = Path(__file__).parent.parent / "data" / "captured_images"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")

# serve splat files statically
SCENES_DIR = Path(__file__).parent.parent / "data" / "scenes"
SCENES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/scenes", StaticFiles(directory=str(SCENES_DIR)), name="scenes")


@app.get("/health")
def health():
    return {"status": "ok"}
