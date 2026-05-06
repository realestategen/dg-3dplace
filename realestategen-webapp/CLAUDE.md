# RealEstateGen DG-3DGS Web App

## Project Overview
Web app for Diffusion-Guided 3D Gaussian Splatting (DG-3DGS). Users upload videos, create 3DGS scenes, and interactively view them with snapshot capture.

## Stack
- **Backend:** FastAPI + SQLite (SQLAlchemy 2.0+), Python 3.13, venv at `backend/.venv/`
- **Frontend:** React 18 + TypeScript + Vite 4, port 5174
- **3DGS Viewer:** `@mkkellogg/gaussian-splats-3d` (WebGL, in-browser)
- **Processing:** nerfstudio Docker container

## Ports
- Backend API: **8765** (ports 8000 and 8080 are occupied by Jitsi on this machine)
- Frontend dev: **5174**
- API docs: http://localhost:8765/docs

## Start
```bash
./start.sh   # starts both backend (8765) and frontend (5174)
```

Or manually:
```bash
# backend
cd backend && .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8765 --reload

# frontend
cd frontend && npm run dev -- --port 5174
```

## Docker — CRITICAL
**Always use `sudo docker run --runtime=nvidia`** — NOT `--gpus all` (fails on this machine, CDI mode) and NOT `--no-gpu` (too slow).

Sudo password: `tuf@cseg2` — pipe via stdin with `sudo -S`.

```bash
echo "tuf@cseg2" | sudo -S docker run --runtime=nvidia --rm \
  -v <workspace>:/workspace \
  nerfstudio/nerfstudio:latest <command>
```

## Processing Pipeline (backend/processing.py)
Three Docker steps run in a background thread per scene:

1. **Frame extraction + COLMAP**
   ```
   ns-process-data video --data /workspace/input_video.mp4 --output-dir /workspace/data
   ```

2. **Train splatfacto** (7000 iterations)
   ```
   ns-train splatfacto --data /workspace/data --output-dir /workspace/output \
     --max-num-iterations 7000 colmap \
     --colmap-path colmap/sparse/0 --images-path images --downscale-factor 1
   ```
   - COLMAP output lands at `data/colmap/sparse/0` (not `data/sparse/0`)

3. **Export splat**
   ```
   ns-export gaussian-splat --load-config /workspace/output/data/splatfacto/<timestamp>/config.yml \
     --output-dir /workspace/output/splat_export
   ```
   - Output: `output/splat_export/splat.ply`

After each Docker step, fix permissions with:
```bash
echo "tuf@cseg2" | sudo -S chmod -R 777 <workspace>
```

## Data Directories
```
data/
  videos/            # uploaded MP4s
  scenes/{id}/       # per-scene nerfstudio workspace
    input_video.mp4
    data/            # ns-process-data output (frames, COLMAP)
    output/          # ns-train output + splat_export/splat.ply
  captured_images/   # PNG snapshots from viewer
```

## API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/videos` | Upload video |
| GET | `/api/videos` | List videos |
| GET | `/api/videos/{id}/stream` | Stream video |
| POST | `/api/scenes` | Start 3DGS processing |
| GET | `/api/scenes/{id}/status` | Poll status + log tail |
| GET | `/api/scenes/{id}/splat` | Download .ply |
| POST | `/api/captures` | Save snapshot (base64 PNG) |
| GET | `/api/captures/scene/{id}` | List captures |
| GET | `/api/captures/{id}/image` | Get capture image |

## Sample Videos
Located at `sample-videos/room_01.mp4` (3.3MB) and `room_02.mov` (29MB).

## Database
SQLite at `backend/realestategen.db`. Tables: `videos`, `scenes`, `captures`.
Scene statuses: `pending` → `processing_frames` → `training` → `exporting` → `done` / `failed`.
