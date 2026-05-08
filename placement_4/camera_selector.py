"""Interactive 3DGS camera selector via GaussianSplats3D in the browser."""

import os, threading, time, webbrowser, shutil
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_file
from scipy.spatial.transform import Rotation as ScipyRotation


# ── HTML template — scene bounds injected at serve time ──────────────────────
# Placeholders: {cx} {cy} {cz} = scene centre (Three.js Y-up frame)
#               {cam_dist}     = initial camera distance

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DG-3DPlace — Camera Selector</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#111; color:#eee; font-family:system-ui,sans-serif;
         display:flex; flex-direction:column; height:100vh; overflow:hidden; }}
  #toolbar {{ display:flex; align-items:center; gap:14px; padding:10px 18px;
             background:#1a1a2e; border-bottom:1px solid #333; flex-shrink:0; }}
  #toolbar h1 {{ font-size:15px; font-weight:600; color:#a78bfa; }}
  #toolbar span {{ font-size:12px; color:#888; }}
  #select-btn {{ margin-left:auto; padding:9px 22px; border:none; border-radius:8px;
                background:#7c3aed; color:#fff; font-size:14px; font-weight:600;
                cursor:pointer; transition:background .2s; }}
  #select-btn:hover {{ background:#6d28d9; }}
  #select-btn.done {{ background:#16a34a; cursor:default; }}
  #canvas-wrap {{ flex:1; position:relative; }}
  canvas {{ width:100%!important; height:100%!important; display:block; }}
  #status {{ position:absolute; bottom:14px; left:50%; transform:translateX(-50%);
            background:rgba(0,0,0,.72); padding:7px 18px; border-radius:20px;
            font-size:12px; color:#ccc; pointer-events:none; white-space:nowrap; }}
  #cam-info {{ position:absolute; top:10px; left:14px; background:rgba(0,0,0,.58);
              padding:7px 13px; border-radius:8px; font-size:11px; color:#9ca3af;
              font-family:monospace; line-height:1.7; pointer-events:none; display:none; }}
  #progress {{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
              text-align:center; pointer-events:none; }}
  #progress p {{ font-size:13px; color:#a78bfa; margin-top:10px; }}
  .spinner {{ width:44px; height:44px; border:4px solid #333;
             border-top-color:#a78bfa; border-radius:50%;
             animation:spin .8s linear infinite; margin:0 auto; }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
</style>
</head>
<body>
<div id="toolbar">
  <h1>DG-3DPlace &nbsp;·&nbsp; Camera Selector</h1>
  <span>Orbit: left-drag &nbsp;|&nbsp; Pan: right-drag &nbsp;|&nbsp; Zoom: scroll</span>
  <button id="select-btn" onclick="selectCamera()">&#10003; Select this view</button>
</div>
<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
  <div id="progress">
    <div class="spinner"></div>
    <p id="progress-text">Downloading scene…</p>
  </div>
  <div id="cam-info"></div>
  <div id="status">Loading…</div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.176.0/build/three.module.js",
    "@mkkellogg/gaussian-splats-3d": "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import * as GS3D from '@mkkellogg/gaussian-splats-3d';

// Scene centre and camera distance injected by Python
const SCENE_CENTER = new THREE.Vector3({cx}, {cy}, {cz});
const CAM_DIST     = {cam_dist};

const canvas   = document.getElementById('canvas');
const status   = document.getElementById('status');
const camInfo  = document.getElementById('cam-info');
const progress = document.getElementById('progress');
const progTxt  = document.getElementById('progress-text');

const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

function resize() {{
  const w = canvas.parentElement.clientWidth;
  const h = canvas.parentElement.clientHeight;
  renderer.setSize(w, h, false);
  if (window._viewer && window._viewer.camera) {{
    window._viewer.camera.aspect = w / h;
    window._viewer.camera.updateProjectionMatrix();
  }}
}}
window.addEventListener('resize', resize);
resize();

// Initial camera position: offset from scene centre along -Z + a bit of Y
const initPos = SCENE_CENTER.clone().add(new THREE.Vector3(0, CAM_DIST * 0.3, CAM_DIST));

const viewer = new GS3D.Viewer({{
  renderer,
  selfDrivenMode:         false,
  sharedMemoryForWorkers: true,
  dynamicScene:           false,
  webXRMode:              GS3D.WebXRMode.None,
  renderMode:             GS3D.RenderMode.Always,
  sceneRevealMode:        GS3D.SceneRevealMode.Instant,
  camera: new THREE.PerspectiveCamera(
    60,
    canvas.parentElement.clientWidth / canvas.parentElement.clientHeight,
    0.01, 5000
  ),
  initialCameraPosition: [initPos.x, initPos.y, initPos.z],
  initialCameraLookAt:   [SCENE_CENTER.x, SCENE_CENTER.y, SCENE_CENTER.z],
}});
window._viewer = viewer;

// The 3DGS .ply is stored in Z-up world coords.
// Three.js is Y-up.  Rotate the splat scene -90° around X so Z-up → Y-up.
// quaternion for -90° around X: x=sin(-45°)=-√2/2, y=0, z=0, w=cos(-45°)=√2/2
const ROT_XYZW = [-0.7071068, 0, 0, 0.7071068];

progTxt.textContent = 'Downloading scene (first load may take ~20 s)…';
viewer.addSplatScene('/scene.ply', {{
  splatAlphaRemovalThreshold: 5,
  showLoadingUI:  false,
  progressiveLoad: true,
  rotation: ROT_XYZW,
}})
.then(() => {{
  progress.style.display = 'none';
  camInfo.style.display  = 'block';
  status.textContent = 'Scene loaded — orbit freely, then click  ✓ Select this view';
  animate();
}})
.catch(err => {{
  document.querySelector('.spinner').style.display = 'none';
  progTxt.textContent  = '❌ ' + (err && err.message ? err.message : String(err));
  progTxt.style.color  = '#f87171';
  console.error('GS3D error:', err);
}});

function animate() {{
  requestAnimationFrame(animate);
  viewer.update();
  viewer.render();
  const c = viewer.camera;
  const p = c.position;
  const q = c.quaternion;
  camInfo.innerHTML =
    '<b style="color:#c4b5fd">Camera</b><br>' +
    'pos : ' + p.x.toFixed(3) + ', ' + p.y.toFixed(3) + ', ' + p.z.toFixed(3) + '<br>' +
    'quat: ' + q.x.toFixed(3) + ', ' + q.y.toFixed(3) + ', ' + q.z.toFixed(3) + ', ' + q.w.toFixed(3) + '<br>' +
    'fov : ' + c.fov.toFixed(1) + '°';
}}

window.selectCamera = function() {{
  const c = viewer.camera;
  c.updateMatrixWorld(true);
  fetch('/select-camera', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{
      position:        [c.position.x, c.position.y, c.position.z],
      quaternion_xyzw: [c.quaternion.x, c.quaternion.y, c.quaternion.z, c.quaternion.w],
      c2w_col_major:   Array.from(c.matrixWorld.elements),
      w2c_col_major:   Array.from(c.matrixWorldInverse.elements),
      fov_deg:         c.fov,
      aspect:          c.aspect,
      render_width:    canvas.parentElement.clientWidth,
      render_height:   canvas.parentElement.clientHeight,
    }}),
  }})
  .then(r => r.json())
  .then(d => {{
    if (d.ok) {{
      const btn = document.getElementById('select-btn');
      btn.textContent = '✓ Camera saved — pipeline continuing…';
      btn.classList.add('done');
      status.textContent = 'Camera saved. You can close this tab.';
    }} else {{
      status.textContent = '❌ ' + d.error;
    }}
  }})
  .catch(err => {{ status.textContent = '❌ Network error: ' + err; }});
}};
</script>
</body>
</html>"""


def _compute_scene_bounds(ply_path: str):
    """Return (center_zup, extent) in the Z-up world frame from PLY xyz."""
    import plyfile
    v = plyfile.PlyData.read(ply_path)["vertex"]
    xs = v["x"].astype(np.float64)
    ys = v["y"].astype(np.float64)
    zs = v["z"].astype(np.float64)
    center = np.array([xs.mean(), ys.mean(), zs.mean()])
    extent = max(np.ptp(xs), np.ptp(ys), np.ptp(zs))
    return center, float(extent)


def _zup_center_to_threejs(center_zup: np.ndarray):
    """Convert scene centre from Z-up → Three.js Y-up.
    The scene rotation applied in the viewer is -90° around X,
    which maps: [X, Y, Z]_zup → [X, Z, -Y]_threejs.
    """
    x, y, z = center_zup
    return float(x), float(z), float(-y)


def _downsample_ply_for_viewer(src_ply: str, dst_ply: str,
                                max_splats: int = 300_000) -> str:
    import plyfile
    pd = plyfile.PlyData.read(src_ply)
    v  = pd["vertex"]
    N  = len(v)
    if N <= max_splats:
        shutil.copy(src_ply, dst_ply)
        print(f"Viewer PLY: {N:,} splats (no downsample needed)")
        return dst_ply

    op  = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    idx = np.argpartition(op, -max_splats)[-max_splats:]
    idx = idx[np.argsort(-op[idx])]
    sub = v[idx]
    el  = plyfile.PlyElement.describe(sub, "vertex")
    plyfile.PlyData([el], text=False).write(dst_ply)
    mb = os.path.getsize(dst_ply) / 1024 / 1024
    print(f"Viewer PLY: {N:,} → {max_splats:,} splats  ({mb:.1f} MB)")
    return dst_ply


def _build_camera_state(data: dict, render_w: int, render_h: int,
                         fov_deg: float) -> dict:
    """Convert Three.js camera payload → pipeline camera_state dict.

    The viewer applies a -90° X rotation to the splat scene so Three.js Y-up
    matches Z-up world coords.  We undo that rotation on the camera matrix so
    the saved position/wxyz are in the original Z-up world frame.

    SceneCamera expects wxyz in OpenGL/Z-up convention and applies the GL→CV
    flip internally, so we must NOT pre-apply it here.
    """
    # Column-major (WebGL/Three.js) → row-major numpy
    c2w_gl = np.array(data["c2w_col_major"], dtype=np.float64).reshape(4, 4).T

    # Undo the -90° X scene rotation by applying +90° X to the camera matrix.
    # This converts camera from Three.js Y-up frame → Z-up world (OpenGL convention).
    angle    = np.pi / 2.0
    Rx_inv   = np.array([[1, 0,            0,           0],
                          [0, np.cos(angle), -np.sin(angle), 0],
                          [0, np.sin(angle),  np.cos(angle), 0],
                          [0, 0,            0,           1]], dtype=np.float64)
    c2w_gl_zup = Rx_inv @ c2w_gl   # OpenGL c2w in Z-up world

    # Extract position and wxyz directly from the OpenGL/Z-up c2w.
    # SceneCamera interprets wxyz as an OpenGL quaternion and applies
    # its own GL→CV flip — so we must not apply the flip here.
    position = c2w_gl_zup[:3, 3].copy()
    q_xyzw   = ScipyRotation.from_matrix(c2w_gl_zup[:3, :3]).as_quat()
    wxyz     = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    fov_rad = np.deg2rad(float(data.get("fov_deg", fov_deg)))
    w       = int(data.get("render_width",  render_w))
    h       = int(data.get("render_height", render_h))
    fx      = (w / 2.0) / np.tan(fov_rad / 2.0)
    K       = np.array([[fx, 0, w/2.0],
                        [0, fx, h/2.0],
                        [0,  0,    1.0]], dtype=np.float64)

    # Build w2c_cv the same way SceneCamera does (for completeness in the dict)
    w2c_gl_zup = np.linalg.inv(c2w_gl_zup)
    w2c_cv = w2c_gl_zup.copy()
    w2c_cv[1, :] *= -1
    w2c_cv[2, :] *= -1

    return {
        "position": position, "wxyz": wxyz,
        "fov_rad": fov_rad, "render_width": w, "render_height": h,
        "intrinsics": K, "extrinsics_w2c": w2c_cv, "c2w": c2w_gl_zup,
        "cam_idx": 0, "azimuth_rad": 0.0, "azimuth_deg": 0.0,
        "scene_center": position, "orbit_radius": 1.0,
        "camera_height_offset": 0.0, "num_cameras": 1,
    }


def run_interactive_camera_selector(
    ply_path: str,
    session_dir: str,
    camera_state_path: str,
    render_w: int  = 1280,
    render_h: int  = 720,
    fov_deg: float = 60.0,
    port: int      = 7860,
    max_viewer_splats: int = 50_000,
) -> dict:

    # 1. Downsample for browser
    viewer_ply = os.path.join(session_dir, "scene_viewer_downsampled.ply")
    _downsample_ply_for_viewer(ply_path, viewer_ply, max_splats=max_viewer_splats)

    # 2. Compute scene bounds → inject into HTML so camera starts near scene
    center_zup, extent = _compute_scene_bounds(viewer_ply)
    cx, cy, cz = _zup_center_to_threejs(center_zup)
    cam_dist   = round(float(extent) * 1.2, 4)
    print(f"Scene centre (Z-up): {np.round(center_zup, 3)}  extent: {extent:.3f}")
    print(f"Three.js centre: ({cx:.3f}, {cy:.3f}, {cz:.3f})  cam_dist: {cam_dist:.3f}")

    html = _HTML_TEMPLATE.format(cx=cx, cy=cy, cz=cz, cam_dist=cam_dist)

    result: list = []
    app = Flask(__name__)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.after_request
    def _sec(response):
        response.headers["Cross-Origin-Opener-Policy"]   = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response

    @app.get("/")
    def index():
        return Response(html, mimetype="text/html")

    @app.get("/scene.ply")
    def serve_ply():
        return send_file(viewer_ply, mimetype="application/octet-stream",
                         conditional=True)

    @app.post("/select-camera")
    def select_camera():
        try:
            state = _build_camera_state(
                request.get_json(force=True), render_w, render_h, fov_deg)
            result.append(state)
            return jsonify(ok=True)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port,
                               debug=False, use_reloader=False),
        daemon=True,
    ).start()

    url = f"http://localhost:{port}"
    print(f"\n{'='*58}")
    print(f"  3DGS Camera Selector  →  {url}")
    print(f"  Orbit freely, pick your view, click  ✓ Select this view")
    print(f"{'='*58}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    print("Waiting for camera selection…", flush=True)
    while not result:
        time.sleep(0.25)

    state = result[0]
    torch.save(state, camera_state_path)
    pos = state["position"]
    print(f"Camera: pos={np.round(pos, 4)}  fov={np.degrees(state['fov_rad']):.1f}°")
    print(f"Saved → {camera_state_path}")
    return state
