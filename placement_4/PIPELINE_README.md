# DG-3DPlace Pipeline — Deep Technical Reference

## Table of Contents

1. [What Is a 3DGS Checkpoint?](#1-what-is-a-3dgs-checkpoint)
2. [Scene Coordinate System and Conventions](#2-scene-coordinate-system-and-conventions)
3. [SceneCamera — Unified Camera Model](#3-scenecamera--unified-camera-model)
4. [Rendering with gsplat](#4-rendering-with-gsplat)
5. [Interactive Camera Selection](#5-interactive-camera-selection)
6. [Depth Map Rendering and Unprojection](#6-depth-map-rendering-and-unprojection)
7. [Object Detection in 2D (OWLv2)](#7-object-detection-in-2d-owlv2)
8. [Gemini Cutout and Silhouette Mask](#8-gemini-cutout-and-silhouette-mask)
9. [Depth-Aware Gaussian Selection](#9-depth-aware-gaussian-selection)
   - 9.1 [Pass 0 — 2D Bbox / Mask Filter](#91-pass-0--2d-bbox--mask-filter)
   - 9.2 [Pass 1 — Occlusion Consistency](#92-pass-1--occlusion-consistency)
   - 9.3 [Pass 2 — RANSAC Dominant-Plane Vote](#93-pass-2--ransac-dominant-plane-vote)
   - 9.4 [Pass 2b — Spatial-Locality Connected-Component Refinement](#94-pass-2b--spatial-locality-connected-component-refinement)
   - 9.5 [MAD Fallback](#95-mad-fallback)
10. [Support-Surface Estimation](#10-support-surface-estimation)
    - 10.1 [RANSAC Plane Fit](#101-ransac-plane-fit)
    - 10.2 [Tangent-Basis Footprint Size](#102-tangent-basis-footprint-size)
    - 10.3 [Tilt Clamping](#103-tilt-clamping)
    - 10.4 [Depth-Map Fallback](#104-depth-map-fallback)
11. [Surface-Aligned Rotation (Rodrigues Formula)](#11-surface-aligned-rotation-rodrigues-formula)
12. [Robust Scale Estimation](#12-robust-scale-estimation)
13. [GLB to Gaussians Conversion](#13-glb-to-gaussians-conversion)
    - 13.1 [Why GLB and Gaussian Splatting Together?](#131-why-glb-and-gaussian-splatting-together)
    - 13.2 [Coordinate Conversion — GLB Y-up to Scene Z-up](#132-coordinate-conversion--glb-y-up-to-scene-z-up)
    - 13.3 [Area-Weighted Surface Sampling](#133-area-weighted-surface-sampling)
    - 13.4 [Texture Color Sampling](#134-texture-color-sampling)
    - 13.5 [Quaternion Transformation (Basis Change vs Object Rotation)](#135-quaternion-transformation-basis-change-vs-object-rotation)
    - 13.6 [Adaptive Gaussian Scale Radius](#136-adaptive-gaussian-scale-radius)
    - 13.7 [Slope-Aware Support Lift](#137-slope-aware-support-lift)
    - 13.8 [mesh2splat PLY Path](#138-mesh2splat-ply-path)
    - 13.9 [Train Mode — Blender + COLMAP + 3DGS Trainer](#139-train-mode--blender--colmap--3dgs-trainer)
14. [Checkpoint Merging — Writing the Combined Scene](#14-checkpoint-merging--writing-the-combined-scene)
15. [End-to-End Data Flow Diagram](#15-end-to-end-data-flow-diagram)
16. [Library Choices and Justifications](#16-library-choices-and-justifications)
17. [Novelty Summary](#17-novelty-summary)
18. [References](#18-references)

---

## 1. What Is a 3DGS Checkpoint?

A **3D Gaussian Splatting (3DGS) checkpoint** (`.ckpt` file) is a PyTorch `state_dict` saved with `torch.save`. It stores the parameters of a scene that has been reconstructed from multi-view photos using the 3DGS training procedure [Kerbl et al. 2023]. The key fields under the `"pipeline"` key are:

| Field | Shape | Meaning |
|---|---|---|
| `_model.means` | `(N, 3)` float32 | World-space centre of each Gaussian, XYZ |
| `_model.scales` | `(N, 3)` float32 | Log-scale of each Gaussian along its 3 principal axes |
| `_model.quats` | `(N, 4)` float32 | Orientation of each Gaussian as a unit quaternion (w, x, y, z) |
| `_model.features_dc` | `(N, 1, 3)` or `(N, 3)` float32 | Zeroth-order (DC) Spherical Harmonic colour coefficient |
| `_model.features_rest` | `(N, 15, 3)` float32 | Higher-order SH coefficients (degrees 1-3), typically small |
| `_model.opacities` | `(N, 1)` float32 | Per-Gaussian opacity stored as a **logit** (inverse-sigmoid) |

**Why logit opacity?** The sigmoid function maps any real number to `(0,1)`. Storing `logit(opacity)` lets the optimiser work in an unbounded real space while the renderer applies `sigmoid` at runtime to get a valid probability. A logit of `0` = 50% opacity, `5` ≈ 99.3% opacity (effectively opaque), `-5` ≈ 0.7% (near-transparent).

**Why log-scale?** The renderer computes `exp(log_scale)` to get true standard deviations. Log-space lets the optimiser move from tiny to large in equal gradient steps — the same reason neural networks learn `log_std` in variational autoencoders.

**Spherical Harmonics (SH).** Each Gaussian has view-dependent colour encoded in SH coefficients. The DC term gives the base colour; higher-degree terms capture specular-like highlights. During placement we use only the DC term (constant colour), which renders correctly from any angle.

**How to read a checkpoint:**

```python
ckpt  = torch.load("bench_park.ckpt", map_location="cpu", weights_only=False)
state = ckpt["pipeline"]
means       = state["_model.means"].numpy()      # (N, 3)
opacities_raw = state["_model.opacities"].numpy()  # (N, 1) logit
opacities   = 1 / (1 + np.exp(-opacities_raw))  # sigmoid → real opacity
```

The scene might contain `N = 200 000 – 3 000 000` Gaussians depending on scene complexity and training duration.

---

## 2. Scene Coordinate System and Conventions

Understanding the coordinate system is critical throughout the pipeline because the same 3D point is expressed differently in different spaces.

### World space (scene Z-up)

The trained 3DGS scene uses a **right-handed Z-up coordinate system** (common in architectural and civil-engineering CAD). `+X` = right, `+Y` = forward/into the scene, `+Z` = up. All Gaussian means (`_model.means`) live here.

### Camera conventions

Two camera conventions coexist:

| Convention | +Z direction | +Y direction | Used by |
|---|---|---|---|
| **OpenGL** | Away from camera (behind scene) | Up | `c2w` matrix, viser viewer, camera quaternion storage |
| **OpenCV** | Into scene (forward) | Down | `w2c` matrix, gsplat rasteriser, depth map, projection |

The pipeline stores cameras in **OpenGL** quaternion form (as they come from viser) but converts `w2c` to **OpenCV** for gsplat and projection. The conversion flips rows 1 and 2 of the view matrix:

```python
w2c_opencv = w2c_opengl.copy()
w2c_opencv[1, :] *= -1   # flip Y row
w2c_opencv[2, :] *= -1   # flip Z row
```

**Why this matters:** Every depth value, pixel coordinate, and back-projected 3D point produced in this pipeline is in OpenCV camera space unless explicitly converted back through `np.linalg.inv(camera.w2c)`.

### GLB/OBJ object space (Y-up)

3D objects from GLB/OBJ exporters use the glTF convention: Y-up. When inserting an object into the Z-up scene, every position must be remapped:

```
[X, Y, Z]_glb → [X, -Z, Y]_scene
```

This is the `_COORD_CONV` matrix in `glb_to_gaussians.py`:

```python
_COORD_CONV = np.array([[1, 0, 0],
                         [0, 0,-1],
                         [0, 1, 0]], dtype=np.float64)
```

Rotating the positions also changes the orientation of every Gaussian. Quaternions representing orientations must be transformed via the **basis-change sandwich**: `q_new = q_R * q_old * q_R^{-1}` (see §13.5).

---

## 3. SceneCamera — Unified Camera Model

**File:** `updated_pipeline.py`, class `SceneCamera` (line 146)

Every camera in the pipeline is wrapped in a single `SceneCamera` object. This eliminates the bug class where one subsystem uses OpenGL matrices and another uses OpenCV matrices.

### Construction

```python
cam = SceneCamera(position, wxyz, fov_rad, width, height)
```

| Parameter | Meaning |
|---|---|
| `position` | Camera world-space position (3,) |
| `wxyz` | Camera orientation quaternion in `(w, x, y, z)` order (OpenGL convention from viser) |
| `fov_rad` | Vertical field of view in radians |
| `width`, `height` | Render resolution in pixels |

### Intrinsic matrix derivation

From vertical FOV, using the standard pinhole model:

```
fy = (height / 2) / tan(fov_rad / 2)
fx = fy          (square pixels — equal horizontal and vertical focal length)
cx = width  / 2
cy = height / 2
```

The full intrinsic matrix `K` is:

```
K = [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  1 ]]
```

This is the standard perspective camera model [Hartley & Zisserman 2003, Chapter 6]. `fx` and `fy` are focal lengths in pixel units; `cx, cy` is the principal point (image centre for a centred lens).

### Projection: world → pixel

```python
u, v, z, valid = cam.project(points_3d)
```

Transforms world points through the OpenCV `w2c` matrix and applies the pinhole equations:

```
[Xc, Yc, Zc, 1]^T  =  w2c @ [X_world, Y_world, Z_world, 1]^T

u = fx * Xc / Zc + cx
v = fy * Yc / Zc + cy
```

Points with `Zc <= 0.1` (behind or too close to camera) are marked invalid. This projection is used throughout the pipeline to find which pixel every Gaussian falls on — essential for depth comparison and 2D mask tests.

---

## 4. Rendering with gsplat

**File:** `updated_pipeline.py`, `render_gaussians()` (line 214)

**Library:** `gsplat` — GPU-accelerated differentiable 3D Gaussian rasteriser [Ye et al. 2024].

### Why gsplat?

gsplat implements the alpha-composited tile-based rasterisation algorithm from the original 3DGS paper [Kerbl et al. 2023] with CUDA kernels, making it orders of magnitude faster than CPU alternatives. It is the standard renderer used in nearly all 3DGS training and inference pipelines. Critically for this work, it supports `render_mode="RGB+ED"` which simultaneously returns an RGB image and an **expected depth** (ED) map in a single forward pass — no second render needed.

### What goes into the rasteriser

Before passing data to gsplat, several conversions happen:

```python
# SH DC → RGB colour (inverse of training encoding):
colors_rgb = clip(C0 * features_dc + 0.5, 0, 1)

# Logit → sigmoid opacity:
opacities = 1 / (1 + exp(-opacities_raw))

# Log-scale → actual scale:
scales = exp(log_scales)

# Normalise quaternions (floating-point drift can de-normalise them):
quats = quats / ||quats||
```

The `C0 = 0.28209479177387814` constant is the zeroth-degree SH basis function `Y_0^0 = 1 / (2*sqrt(π))`. The SH encoding convention is `f_dc = (RGB - 0.5) / C0`, so decoding is `RGB = C0 * f_dc + 0.5`.

### Depth mode (`render_mode="RGB+ED"`)

When `return_depth=True`, gsplat appends a depth channel to the render. The output is `(H, W, 4)` — the first 3 channels are RGB, the 4th is camera-space Z (positive forward, in scene-unit metres). This is the **alpha-composited expected depth**: for each pixel, the depth value is a weighted sum over all Gaussians contributing to that pixel, weighted by their accumulated alpha. Front Gaussians dominate, making the depth map approximate the **visible surface depth** at each pixel.

This depth map is the cornerstone of the occlusion-consistency filter (§9.2).

---

## 5. Interactive Camera Selection

**File:** `updated_pipeline.py`, `select_camera_and_render()` (line 931)

The user chooses the observation angle from which the object will be placed. The workflow:

1. **Export checkpoint to PLY** — `ckpt_to_ply()` reads the checkpoint and writes a standard Gaussian PLY file (standard format from the original 3DGS paper). The PLY viewer can display this without any custom runtime.

2. **Launch interactive browser viewer** — `run_interactive_camera_selector()` starts a `viser`-based 3D viewer in the browser (port 7860). The user orbits/zooms/tilts to the desired viewpoint and clicks a button.

3. **Save camera state** — viser returns position `(x,y,z)`, orientation quaternion `(w,x,y,z)`, FOV, and render dimensions as a `dict` written to `selected_camera_state.pt` via `torch.save`.

4. **Render selected view** — the pipeline immediately renders the scene from that camera using `render_gaussians` and saves `selected_camera_view.png` as a reference image. This image becomes the base layer for Gemini's diffusion edit.

Why save the camera state separately? The camera must be reproduced exactly later when generating the depth map and projecting Gaussians. A saved `.pt` file ensures the same numbers are used across sessions, Python processes, and conda environments.

---

## 6. Depth Map Rendering and Unprojection

**File:** `updated_pipeline.py`, `render_depth_map()` (line 278), `unproject_depth_to_world()` (line 287)

### Rendering

```python
depth_map = render_depth_map(means, scales, quats, features_dc, opacities_raw, cam)
```

This calls `render_gaussians` with `return_depth=True` and returns only the `(H, W)` float32 depth array in camera-space Z.

The depth map is saved as:
- `placement_depth_map.png` — jet-coloured for human inspection
- `placement_depth_map.png.raw.npy` — raw float32 for numerical use

### Unprojection (pixel → 3D world point)

Given a pixel `(u, v)` and a depth value `d` (camera-space Z), the 3D position in **camera space** is:

```
Xc = (u - cx) * d / fx
Yc = (v - cy) * d / fy
Zc = d
```

Converting to **world space** requires multiplying by the inverse of the world-to-camera matrix:

```python
c2w_cv = np.linalg.inv(camera.w2c)            # 4×4
P_world = c2w_cv @ [Xc, Yc, Zc, 1.0]          # homogeneous
return P_world[:3]
```

This is the standard **pinhole inverse projection** [Hartley & Zisserman 2003, Chapter 6.1]. The implementation samples a `window × window` patch around the requested pixel and uses the **median** of valid (>0.01) depth values — this suppresses depth noise from the Gaussian alpha-compositing process without requiring any extra filtering model.

---

## 7. Object Detection in 2D (OWLv2)

**File:** `updated_pipeline.py`, `detect_prompt_box_with_owlv2()` (line 1071)

After the user selects a camera and Gemini edits the scene image to add the new object, the pipeline must find the object's 2D bounding box in the edited image. It uses **OWLv2** (Open-Vocabulary Object Detection via Vision Transformer) [Minderer et al. 2023].

### Why OWLv2?

Traditional COCO-trained detectors (YOLO, Faster-RCNN) are limited to their ~80 fixed classes. The user can request "a red leather armchair" or "a traditional wooden coffee table" — arbitrary open-vocabulary descriptions. OWLv2 accepts a free-text query and matches it against image patches using a CLIP-style vision-language model, returning boxes and confidence scores.

### Query generation strategy

A single query string often gives poor recall on uncommon phrasings. The pipeline generates multiple query variants from the user's prompt:

```python
query_variants = [
    prompt_raw,
    f"a photo of {prompt_raw}",    # CLIP-style prefix
    object_class_noun,              # extracted known keyword (e.g. "car")
    f"a photo of a {noun}",
    individual_tokens[:3],          # first 3 non-stopword tokens
]
```

All variants are passed as a batch to `Owlv2ForObjectDetection`. The highest-confidence box across all variants is selected. The score threshold is 0.06 (low, because OWLv2 is already conservative and real scene images may not look like the CLIP training distribution).

### Resolution alignment

Gemini may return the edited image at a slightly different resolution than the camera's `render_width × render_height`. All downstream pixel coordinates (OWLv2 bbox, mask, Gaussian projection) must live in the same pixel space. Before detection runs, the pipeline checks and resizes the input image if needed:

```python
if (actual_w, actual_h) != (expected_w, expected_h):
    image.resize((expected_w, expected_h), Image.LANCZOS).save(resized_path)
```

Without this, a box detected at pixel `(640, 360)` in a 1280×720 image would be interpreted as `(640, 360)` in a 960×540 camera — everything shifts.

---

## 8. Gemini Cutout and Silhouette Mask

**File:** `updated_pipeline.py`, `_extract_cutout_silhouette()` (line 1241), `_match_cutout_inside_bbox()` (line 1281)

OWLv2 provides an axis-aligned bounding box, but the box always contains background. A tight **per-pixel silhouette mask** greatly reduces the number of background Gaussians passed to the depth-aware filter (§9) and improves placement accuracy.

### Cutout extraction

Gemini generates a separate "cutout" image of the object on a white or transparent background. The silhouette is extracted by:

1. Reading the RGBA image. If the alpha channel has any values below 250, use `alpha > 10` as the mask. This handles PNG cutouts with real transparency.
2. Otherwise, detect near-white background pixels (`max_channel >= 245` and low saturation) and invert.
3. Apply binary morphological **opening** (removes isolated pixel noise), **closing** (fills small holes), and `fill_holes` (fills enclosed regions) using a 3×3 structuring element.
4. Keep only the **largest connected component** (`_select_best_component`) — suppresses background artefacts that match the near-white test or alpha-channel compression noise.
5. Crop to the tight bounding box of the remaining mask.

### Template matching into the detection bbox

The cutout silhouette is then matched to the actual position of the object within the OWLv2 bbox using **normalised cross-correlation (NCC)**:

```
NCC = E[(patch_grey - μ_patch)(template_grey - μ_template)] / (σ_patch * σ_template)
```

NCC is invariant to brightness and contrast differences, making it robust to Gemini changing the object's lighting when placing it in the scene. The match is searched over:
- 5 scale candidates (±20% of the tight-fit scale)
- A spatial grid of offsets within the bbox (±15% in each direction, step 2 pixels)

The best `(offset_x, offset_y, scale)` combination is selected by highest NCC minus a small distance-from-bbox-centre penalty. The resulting full-resolution binary mask is saved as `added_object_mask.png` and reused in Gaussian selection, surface estimation, and scale computation.

---

## 9. Depth-Aware Gaussian Selection

**File:** `updated_pipeline.py`, `select_object_gaussians_depth_aware()` (line 399)

This is the most critical step for accurate initial placement. It answers: **"Which Gaussians in the scene actually belong to the surface the detected object is resting on?"**

### Why naive 2D-bbox selection fails

Every Gaussian in the scene is projected onto the camera plane. A naive test keeps all Gaussians whose projections land inside the OWLv2 bbox. But a single camera ray can pierce multiple real 3D surfaces: a bench seat, a far wall behind it, and the floor in front of it. All three project to pixels inside the box. A 2D-only test cannot distinguish them. The object's apparent 3D bounding box inflates to span bench+wall+floor, making its derived scale and position wrong.

### 9.1 Pass 0 — 2D Bbox / Mask Filter

```python
in_bbox = valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2) & (opacity > threshold)
if mask is not None:
    in_bbox = in_bbox & mask[vi, ui]
```

All Gaussians whose projections fall inside the bbox (or tighter silhouette mask, if available) and above the opacity threshold are **candidates**. This is a necessary but not sufficient condition.

The opacity threshold (`OPACITY_THRESHOLD = 0.1`) rejects near-transparent Gaussians that 3DGS creates at empty space locations (they are real artefacts of training but do not correspond to visible surfaces).

### 9.2 Pass 1 — Occlusion Consistency

For each candidate Gaussian, look up the depth the scene rendered at that Gaussian's projected pixel:

```python
surf_depth = depth_map[vi, ui]
occlusion_ok = abs(z_gaussian - surf_depth) < depth_tol_abs   # default 0.12 scene units
```

The rendered depth map is the **alpha-composited visible-surface depth** (§4). A Gaussian that is further from the camera than what was rendered at its pixel is **occluded** — something is in front of it. That something is not the object being detected. Occluded Gaussians are dropped.

**Why this rejects the wall but not the bench.** The bench is the visible surface in the bbox; it renders at shallow depth. The wall is behind it; it renders at its own pixel (beyond the bench) but the bench occludes it at the bench's pixel locations. The wall's Gaussians have `z_gaussian >> surf_depth` at bench pixels → rejected.

**What this cannot do.** Pass 1 works in 1D camera-depth space. It cannot separate two genuinely visible surfaces at similar depth but different 3D position — e.g. a teddy bear on a bench at `z≈2.1m`, whose feet overlap the floor at `z≈2.05m` and whose body overlaps the bench at `z≈2.1m`. Both are visible and occlusion-consistent. A purely depth-scalar test cannot tell them apart.

### 9.3 Pass 2 — RANSAC Dominant-Plane Vote

**The key novelty:** after Pass 1, the surviving candidates are fit with a RANSAC plane in full 3D position space (not 1D depth), and only the inliers of the **dominant** (most-inliers) plane are kept.

#### Why a dominant-plane vote works

The object's support surface — the bench seat — is a large, coherent, flat patch of Gaussians. The floor sliver at the teddy's feet and the wall behind the head are small patches, each on their own (different) planes. A RANSAC fit over all Pass-1 survivors finds the plane that fits the **most** points. Because the bench has far more Gaussians than the floor sliver or wall sliver, the bench's plane wins. Points on the floor or wall are not inliers of the bench plane and are dropped.

#### RANSAC algorithm (`fit_plane_ransac`, line 594)

The Randomized Consensus (RANSAC) algorithm [Fischler & Bolles 1981] is the standard technique for fitting a geometric model to noisy data with outliers.

```
repeat max_iters times:
    sample 3 random points → define a candidate plane
    normal = cross(p1-p0, p2-p0)  (cross product → plane normal)
    count inliers: points where |dot(point-p0, normal)| < dist_thresh
    keep track of best (most inliers) plane

refine: SVD total-least-squares on the best inlier set
    A = inlier_pts - mean(inlier_pts)
    U, S, Vt = SVD(A)
    normal = Vt[-1]   ← last row of Vt = direction of minimum variance
```

The SVD refinement is **total least squares** — it minimises orthogonal distances to the plane (algebraic distance from every point to the hyperplane), which is the correct objective for fitting a plane in 3D point clouds [Ahn et al. 2002]. Plain least-squares (fitting `z = ax + by + c`) biases the normal toward vertical planes.

#### Normal orientation convention

After SVD, the normal direction is ambiguous (both `n` and `-n` define the same plane). The convention is to ensure `n[2] > 0`, meaning the normal always points upward (toward the sky). This is physically meaningful: support surfaces have their normals pointing toward the object resting on them, i.e. upward in a gravity-aligned scene.

### 9.4 Pass 2b — Spatial-Locality Connected-Component Refinement

A finite-threshold RANSAC plane inlier test has one remaining failure mode: a **coincidental coplanar outlier**. Consider a wall at `y = 5m` with Gaussians at heights `z ≈ 1.2m`. If the bench plane has a slight tilt in Y (say, the normal has a small Y component), the plane equation `a*x + b*y + c*z = d` can be satisfied by wall points even though they are geometrically far away — the large `b*5.0` term is cancelled by the `c*1.2` term for certain normal orientations.

The solution is `_largest_spatial_cluster()` (line 351):

```
1. Build a k-d tree over the RANSAC inlier points.
2. Compute the median nearest-neighbour distance within those points.
3. Link all pairs of inliers within radius = 4 × median_nn_distance.
4. Find connected components of the resulting graph.
5. Keep only the largest connected component.
```

Wall Gaussians at `y=5m` are spatially isolated from bench Gaussians at `y=0–0.6m`. Even if both satisfy the bench plane equation, they cannot be in the same connected component at radius `~4 × 0.02m = 0.08m`. The wall cluster is a small satellite and is discarded; the bench cluster is the large majority and is kept.

**Why 4× median nearest-neighbour?** The k-d tree radius is set adaptively to 4× the median point-to-nearest-neighbour distance within the inlier set. This means the graph links Gaussians that are naturally adjacent in the reconstruction (within one "Gaussian spacing") but cannot accidentally jump across sparse regions or disconnected geometry. The `max(4*median, 0.05)` floor ensures connectivity even when the inlier set is very small (e.g. 8-10 Gaussians).

**Library:** `scipy.spatial.cKDTree` (C extension, O(N log N) build, O(M log N) query), `scipy.sparse.csgraph.connected_components` (Union-Find, O(N+E) where E = edges).

### 9.5 MAD Fallback

If Pass 2 cannot fit any RANSAC plane (fewer than `min_plane_inliers` points survive Pass 1, or the scene is geometrically too sparse), the pipeline falls back to a **Median Absolute Deviation** filter on camera-space depth scalars:

```python
med = median(z_stage1)
mad = median(|z_stage1 - med|) + 1e-6
keep = |z - med| < max(depth_tol_abs, mad_k * mad)
```

MAD is a robust scale estimator [Hampel 1974] — unlike standard deviation, a single outlier cannot inflate it significantly. The fallback is labelled `"mad_depth_fallback"` in the info dict so it is traceable in logs.

---

## 10. Support-Surface Estimation

**File:** `updated_pipeline.py`, `estimate_support_surface_robust()` (line 682)

After Gaussian selection, `object_indices` contains only the Gaussians on the dominant support surface. These are already the right 3D points — correctly positioned in world space, at the correct depth, on the correct surface. The support-surface estimator fits a plane directly to them to extract:
- **Position** (where to place the object's base)
- **Normal** (how the surface is tilted — for rotation)
- **Footprint size** (how big the object should be — for scale)

### 10.1 RANSAC Plane Fit

`fit_plane_ransac(target_means, ...)` is run again, this time on the already-clean filtered Gaussians. Why RANSAC again when the points are already clean? Because `object_indices` can still contain a few stray Gaussians from furniture edges, specular highlights, or silhouette noise — RANSAC's outlier tolerance makes the plane estimate robust to these residuals. A plain SVD on all points would be slightly perturbed by even 5–10% outliers.

The minimum inlier count is `max(8, N/4)` — at least a quarter of the filtered points must agree on a plane for it to be trusted. This guards against degenerate cases where most filtered Gaussians are noise.

### 10.2 Tangent-Basis Footprint Size

After fitting the plane, the footprint (the "shadow" of the Gaussians projected onto the plane) is measured by projecting onto two orthonormal tangent axes:

```python
u_axis, v_axis = _plane_tangent_basis(normal)

# Project all inlier points onto the plane's tangent basis:
centered = footprint_pts - mean(footprint_pts)
pu = centered @ u_axis   # coordinate along first tangent axis
pv = centered @ v_axis   # coordinate along second tangent axis

# Percentile-trimmed extent along each axis:
eu = percentile(pu, 95) - percentile(pu, 5)
ev = percentile(pv, 95) - percentile(pv, 5)
footprint_size = max(eu, ev)
```

**Why percentile-trimmed (5th–95th) extent?** Min/max extent is dominated by the single most extreme outlier point. The inter-percentile range is a robust measure of spread [Rousseeuw & Leroy 1987]. Using 5%–95% effectively ignores the 5% most extreme Gaussians on each side.

**Why `max(eu, ev)` not `min(eu, ev)`?** The object needs to *cover* the filtered Gaussians. The footprint's longer dimension is the binding constraint — a narrow but tall bench needs a tall scale, not a narrow one. Min was used previously and consistently underestized elongated objects.

**`_plane_tangent_basis`:** Builds an orthonormal pair `(u, v)` spanning the plane perpendicular to `normal` using a single cross product:

```python
helper = [1, 0, 0] if |normal[0]| < 0.9 else [0, 1, 0]   # avoid parallel
u = cross(normal, helper);  u /= ||u||
v = cross(normal, u)
```

This is the Gram-Schmidt-style construction for a plane orthonormal frame [Golub & Van Loan 2013, §5.1].

### 10.3 Tilt Clamping

`_clamp_tilt(normal, max_tilt_deg=35.0)` guards against degenerate plane fits producing physically impossible tilts (e.g. a vertical wall being nominated as a support surface, with a nearly horizontal normal). If the angle between `normal` and world-up exceeds `max_tilt_deg`, the normal is blended toward world-up at exactly `max_tilt_deg` using Rodrigues' rotation formula:

```python
clamped = up * cos(max_tilt_deg) + cross(axis, up) * sin(max_tilt_deg) + axis * dot(axis, up) * (1 - cos(max_tilt_deg))
```

35° is a generous ramp angle — most real floors are within 5°, gentle ramps up to ~15°, aggressive ramps up to ~30°.

### 10.4 Depth-Map Fallback

When fewer than 8 Gaussians are available for the primary path (very sparse detection), the pipeline falls back to unprojecting the **depth map itself** through the silhouette mask:

```python
for each pixel (u, v) in the silhouette mask:
    d = depth_map[v, u]
    Xc = (u - cx) * d / fx;  Yc = (v - cy) * d / fy;  Zc = d
    world_pt = inv(w2c) @ [Xc, Yc, Zc, 1]
```

This produces a dense point cloud of world-space surface points visible through the mask, and `fit_plane_ransac` is run on those instead.

---

## 11. Surface-Aligned Rotation (Rodrigues Formula)

**File:** `updated_pipeline.py`, `compute_surface_alignment_rotation()` (line 790)

Most object insertion pipelines simply place objects with identity rotation (always perfectly vertical). On a ramp, tilted table, or angled surface, an upright object would visually float or sink into the surface. The rotation must align the object's local up-axis to the surface normal.

The **shortest-arc rotation** is the minimal rotation from `world_up = [0,0,1]` to `surface_normal = n`. No extra yaw is introduced — only the tilt strictly required to match the slope. This is computed via the **Rodrigues rotation formula** [Rodrigues 1840]:

```python
v = cross(up, n)        # rotation axis
s = ||v||               # sin of rotation angle
c = dot(up, n)          # cos of rotation angle

# Skew-symmetric cross-product matrix for v:
vx = [[0, -v2, v1],
      [v2,  0, -v0],
      [-v1, v0,  0]]

R = I + vx + vx @ vx * (1 - c) / s^2
```

This is the equivalent of the **cross-product formulation of Rodrigues** [Slabaugh 1999]. It is numerically stable for any angle except exactly 180°, which is handled separately (`normal` pointing straight down → 180° flip about X-axis). The resulting 3×3 rotation matrix is passed to `glb_to_gaussians` and applied to every Gaussian's position and orientation quaternion.

**Why this is novel in the context of object placement in 3DGS scenes:** Prior work on 3DGS object insertion (e.g., GaussianEditor [Chen et al. 2023], GaussianObject [Yang et al. 2024]) does not address surface-normal-aligned rotation for initial placement. Objects are typically placed with manual rotation or identity rotation. This pipeline derives the surface orientation automatically from the scene's own Gaussian data, making it the first fully automated surface-aware placement system for 3DGS scenes (to the authors' knowledge).

---

## 12. Robust Scale Estimation

**File:** `updated_pipeline.py`, `estimate_object_scale_robust()` (line 817)

The scale passed to `glb_to_gaussians` tells it to size the object so its largest dimension equals `target_scale * scale_factor` metres (with `scale_factor=0.4` providing a slight shrink to avoid the object being too large on first placement).

### Scale precedence

1. **`footprint_size`** (primary): the percentile-trimmed in-plane extent of the filtered Gaussians (§10.2). This is purely derived from the 3D evidence — "how large is the surface area the object sits on?" It is the most reliable estimate because the filtered Gaussians have already been validated by depth and RANSAC.

2. **`target_extent` bbox diagonal** (fallback 1): min of the X and Y extents of the raw filtered Gaussian bounding box. Less reliable than footprint_size because the raw bbox is not outlier-trimmed, but better than no 3D information.

3. **2D back-projection** (fallback 2 / cross-check only): using the pinhole camera model, a 2D width in pixels at a known depth gives a real-world width:

```
real_width  = width_pixels  * depth / fx
real_height = height_pixels * depth / fy
scale_2d    = max(real_width, real_height)
```

This is computed but **deliberately not blended** into the final scale value when a better estimate exists. When 2D back-projection was previously merged into the final scale (via geometric mean), depth noise and tight mask sizes consistently shrunk the scale 30–50% below the correct value.

A **cross-check warning** is logged if the 2D estimate disagrees with the chosen scale by more than 2.5× — this indicates either the mask or depth map has quality issues and the result should be manually verified.

---

## 13. GLB to Gaussians Conversion

**File:** `glb_to_gaussians.py`

After the pipeline determines where and how to place the object (position, rotation, scale, support surface), it must convert the 3D object mesh (GLB format) into the Gaussian format expected by the scene checkpoint.

### 13.1 Why GLB and Gaussian Splatting Together?

3DGS training requires many posed images and GPU hours. For *inserted* objects, we only have a mesh (from Gemini/TripoSR or any 3D generator). The conversion pipeline solves this by either:
- **Sample mode** (fast): directly sample points on the mesh surface, assign colour from texture, and initialise Gaussian parameters analytically.
- **Train mode** (quality): render synthetic views from Blender, run COLMAP for camera poses, and train a full 3DGS model — producing a high-quality textured Gaussian representation.

The sample mode is used as the fast default; the train mode is attempted first in the main pipeline for better quality and falls back to sample mode if it fails.

### 13.2 Coordinate Conversion — GLB Y-up to Scene Z-up

GLB files (glTF 2.0 standard) use a Y-up right-handed coordinate system. The scene uses Z-up. The conversion:

```
[X, Y, Z]_glb → [X, -Z, Y]_scene
```

is implemented as matrix multiplication with:

```python
_COORD_CONV = [[1, 0,  0],
               [0, 0, -1],
               [0, 1,  0]]
```

This is applied to every point `pts_scene = _COORD_CONV @ pts.T).T`. The same transformation must be applied to Gaussian orientation quaternions (§13.5) — skipping the quaternion transform would make the object's internal structure misaligned with its new position.

### 13.3 Area-Weighted Surface Sampling

The mesh is sampled with points distributed proportionally to surface area:

```python
# Triangle areas from cross product of edge vectors:
cross = np.cross(v1 - v0, v2 - v0)
areas = 0.5 * ||cross||

# Sample triangles proportionally:
face_idx = np.random.choice(n_faces, size=n_points, p=areas/total_area)

# Uniform sampling within each triangle using barycentric coordinates:
r1, r2 ~ Uniform(0, 1)
b0 = 1 - sqrt(r1);   b1 = sqrt(r1) * (1 - r2);   b2 = sqrt(r1) * r2
point = b0*v0 + b1*v1 + b2*v2
```

The barycentric sampling formula `(b0, b1, b2) = (1-√r1, √r1*(1-r2), √r1*r2)` produces a **uniform distribution over the triangle** [Shirley & Chiu 1997] — naive uniform sampling of `(r1, r2)` would concentrate points near one corner. Area-weighting ensures large faces are represented proportionally — a small triangulated nose would otherwise be equally sampled as the large flat tabletop.

For multi-mesh GLBs, each mesh's sample count is proportional to its surface area contribution, ensuring the total `num_gaussians` is distributed realistically.

### 13.4 Texture Color Sampling

For each sampled surface point, colour is extracted from the mesh's material in this priority order:

1. **Face colours** — direct per-face RGBA in `visual.face_colors`
2. **Vertex colours** — barycentric interpolation of per-vertex RGBA: `color = b0*c_v0 + b1*c_v1 + b2*c_v2`
3. **UV texture map** — bilinear interpolation of the material's albedo texture at the sampled UV coordinates
4. **Default grey** `[0.65, 0.65, 0.65]` — fallback if no colour information is available

**Bilinear texture sampling** (`_sample_texture_bilinear`): converts `(u, v)` UV coordinates to pixel indices `(x_float, y_float)`, takes the four surrounding integer pixels, and blends them with bilinear weights:

```
color = (1-wx)*(1-wy)*c00 + wx*(1-wy)*c01 + (1-wx)*wy*c10 + wx*wy*c11
```

This is the standard bilinear interpolation formula, smoother than nearest-neighbour for objects viewed at non-integer pixel sizes.

**Library:** `trimesh` handles multi-format mesh loading (GLB, OBJ, PLY) with automatic material/UV extraction. It is preferred over Open3D because trimesh natively loads glTF 2.0 materials with proper UV unwrapping, which Open3D does not support.

### 13.5 Quaternion Transformation (Basis Change vs Object Rotation)

There are two distinct quaternion operations applied to Gaussian orientations:

**A. Coordinate-system basis change** (`_transform_quats_by_matrix`, line 170):

When the coordinate axes are remapped by `_COORD_CONV`, every Gaussian's internal orientation must be re-expressed in the new basis. This is the **similarity transform** or **conjugate sandwich**:

```
R_new = R_coord @ R_old @ R_coord^T
```

In quaternion algebra: `q_new = q_R * q_old * q_R^{-1}` where `q_R` is the quaternion corresponding to `_COORD_CONV`.

This is NOT a rotation of the object — it is a change of the reference frame. The object looks identical; only the coordinates used to describe it change.

**B. Object placement rotation** (`_left_multiply_quats`, line 184):

The surface-alignment rotation `R_place` (§11) actively rotates the object in world space. This is a **left multiplication**:

```
R_final = R_place @ R_old_in_world
```

In quaternion algebra: `q_final = q_place * q_old`

Why left-multiply? Because `R_place` is expressed in world space, and we want it applied after the object is already in its local world-space orientation. Left-multiplying pre-pends the world-space rotation on top of the existing orientation.

**Library:** `scipy.spatial.transform.Rotation` handles the conversions between rotation matrix and quaternion with numerically stable algorithms (quaternion normalisation, SLERP, etc.). Direct quaternion multiplication is avoided in favour of this abstraction to prevent handedness mistakes.

### 13.6 Adaptive Gaussian Scale Radius

Each point sampled on the mesh becomes a Gaussian. The Gaussian's log-scale is initialised to:

```python
adaptive_radius = (np.ptp(pts_scene, axis=0).prod() / max(1, N)) ** (1.0 / 3.0) * 0.35
log_scale = log(max(adaptive_radius, 1e-7))
```

`np.ptp` = peak-to-peak = range. `prod()` gives the volume of the bounding box. `(volume / N) ^ (1/3)` is the side length of a cube if the N points were uniformly distributed in that volume — the "average spacing" between Gaussians. Multiplying by 0.35 makes each Gaussian slightly smaller than this spacing to avoid excessive overlap while still covering the surface.

This is an analytical approximation of the scale that a 3DGS optimiser would converge to — it produces a visually solid-looking object immediately without any training.

### 13.7 Slope-Aware Support Lift

**File:** `glb_to_gaussians.py`, `_lift_to_support()` (line 19)

After positioning and rotating the Gaussian cloud, it is lifted vertically until its lowest point exactly touches the support surface. On a flat surface, this is a simple scalar shift:

```python
means[:, 2] += support_z - means[:, 2].min()
```

On a tilted surface (slope, ramp, angled tabletop), different Gaussians of the same object are at different heights relative to the floor — the downhill edge needs a larger shift than the uphill edge. A global scalar shift leaves the downhill side floating. The slope-aware solution evaluates the plane height at every Gaussian's own `(x, y)`:

```
plane_z(x, y) = anchor[2] - (n[0]*(x - anchor[0]) + n[1]*(y - anchor[1])) / n[2]
```

This is the standard **implicit plane equation** solved for z: `n · (X - P) = 0  →  z = P[2] - (n[0]*(x-P[0]) + n[1]*(y-P[1])) / n[2]`.

Then:
```python
clearance = means[:, 2] - plane_z(means[:, 0], means[:, 1])
means[:, 2] -= clearance.min()   # shift so worst-case clearance = 0
```

The object shifts in world-Z by the single most negative clearance (the point furthest below the slope). After the shift, exactly one Gaussian touches the slope surface; all others are above it by the correct slope-consistent amount.

**Why this matters:** On a 15° ramp with an object that is 0.5m wide, the height difference between the front and back edge is `0.5 * sin(15°) ≈ 0.13m`. A flat-Z approach would either sink the front edge 0.13m or float the back edge 0.13m. The per-point plane evaluation eliminates both.

### 13.8 mesh2splat PLY Path

When a mesh2splat-trained PLY is available alongside the GLB, `mesh2splat_ply_to_gaussians()` (line 73) is used instead of the sample-mode fallback. mesh2splat is an external tool that trains a proper 3DGS model for the object mesh, producing realistic per-Gaussian sizes, orientations, and SH coefficients. The PLY format is the standard 3DGS output format with named properties: `x, y, z, f_dc_0/1/2, f_rest_0…44, opacity, scale_0/1/2, rot_0/1/2/3`.

The pipeline reads these with `plyfile.PlyData.read()` and applies the same 6-step transformation (centre → coord-conv → scale → rotation → translation → lift) identically to the sample path, but using the trained quaternions and log-scales rather than analytically initialised ones. This produces dramatically better visual quality because the trained Gaussians have accurate per-orientation ellipsoidal shapes.

### 13.9 Train Mode — Blender + COLMAP + 3DGS Trainer

The train mode produces the highest quality conversion:

1. **Blender renders** (`_write_blender_render_script`, `_render_synthetic_views`): a Python script is written that drives Blender headlessly (`blender -b -P script.py`) to render `num_views` images (default 48) of the object from orbit cameras at two elevation angles (±15°), with realistic lighting from an area light. Camera parameters (FOV, world matrix per view) are saved as JSON for later stages.

2. **COLMAP reconstruction** (`_run_colmap`): the rendered images are fed to COLMAP [Schönberger & Frahm 2016] for structure-from-motion. COLMAP extracts SIFT features, matches them, and runs incremental reconstruction to produce accurate camera poses. Even though the cameras are known (from Blender), COLMAP registration grounds the poses in metric scale and validates their consistency.

3. **3DGS training** (`_run_external_training`): a trainer is called with the images, cameras, and sparse reconstruction. After `train_steps` iterations, it outputs a Gaussian `.pt` file which is loaded and normalised via `_normalize_trained_gaussian_dict`.

The result is a fully trained 3DGS object with per-Gaussian sizes, orientations, and colour that accurately represent the original mesh's appearance from all angles — far better than point-cloud initialisation alone.

---

## 14. Checkpoint Merging — Writing the Combined Scene

**File:** `updated_pipeline.py`, `add_object_to_scene()` (line 1880+)

After converting the object to Gaussians, the scene and object Gaussian tensors are **concatenated** along dimension 0:

```python
state["_model.means"]       = cat([scene_means,    object_means])     # (N+M, 3)
state["_model.scales"]      = cat([scene_scales,   object_scales])    # (N+M, 3)
state["_model.quats"]       = cat([scene_quats,    object_quats])     # (N+M, 4)
state["_model.features_dc"] = cat([scene_fdc,      object_fdc])       # (N+M, 1, 3)
state["_model.features_rest"]= cat([scene_frest,   object_frest])     # (N+M, 15, 3)
state["_model.opacities"]   = cat([scene_opacities, object_opacities]) # (N+M, 1)
```

The merged checkpoint is saved with `torch.save(ckpt, OUTPUT_PATH)`. The number of object Gaussians is stored separately in `ckpt["num_object_gaussians"]` so downstream optimisation can identify which Gaussians to train (the new object) vs keep frozen (the scene background).

**Why this works:** 3DGS rendering is simply alpha-compositing all Gaussians from front-to-back at each pixel. Adding more Gaussians to the state dict is equivalent to adding geometry to the scene — the renderer sees a combined scene with `N+M` Gaussians. No retraining is needed for the initial placement; the object is already coloured and sized correctly. Post-placement optimisation (separate stage) refines the object's position, size, and colour to best match the background scene's lighting.

---

## 15. End-to-End Data Flow Diagram

```
CKPT (N Gaussians)
        │
        ├──► ckpt_to_ply() ──► PLY file
        │                          │
        │                    viser web viewer
        │                          │ user selects camera
        │                    camera_state.pt
        │                          │
        ├──► render_gaussians() ──► selected_camera_view.png
        │    (selected camera)
        │
        │  [Gemini edits selected_camera_view.png → diffusion_edited.png]
        │
        ├──► render_depth_map() ──► depth_map (H×W float32)
        │
        │  OWLv2 detection on diffusion_edited.png
        │  → bbox (x1,y1,x2,y2) + confidence
        │
        │  Gemini cutout silhouette
        │  → silhouette mask (H×W bool)
        │  → NCC template matching → placement_mask (H×W bool)
        │
        ├──► select_object_gaussians_depth_aware()
        │    Pass 0: 2D bbox+mask filter    → raw candidates
        │    Pass 1: occlusion consistency  → depth-consistent candidates
        │    Pass 2: RANSAC dominant plane  → single-surface inliers
        │    Pass 2b: spatial cluster       → spatially contiguous inliers
        │    → object_indices (K Gaussians)
        │
        ├──► estimate_support_surface_robust()
        │    RANSAC plane on means[object_indices]
        │    Tangent-basis footprint → footprint_size
        │    Tilt clamp → surface_normal
        │    → support_point, support_normal, footprint_size
        │
        ├──► compute_surface_alignment_rotation(surface_normal)
        │    Rodrigues formula (shortest arc up→normal)
        │    → placement_rotation (3×3)
        │
        ├──► estimate_object_scale_robust()
        │    Primary: footprint_size
        │    Fallback: gaussian bbox / 2D backprojection
        │    → scale (metres)
        │
        └──► glb_to_gaussians() or mesh2splat_ply_to_gaussians()
             1. Centre object
             2. Coord conv (Y-up → Z-up)
             3. Scale to footprint_size
             4. Apply placement_rotation (Rodrigues rotation)
             5. Translate to support_point XY
             6. Slope-aware lift to support surface
             → object_gaussians (M Gaussians)

                    CKPT (N + M Gaussians)
                    saved → room_with_object.ckpt
```

---

## 16. Library Choices and Justifications

| Library | Role | Why This One |
|---|---|---|
| `torch` / PyTorch | Tensor operations, checkpoint I/O, GPU computation | 3DGS checkpoints are natively PyTorch `.pt`/`.ckpt` files. All Gaussian data is `torch.Tensor`. GPU operations for rendering require CUDA tensors. |
| `gsplat` | 3DGS GPU rasterisation | The standard open-source 3DGS renderer [Ye et al. 2024]. GPU tile-based rasterisation is 1000× faster than any CPU alternative. Supports `RGB+ED` depth render in a single pass. |
| `numpy` | Numerical computation (geometry, matrix ops, depth maps) | Fast C-backed array operations. All geometry (plane fitting, projections, quaternion handling) runs in numpy because PyTorch overhead is not justified for single-shot operations on small arrays. |
| `scipy.spatial.cKDTree` | Nearest-neighbour queries for spatial clustering | C-backed k-d tree, O(N log N) build and O(M log N) query. Significantly faster than scikit-learn's `NearestNeighbors` for this use case. Used for the connected-component spatial filter. |
| `scipy.sparse.csgraph` | Connected-component labelling on point graphs | Sparse matrix representation avoids N² memory for large point clouds. The `connected_components` function uses union-find internally, O(N+E). |
| `scipy.spatial.transform.Rotation` | Quaternion↔matrix conversions, rotation composition | Numerically stable implementations of quaternion normalisation, composition, and conversion. Avoids all manual quaternion arithmetic that is error-prone in handedness and normalisation. |
| `scipy.ndimage` | Binary morphological operations on masks | Industry-standard image morphology (opening, closing, fill_holes, label). Used for silhouette mask cleaning and connected-component detection on 2D masks. |
| `trimesh` | GLB/OBJ mesh loading with materials and UV | Best Python library for glTF 2.0 material support. Loads multi-mesh GLB scenes with proper UV unwrapping, which Open3D and PyMesh do not reliably support. |
| `plyfile` | PLY file reading for mesh2splat output | Lightweight reader for binary/ASCII PLY with named properties. The standard Gaussian PLY format uses named vertex properties (`x, y, z, f_dc_0, scale_0, rot_0, ...`) which plyfile reads directly by name. |
| `PIL` / Pillow | Image I/O, resizing, format conversion | Universal image I/O. Used for reading/writing PNG/JPEG and for the LANCZOS-quality resize that aligns input images to camera resolution. |
| `transformers` (HuggingFace) | OWLv2 detection, optional SAM refinement | Standard interface to pre-trained vision-language models. OWLv2 and SAM are both available as HuggingFace model cards with one-line download. |
| `matplotlib` | Depth map visualisation, bbox overlay images | Quick visualisation of scalar fields (jet colormap for depth) and geometry overlays (bbox rectangles). Not performance-critical; only used for diagnostic outputs. |
| `psutil` | CPU/memory resource tracking | Lightweight process introspection for the detection resource report (timing, memory delta). |

---

## 17. Novelty Summary

The following techniques in this pipeline are either novel or represent a novel combination not previously applied to 3DGS object placement:

### 1. RANSAC Dominant-Plane Vote for Gaussian Selection (§9.3)

Prior 3DGS editing work selects Gaussians using 2D masks or rough 3D distance thresholds. This pipeline uses a full 3D RANSAC plane vote over occlusion-consistent candidates to find the **majority-vote support surface**. This is the first method (to the authors' knowledge) that uses plane consensus voting to separate a teddy bear's bench-contact Gaussians from its floor-contact and wall-contact Gaussians using only the scene's own 3D data.

### 2. Spatial-Locality Connected-Component Post-Filtering (§9.4)

Standard RANSAC returns all inliers of the winning plane equation, which can include spatially disjoint but equation-coincident points. Applying graph-connectivity clustering on the RANSAC inlier set — with adaptive radius based on the inlier set's own nearest-neighbour spacing — is a novel combination that robustly removes these false positives while preserving elongated or curved support surfaces.

### 3. Direct Plane Fit on Filtered Gaussians for Surface Estimation (§10)

Previous approaches (and the first version of this pipeline) searched for the support surface by looking at scene Gaussians *outside* the object region and inferring the surface below. This paper's approach fits the plane *directly to the depth-validated object_indices Gaussians themselves*, which are already correctly positioned at the right world-space height. This eliminates the large positional error that arose from using the wrong set of reference points.

### 4. Per-Point Slope-Aware Lift (§13.7)

The slope-aware lift evaluates the support plane's height `z_plane(x, y)` under every Gaussian individually, rather than using a single scalar `z_min`. This makes the lift correct on any tilted surface without any parameter tuning. To the authors' knowledge, no prior 3DGS object insertion paper handles slope placement at this geometric precision.

### 5. Percentile-Trimmed In-Plane Footprint Scale (§10.2)

Deriving the object's required real-world scale from the trimmed extent of the filtered Gaussians projected onto their own support plane — rather than from 2D pixel measurements + depth back-projection — is more direct and more robust. The 2D+depth approach is retained only as a cross-check, not as an input to the final scale.

### 6. Rodrigues Shortest-Arc Placement Rotation (§11)

Using the Rodrigues formula to compute the minimal rotation from world-up to surface-normal provides the correct surface-flush tilt for any slope without introducing any spurious yaw. This is geometrically exact and numerically simple, and is applied here for the first time to 3DGS initial placement.

---

## 18. References

[Kerbl et al. 2023] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM Transactions on Graphics (SIGGRAPH 2023). https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[Ye et al. 2024] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Justin Kerr, Matias Turkulainen, Angjoo Kanazawa. "gsplat: An Open-Source Library for Gaussian Splatting." arXiv:2409.06765. https://arxiv.org/abs/2409.06765

[Minderer et al. 2023] Matthias Minderer, Alexey Gritsenko, Neil Houlsby. "Scaling Open-Vocabulary Object Detection." NeurIPS 2023. https://arxiv.org/abs/2306.09683

[Fischler & Bolles 1981] Martin A. Fischler, Robert C. Bolles. "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography." Communications of the ACM, 24(6):381–395. The canonical RANSAC paper.

[Hartley & Zisserman 2003] Richard Hartley, Andrew Zisserman. "Multiple View Geometry in Computer Vision." Cambridge University Press, 2nd ed. The standard reference for camera models, projection, homography, and epipolar geometry.

[Ahn et al. 2002] Soon Ju Ahn, Wolfgang Rauh, Hans-Jürgen Warnecke. "Least-squares orthogonal distances fitting of circle, sphere, ellipse, hyperbola, and parabola." Pattern Recognition, 34(12):2283–2303. On total-least-squares (orthogonal distance) fitting for geometric primitives.

[Golub & Van Loan 2013] Gene H. Golub, Charles F. Van Loan. "Matrix Computations." Johns Hopkins University Press, 4th ed. Standard reference for SVD and its applications to least-squares problems.

[Rousseeuw & Leroy 1987] Peter J. Rousseeuw, Annick M. Leroy. "Robust Regression and Outlier Detection." Wiley. Comprehensive treatment of MAD, percentile trimming, and other robust estimators.

[Hampel 1974] Frank R. Hampel. "The Influence Curve and Its Role in Robust Estimation." Journal of the American Statistical Association, 69(346):383–393. Original MAD estimator paper.

[Rodrigues 1840] Olinde Rodrigues. "Des lois géométriques qui régissent les déplacements d'un système solide dans l'espace." Journal de Mathématiques Pures et Appliquées, 5:380–440. Original Rodrigues rotation formula.

[Slabaugh 1999] Gregory G. Slabaugh. "Computing Euler angles from a rotation matrix." Technical Note, 1999. Accessible treatment of Rodrigues formula and rotation representations.

[Shirley & Chiu 1997] Peter Shirley, Kenneth Chiu. "A Low Distortion Map Between Disk and Square." Journal of Graphics Tools, 2(3):45–52. Derivation of the uniform barycentric sampling formula `(1-√r1, √r1*(1-r2), √r1*r2)`.

[Schönberger & Frahm 2016] Johannes L. Schönberger, Jan-Michael Frahm. "Structure-from-Motion Revisited." CVPR 2016. The COLMAP paper — feature extraction, matching, and incremental SfM.

[Chen et al. 2023] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, Guosheng Lin. "GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting." arXiv:2311.14521.

[Yang et al. 2024] Zhen Yang et al. "GaussianObject: Just Taking Four Images to Get A High-Quality 3D Object with Gaussian Splatting." arXiv:2402.10259.
