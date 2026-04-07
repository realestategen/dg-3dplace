import os
import re
import sys
import argparse
import subprocess
from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np
from PIL import Image

from gemini_image_gen import generate_object_cutout_with_gemini


_HY3D_PIPELINE = None
_HY3D_PAINT_PIPELINE = None


def _resolve_hunyuan_paths() -> Tuple[str, str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    hy_dir = os.path.join(os.path.dirname(here), "Hunyuan3D-2.1")
    hy_shape = os.path.join(hy_dir, "hy3dshape")
    hy_paint = os.path.join(hy_dir, "hy3dpaint")
    return hy_dir, hy_shape, hy_paint


def _resolve_hunyuan_paint_assets() -> Dict[str, str]:
    hy_dir, _, hy_paint = _resolve_hunyuan_paths()
    return {
        "realesrgan_ckpt_path": os.path.join(hy_paint, "ckpt", "RealESRGAN_x4plus.pth"),
        "multiview_cfg_path": os.path.join(hy_paint, "cfgs", "hunyuan-paint-pbr.yaml"),
        "custom_pipeline": os.path.join(hy_paint, "hunyuanpaintpbr"),
        "hy_dir": hy_dir,
    }


def _setup_hunyuan_imports() -> None:
    hy_dir, hy_shape, hy_paint = _resolve_hunyuan_paths()
    for p in [hy_dir, hy_shape, hy_paint]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _get_shape_pipeline(model_path: str = "tencent/Hunyuan3D-2.1"):
    global _HY3D_PIPELINE
    if _HY3D_PIPELINE is not None:
        return _HY3D_PIPELINE

    _setup_hunyuan_imports()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    _HY3D_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    return _HY3D_PIPELINE


def _get_paint_pipeline(max_num_view: int = 6, resolution: int = 512):
    global _HY3D_PAINT_PIPELINE
    if _HY3D_PAINT_PIPELINE is not None:
        return _HY3D_PAINT_PIPELINE

    _setup_hunyuan_imports()
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(max_num_view=max_num_view, resolution=resolution)
    paint_assets = _resolve_hunyuan_paint_assets()

    if os.path.exists(paint_assets["realesrgan_ckpt_path"]):
        conf.realesrgan_ckpt_path = paint_assets["realesrgan_ckpt_path"]
    if os.path.exists(paint_assets["multiview_cfg_path"]):
        conf.multiview_cfg_path = paint_assets["multiview_cfg_path"]
    if os.path.exists(paint_assets["custom_pipeline"]):
        conf.custom_pipeline = paint_assets["custom_pipeline"]

    _HY3D_PAINT_PIPELINE = Hunyuan3DPaintPipeline(conf)
    return _HY3D_PAINT_PIPELINE


def _run_hunyuan_paint_subprocess(
    mesh_path: str,
    image_path: str,
    output_mesh_path: str,
    texture_env: str,
    max_num_view: int,
    resolution: int,
) -> Tuple[bool, str]:
    """Run Hunyuan paint in a separate conda env (bpy-enabled)."""
    texture_env = (texture_env or "").strip()
    if not texture_env:
        return False, "No texture subprocess env configured"

    here = os.path.dirname(os.path.abspath(__file__))
    worker_path = os.path.join(here, "hunyuan_paint_worker.py")
    if not os.path.exists(worker_path):
        return False, f"Worker script missing: {worker_path}"

    cmd = [
        "conda",
        "run",
        "-n",
        texture_env,
        "python",
        worker_path,
        "--mesh",
        mesh_path,
        "--image",
        image_path,
        "--output",
        output_mesh_path,
        "--views",
        str(max_num_view),
        "--resolution",
        str(resolution),
    ]

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        return False, f"Failed to launch subprocess painter: {e}"

    if result.returncode != 0:
        err = (result.stderr or "").strip()
        out = (result.stdout or "").strip()
        message = err if err else out
        return False, f"Subprocess painter failed: {message}"

    return True, (result.stdout or "").strip()


def detect_bbox_with_owlv2(image_path: str, prompt: str, score_threshold: float = 0.06) -> Optional[Tuple[float, float, float, float]]:
    """Detect object bbox with OWLv2 using full prompt + simplified query variants."""
    try:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
    except Exception:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        prompt_raw = (prompt or "").strip()
        prompt_l = prompt_raw.lower()
        stop_words = {
            "a", "an", "the", "on", "in", "at", "near", "next", "to", "of", "with",
            "and", "or", "under", "over", "behind", "front", "left", "right",
            "red", "blue", "green", "yellow", "white", "black", "brown", "gray",
            "small", "large", "big",
        }
        known_targets = ["car", "bench", "vase", "laptop", "chair", "table", "sofa", "plant", "bottle"]

        query_variants = []
        for q in [prompt_raw, f"a photo of {prompt_raw}"]:
            q = q.strip()
            if q and q not in query_variants:
                query_variants.append(q)

        for obj in known_targets:
            if re.search(rf"\b{obj}\b", prompt_l):
                for q in [obj, f"a photo of a {obj}"]:
                    if q not in query_variants:
                        query_variants.append(q)

        prompt_tokens = [t for t in re.split(r"[^a-z0-9]+", prompt_l) if len(t) >= 3 and t not in stop_words]
        for tok in prompt_tokens[:3]:
            if tok not in query_variants:
                query_variants.append(tok)

        inputs = processor(text=[query_variants], images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
        res0 = results[0]
        if len(res0["scores"]) == 0:
            return None

        scores = res0["scores"].detach().cpu().numpy()
        boxes = res0["boxes"].detach().cpu().numpy()
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) < score_threshold:
            return None

        x1, y1, x2, y2 = boxes[best_idx]
        return float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return None


def _crop_with_padding(image: Image.Image, bbox: Tuple[float, float, float, float], pad_ratio: float = 0.12) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = image.size

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(8.0, bw * pad_ratio)
    pad_y = max(8.0, bh * pad_ratio)

    cx1 = int(max(0, np.floor(x1 - pad_x)))
    cy1 = int(max(0, np.floor(y1 - pad_y)))
    cx2 = int(min(w, np.ceil(x2 + pad_x)))
    cy2 = int(min(h, np.ceil(y2 + pad_y)))

    return image.crop((cx1, cy1, cx2, cy2))


def _remove_bg_if_available(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove
        return remove(image)
    except Exception:
        return image


def _remove_white_background(image: Image.Image, threshold: int = 250, alpha_cutoff: int = 8) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    white_mask = (rgb[:, :, 0] >= threshold) & (rgb[:, :, 1] >= threshold) & (rgb[:, :, 2] >= threshold)
    alpha[white_mask] = 0
    arr[:, :, 3] = np.where(alpha <= alpha_cutoff, 0, alpha)
    return Image.fromarray(arr)


def _foreground_ratio(image: Image.Image) -> float:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
    return float((alpha > 0).mean())


def _make_centered_canvas(image: Image.Image, padding_ratio: float = 0.18) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
    ys, xs = np.where(alpha > 0)

    if len(xs) == 0 or len(ys) == 0:
        return rgba

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    obj = rgba.crop((x1, y1, x2, y2))

    obj_w, obj_h = obj.size
    canvas_size = int(max(obj_w, obj_h) * (1.0 + 2.0 * padding_ratio))
    canvas_size = max(canvas_size, max(obj_w, obj_h))

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    offset_x = (canvas_size - obj_w) // 2
    offset_y = (canvas_size - obj_h) // 2
    canvas.alpha_composite(obj, (offset_x, offset_y))
    return canvas


def _prepare_shape_input(
    crop: Image.Image,
    prompt: str,
    session_dir: Optional[str] = None,
    use_gemini: bool = True,
    require_gemini: bool = False,
    api_key: Optional[str] = None,
) -> Image.Image:
    """Create a conservative object-focused image for Hunyuan3D."""
    if crop.mode != "RGBA":
        crop_rgba = crop.convert("RGBA")
    else:
        crop_rgba = crop.copy()

    gemini_cutout_path = None
    cleaned_cutout_path = None
    if use_gemini:
        api_key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        if api_key:
            try:
                if session_dir:
                    os.makedirs(session_dir, exist_ok=True)
                    gemini_cutout_path = os.path.join(session_dir, "gemini_object_cutout.png")
                    cleaned_cutout_path = os.path.join(session_dir, "gemini_object_cleaned.png")
                else:
                    gemini_cutout_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_object_cutout.png")
                    cleaned_cutout_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_object_cleaned.png")

                generate_object_cutout_with_gemini(
                    api_key=api_key,
                    input_image_path=_save_temp_crop(crop_rgba, session_dir),
                    object_prompt=prompt,
                    output_image_path=gemini_cutout_path,
                    width=max(512, crop_rgba.width),
                    height=max(512, crop_rgba.height),
                )

                with Image.open(gemini_cutout_path) as generated:
                    prepared = generated.convert("RGBA")
                prepared = _remove_white_background(prepared)
                if cleaned_cutout_path:
                    prepared.save(cleaned_cutout_path)
                print(f"Using Gemini object cutout: {gemini_cutout_path}")
                if cleaned_cutout_path:
                    print(f"Saved cleaned object PNG: {cleaned_cutout_path}")
                return _make_centered_canvas(prepared)
            except Exception:
                print("Gemini cutout failed; falling back to local crop cleanup.")
                if require_gemini:
                    raise RuntimeError("Gemini cutout failed and strict mode is enabled.")
                pass

        elif require_gemini:
            raise RuntimeError("Gemini cutout is required but GEMINI_API_KEY/GOOGLE_API_KEY is not set.")

    prepared = crop_rgba
    if _foreground_ratio(crop_rgba) > 0.95:
        try:
            candidate = _remove_bg_if_available(crop_rgba.convert("RGB")).convert("RGBA")
            ratio = _foreground_ratio(candidate)
            if 0.03 <= ratio <= 0.95:
                prepared = candidate
        except Exception:
            prepared = crop_rgba

    prepared = _remove_white_background(prepared)
    return _make_centered_canvas(prepared)


def _save_temp_crop(crop: Image.Image, session_dir: Optional[str]) -> str:
    base_dir = session_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    temp_path = os.path.join(base_dir, "_gemini_cutout_input.png")
    crop.save(temp_path)
    return temp_path


def _save_temp_paint_input(image: Image.Image, session_dir: Optional[str]) -> str:
    base_dir = session_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, "_paint_input.png")
    image.convert("RGBA").save(path)
    return path


def _apply_image_colors_to_mesh(mesh_obj, source_image_path: str) -> bool:
    """Apply per-vertex colors by projecting source image colors to mesh vertices."""
    if not os.path.exists(source_image_path):
        print(f"[!] Source image not found: {source_image_path}")
        return False

    try:
        source_img = Image.open(source_image_path).convert("RGBA")
        arr = np.array(source_img, dtype=np.uint8)
        h, w = arr.shape[:2]

        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        fg_mask = alpha > 20
        if not np.any(fg_mask):
            print("[!] Source image has no foreground alpha for color projection")
            return False

        # Robust fallback color for projected pixels that hit transparent regions.
        fallback_rgb = np.median(rgb[fg_mask], axis=0).astype(np.uint8)

        if not hasattr(mesh_obj, "vertices"):
            return False

        verts = np.asarray(mesh_obj.vertices, dtype=np.float32)
        if verts.size == 0:
            return False

        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)

        # Planar projection: x -> u, y -> v, works well for upright object crops.
        u = (verts[:, 0] - mins[0]) / spans[0]
        v = 1.0 - ((verts[:, 1] - mins[1]) / spans[1])

        x = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
        y = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)

        sampled_rgba = arr[y, x]
        sampled_rgb = sampled_rgba[:, :3]
        sampled_a = sampled_rgba[:, 3:4]

        # Replace transparent hits with a representative foreground color.
        projected_rgb = np.where(sampled_a > 20, sampled_rgb, fallback_rgb)
        vertex_rgba = np.concatenate(
            [projected_rgb.astype(np.uint8), np.full((len(projected_rgb), 1), 255, dtype=np.uint8)],
            axis=1,
        )

        mesh_obj.visual.vertex_colors = vertex_rgba
        print("Applied projected vertex colors from source image")
        return True
    except Exception as e:
        print(f"[!] Color projection failed: {e}")
        return False

    return False


def _export_colored_glb(mesh_obj, output_obj_path: str) -> Optional[str]:
    """Export a GLB companion file for reliable vertex-color visualization."""
    try:
        glb_path = os.path.splitext(output_obj_path)[0] + ".glb"
        mesh_obj.export(glb_path)
        return glb_path
    except Exception:
        return None


def generate_obj_from_prompt_image(
    image_path: str,
    prompt: str,
    output_obj_path: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    session_dir: Optional[str] = None,
    model_path: str = "tencent/Hunyuan3D-2.1",
    require_gemini_cutout: bool = False,
    api_key: Optional[str] = None,
    enable_texture: bool = True,
    texture_views: int = 6,
    texture_resolution: int = 512,
    texture_env: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate 3D OBJ from an image region and prompt using Hunyuan3D shape model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_obj_path)), exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    if bbox is None:
        bbox = detect_bbox_with_owlv2(image_path, prompt)
        if bbox is None:
            raise RuntimeError("Could not detect object bbox with OWLv2 for 2D->3D generation.")

    crop = _crop_with_padding(image, bbox)
    crop_processed = _prepare_shape_input(
        crop,
        prompt=prompt,
        session_dir=session_dir,
        require_gemini=require_gemini_cutout,
        api_key=api_key,
    )

    gemini_cutout_path = None
    gemini_cleaned_path = None
    if session_dir:
        gemini_cutout_candidate = os.path.join(session_dir, "gemini_object_cutout.png")
        gemini_cleaned_candidate = os.path.join(session_dir, "gemini_object_cleaned.png")
        if os.path.exists(gemini_cutout_candidate):
            gemini_cutout_path = gemini_cutout_candidate
        if os.path.exists(gemini_cleaned_candidate):
            gemini_cleaned_path = gemini_cleaned_candidate

    debug_crop_path = None
    object_only_png_path = None
    if session_dir:
        os.makedirs(session_dir, exist_ok=True)
        debug_crop_path = os.path.join(session_dir, "detected_object_crop.png")
        object_only_png_path = os.path.join(session_dir, "detected_object_only.png")
        crop.save(debug_crop_path)
        crop_processed.save(object_only_png_path)

    pipeline = _get_shape_pipeline(model_path=model_path)
    mesh_untextured = pipeline(image=crop_processed)[0]
    mesh_untextured.export(output_obj_path)

    textured_output_path = None
    texture_error = None
    texture_method = "shape_only"
    mtl_path = None
    albedo_path = None
    colored_glb_path = None

    if enable_texture:
        paint_input_path = object_only_png_path or _save_temp_paint_input(crop_processed, session_dir)

        # Try Hunyuan paint pipeline first
        try:
            paint_pipeline = _get_paint_pipeline(max_num_view=texture_views, resolution=texture_resolution)
            textured_output_path = paint_pipeline(
                mesh_path=output_obj_path,
                image_path=paint_input_path,
                output_mesh_path=output_obj_path,
            )
            print("Using Hunyuan paint pipeline textures.")
            texture_method = "hunyuan_paint"

            obj_root, _ = os.path.splitext(output_obj_path)
            mtl_candidate = obj_root + ".mtl"
            albedo_candidate = obj_root + ".jpg"
            if os.path.exists(mtl_candidate):
                mtl_path = mtl_candidate
            if os.path.exists(albedo_candidate):
                albedo_path = albedo_candidate

        except Exception as e:
            texture_error = str(e)
            print(f"Hunyuan paint failed ({texture_error}), trying fallback color projection...")

            # Preferred fallback: run Hunyuan paint in separate bpy-enabled env.
            ok, msg = _run_hunyuan_paint_subprocess(
                mesh_path=output_obj_path,
                image_path=paint_input_path,
                output_mesh_path=output_obj_path,
                texture_env=texture_env or os.environ.get("HY3D_TEXTURE_ENV", ""),
                max_num_view=texture_views,
                resolution=texture_resolution,
            )
            if ok:
                textured_output_path = output_obj_path
                texture_error = None
                texture_method = "hunyuan_paint_subprocess"
                print("Using Hunyuan paint via subprocess env.")

                obj_root, _ = os.path.splitext(output_obj_path)
                mtl_candidate = obj_root + ".mtl"
                albedo_candidate = obj_root + ".jpg"
                if os.path.exists(mtl_candidate):
                    mtl_path = mtl_candidate
                if os.path.exists(albedo_candidate):
                    albedo_path = albedo_candidate
            else:
                print(f"Subprocess painter unavailable: {msg}")

            # Fallback: apply colors directly from the input image (no Blender needed)
            if textured_output_path is None:
                try:
                    mesh_obj = mesh_untextured.copy()
                    if _apply_image_colors_to_mesh(mesh_obj, paint_input_path):
                        mesh_obj.export(output_obj_path)
                        print(f"Applied image colors, saved to {output_obj_path}")
                        textured_output_path = output_obj_path
                        colored_glb_path = _export_colored_glb(mesh_obj, output_obj_path)
                        texture_error = None
                        texture_method = "vertex_color_projection"
                    else:
                        print("Image color projection also failed, keeping untextured mesh.")
                except Exception as fallback_error:
                    print(f"Fallback color projection failed: {fallback_error}")

    return {
        "output_obj_path": output_obj_path,
        "textured_output_path": textured_output_path,
        "colored_glb_path": colored_glb_path,
        "texture_method": texture_method,
        "mtl_path": mtl_path,
        "albedo_path": albedo_path,
        "texture_error": texture_error,
        "bbox": bbox,
        "crop_path": debug_crop_path,
        "object_only_png_path": object_only_png_path,
        "gemini_object_cutout_path": gemini_cutout_path,
        "gemini_object_cleaned_path": gemini_cleaned_path,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D OBJ from 2D image + prompt using Hunyuan3D.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", required=True, help="Prompt describing the object")
    parser.add_argument("--output", required=True, help="Output OBJ path")
    parser.add_argument("--session-dir", default="", help="Optional debug output folder")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, help="Optional bbox x1 y1 x2 y2")
    parser.add_argument("--require-gemini-cutout", action="store_true", help="Fail if Gemini cutout is not used")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key")
    parser.add_argument("--no-texture", action="store_true", help="Disable Hunyuan paint texturing step")
    parser.add_argument("--texture-views", type=int, default=6, help="Texture views for Hunyuan paint (6-9)")
    parser.add_argument("--texture-resolution", type=int, default=512, help="Texture resolution for Hunyuan paint")
    parser.add_argument("--texture-env", default=os.environ.get("HY3D_TEXTURE_ENV", ""), help="Optional conda env name for subprocess Hunyuan paint")
    return parser.parse_args()


def main():
    args = _parse_args()
    bbox = tuple(args.bbox) if args.bbox is not None else None
    result = generate_obj_from_prompt_image(
        image_path=args.image,
        prompt=args.prompt,
        output_obj_path=args.output,
        bbox=bbox,
        session_dir=args.session_dir or None,
        require_gemini_cutout=bool(args.require_gemini_cutout),
        api_key=(args.api_key or "").strip() or None,
        enable_texture=not bool(args.no_texture),
        texture_views=max(6, min(9, int(args.texture_views))),
        texture_resolution=int(args.texture_resolution),
        texture_env=(args.texture_env or "").strip() or None,
    )
    print("2D->3D generation completed")
    print(f"OBJ: {result['output_obj_path']}")
    print(f"BBox: {result['bbox']}")
    if result.get("crop_path"):
        print(f"Crop: {result['crop_path']}")
    if result.get("object_only_png_path"):
        print(f"Object-only PNG: {result['object_only_png_path']}")
    if result.get("gemini_object_cutout_path"):
        print(f"Gemini cutout PNG: {result['gemini_object_cutout_path']}")
    if result.get("gemini_object_cleaned_path"):
        print(f"Gemini cleaned PNG: {result['gemini_object_cleaned_path']}")
    if result.get("textured_output_path"):
        print(f"Textured mesh: {result['textured_output_path']}")
    if result.get("colored_glb_path"):
        print(f"Colored GLB: {result['colored_glb_path']}")
    if result.get("texture_method"):
        print(f"Texture method: {result['texture_method']}")
    if result.get("mtl_path"):
        print(f"MTL: {result['mtl_path']}")
    if result.get("albedo_path"):
        print(f"Albedo: {result['albedo_path']}")
    if result.get("texture_error") and not result.get("textured_output_path"):
        print(f"Texture step failed: {result['texture_error']}")


if __name__ == "__main__":
    main()



