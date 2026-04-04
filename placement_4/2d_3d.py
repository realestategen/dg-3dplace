import os
import re
import sys
import argparse
from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np
from PIL import Image

from gemini_image_gen import generate_object_cutout_with_gemini


_HY3D_PIPELINE = None


def _resolve_hunyuan_paths() -> Tuple[str, str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    hy_dir = os.path.join(os.path.dirname(here), "Hunyuan3D-2.1")
    hy_shape = os.path.join(hy_dir, "hy3dshape")
    hy_paint = os.path.join(hy_dir, "hy3dpaint")
    return hy_dir, hy_shape, hy_paint


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
    return Image.fromarray(arr, mode="RGBA")


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


def generate_obj_from_prompt_image(
    image_path: str,
    prompt: str,
    output_obj_path: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    session_dir: Optional[str] = None,
    model_path: str = "tencent/Hunyuan3D-2.1",
    require_gemini_cutout: bool = False,
    api_key: Optional[str] = None,
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

    return {
        "output_obj_path": output_obj_path,
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


if __name__ == "__main__":
    main()
