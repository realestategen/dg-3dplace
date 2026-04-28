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


def _resolve_hunyuan_paths() -> Tuple[str, str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    hy_dir = os.path.join(os.path.dirname(here), "Hunyuan3D-2.1")
    hy_shape = os.path.join(hy_dir, "hy3dshape")
    hy_paint = os.path.join(hy_dir, "hy3dpaint")
    return hy_dir, hy_shape, hy_paint


def _run_step1_shape_subprocess(
    input_image_path: str,
    output_mesh_path: str,
    conda_env: str = "hunyuan",
) -> Tuple[bool, str]:
    """Run Hunyuan3D step1_shape.py via subprocess in conda env."""
    hy_dir, _, _ = _resolve_hunyuan_paths()
    step1_script = os.path.join(hy_dir, "step1_shape.py")
    
    if not os.path.exists(step1_script):
        return False, f"step1_shape.py not found at {step1_script}"
    
    input_image_abs = os.path.abspath(input_image_path)
    if not os.path.exists(input_image_abs):
        return False, f"Input image not found: {input_image_abs}"
    
    output_mesh_abs = os.path.abspath(output_mesh_path)
    os.makedirs(os.path.dirname(output_mesh_abs), exist_ok=True)
    
    # Create wrapper that runs from Hunyuan3D-2.1 directory
    step1_wrapper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_step1_wrapper.py")
    processed_image_abs = os.path.splitext(input_image_abs)[0] + "_processed.png"
    
    try:
        with open(step1_wrapper_path, 'w') as f:
            f.write(f"""import sys
import os

# Change to Hunyuan3D-2.1 directory so relative imports work
os.chdir(r'{hy_dir}')
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Set input/output paths
INPUT_IMAGE_PATH = r'{input_image_abs}'
PROCESSED_IMAGE_PATH = r'{processed_image_abs}'
MESH_OUTPUT_PATH = r'{output_mesh_abs}'

# Execute step1_shape.py code
with open(r'{step1_script}') as code_file:
    code = code_file.read()
    # Replace the default paths with our absolute paths
    code = code.replace('INPUT_IMAGE_PATH = "input/demo.png"', 
                       f'INPUT_IMAGE_PATH = r"{input_image_abs}"')
    code = code.replace('PROCESSED_IMAGE_PATH = "input/demo_no_bg.png"',
                       f'PROCESSED_IMAGE_PATH = r"{processed_image_abs}"')
    code = code.replace('MESH_OUTPUT_PATH = "intermediate_mesh/mesh.obj"',
                       f'MESH_OUTPUT_PATH = r"{output_mesh_abs}"')
    exec(code)
""")
    except Exception as e:
        return False, f"Failed to create step1 wrapper: {e}"
    
    cmd = ["conda", "run", "-n", conda_env, "python", step1_wrapper_path]
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600, cwd=hy_dir)
    except subprocess.TimeoutExpired:
        return False, "Step1 shape generation timed out (>10 min)"
    except Exception as e:
        return False, f"Failed to launch step1: {e}"
    
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        out = (result.stdout or "").strip()
        message = err if err else out
        return False, f"Step1 failed: {message}\nStdout: {out}"
    
    if not os.path.exists(output_mesh_abs):
        return False, f"Step1 did not produce mesh at {output_mesh_abs}\nStdout: {result.stdout}\nStderr: {result.stderr}"
    
    return True, f"Generated mesh: {output_mesh_abs}"


def _run_step2_paint_subprocess(
    mesh_path: str,
    image_path: str,
    output_folder: str,
    conda_env: str = "hunyuan",
    max_num_view: int = 9,
    resolution: int = 512,
) -> Tuple[bool, str]:
    """Run Hunyuan3D step2_paint.py via subprocess in conda env."""
    hy_dir, _, _ = _resolve_hunyuan_paths()
    step2_script = os.path.join(hy_dir, "step2_paint.py")
    
    if not os.path.exists(step2_script):
        return False, f"step2_paint.py not found at {step2_script}"
    
    mesh_abs = os.path.abspath(mesh_path)
    if not os.path.exists(mesh_abs):
        return False, f"Input mesh not found: {mesh_abs}"
    
    image_abs = os.path.abspath(image_path)
    if not os.path.exists(image_abs):
        return False, f"Input image not found: {image_abs}"
    
    output_abs = os.path.abspath(output_folder)
    os.makedirs(output_abs, exist_ok=True)
    
    # Create wrapper that runs from Hunyuan3D-2.1 directory
    step2_wrapper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_step2_wrapper.py")

    def _run_once(views: int, res: int) -> Tuple[bool, str]:
        try:
            with open(step2_wrapper_path, 'w') as f:
                f.write(f"""import sys
import os

# Change to Hunyuan3D-2.1 directory so relative imports work
os.chdir(r'{hy_dir}')
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Set input/output paths
IMAGE_INPUT = r'{image_abs}'
MESH_INPUT = r'{mesh_abs}'
OUTPUT_DIR = r'{output_abs}'

# Execute step2_paint.py code
with open(r'{step2_script}') as code_file:
    code = code_file.read()
    # Replace the default paths with our absolute paths
    code = code.replace('IMAGE_INPUT = "input/demo_no_bg.png"',
                       f'IMAGE_INPUT = r"{image_abs}"')
    code = code.replace('MESH_INPUT = "intermediate_mesh/mesh.obj"',
                       f'MESH_INPUT = r"{mesh_abs}"')
    code = code.replace('OUTPUT_DIR = "output"',
                       f'OUTPUT_DIR = r"{output_abs}"')
    code = code.replace(
        'paint_config = Hunyuan3DPaintConfig(max_num_view=9, resolution=512)',
        'paint_config = Hunyuan3DPaintConfig(max_num_view={views}, resolution={res})'
    )
    exec(code)
""")
        except Exception as e:
            return False, f"Failed to create step2 wrapper: {e}"

        cmd = ["conda", "run", "-n", conda_env, "python", step2_wrapper_path]

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=1800, cwd=hy_dir)
        except subprocess.TimeoutExpired:
            return False, f"Step2 paint generation timed out (>30 min) at views={views}, resolution={res}"
        except Exception as e:
            return False, f"Failed to launch step2 at views={views}, resolution={res}: {e}"

        if result.returncode != 0:
            err = (result.stderr or "").strip()
            out = (result.stdout or "").strip()
            message = err if err else out
            return False, f"Step2 failed at views={views}, resolution={res}: {message}\nStdout: {out}"

        return True, f"Painted mesh in: {output_abs} (views={views}, resolution={res})"

    # Retry ladder for low-VRAM cases.
    attempts = [
        (int(max_num_view), int(resolution)),
        (6, 384),
        (6, 320),
        (4, 256),
    ]

    seen = set()
    errors = []
    for views, res in attempts:
        key = (max(1, views), max(128, res))
        if key in seen:
            continue
        seen.add(key)

        ok, msg = _run_once(key[0], key[1])
        if ok:
            if key != (int(max_num_view), int(resolution)):
                print(f"Step2 recovered with reduced settings: views={key[0]}, resolution={key[1]}")
            return True, msg

        errors.append(msg)
        lower_msg = msg.lower()
        is_oom = ("outofmemoryerror" in lower_msg) or ("cuda out of memory" in lower_msg)
        if not is_oom:
            # Non-memory failures should not be retried with lower settings.
            return False, msg

        print(f"Step2 OOM at views={key[0]}, resolution={key[1]}; retrying with lower memory settings...")

    return False, "\n---\n".join(errors)


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


def _save_temp_crop(crop: Image.Image, session_dir: Optional[str]) -> str:
    base_dir = session_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    temp_path = os.path.join(base_dir, "_gemini_cutout_input.png")
    crop.save(temp_path)
    return temp_path


def _collect_textured_outputs(output_dir: str, untextured_obj_path: str) -> Dict[str, Optional[str]]:
    """Find textured mesh artifacts generated by Step 2.

    Prefers GLB for color fidelity, then OBJ with companion MTL/texture.
    """
    if not os.path.isdir(output_dir):
        return {
            "output_glb_path": None,
            "textured_output_path": None,
            "mtl_path": None,
            "albedo_path": None,
            "output_color_mesh_path": None,
        }

    untextured_abs = os.path.abspath(untextured_obj_path)
    glb_candidates = []
    obj_candidates = []
    mtl_candidates = []
    tex_candidates = []

    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        lower = name.lower()
        if lower.endswith(".glb"):
            glb_candidates.append(path)
        elif lower.endswith(".obj"):
            obj_candidates.append(path)
        elif lower.endswith(".mtl"):
            mtl_candidates.append(path)
        elif lower.endswith((".png", ".jpg", ".jpeg")):
            tex_candidates.append(path)

    glb_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    obj_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    mtl_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    tex_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    output_glb_path = glb_candidates[0] if glb_candidates else None

    # Prefer an OBJ that is not the untextured one.
    textured_output_path = None
    for obj in obj_candidates:
        if os.path.abspath(obj) != untextured_abs:
            textured_output_path = obj
            break

    # If only one OBJ exists, keep it as potential textured OBJ only when MTL exists.
    if textured_output_path is None and obj_candidates:
        only_obj = obj_candidates[0]
        stem, _ = os.path.splitext(only_obj)
        mtl_for_obj = stem + ".mtl"
        if os.path.exists(mtl_for_obj):
            textured_output_path = only_obj

    mtl_path = None
    if textured_output_path:
        stem, _ = os.path.splitext(textured_output_path)
        candidate = stem + ".mtl"
        if os.path.exists(candidate):
            mtl_path = candidate
    if mtl_path is None and mtl_candidates:
        mtl_path = mtl_candidates[0]

    albedo_path = tex_candidates[0] if tex_candidates else None

    # Prefer GLB for color extraction, then textured OBJ.
    output_color_mesh_path = output_glb_path or textured_output_path

    return {
        "output_glb_path": output_glb_path,
        "textured_output_path": textured_output_path,
        "mtl_path": mtl_path,
        "albedo_path": albedo_path,
        "output_color_mesh_path": output_color_mesh_path,
    }


def generate_obj_from_prompt_image(
    image_path: str,
    prompt: str,
    output_obj_path: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    session_dir: Optional[str] = None,
    require_gemini_cutout: bool = False,
    api_key: Optional[str] = None,
    conda_env: str = "hunyuan",
) -> Dict[str, Any]:
    """Generate colored 3D OBJ from 2D image using Hunyuan3D-2.1 workflow.
    
    Workflow:
    1. Generate object cutout with Gemini API (white background)
    2. Run step1_shape.py via subprocess -> untextured mesh
    3. Run step2_paint.py via subprocess -> colored mesh
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_obj_path)), exist_ok=True)
    if session_dir:
        os.makedirs(session_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    # Step 0: Detect object bbox if not provided
    if bbox is None:
        bbox = detect_bbox_with_owlv2(image_path, prompt)
        if bbox is None:
            raise RuntimeError("Could not detect object bbox with OWLv2 for 2D->3D generation.")

    crop = _crop_with_padding(image, bbox)
    
    # Step 1: Generate Gemini object cutout with white background
    gemini_cutout_path = None
    if session_dir:
        gemini_cutout_path = os.path.join(session_dir, "gemini_object_cutout.png")
    else:
        gemini_cutout_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_object_cutout.png")
    
    api_key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    
    if not api_key and require_gemini_cutout:
        raise RuntimeError("Gemini API key is required but not found.")
    
    if api_key:
        print(f"Generating Gemini object cutout...")
        try:
            generate_object_cutout_with_gemini(
                api_key=api_key,
                input_image_path=_save_temp_crop(crop, session_dir),
                object_prompt=prompt,
                output_image_path=gemini_cutout_path,
                width=max(512, crop.width),
                height=max(512, crop.height),
            )
            print(f"✓ Gemini cutout saved to: {gemini_cutout_path}")
            
            # Verify the cutout was created
            if not os.path.exists(gemini_cutout_path):
                raise FileNotFoundError(f"Gemini didn't create output file")
            
            shape_input_image = gemini_cutout_path
        except Exception as e:
            if require_gemini_cutout:
                raise RuntimeError(f"Gemini cutout generation failed: {e}")
            print(f"Gemini cutout failed ({e}), using crop as fallback")
            shape_input_image = _save_temp_crop(crop, session_dir)
    else:
        if require_gemini_cutout:
            raise RuntimeError("Gemini cutout is required but GEMINI_API_KEY is not set.")
        shape_input_image = _save_temp_crop(crop, session_dir)

    # Step 2: Run Hunyuan3D step1_shape.py to generate untextured mesh
    print(f"Running Hunyuan3D step1 (shape generation)...")
    ok, msg = _run_step1_shape_subprocess(
        input_image_path=shape_input_image,
        output_mesh_path=output_obj_path,
        conda_env=conda_env,
    )
    
    if not ok:
        raise RuntimeError(f"Step1 shape generation failed: {msg}")
    
    print(f"✓ Untextured mesh generated: {output_obj_path}")
    
    # Step 3: Run Hunyuan3D step2_paint.py to paint the mesh
    print(f"Running Hunyuan3D step2 (texture painting)...")
    output_dir = os.path.dirname(os.path.abspath(output_obj_path))
    ok, msg = _run_step2_paint_subprocess(
        mesh_path=output_obj_path,
        image_path=shape_input_image,
        output_folder=output_dir,
        conda_env=conda_env,
    )
    
    if not ok:
        raise RuntimeError(f"Step2 paint generation failed: {msg}")
    
    print(f"✓ Painted mesh completed in: {output_dir}")

    textured_outputs = _collect_textured_outputs(output_dir=output_dir, untextured_obj_path=output_obj_path)
    output_glb_path = textured_outputs.get("output_glb_path")
    textured_output_path = textured_outputs.get("textured_output_path")
    mtl_path = textured_outputs.get("mtl_path")
    albedo_path = textured_outputs.get("albedo_path")
    output_color_mesh_path = textured_outputs.get("output_color_mesh_path")

    if output_color_mesh_path:
        print(f"✓ Color mesh selected: {output_color_mesh_path}")
    else:
        print("Warning: no textured mesh artifact detected; pipeline may fall back to flat color.")
    
    return {
        "output_obj_path": output_obj_path,
        "output_glb_path": output_glb_path,
        "textured_output_path": textured_output_path,
        "mtl_path": mtl_path,
        "albedo_path": albedo_path,
        "output_color_mesh_path": output_color_mesh_path,
        "gemini_object_cutout_path": gemini_cutout_path,
        "bbox": bbox,
        "conda_env": conda_env,
    }


def generate_obj_from_cutout_image(
    cutout_image_path: str,
    output_obj_path: str,
    session_dir: Optional[str] = None,
    conda_env: str = "hunyuan",
) -> Dict[str, Any]:
    """Generate colored 3D object from an existing object cutout image.

    This path skips all Gemini usage and is intended for debug reruns from a saved session.
    """
    if not os.path.exists(cutout_image_path):
        raise FileNotFoundError(f"Cutout image not found: {cutout_image_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_obj_path)), exist_ok=True)
    if session_dir:
        os.makedirs(session_dir, exist_ok=True)

    shape_input_image = os.path.abspath(cutout_image_path)

    print("Running Hunyuan3D step1 (shape generation) from existing cutout...")
    ok, msg = _run_step1_shape_subprocess(
        input_image_path=shape_input_image,
        output_mesh_path=output_obj_path,
        conda_env=conda_env,
    )
    if not ok:
        raise RuntimeError(f"Step1 shape generation failed: {msg}")

    print(f"✓ Untextured mesh generated: {output_obj_path}")

    print("Running Hunyuan3D step2 (texture painting) from existing cutout...")
    output_dir = os.path.dirname(os.path.abspath(output_obj_path))
    ok, msg = _run_step2_paint_subprocess(
        mesh_path=output_obj_path,
        image_path=shape_input_image,
        output_folder=output_dir,
        conda_env=conda_env,
    )
    if not ok:
        raise RuntimeError(f"Step2 paint generation failed: {msg}")

    print(f"✓ Painted mesh completed in: {output_dir}")

    textured_outputs = _collect_textured_outputs(output_dir=output_dir, untextured_obj_path=output_obj_path)
    output_glb_path = textured_outputs.get("output_glb_path")
    textured_output_path = textured_outputs.get("textured_output_path")
    mtl_path = textured_outputs.get("mtl_path")
    albedo_path = textured_outputs.get("albedo_path")
    output_color_mesh_path = textured_outputs.get("output_color_mesh_path")

    if output_color_mesh_path:
        print(f"✓ Color mesh selected: {output_color_mesh_path}")
    else:
        print("Warning: no textured mesh artifact detected; pipeline may fall back to flat color.")

    return {
        "output_obj_path": output_obj_path,
        "output_glb_path": output_glb_path,
        "textured_output_path": textured_output_path,
        "mtl_path": mtl_path,
        "albedo_path": albedo_path,
        "output_color_mesh_path": output_color_mesh_path,
        "gemini_object_cutout_path": shape_input_image,
        "bbox": None,
        "conda_env": conda_env,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate colored 3D object with Hunyuan3D-2.1 workflow")
    parser.add_argument("--image", default="", help="Input image path")
    parser.add_argument("--prompt", default="", help="Prompt describing the object")
    parser.add_argument("--output", required=True, help="Output OBJ path")
    parser.add_argument("--cutout", default="", help="Existing cutout image to skip Gemini and run only Hunyuan steps")
    parser.add_argument("--session-dir", default="", help="Optional debug output folder")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, help="Optional bbox x1 y1 x2 y2")
    parser.add_argument("--require-gemini-cutout", action="store_true", help="Fail if Gemini cutout is not used")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key")
    parser.add_argument("--conda-env", default="hunyuan", help="Conda environment with Hunyuan3D")
    return parser.parse_args()


def main():
    args = _parse_args()
    bbox = tuple(args.bbox) if args.bbox is not None else None
    
    try:
        if (args.cutout or "").strip():
            result = generate_obj_from_cutout_image(
                cutout_image_path=args.cutout.strip(),
                output_obj_path=args.output,
                session_dir=args.session_dir or None,
                conda_env=args.conda_env,
            )
        else:
            if not (args.image or "").strip():
                raise ValueError("--image is required when --cutout is not provided")
            if not (args.prompt or "").strip():
                raise ValueError("--prompt is required when --cutout is not provided")
            result = generate_obj_from_prompt_image(
                image_path=args.image,
                prompt=args.prompt,
                output_obj_path=args.output,
                bbox=bbox,
                session_dir=args.session_dir or None,
                require_gemini_cutout=bool(args.require_gemini_cutout),
                api_key=(args.api_key or "").strip() or None,
                conda_env=args.conda_env,
            )
        
        print("\n" + "="*60)
        print("✓ 3D Generation Complete!")
        print("="*60)
        print(f"Output OBJ: {result['output_obj_path']}")
        if result.get('output_glb_path'):
            print(f"Output GLB: {result['output_glb_path']}")
        print(f"BBox: {result['bbox']}")
        if result.get('gemini_object_cutout_path'):
            print(f"Gemini Cutout: {result['gemini_object_cutout_path']}")
        print("="*60)
    
    except Exception as e:
        print(f"\n✗ Generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



