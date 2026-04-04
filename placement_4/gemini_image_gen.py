import os
import json
import base64
import argparse
import time
import urllib.request
import urllib.parse
import urllib.error

from PIL import Image


DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-image-preview")
DEFAULT_GEMINI_IMAGE_MODEL_FALLBACKS = [
    "gemini-2.5-flash-image-preview",
    "gemini-2.5-flash-image",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-exp-image-generation",
]


def generate_diffusion_image_with_gemini(
    api_key,
    input_image_path,
    object_prompt,
    output_image_path,
    width,
    height,
    model=DEFAULT_GEMINI_MODEL,
    fallback_models=None,
    max_retries=3,
    initial_retry_delay_sec=2.0,
    debug_response_path=None,
):
    """Calls Gemini image-edit API and saves the generated image to output_image_path."""
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    with open(input_image_path, "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        f"Add '{object_prompt}' to this image. "
        "Do not change anything else in the scene. "
        f"Keep resolution exactly {width}x{height}. "
        "Return only the edited image."
    )

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                ]
            }
        ]
    }

    model_fallbacks = fallback_models or DEFAULT_GEMINI_IMAGE_MODEL_FALLBACKS

    candidate_models = []
    for m in [model] + list(model_fallbacks):
        m_clean = (m or "").strip()
        if m_clean and m_clean not in candidate_models:
            candidate_models.append(m_clean)

    response = None
    last_error = None
    for model_name in candidate_models:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?"
            f"key={urllib.parse.quote(api_key)}"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    response = json.loads(resp.read().decode("utf-8"))
                print(f"Gemini request succeeded with model: {model_name}")
                break
            except urllib.error.HTTPError as e:
                err_text = ""
                try:
                    err_text = e.read().decode("utf-8", errors="replace")
                except Exception:
                    err_text = str(e)
                last_error = f"HTTP {e.code} for model {model_name}: {err_text}"

                # Invalid model/request; try next fallback model immediately.
                if e.code in (400, 404):
                    break

                # Quota/rate-limit; retry same model with exponential backoff.
                if e.code == 429 and attempt < max_retries:
                    wait_s = initial_retry_delay_sec * (2 ** (attempt - 1))
                    print(
                        f"429 rate/quota limit for model {model_name}. "
                        f"Retrying in {wait_s:.1f}s (attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(wait_s)
                    continue

                if e.code == 429:
                    break

                raise RuntimeError(f"Gemini API call failed: {last_error}")
            except Exception as e:
                last_error = f"Model {model_name} failed: {e}"
                break

        if response is not None:
            break

    if response is None:
        quota_hint = (
            " If this is HTTP 429/RESOURCE_EXHAUSTED, your key has hit quota or billing limits. "
            "Check billing/quota, wait for reset, reduce request rate, or switch to another key/project."
        )
        raise RuntimeError(
            "Gemini API call failed for all candidate models. "
            f"Tried: {candidate_models}. Last error: {last_error}.{quota_hint}"
        )

    if debug_response_path:
        with open(debug_response_path, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        print(f"Saved raw Gemini response to {debug_response_path}")

    candidates = response.get("candidates", [])
    image_data_b64 = None
    image_mime = "image/png"
    for cand in candidates:
        parts = cand.get("content", {}).get("parts", [])
        for part in parts:
            # Gemini responses may use either inlineData (camelCase) or inline_data (snake_case).
            inline_data = part.get("inline_data") or part.get("inlineData")
            if inline_data and inline_data.get("data"):
                image_data_b64 = inline_data.get("data")
                image_mime = inline_data.get("mime_type") or inline_data.get("mimeType") or "image/png"
                break
        if image_data_b64:
            break

    if not image_data_b64:
        candidate_shapes = []
        for idx, cand in enumerate(candidates):
            parts = cand.get("content", {}).get("parts", [])
            part_keys = [sorted(list(p.keys())) for p in parts]
            candidate_shapes.append({"candidate": idx, "parts": part_keys})
        raise RuntimeError(
            "Gemini response did not include an edited image. "
            f"Response keys: {list(response.keys())}. Candidate part keys: {candidate_shapes}. "
            "Use --debug-response-json <path> to inspect full response."
        )

    image_bytes_out = base64.b64decode(image_data_b64)
    with open(output_image_path, "wb") as f:
        f.write(image_bytes_out)

    with Image.open(output_image_path) as generated:
        if generated.size != (width, height):
            generated = generated.resize((width, height), Image.LANCZOS)
            generated.save(output_image_path)

    print(
        f"Gemini edited image saved to {output_image_path} "
        f"(mime={image_mime}, size={width}x{height})"
    )

    return output_image_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an edited image using Gemini image API."
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--object", required=True, help="Object to add to image")
    parser.add_argument("--output", required=True, help="Path to save edited image")
    parser.add_argument("--width", type=int, default=1280, help="Target output width")
    parser.add_argument("--height", type=int, default=720, help="Target output height")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per model for transient errors")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Initial retry delay in seconds (exponential backoff)")
    parser.add_argument(
        "--debug-response-json",
        default="",
        help="Optional path to save raw Gemini JSON response for debugging",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    api_key = (args.api_key or "").strip()
    if not api_key:
        raise SystemExit("Missing API key. Set GEMINI_API_KEY or pass --api-key.")

    generate_diffusion_image_with_gemini(
        api_key=api_key,
        input_image_path=args.input,
        object_prompt=args.object,
        output_image_path=args.output,
        width=args.width,
        height=args.height,
        model=args.model,
        max_retries=max(1, args.max_retries),
        initial_retry_delay_sec=max(0.0, args.retry_delay),
        debug_response_path=(args.debug_response_json or "").strip() or None,
    )


if __name__ == "__main__":
    main()


# python gemini_image_gen.py \
#   --input bench_added.png \
#   --object "car" \
#   --output gemini_diffusion_added.png \
#   --width 1280 \
#   --height 720