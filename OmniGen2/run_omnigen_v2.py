import torch
import os
from PIL import Image
from diffusers.utils import load_image

# --- 1. EXACT IMPORT FROM YOUR INFERENCE.PY ---
try:
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this from the 'OmniGen2' root folder.")
    exit()

# --- Configuration ---
INPUT_IMAGE = "input_images/background2.jpeg" 
OUTPUT_IMAGE = "output_images/output4.png"
MODEL_NAME = "OmniGen2/OmniGen2" # HuggingFace ID

# V2 Prompt: Explicitly reference the image for editing
# PROMPT = "<img><|image_1|></img> Add a cozy sectional sofa set arranged in the center of the room."

# PROMPT = (
#     "<img><|image_1|></img> Detect the empty floor space and transform this room into a "
#     "modern living room. Place a beige L-shaped sectional sofa in the center facing forward. "
#     "Add a rectangular wooden coffee table in front of the sofa. Place a tall potted fiddle-leaf fig plant "
#     "in the back left corner. Lay down a textured white rug under the furniture setup. "
#     "Keep the original windows and walls unchanged."
# )

# PROMPT = (
#     "<img><|image_1|></img> Edit this image to fully furnish the room: "
#     "1. Place a gray fabric sofa on the left wall. "
#     "2. Add a sleek TV stand with a television on the right wall opposite the sofa. "
#     "3. Place a round glass coffee table in the middle. "
#     "4. Add a cozy armchair in the foreground corner. "
#     "Ensure realistic lighting and shadows from the window."
# )

# PROMPT = (
#     "<img><|image_1|></img> Transform this into a high-end interior design photograph. "
#     "Add a beige linen sectional sofa in the center with plush throw pillows. "
#     "Include a rustic oak coffee table and a soft wool rug. "
#     "Lighting: Warm golden hour sunlight streaming through the window, creating soft shadows. "
#     "Style: 8k resolution, photorealistic, cinematic lighting, cozy atmosphere, highly detailed textures."
# )

PROMPT = "<img><|image_1|></img> Add a wooden table in the center of the empty room."

def main():
    print(f"Loading OmniGen V2 model: {MODEL_NAME}...")
    
    # 2. Load Pipeline using the correct class
    # We use bfloat16 for memory efficiency (Crucial for 16GB VRAM)
    pipe = OmniGen2Pipeline.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # --- FIX FOR "AttributeError: enable_teacache" ---
    # The local pipeline expects this attribute, but the downloaded model doesn't have it.
    # We manually set it to False to prevent the crash.
    pipe.transformer.enable_teacache = False
    # -------------------------------------------------

    # 3. Memory Optimization (REQUIRED for 16GB VRAM)
    # This moves the model to CPU when not processing to save GPU space
    pipe.enable_model_cpu_offload()
    
    # Check input
    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: Could not find {INPUT_IMAGE}")
        return

    image = load_image(INPUT_IMAGE)
    w, h = image.size
    
    # Align dimensions to multiples of 16
    w = w - (w % 16)
    h = h - (h % 16)
    
    print(f"Processing: {INPUT_IMAGE} ({w}x{h})")
    print(f"Prompt: {PROMPT}")

    # 4. Generate
    # Note: If 'input_images' fails, some V2 versions use 'image' or 'images'
    # The dictionary key usage suggests standard diffusers syntax might apply
    generator = torch.Generator(device="cpu").manual_seed(42)

    result = pipe(
        prompt=PROMPT,
        input_images=[image],      # V2 list format
        height=h,
        width=w,
        text_guidance_scale=3.0,   # <--- Renamed from guidance_scale
        image_guidance_scale=1.6,   # <--- Renamed from img_guidance_scale
        #max_input_image_size=1024, # Limit size to prevent OOM
        generator=generator
    )

    # 5. Save
    # The pipeline usually returns a standard Diffusers output object
    output = result.images[0]
    output.save(OUTPUT_IMAGE)
    print(f"Success! Image saved as {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()